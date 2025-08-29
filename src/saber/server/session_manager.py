"""
SessionManager implementation for SABER domain server.

The SessionManager serves as the central orchestrator for managing client sessions
and coordinating all server components. REST API functionality is handled by SessionAPI.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from ..logging_config import (
    get_cleanup_logger,
    get_session_manager_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
    log_session_end,
)
from .api.session_mcp_api import SessionMCPAPI
from .api.session_rest_api import SessionRestAPI
from .base import Action, CommandResult
from .benchmarks.benchmark_info import BenchmarkInfo
from .benchmarks.benchmark_manager import BenchmarkManager
from .episodes.constants import EpisodeResponseKeys, EpisodeTerminationReason
from .episodes.episode_manager import EpisodeManager
from .evaluation.evaluation_manager import EvaluationManager
from .execution.cleanup.cleanup_manager import ContainerCleanupManager
from .execution.cleanup.cleanup_reason import CleanupReason
from .execution.execution_manager import ExecutionManager
from .execution.sandbox.permanent_environment_manager import PermanentEnvironmentManager
from .policy.policy_manager import PolicyDocument, PolicyManager

logger = get_session_manager_logger(__name__)
cleanup_logger = get_cleanup_logger(__name__)


class ClientSession(BaseModel):
    """Represents an active client session."""

    model_config = ConfigDict()

    session_id: str = Field(..., description="Unique session identifier")
    client_id: str = Field(..., description="Client identifier")
    current_episode_id: Optional[str] = Field(None, description="Current active episode ID")
    current_task_id: Optional[str] = Field(None, description="Current active task ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    is_active: bool = Field(default=True, description="Whether session is active")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context data")

    @field_serializer("created_at", "last_activity")
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime fields to ISO format."""
        return value.isoformat()

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.utcnow()


class SessionManager:
    """
    Central orchestrator for SABER domain server.

    Manages multiple concurrent client sessions and coordinates all server components.
    REST API functionality is delegated to SessionAPI.
    """

    def __init__(
        self,
        domain_name: str,
        config_dir: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        mcp_host: str = "0.0.0.0",
        mcp_port: int = 3001,
        session_timeout_minutes: int = 30,
        cleanup_interval_minutes: int = 5,
    ):
        """
        Initialize the SessionManager.

        Args:
            domain_name: Name of the security domain (e.g., 'malware_classification')
            config_dir: Path to configuration directory containing tasks.yaml and other config files
            host: REST server host address
            port: REST server port
            mcp_host: MCP server host address
            mcp_port: MCP server port
            session_timeout_minutes: Minutes of inactivity before a session times out
            cleanup_interval_minutes: Minutes between cleanup checks
        """
        self.domain_name = domain_name
        self.config_dir = config_dir
        self.host = host
        self.port = port
        self.mcp_host = mcp_host
        self.mcp_port = mcp_port
        self.session_timeout_minutes = session_timeout_minutes
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.active_sessions: Dict[str, ClientSession] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()

        # Initialize server components
        logger.info(f"Initializing SessionManager for domain '{domain_name}' with config_dir '{config_dir}'")

        self.benchmark_manager = BenchmarkManager(domain_name, config_dir)
        self.episode_manager = EpisodeManager()
        self.execution_manager = ExecutionManager(config_dir)
        self.policy_manager = PolicyManager(domain_name)
        self.evaluation_manager = EvaluationManager()
        self.cleanup_manager = ContainerCleanupManager(self.execution_manager._sandbox_manager)

        # Initialize permanent environment manager with enhanced logging config
        permanent_config = {
            "domain": domain_name,
            "config_dir": config_dir,
            "logs_directory": "/app/logs",  # Ensure this is mounted to host
            "enable_logging": True,
        }
        self.permanent_environment_manager = PermanentEnvironmentManager(permanent_config)

        # Initialize protocol handlers
        self.rest_api = SessionRestAPI(self, host, port)
        self.mcp_api = SessionMCPAPI(self, mcp_host, mcp_port)

        logger.info(
            f"SessionManager initialized for domain '{domain_name}' on REST:{host}:{port}, MCP:{mcp_host}:{mcp_port}"
        )

    async def start_server(self) -> None:
        """Start both REST and MCP servers concurrently with session cleanup."""
        import asyncio

        # Start permanent environment if configured
        await self._start_permanent_environment()

        # Start the session cleanup task
        self.cleanup_task = asyncio.create_task(self._session_cleanup_loop())

        # Start both servers concurrently using asyncio tasks
        rest_task = asyncio.create_task(self.rest_api.start_server())
        mcp_task = asyncio.create_task(self.mcp_api.start_mcp_server())

        # Wait for both to complete (they run indefinitely)
        await asyncio.gather(rest_task, mcp_task, self.cleanup_task)

    async def shutdown(self) -> None:
        """Shutdown the SessionManager and cleanup resources."""
        logger.info(f"Shutting down SessionManager for domain {self.domain_name}")

        # Signal shutdown to stop cleanup loop
        self.shutdown_event.set()

        # Cancel cleanup task if running
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Stop permanent environment
        if self.permanent_environment_manager.is_running():
            logger.info("Stopping permanent environment...")
            try:
                self.permanent_environment_manager.stop_permanent_environment()
                logger.info("Permanent environment stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping permanent environment: {e}")

        # Shutdown MCP server first
        await self.mcp_api.shutdown_mcp_server()

        # Cleanup all active sessions
        session_ids = list(self.active_sessions.keys())
        cleanup_logger.info(
            f"Shutdown cleanup initiated: active_sessions_count={len(session_ids)}, session_list={session_ids}"
        )
        for session_id in session_ids:
            cleanup_logger.info(f"Terminating session {session_id} during shutdown")
            await self.terminate_session(session_id)

        logger.info("SessionManager shutdown complete")

    async def create_session(self, client_id: str) -> ClientSession:
        """
        Create a new client session.

        Args:
            client_id: Identifier for the connecting client

        Returns:
            ClientSession object for the new session

        Raises:
            HTTPException: If the server is not ready for session creation
        """
        # Check if the sandbox manager is ready before creating sessions
        if not self.execution_manager.is_sandbox_ready():
            raise HTTPException(status_code=503, detail="Server is still initializing. Please try again in a moment.")

        session_id = str(uuid.uuid4())
        session = ClientSession(
            session_id=session_id, client_id=client_id, current_episode_id=None, current_task_id=None
        )

        self.active_sessions[session_id] = session

        # Log session creation with evaluation manager (ignore failures)
        try:
            await self.evaluation_manager.log_session_start(session_id, client_id)
        except Exception as e:
            logger.warning(f"Failed to log session start: {str(e)}")

        logger.info(f"Session created {session_id} for client {client_id}")
        return session

    async def terminate_session(self, session_id: str) -> None:
        """
        Terminate a client session and cleanup resources.

        Args:
            session_id: ID of the session to terminate
        """
        log_session_end(logger, session_id, "SessionManager termination")
        session = self._get_session(session_id)
        session.is_active = False

        # End any active episode
        if session.current_episode_id:
            try:
                logger.info(
                    f"Ending active episode {session.current_episode_id} during session termination "
                    f"for session {session_id}"
                )
                self.episode_manager.end_episode(session_id, EpisodeTerminationReason.SESSION_TERMINATED)
                session.current_episode_id = None
                session.current_task_id = None
            except Exception as e:
                logger.warning(f"Error ending episode during session termination: {str(e)}")

        # Log session end with evaluation manager (ignore failures)
        try:
            await self.evaluation_manager.log_session_end(session_id)
        except Exception as e:
            logger.warning(f"Failed to log session end: {str(e)}")

        # Cleanup execution resources (Docker containers) using unified cleanup manager
        try:
            log_operation_start(logger, "Unified container cleanup", session_id)
            cleanup_success = self.cleanup_manager.cleanup_session(
                session_id, CleanupReason.SESSION_TERMINATED, {"manual_termination": True}
            )
            if cleanup_success:
                log_operation_success(logger, "Unified container cleanup", session_id)
            else:
                logger.warning(f"Unified container cleanup reported failure for session {session_id}")
        except Exception as e:
            log_operation_failure(logger, "Unified container cleanup", str(e), session_id)

        # Remove session from active sessions
        logger.info(f"Session {session_id} removed from active sessions")
        del self.active_sessions[session_id]

        logger.info(f"Terminated session {session_id}")

    async def start_episode(self, session_id: str, task_id: str) -> Any:
        """
        Initialize episode for a task.

        Args:
            session_id: ID of the client session
            task_id: ID of the task to start

        Returns:
            Episode object for the started episode
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Get the task object to access its configuration
        task = self.benchmark_manager.get_task(task_id)

        # Start episode through episode manager
        episode = self.episode_manager.start_episode(
            session_id=session_id, task_id=task_id, initial_context=task.initial_context.copy()
        )

        # Configure execution manager with task object only
        self.execution_manager.configure_for_task(session_id, task)

        # Configure policy manager with task object
        self.policy_manager.configure_for_task(session_id, task)

        # Configure episode manager with task object
        self.episode_manager.configure_for_task(session_id, task)

        session.current_episode_id = episode.episode_id
        session.current_task_id = task_id

        # Log episode start with evaluation manager (ignore failures)
        try:
            await self.evaluation_manager.log_episode_start(session_id, episode.episode_id, task_id)
        except Exception as e:
            logger.warning(f"Failed to log episode start: {e}")

        logger.info(f"Started episode {episode.episode_id} for task {task_id} in session {session_id}")
        return episode

    def get_benchmark_info(self) -> BenchmarkInfo:
        """
        Get complete benchmark information for client-side orchestration.

        Returns:
            BenchmarkInfo object with all tasks and their episode attempt configurations
        """
        return self.benchmark_manager.get_benchmark_info()

    async def end_episode(
        self, session_id: str, reason: str = EpisodeTerminationReason.COMPLETED, result: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End the current episode for a session.

        Args:
            session_id: ID of the client session
            reason: Reason for episode termination (use EpisodeTerminationReason enum values)
            result: Optional result/submission from the episode (e.g., captured flag)

        Returns:
            Dictionary with episode termination information
        """
        session = self._get_session(session_id)
        session.update_activity()

        completed_task_id = session.current_task_id
        success = EpisodeTerminationReason.is_success(reason)

        # End episode through episode manager, passing the result
        self.episode_manager.end_episode(session_id, reason, result)

        # Clear current episode and task from session
        session.current_episode_id = None
        session.current_task_id = None

        logger.info(f"Ended episode for session {session_id} with reason: {reason}")

        # Build response info - client handles orchestration now
        response: Dict[str, Any] = {
            EpisodeResponseKeys.EPISODE_ENDED.value: True,
            EpisodeResponseKeys.PREVIOUS_TASK_ID.value: completed_task_id,
            EpisodeResponseKeys.SUCCESS.value: success,
            EpisodeResponseKeys.REASON.value: reason,
        }

        return response

    async def execute_action(self, session_id: str, action: Action) -> CommandResult:
        """
        Execute action for MCP integration.

        Args:
            session_id: ID of the client session
            action: Action object containing command and parameters

        Returns:
            CommandResult with execution results
        """
        session = self._get_session(session_id)
        session.update_activity()

        if not session.current_episode_id:
            return CommandResult.error_result(error="No active episode in session")

        try:
            log_operation_start(
                logger, "Action execution", session_id, tool=action.tool_name, episode=session.current_episode_id
            )

            context = {"session_id": session_id}
            command_result = await self.execution_manager.step(action, context)

            if command_result.success:
                log_operation_success(logger, "Action execution", session_id, tool=action.tool_name)
            else:
                log_operation_failure(logger, "Action execution", command_result.error or "Unknown error", session_id)
            step_result = self.episode_manager.step(session_id, action, command_result)

            # Log action with evaluation manager (ignore failures)
            try:
                await self.evaluation_manager.log_action(
                    session_id=session_id, episode_id=session.current_episode_id, action=action, result=command_result
                )
            except Exception as e:
                logger.warning(f"Failed to log action: {e}")

            # End episode if step indicates completion OR if EpisodeManager indicates termination
            if step_result.step.done:
                self.episode_manager.end_episode(session_id, EpisodeTerminationReason.COMPLETED)
                session.current_episode_id = None
                session.current_task_id = None
                # Add termination metadata to command result
                if not hasattr(command_result, "metadata") or command_result.metadata is None:
                    command_result.metadata = {}
                command_result.metadata["episode_terminated"] = True
                command_result.metadata["termination_reason"] = EpisodeTerminationReason.COMPLETED
                try:
                    await self.evaluation_manager.log_episode_end(session_id, EpisodeTerminationReason.COMPLETED)
                except Exception as e:
                    logger.warning(f"Failed to log episode end: {e}")
            elif step_result.should_terminate:
                termination_reason = step_result.termination_reason or EpisodeTerminationReason.TERMINATED
                self.episode_manager.end_episode(session_id, termination_reason)
                session.current_episode_id = None
                session.current_task_id = None
                # Add termination metadata to command result
                if not hasattr(command_result, "metadata") or command_result.metadata is None:
                    command_result.metadata = {}
                command_result.metadata["episode_terminated"] = True
                command_result.metadata["termination_reason"] = termination_reason
                try:
                    await self.evaluation_manager.log_episode_end(session_id, termination_reason)
                except Exception as e:
                    logger.warning(f"Failed to log episode end: {e}")
                logger.info(f"🏗️ Episode ended due to termination condition: {termination_reason}")

            return command_result

        except Exception as e:
            logger.error(f"Command execution failed in session {session_id}: {e}")

            # Remove episode tracking and trigger immediate cleanup on any error
            try:
                self.episode_manager.remove_episode_on_error(session_id, e)
                session.current_episode_id = None
                session.current_task_id = None
                logger.info("Episode removed from tracking due to error")

                # Trigger immediate container cleanup using unified cleanup manager
                cleanup_success = self.cleanup_manager.cleanup_session(
                    session_id, CleanupReason.ERROR_TRIGGERED, {"error": str(e), "error_type": type(e).__name__}
                )
                if cleanup_success:
                    logger.info("Error-triggered container cleanup completed successfully")
                else:
                    logger.warning("Error-triggered container cleanup reported failure")

            except Exception as cleanup_error:
                logger.error(f"Failed to handle error cleanup: {cleanup_error}")

            return CommandResult.error_result(error=str(e))

    async def get_current_task(self, session_id: str) -> Any:
        """
        Get current task object with episode context.

        Args:
            session_id: ID of the client session

        Returns:
            Task object for the current episode
        """
        session = self._get_session(session_id)
        session.update_activity()

        if not session.current_episode_id:
            raise HTTPException(status_code=400, detail="No active episode in session")

        # Get current episode and task info from episode manager
        episode = self.episode_manager.get_current_episode(session_id)
        if not episode:
            raise HTTPException(status_code=400, detail="No active episode found")

        task = self.benchmark_manager.get_task(episode.task_id)
        return task

    def get_policy(self, session_id: str) -> PolicyDocument:
        """
        Get domain policy document.

        Args:
            session_id: ID of the client session

        Returns:
            PolicyDocument for the domain
        """
        session = self._get_session(session_id)
        session.update_activity()

        return self.policy_manager.get_policy()

    def _get_session(self, session_id: str) -> ClientSession:
        """
        Get session by ID with validation.

        Args:
            session_id: ID of the session to retrieve

        Returns:
            ClientSession object

        Raises:
            HTTPException: If session not found or inactive
        """
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = self.active_sessions[session_id]
        if not session.is_active:
            raise HTTPException(status_code=400, detail="Session is not active")

        return session

    async def _session_cleanup_loop(self) -> None:
        """Periodic cleanup of inactive sessions."""
        logger.info(
            f"Starting session cleanup loop - timeout: {self.session_timeout_minutes}min, "
            f"check interval: {self.cleanup_interval_minutes}min"
        )

        while not self.shutdown_event.is_set():
            try:
                await self._cleanup_inactive_sessions()

                # Wait for cleanup interval or shutdown signal
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=self.cleanup_interval_minutes * 60)
                    # If shutdown event is set, exit the loop
                    break
                except asyncio.TimeoutError:
                    # Timeout is expected - continue with next cleanup cycle
                    continue

            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}")
                # Wait a bit before retrying to avoid tight error loops
                await asyncio.sleep(30)

    async def _cleanup_inactive_sessions(self) -> None:
        """Check for and cleanup inactive sessions."""
        current_time = datetime.utcnow()
        timeout_threshold = timedelta(minutes=self.session_timeout_minutes)
        sessions_to_cleanup = []

        for session_id, session in self.active_sessions.items():
            time_since_activity = current_time - session.last_activity

            if time_since_activity > timeout_threshold:
                logger.info(
                    f"Session {session_id} timed out after " f"{time_since_activity.total_seconds():.1f}s of inactivity"
                )
                sessions_to_cleanup.append(session_id)

        # Cleanup identified sessions
        for session_id in sessions_to_cleanup:
            try:
                cleanup_logger.info(f"Cleaning up inactive session {session_id} due to timeout")
                await self.terminate_session(session_id)
                logger.info(f"Cleaned up inactive session {session_id}")
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions."""
        current_time = datetime.utcnow()
        stats: Dict[str, Any] = {
            "total_sessions": len(self.active_sessions),
            "timeout_minutes": self.session_timeout_minutes,
            "cleanup_interval_minutes": self.cleanup_interval_minutes,
            "sessions": [],
        }

        for session_id, session in self.active_sessions.items():
            time_since_activity = current_time - session.last_activity
            stats["sessions"].append(
                {
                    "session_id": session_id,
                    "client_id": session.client_id,
                    "current_episode_id": session.current_episode_id,
                    "current_task_id": session.current_task_id,
                    "uptime_seconds": (current_time - session.created_at).total_seconds(),
                    "time_since_activity_seconds": time_since_activity.total_seconds(),
                    "is_active": session.is_active,
                }
            )

        return stats

    def get_episode_config(self, session_id: str) -> Dict[str, Any]:
        """
        Get episode configuration for monitoring.

        Args:
            session_id: Session ID

        Returns:
            Episode configuration including max_steps and other settings from the actual task
        """
        session = self._get_session(session_id)

        # Delegate to BenchmarkManager which has all the task configuration logic
        if session.current_task_id:
            try:
                episode_config = self.benchmark_manager.get_episode_config(session.current_task_id)
                # Add session-specific info
                episode_config["session_id"] = session_id
                return episode_config
            except Exception as e:
                logger.warning(f"Failed to get episode configuration for {session.current_task_id}: {e}")

        # Return empty config if no current task or on error
        return {"session_id": session_id}

    def should_terminate_episode(self, session_id: str) -> tuple[bool, str]:
        """
        Check if the current episode should be terminated.

        Args:
            session_id: Session ID to check

        Returns:
            Tuple of (should_terminate, reason)
        """
        return self.episode_manager.should_terminate_episode(session_id)

    async def _start_permanent_environment(self) -> None:
        """Start permanent environment if configured."""
        permanent_env_name = self.benchmark_manager.config_loader.get_permanent_environment()
        if not permanent_env_name:
            logger.info("No permanent environment configured")
            return

        try:
            logger.info(f"Starting permanent environment: {permanent_env_name}")

            # Load permanent environment specification
            if self.execution_manager._environment_loader is None:
                raise RuntimeError("Environment loader is not initialized")
            permanent_env_spec = self.execution_manager._environment_loader.load_permanent_environment(
                permanent_env_name
            )

            # Start the permanent environment
            self.permanent_environment_manager.start_permanent_environment(permanent_env_spec)

            logger.info(f"Permanent environment '{permanent_env_name}' started successfully")

        except Exception as e:
            logger.error(f"Failed to start permanent environment '{permanent_env_name}': {e}")
            raise
