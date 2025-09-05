"""
SessionManager implementation for SABER domain server.

The SessionManager serves as the central orchestrator for managing client sessions
and coordinating all server components. REST API functionality is handled by SessionAPI.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from ..api_models import EpisodeEndResponse
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
from .episodes.constants import EpisodeTerminationReason
from .episodes.episode_manager import EpisodeManager
from .evaluation.evaluation_manager import EvaluationManager
from .execution.cleanup.cleanup_reason import CleanupReason
from .execution.execution_manager import ExecutionManager
from .policy.policy_manager import PolicyDocument, PolicyManager

logger = get_session_manager_logger(__name__)
cleanup_logger = get_cleanup_logger(__name__)


class ClientSession(BaseModel):
    """Represents an active client session with support for multiple episodes."""

    model_config = ConfigDict()

    session_id: str = Field(..., description="Unique session identifier")
    client_id: str = Field(..., description="Client identifier")

    # Multi-episode support
    active_episode_ids: List[str] = Field(default_factory=list, description="Currently active episode IDs")
    episode_history: List[str] = Field(
        default_factory=list, description="All completed episode IDs in chronological order"
    )

    # Task orchestration support
    task_queue: List[str] = Field(default_factory=list, description="Queued tasks for orchestration")

    # Session metadata
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

    def add_active_episode(self, episode_id: str) -> None:
        """Add an episode to the active episodes list."""
        if episode_id not in self.active_episode_ids:
            self.active_episode_ids.append(episode_id)
            self.update_activity()

    def remove_active_episode(self, episode_id: str) -> None:
        """Remove an episode from active episodes."""
        if episode_id in self.active_episode_ids:
            self.active_episode_ids.remove(episode_id)
            self.update_activity()

    def complete_episode(self, episode_id: str) -> None:
        """Move an episode from active to history."""
        if episode_id in self.active_episode_ids:
            self.active_episode_ids.remove(episode_id)
            if episode_id not in self.episode_history:
                self.episode_history.append(episode_id)
            self.update_activity()

    def add_task_to_queue(self, task_id: str) -> None:
        """Add a task to the orchestration queue."""
        if task_id not in self.task_queue:
            self.task_queue.append(task_id)
            self.update_activity()

    def get_next_task(self) -> Optional[str]:
        """Get and remove the next task from the queue."""
        if self.task_queue:
            task_id = self.task_queue.pop(0)
            self.update_activity()
            return task_id
        return None

    def has_active_episodes(self) -> bool:
        """Check if session has any active episodes."""
        return len(self.active_episode_ids) > 0

    def get_episode_count(self) -> Dict[str, int]:
        """Get episode counts for analytics."""
        return {
            "active": len(self.active_episode_ids),
            "completed": len(self.episode_history),
            "total": len(self.active_episode_ids) + len(self.episode_history),
        }


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
        self.cleanup_task: Optional[asyncio.Task[None]] = None
        self.shutdown_event = asyncio.Event()

        # Initialize server components
        logger.info(f"Initializing SessionManager for domain '{domain_name}' with config_dir '{config_dir}'")

        self.benchmark_manager = BenchmarkManager(domain_name, config_dir)
        self.episode_manager = EpisodeManager()

        # Initialize execution manager first
        self.execution_manager = ExecutionManager(config_dir)

        # Initialize permanent environment manager through ExecutionManager for unified container lifecycle
        permanent_config = {
            "domain": domain_name,
            "config_dir": config_dir,
            "enable_logging": True,
        }
        self.execution_manager.initialize_permanent_environment_manager(permanent_config)

        self.policy_manager = PolicyManager(domain_name)
        self.evaluation_manager = EvaluationManager()

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

        # Stop permanent environment through ExecutionManager
        if self.execution_manager.is_permanent_environment_running():
            logger.info("Stopping permanent environment...")
            try:
                self.execution_manager.stop_permanent_environment()
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
        session = ClientSession(session_id=session_id, client_id=client_id)

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

        # End all active episodes (this will trigger individual episode cleanup)
        if session.has_active_episodes():
            try:
                logger.info(
                    f"Ending {len(session.active_episode_ids)} active episodes during session termination "
                    f"for session {session_id}: {session.active_episode_ids}"
                )
                # End all active episodes - each will trigger its own cleanup
                for episode_id in session.active_episode_ids.copy():  # Copy to avoid modification during iteration
                    try:
                        # Call end_episode which will handle both episode termination AND container cleanup
                        await self.end_episode(session_id, episode_id, EpisodeTerminationReason.SESSION_TERMINATED)
                        logger.info(f"Successfully ended episode {episode_id}")
                    except Exception as e:
                        logger.warning(f"Error ending episode {episode_id}: {str(e)}")

                # Clear remaining state
                session.active_episode_ids.clear()
                session.task_queue.clear()
            except Exception as e:
                logger.warning(f"Error ending episodes during session termination: {str(e)}")

        # Log session end with evaluation manager (ignore failures)
        try:
            await self.evaluation_manager.log_session_end(session_id)
        except Exception as e:
            logger.warning(f"Failed to log session end: {str(e)}")

        # Check for any orphaned episodes that might still need cleanup
        try:
            log_operation_start(logger, "Orphaned episode cleanup check", session_id)
            episodes_to_cleanup = list(session.episode_history)
            logger.info(f"Checking for orphaned episodes: {episodes_to_cleanup}")

            cleanup_success = True
            for episode_id in episodes_to_cleanup:
                try:
                    logger.info(f"🧹 Checking orphaned cleanup for episode {episode_id}")
                    episode_cleanup = self.execution_manager.cleanup_episode(
                        episode_id,
                        CleanupReason.SESSION_TERMINATED,
                        {"manual_termination": True, "orphaned_check": True},
                    )
                    if not episode_cleanup:
                        cleanup_success = False
                        logger.warning(f"Orphaned episode cleanup failed for {episode_id}")
                    else:
                        logger.info(f"✅ Orphaned episode cleanup succeeded for {episode_id}")
                except Exception as e:
                    cleanup_success = False
                    logger.warning(f"Orphaned episode cleanup error for {episode_id}: {e}")

            if cleanup_success:
                log_operation_success(logger, "Orphaned episode cleanup check", session_id)
            else:
                logger.warning(f"Orphaned episode cleanup check reported failures for session {session_id}")
        except Exception as e:
            log_operation_failure(logger, "Orphaned episode cleanup check", str(e), session_id)

        # Mark session as inactive
        session.is_active = False

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

        # Configure execution manager with task object and episode ID for unique container naming
        self.execution_manager.configure_for_task(episode.episode_id, task, session_id=session_id)

        # Configure policy manager with task object and episode ID
        self.policy_manager.configure_for_episode(episode.episode_id, session_id, task)

        # Configure episode manager with task object
        self.episode_manager.configure_for_task(episode.episode_id, task)

        # Add episode to session's active episodes
        session.add_active_episode(episode.episode_id)

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
        self,
        session_id: str,
        episode_id: str,
        reason: str = EpisodeTerminationReason.COMPLETED,
        result: Optional[str] = None,
    ) -> EpisodeEndResponse:
        """
        End a specific episode for a session.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode to end
            reason: Reason for episode termination (use EpisodeTerminationReason enum values)
            result: Optional result/submission from the episode (e.g., captured flag)

        Returns:
            Dictionary with episode termination information
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Validate episode is active in this session
        if episode_id not in session.active_episode_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Episode {episode_id} is not active in session {session_id}. "
                f"Active episodes: {session.active_episode_ids}",
            )

        # Get the task_id from the episode before ending it
        episode = self.episode_manager.get_episode_by_id(episode_id)
        completed_task_id = episode.task_id if episode else None
        success = EpisodeTerminationReason.is_success(reason)

        # End episode through episode manager, passing the result
        self.episode_manager.end_episode(episode_id, reason, result)

        # Cleanup episode containers immediately when episode ends
        try:
            logger.info(f"🧹 Starting episode cleanup for {episode_id}")
            episode_cleanup = self.execution_manager.cleanup_episode(
                episode_id, CleanupReason.EPISODE_COMPLETED, {"episode_end_reason": reason}
            )
            if episode_cleanup:
                logger.info(f"✅ Episode cleanup completed for {episode_id}")
            else:
                logger.warning(f"❌ Episode cleanup failed for {episode_id}")
        except Exception as e:
            logger.error(f"Episode cleanup error for {episode_id}: {e}")

        # Move episode from active to history in session
        session.complete_episode(episode_id)

        logger.info(f"Ended episode {episode_id} for session {session_id} with reason: {reason}")

        # Build response using proper API model - client handles orchestration now
        return EpisodeEndResponse(
            episode_ended=True,
            episode_id=episode_id,
            success=success,
            reason=reason,
            previous_task_id=completed_task_id,
            active_episodes_remaining=len(session.active_episode_ids),
        )

    async def execute_action(self, session_id: str, episode_id: str, action: Action) -> CommandResult:
        """
        Execute action for MCP integration.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode to execute action against
            action: Action object containing command and parameters

        Returns:
            CommandResult with execution results
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Validate episode is active in this session
        if episode_id not in session.active_episode_ids:
            return CommandResult.error_result(
                error=f"Episode {episode_id} is not active in session {session_id}. "
                f"Active episodes: {session.active_episode_ids}"
            )

        try:
            log_operation_start(logger, "Action execution", session_id, tool=action.tool_name, episode=episode_id)

            context = {"session_id": session_id, "episode_id": episode_id}
            command_result = await self.execution_manager.step(action, context)

            if command_result.success:
                log_operation_success(logger, "Action execution", session_id, tool=action.tool_name)
            else:
                log_operation_failure(logger, "Action execution", command_result.error or "Unknown error", session_id)

            step_result = self.episode_manager.step(episode_id, action, command_result)

            # Log action with evaluation manager (ignore failures)
            try:
                await self.evaluation_manager.log_action(
                    session_id=session_id, episode_id=episode_id, action=action, result=command_result
                )
            except Exception as e:
                logger.warning(f"Failed to log action: {e}")

            # End episode if step indicates completion OR if EpisodeManager indicates termination
            if step_result.step.done:
                self.episode_manager.end_episode(episode_id, EpisodeTerminationReason.COMPLETED)
                session.complete_episode(episode_id)
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
                self.episode_manager.end_episode(episode_id, termination_reason)
                session.complete_episode(episode_id)
                # Add termination metadata to command result
                if not hasattr(command_result, "metadata") or command_result.metadata is None:
                    command_result.metadata = {}
                command_result.metadata["episode_terminated"] = True
                command_result.metadata["termination_reason"] = termination_reason
                try:
                    await self.evaluation_manager.log_episode_end(session_id, termination_reason)
                except Exception as e:
                    logger.warning(f"Failed to log episode end: {e}")
                logger.info(f"🏗️ Episode {episode_id} ended due to termination condition: {termination_reason}")

            return command_result

        except Exception as e:
            logger.error(f"Command execution failed in session {session_id}, episode {episode_id}: {e}")

            # Remove episode tracking and trigger immediate cleanup on any error
            try:
                self.episode_manager.remove_episode_on_error(episode_id, e)
                session.remove_active_episode(episode_id)
                logger.info(f"Episode {episode_id} removed from tracking due to error")

                # Trigger immediate container cleanup through ExecutionManager
                cleanup_success = self.execution_manager.cleanup_session(
                    session_id, CleanupReason.ERROR_TRIGGERED, {"error": str(e), "error_type": type(e).__name__}
                )
                if cleanup_success:
                    logger.info("Error-triggered container cleanup completed successfully")
                else:
                    logger.warning("Error-triggered container cleanup reported failure")

            except Exception as cleanup_error:
                logger.error(f"Failed to handle error cleanup: {cleanup_error}")

            return CommandResult.error_result(error=str(e))

    async def execute_episode_action(self, session_id: str, episode_id: str, action: Action) -> CommandResult:
        """
        Execute action in context of specific episode.

        This is an alias for execute_action to match the implementation plan naming.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode to execute action against
            action: Action object containing command and parameters

        Returns:
            CommandResult with execution results
        """
        return await self.execute_action(session_id, episode_id, action)

    async def get_current_task(self, session_id: str, episode_id: str) -> Any:
        """
        Get task object for a specific episode.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode

        Returns:
            Task object for the specified episode
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Validate episode is active in this session
        if episode_id not in session.active_episode_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Episode {episode_id} is not active in session {session_id}. "
                f"Active episodes: {session.active_episode_ids}",
            )

        # Get episode and task info from episode manager
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            raise HTTPException(status_code=400, detail=f"Episode {episode_id} not found")

        task = self.benchmark_manager.get_task(episode.task_id)
        return task

    def get_policy(self, session_id: str, episode_id: str) -> PolicyDocument:
        """
        Get domain policy document for a specific episode.

        Args:
            session_id: ID of the client session
            episode_id: ID of the episode to get policy for

        Returns:
            PolicyDocument for the domain configured for the episode's task
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Validate episode exists and belongs to session
        if episode_id not in session.active_episode_ids and episode_id not in session.episode_history:
            raise ValueError(f"Episode {episode_id} not found in session {session_id}")

        # Get episode-specific policy that was configured during start_episode
        return self.policy_manager.get_policy(episode_id)

    async def list_session_episodes(self, session_id: str, status_filter: Optional[str] = None) -> List[Any]:
        """
        List all episodes for a session with optional status filtering.

        Args:
            session_id: ID of the client session
            status_filter: Optional filter by episode status

        Returns:
            List of episode objects for the session
        """
        session = self._get_session(session_id)
        session.update_activity()

        episodes = self.episode_manager.get_session_episodes(session_id)

        if status_filter:
            episodes = [ep for ep in episodes if ep.state.value == status_filter]

        return episodes

    async def get_episode_details(self, session_id: str, episode_id: str) -> Any:
        """
        Get detailed information about a specific episode.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode

        Returns:
            Episode object with full details

        Raises:
            HTTPException: If episode not found or doesn't belong to session
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Validate episode belongs to session
        if episode_id not in session.active_episode_ids and episode_id not in session.episode_history:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found in session {session_id}")

        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

        return episode

    def get_tool_event_publisher(self) -> Any:
        """
        Get tool event publisher instance - used by MCP API.

        Returns:
            Tool event publisher instance if available
        """
        # Return the tool event publisher from REST API if available
        if hasattr(self.rest_api, "tool_event_publisher"):
            return self.rest_api.tool_event_publisher
        return None

    def get_episode_by_id(self, episode_id: str) -> Any:
        """
        Get episode by ID - used by MCP API for episode context.

        Args:
            episode_id: ID of the episode to retrieve

        Returns:
            Episode object if found, None otherwise
        """
        return self.episode_manager.get_episode_by_id(episode_id)

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
            episode_counts = session.get_episode_count()
            stats["sessions"].append(
                {
                    "session_id": session_id,
                    "client_id": session.client_id,
                    "active_episode_ids": session.active_episode_ids,
                    "episode_history": session.episode_history,
                    "episode_counts": episode_counts,
                    "task_queue": session.task_queue,
                    "uptime_seconds": (current_time - session.created_at).total_seconds(),
                    "time_since_activity_seconds": time_since_activity.total_seconds(),
                    "is_active": session.is_active,
                }
            )

        return stats

    async def _start_permanent_environment(self) -> None:
        """Start permanent environment if configured through ExecutionManager lifecycle management."""
        permanent_env_name = self.benchmark_manager.config_loader.get_permanent_environment()
        if not permanent_env_name:
            logger.info("No permanent environment configured")
            return

        try:
            logger.info(f"Starting permanent environment: {permanent_env_name}")

            # Load permanent environment specification through ExecutionManager
            if self.execution_manager._environment_loader is None:
                raise RuntimeError("Environment loader is not initialized")
            permanent_env_spec = self.execution_manager._environment_loader.load_permanent_environment(
                permanent_env_name
            )

            # Start permanent environment through ExecutionManager
            self.execution_manager.start_permanent_environment(permanent_env_spec)

            logger.info(f"Permanent environment '{permanent_env_name}' started successfully")

        except Exception as e:
            logger.error(f"Failed to start permanent environment '{permanent_env_name}': {e}")
            raise
