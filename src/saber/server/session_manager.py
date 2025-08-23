"""
SessionManager implementation for SABER domain server.

The SessionManager serves as the central orchestrator for managing client sessions
and coordinating all server components. REST API functionality is handled by SessionAPI.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from .api.session_mcp_api import SessionMCPAPI
from .api.session_rest_api import SessionRestAPI
from .base import Action, CommandResult
from .episodes.episode_manager import EpisodeManager
from .evaluation.evaluation_manager import EvaluationManager
from .execution.execution_manager import ExecutionManager
from .policy.policy_manager import PolicyDocument, PolicyManager
from .tasks.task_manager import TaskManager

logger = logging.getLogger(__name__)


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
        logger.info(f"Initializing SessionManager for domain '{domain_name}' with config dir: {config_dir}")

        self.task_manager = TaskManager(domain_name, config_dir)

        self.episode_manager = EpisodeManager()
        self.execution_manager = ExecutionManager(config_dir)
        self.policy_manager = PolicyManager(domain_name)
        self.evaluation_manager = EvaluationManager()

        # Initialize protocol handlers
        self.rest_api = SessionRestAPI(self, host, port)
        self.mcp_api = SessionMCPAPI(self, mcp_host, mcp_port)

        logger.info(
            f"SessionManager initialized for domain '{domain_name}' on REST:{host}:{port}, MCP:{mcp_host}:{mcp_port}"
        )

    @property
    def app(self) -> Any:
        return self.rest_api.app

    async def start_server(self) -> None:
        """Start both REST and MCP servers concurrently with session cleanup."""
        import asyncio

        # Start the session cleanup task
        self.cleanup_task = asyncio.create_task(self._session_cleanup_loop())

        # Start both servers concurrently using asyncio tasks
        rest_task = asyncio.create_task(self.rest_api.start_server())
        mcp_task = asyncio.create_task(self.mcp_api.start_mcp_server())

        # Wait for both to complete (they run indefinitely)
        await asyncio.gather(rest_task, mcp_task, self.cleanup_task)

    async def shutdown(self) -> None:
        """Shutdown the SessionManager and cleanup resources."""
        logger.info(f"Shutting down SessionManager for domain '{self.domain_name}'")

        # Signal shutdown to stop cleanup loop
        self.shutdown_event.set()

        # Cancel cleanup task if running
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Shutdown MCP server first
        await self.mcp_api.shutdown_mcp_server()

        # Cleanup all active sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
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

        # Register session with MCP API using client_id as MCP client identifier
        # This allows MCP clients to use the client_id when connecting to map to SABER sessions
        self.mcp_api.register_session_for_client(client_id, session_id)

        # Log session creation with evaluation manager (ignore failures)
        try:
            await self.evaluation_manager.log_session_start(session_id, client_id)
        except Exception as e:
            logger.warning(f"Failed to log session start: {e}")

        logger.info(f"Created new session {session_id} for client {client_id}")
        return session

    async def terminate_session(self, session_id: str) -> None:
        """
        Terminate a client session and cleanup resources.

        Args:
            session_id: ID of the session to terminate
        """
        session = self._get_session(session_id)
        session.is_active = False

        # End any active episode
        if session.current_episode_id:
            try:
                self.episode_manager.end_episode(session_id, "session_terminated")
                session.current_episode_id = None
                session.current_task_id = None
            except Exception as e:
                logger.warning(f"Error ending episode during session termination: {e}")

        # Unregister session from MCP API using client_id
        self.mcp_api.unregister_session_for_client(session.client_id)

        # Log session end with evaluation manager (ignore failures)
        try:
            await self.evaluation_manager.log_session_end(session_id)
        except Exception as e:
            logger.warning(f"Failed to log session end: {e}")

        # Cleanup execution resources (Docker containers)
        try:
            self.execution_manager.cleanup_session(session_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup execution resources: {e}")

        # Remove session from active sessions
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
        task = self.task_manager.get_task(task_id)

        # Start episode through episode manager
        episode = self.episode_manager.start_episode(
            session_id=session_id, task_id=task_id, initial_context=task.initial_context.copy()
        )

        # Get the cleanup token for orchestrator coordination
        cleanup_token = self.episode_manager.get_cleanup_token(session_id)

        # Configure execution manager with task object and cleanup token
        self.execution_manager.configure_for_task(session_id, task, cleanup_token)

        session.current_episode_id = episode.episode_id
        session.current_task_id = task_id

        logger.debug(f"Episode cleanup token for session {session_id}: {cleanup_token}")

        # Log episode start with evaluation manager (ignore failures)
        try:
            await self.evaluation_manager.log_episode_start(session_id, episode.episode_id, task_id)
        except Exception as e:
            logger.warning(f"Failed to log episode start: {e}")

        logger.info(f"Started episode {episode.episode_id} for task {task_id} in session {session_id}")
        return episode

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
            # Execute command through execution manager with session context
            context = {"session_id": session_id}
            command_result = await self.execution_manager.step(action, context)

            # Execute step through episode manager
            step_result = self.episode_manager.step(session_id, action, command_result)

            # Log action with evaluation manager (ignore failures)
            try:
                await self.evaluation_manager.log_action(
                    session_id=session_id, episode_id=session.current_episode_id, action=action, result=command_result
                )
            except Exception as e:
                logger.warning(f"Failed to log action: {e}")

            # End episode if step indicates completion
            if step_result.done:
                self.episode_manager.end_episode(session_id, "completed")
                session.current_episode_id = None
                session.current_task_id = None
                try:
                    await self.evaluation_manager.log_episode_end(session_id, "completed")
                except Exception as e:
                    logger.warning(f"Failed to log episode end: {e}")

            return command_result

        except Exception as e:
            logger.error(f"Command execution failed in session {session_id}: {e}")

            # Remove episode tracking immediately on any error - containers will self-terminate
            try:
                self.episode_manager.remove_episode_on_error(session_id, e)
                session.current_episode_id = None
                session.current_task_id = None
                logger.info("Episode removed from tracking due to error - containers will self-terminate")
            except Exception as cleanup_error:
                logger.error(f"Failed to remove episode on error: {cleanup_error}")

            return CommandResult.error_result(error=str(e))

    async def get_current_task(self, session_id: str) -> Dict[str, Any]:
        """
        Get current task information.

        Args:
            session_id: ID of the client session

        Returns:
            Dictionary with current task information
        """
        session = self._get_session(session_id)
        session.update_activity()

        if not session.current_episode_id:
            raise HTTPException(status_code=400, detail="No active episode in session")

        # Get current episode and task info from episode manager
        episode = self.episode_manager.get_current_episode(session_id)
        if not episode:
            raise HTTPException(status_code=400, detail="No active episode found")

        task = self.task_manager.get_task(episode.task_id)

        return {
            "task_id": task.task_id,
            "title": task.title,
            "description": getattr(task, "description", ""),
            "episode_id": episode.episode_id,
            "state": episode.state.value,
            "step_count": len(episode.steps),
            "duration": episode.duration,
        }

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

    def is_episode_over(self, session_id: str) -> tuple[bool, str]:
        """
        Check if the current episode should be terminated.

        Args:
            session_id: ID of the client session

        Returns:
            Tuple of (should_terminate, reason)
        """
        episode = self.episode_manager.get_current_episode(session_id)
        if not episode:
            return True, "no_active_episode"

        if episode.is_complete:
            return True, episode.completion_reason or "completed"

        # Get task configuration for episode limits
        try:
            task = self.task_manager.get_task(episode.task_id)
            episode_config = task.episode_config

            # Check max steps
            max_steps = episode_config.get("max_steps", 20)  # Default to 20
            current_steps = len(episode.steps)

            if current_steps >= max_steps:
                return True, f"max_steps_reached ({current_steps}/{max_steps})"

            # Could add more termination conditions here:
            # - episode timeout
            # - step timeout
            # - resource limits
            # etc.

        except Exception as e:
            logger.warning(f"Failed to check episode termination conditions: {e}")
            # Don't terminate on configuration errors
            pass

        return False, ""

    def get_episode_config(self, session_id: str) -> Dict[str, Any]:
        """
        Get episode configuration for the current episode.

        Args:
            session_id: ID of the client session

        Returns:
            Episode configuration dictionary with defaults
        """
        episode = self.episode_manager.get_current_episode(session_id)
        if not episode:
            return {"max_steps": 20}  # Default config

        try:
            task = self.task_manager.get_task(episode.task_id)
            config = task.episode_config.copy()

            # Apply defaults for missing values
            config.setdefault("max_steps", 20)
            config.setdefault("step_timeout_seconds", 300)
            config.setdefault("episode_timeout_minutes", 30)

            return config
        except Exception as e:
            logger.warning(f"Failed to get episode configuration: {e}")
            return {"max_steps": 20}

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
