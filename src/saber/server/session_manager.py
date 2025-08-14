"""
SessionManager implementation for SABER domain server.

The SessionManager serves as the central orchestrator for managing client sessions
and coordinating all server components. REST API functionality is handled by SessionAPI.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from .api.session_mcp_api import SessionMCPAPI
from .api.session_rest_api import SessionRestAPI
from .evaluation.evaluation_manager import EvaluationManager
from .execution.base import CommandResult
from .execution.execution_manager import ExecutionManager
from .policy.policy_manager import PolicyDocument, PolicyManager
from .tasks.base import Action
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
    ):
        """
        Initialize the SessionManager.

        Args:
            domain_name: Name of the security domain (e.g., 'malware_classification')
            config_dir: Path to configuration directory containing tasks.yaml and environments.yaml
            host: REST server host address
            port: REST server port
            mcp_host: MCP server host address
            mcp_port: MCP server port
        """
        self.domain_name = domain_name
        self.config_dir = config_dir
        self.host = host
        self.port = port
        self.mcp_host = mcp_host
        self.mcp_port = mcp_port
        self.active_sessions: Dict[str, ClientSession] = {}

        # Initialize server components
        logger.info(f"Initializing SessionManager for domain '{domain_name}'")

        self.task_manager = TaskManager(domain_name, config_dir)

        # Get allowed executors from task configuration
        allowed_executors = self.task_manager.get_allowed_executors()
        if allowed_executors:
            logger.info(f"Using executor restriction from task config: {allowed_executors}")
        else:
            logger.info("No executor restriction specified, all executors will be available")

        # Construct execution config path from config directory
        execution_config_path = f"{config_dir}/environments.yaml"
        self.execution_manager = ExecutionManager(
            config_file=execution_config_path, allowed_executors=allowed_executors
        )
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
        """Start both REST and MCP servers concurrently."""
        import asyncio

        # Start both servers concurrently using asyncio tasks
        rest_task = asyncio.create_task(self.rest_api.start_server())
        mcp_task = asyncio.create_task(self.mcp_api.start_mcp_server())

        # Wait for both to complete (they run indefinitely)
        await asyncio.gather(rest_task, mcp_task)

    async def shutdown(self) -> None:
        """Shutdown the SessionManager and cleanup resources."""
        logger.info(f"Shutting down SessionManager for domain '{self.domain_name}'")

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
        """
        session_id = str(uuid.uuid4())
        session = ClientSession(
            session_id=session_id, client_id=client_id, current_episode_id=None, current_task_id=None
        )

        self.active_sessions[session_id] = session

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
                self.task_manager.end_episode(session_id, "session_terminated")
                session.current_episode_id = None
                session.current_task_id = None
            except Exception as e:
                logger.warning(f"Error ending episode during session termination: {e}")

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

        # Get environment specification from task
        environment_spec = self.task_manager.get_task_environment_spec(task_id)
        if environment_spec:
            # Create execution environment for the task
            try:
                self.execution_manager.create_environment(session_id, environment_spec)
                logger.info(f"Created execution environment for session {session_id}, task {task_id}")
            except Exception as e:
                logger.error(f"Failed to create execution environment for session {session_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create execution environment: {e}")

        # Start episode through task manager
        episode = self.task_manager.start_episode(session_id, task_id)
        session.current_episode_id = episode.episode_id
        session.current_task_id = task_id

        # Log episode start with evaluation manager (ignore failures)
        try:
            await self.evaluation_manager.log_episode_start(session_id, episode.episode_id, task_id)
        except Exception as e:
            logger.warning(f"Failed to log episode start: {e}")

        logger.info(f"Started episode {episode.episode_id} for task {task_id} in session {session_id}")
        return episode

    async def execute_command(self, session_id: str, action: Action) -> CommandResult:
        """
        Execute command for MCP integration.

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

            # Execute step through task manager
            step_result = self.task_manager.step(session_id, action, command_result)

            # Log action with evaluation manager (ignore failures)
            try:
                await self.evaluation_manager.log_action(
                    session_id=session_id, episode_id=session.current_episode_id, action=action, result=command_result
                )
            except Exception as e:
                logger.warning(f"Failed to log action: {e}")

            # End episode if step indicates completion
            if step_result.done:
                self.task_manager.end_episode(session_id, "completed")
                session.current_episode_id = None
                session.current_task_id = None
                try:
                    await self.evaluation_manager.log_episode_end(session_id, "completed")
                except Exception as e:
                    logger.warning(f"Failed to log episode end: {e}")

            return command_result

        except Exception as e:
            logger.error(f"Command execution failed in session {session_id}: {e}")
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

        # Get current episode and task info from task manager
        episode = self.task_manager.get_current_episode(session_id)
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
