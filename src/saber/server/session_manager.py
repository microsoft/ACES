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
from pydantic import BaseModel, Field

from .evaluation.evaluation_manager import EvaluationManager
from .execution.execution_manager import ExecutionManager
from .policy.policy_manager import PolicyDocument, PolicyManager
from .session_api import SessionAPI, SessionStepResponse
from .tasks.base import Action
from .tasks.task_manager import TaskManager

logger = logging.getLogger(__name__)


class ClientSession(BaseModel):
    """Represents an active client session."""

    session_id: str = Field(..., description="Unique session identifier")
    client_id: str = Field(..., description="Client identifier")
    current_episode_id: Optional[str] = Field(None, description="Current active episode ID")
    current_task_id: Optional[str] = Field(None, description="Current active task ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    is_active: bool = Field(default=True, description="Whether session is active")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context data")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

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
        tasks_config_path: str,
        execution_config_path: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """
        Initialize the SessionManager.

        Args:
            domain_name: Name of the security domain (e.g., 'malware_classification')
            tasks_config_path: Path to tasks configuration file
            execution_config_path: Path to execution configuration file
            host: Server host address
            port: Server port
        """
        self.domain_name = domain_name
        self.host = host
        self.port = port
        self.active_sessions: Dict[str, ClientSession] = {}

        # Initialize server components
        logger.info(f"Initializing SessionManager for domain '{domain_name}'")

        self.task_manager = TaskManager(domain_name, tasks_config_path)
        self.execution_manager = ExecutionManager(config_file=execution_config_path)
        self.policy_manager = PolicyManager(domain_name)
        self.evaluation_manager = EvaluationManager()

        # Initialize API layer
        self.api = SessionAPI(self, host, port)

        logger.info(f"SessionManager initialized for domain '{domain_name}' on {host}:{port}")

    @property
    def app(self) -> Any:
        return self.api.app

    async def start_server(self) -> None:
        """Start the SessionManager server."""
        await self.api.start_server()

    async def shutdown(self) -> None:
        """Shutdown the SessionManager and cleanup resources."""
        logger.info(f"Shutting down SessionManager for domain '{self.domain_name}'")

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

    async def step(
        self, session_id: str, command: str, parameters: Optional[Dict[str, Any]] = None
    ) -> SessionStepResponse:
        """
        Execute RL step with command.

        Args:
            session_id: ID of the client session
            command: Command to execute
            parameters: Additional command parameters

        Returns:
            SessionStepResponse with execution results and step information
        """
        session = self._get_session(session_id)
        session.update_activity()

        if not session.current_episode_id:
            raise HTTPException(status_code=400, detail="No active episode in session")

        try:
            # Create action for both execution and task managers
            action = Action(tool_name="cli", parameters=parameters or {}, command=command)

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

            # Prepare response
            response = SessionStepResponse(
                success=command_result.success,
                data=command_result.data if command_result.data is not None else {},
                step={
                    "step_number": step_result.step_number,
                    "done": step_result.done,
                    "action": step_result.action.tool_name,
                    "timestamp": step_result.timestamp.isoformat(),
                },
                error=None,
            )

            # End episode if step indicates completion
            if step_result.done:
                self.task_manager.end_episode(session_id, "completed")
                session.current_episode_id = None
                session.current_task_id = None
                try:
                    await self.evaluation_manager.log_episode_end(session_id, "completed")
                except Exception as e:
                    logger.warning(f"Failed to log episode end: {e}")

            return response

        except Exception as e:
            logger.error(f"Step execution failed in session {session_id}: {e}")
            return SessionStepResponse(success=False, data={}, step={}, error=str(e))

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
