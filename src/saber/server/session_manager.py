"""SessionManager implementation for unified client interactions."""

from datetime import datetime
from logging import getLogger
from typing import Any, Dict, Optional

from .tasks.episodes.episode import Action
from .tasks.task_manager import TaskManager
from .tools.base import ToolResult

logger = getLogger(__name__)


class SessionManager:
    """
    Unified endpoint for all client interactions.

    Coordinates session lifecycle, task orchestration, tool execution,
    and integrates with TaskManager for RL episode management.
    """

    def __init__(self, task_manager: TaskManager):
        """
        Initialize SessionManager.

        Args:
            task_manager: TaskManager instance for task and episode management
        """
        self.task_manager = task_manager
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, client_id: str) -> str:
        """
        Create a new client session.

        Args:
            client_id: ID of the client creating the session

        Returns:
            Session ID
        """
        session_id = f"session_{client_id}_{datetime.now().isoformat()}"
        self.active_sessions[session_id] = {"client_id": client_id, "created_at": datetime.now(), "active": True}
        logger.info(f"Created session '{session_id}' for client '{client_id}'")
        return session_id

    def execute_tool(self, session_id: str, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute tool and record episode step.

        This is the key integration point mentioned in Task 2.1 where
        action-response integration happens.

        Args:
            session_id: ID of the session
            tool_name: Name of the tool to execute
            parameters: Parameters for tool execution

        Returns:
            ToolResult from tool execution
        """
        # Create action object for episode tracking
        action = Action(
            tool_name=tool_name, parameters=parameters, timestamp=datetime.now(), command=parameters.get("command")
        )

        # Execute via MCP server (this would be the actual tool execution)
        # For demonstration, create a mock result for DockerCLIExecutor
        result = ToolResult.success_result(
            data={"output": f"Docker CLI executed: {parameters.get('command', 'unknown command')}"},
            execution_time=0.1,
            metadata={"tool": tool_name, "command": parameters.get("command")},
        )

        # Record step in episode (NEW INTEGRATION POINT from Task 2.1)
        try:
            self.task_manager.step(session_id, action, result)
            logger.debug(f"Recorded episode step for session '{session_id}': {tool_name}")
        except Exception as e:
            logger.warning(f"Failed to record episode step: {e}")

        return result

    def start_task_episode(self, session_id: str, task_id: str) -> Dict[str, Any]:
        """
        Start a new task episode using RL-style reset.

        Args:
            session_id: ID of the session
            task_id: ID of the task to start

        Returns:
            Episode information
        """
        episode = self.task_manager.reset(session_id, task_id)
        return {"episode_id": episode.episode_id, "task_id": task_id, "state": episode.state.value}

    def get_current_step_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current RL-style step result for the session.

        Args:
            session_id: ID of the session

        Returns:
            Step result information or None
        """
        try:
            # Create a status query action (no tool result)
            action = Action(tool_name="status_check", parameters={}, command=None)
            step_result = self.task_manager.step(session_id, action)

            return {
                "observation": step_result.observation,
                "done": step_result.done,
                "info": step_result.info,
            }
        except Exception as e:
            logger.error(f"Failed to get step result: {e}")
            return None

    def end_session(self, session_id: str) -> None:
        """
        End a client session and cleanup.

        Args:
            session_id: ID of the session to end
        """
        if session_id in self.active_sessions:
            # End any active episode
            try:
                self.task_manager.end_episode(session_id, "session_ended")
            except Exception as e:
                logger.warning(f"Failed to end episode for session cleanup: {e}")

            # Mark session as inactive
            self.active_sessions[session_id]["active"] = False
            logger.info(f"Ended session '{session_id}'")
