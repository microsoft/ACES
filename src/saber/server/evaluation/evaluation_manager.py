"""
Stub EvaluationManager implementation for SABER domain server.

This is a minimal implementation to support SessionManager development.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EvaluationManager:
    """
    Stub EvaluationManager for tracking and evaluating agent performance.

    This is a minimal implementation to support SessionManager development.
    Full implementation will be added later.
    """

    def __init__(self) -> None:
        """Initialize the EvaluationManager."""
        logger.info("EvaluationManager initialized (stub implementation)")

    async def log_session_start(self, session_id: str, client_id: str) -> None:
        """
        Log session start event.

        Args:
            session_id: ID of the session
            client_id: ID of the client
        """
        logger.info(f"Session started: {session_id} for client {client_id}")

    async def log_session_end(self, session_id: str) -> None:
        """
        Log session end event.

        Args:
            session_id: ID of the session
        """
        logger.info(f"Session ended: {session_id}")

    async def log_episode_start(self, session_id: str, episode_id: str, task_id: str) -> None:
        """
        Log episode start event.

        Args:
            session_id: ID of the session
            episode_id: ID of the episode
            task_id: ID of the task
        """
        logger.info(f"Episode started: {episode_id} for task {task_id} in session {session_id}")

    async def log_episode_end(self, session_id: str, completion_reason: str) -> None:
        """
        Log episode end event.

        Args:
            session_id: ID of the session
            completion_reason: Reason the episode ended
        """
        logger.info(f"Episode ended in session {session_id}: {completion_reason}")

    async def log_action(self, session_id: str, episode_id: str, action: Any, result: Any) -> None:
        """
        Log action execution event.

        Args:
            session_id: ID of the session
            episode_id: ID of the episode
            action: Action that was executed
            result: Result of the action execution
        """
        logger.info(f"Action logged for episode {episode_id} in session {session_id}: {action.tool_name}")

    async def get_trajectory(self, session_id: str) -> list:
        """
        Get trajectory for a session.

        Args:
            session_id: ID of the session

        Returns:
            Empty list (stub implementation)
        """
        return []
