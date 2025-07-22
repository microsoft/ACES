"""TaskSession implementation for task management system."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from .enums import SessionState
from .exceptions import SessionStateException, SubTaskNotFoundException

# Avoid circular imports
if TYPE_CHECKING:
    from .subtask import SubTask
    from .task_manager import TaskManager


class TaskSession:
    """
    Maintains state and context for a client's execution of a domain task.

    A task session tracks the current subtask, completed subtasks, context
    accumulated during execution, and session metadata.
    """

    def __init__(
        self,
        task_id: str,
        client_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize a new task session.

        Args:
            task_id: ID of the task being executed
            client_id: ID of the client executing the task
            initial_context: Initial context for the task
            session_id: Optional session ID (generated if not provided)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.task_id = task_id
        self.client_id = client_id
        self.state = SessionState.CREATED
        self.context = initial_context or {}
        self.completed_subtasks: Set[str] = set()
        self.current_subtask_id: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()

    def advance_to_next(self, task_manager: "TaskManager") -> Optional["SubTask"]:
        """
        Advance to the next available subtask.

        Args:
            task_manager: TaskManager instance for task lookup

        Returns:
            The next subtask to execute, or None if task is complete

        Raises:
            SessionStateException: If session is not in a valid state for advancement
        """
        if self.state not in [SessionState.CREATED, SessionState.ACTIVE]:
            raise SessionStateException(self.session_id, self.state.value, "created or active")

        task = task_manager.get_task(self.task_id)

        # Check if task is complete
        if task.is_complete(self.completed_subtasks):
            self.state = SessionState.COMPLETED
            self.current_subtask_id = None
            self._update_activity()
            return None

        # Get next available subtask
        available_subtasks = task.get_available_subtasks(self.completed_subtasks)

        if not available_subtasks:
            # No subtasks available - might be waiting for dependencies
            return None

        # For MVP, take the first available subtask
        next_subtask = available_subtasks[0]
        self.current_subtask_id = next_subtask.subtask_id
        self.state = SessionState.ACTIVE
        self._update_activity()

        return next_subtask

    def update_context(self, new_context: Dict[str, Any]) -> None:
        """
        Update the session context with new data.

        Args:
            new_context: Dictionary of new context data to merge
        """
        self.context.update(new_context)
        self._update_activity()

    def get_current_subtask(self, task_manager: "TaskManager") -> "SubTask":
        """
        Get the currently active subtask.

        Args:
            task_manager: TaskManager instance for task lookup

        Returns:
            The current subtask

        Raises:
            SubTaskNotFoundException: If current subtask is not found
        """
        if not self.current_subtask_id:
            raise SubTaskNotFoundException(self.task_id, "no_current_subtask")

        return task_manager.get_subtask(self.task_id, self.current_subtask_id)

    def is_active(self) -> bool:
        """
        Check if this session is currently active.

        Returns:
            True if session is in active state
        """
        return self.state == SessionState.ACTIVE

    def complete_subtask(self, subtask_id: str) -> None:
        """
        Mark a subtask as completed.

        Args:
            subtask_id: ID of the subtask to mark as completed
        """
        if subtask_id != self.current_subtask_id:
            # Allow completing any subtask for flexibility
            pass

        self.completed_subtasks.add(subtask_id)
        self._update_activity()

    def pause(self) -> None:
        """Pause the session."""
        if self.state == SessionState.ACTIVE:
            self.state = SessionState.PAUSED
            self._update_activity()

    def resume(self) -> None:
        """Resume a paused session."""
        if self.state == SessionState.PAUSED:
            self.state = SessionState.ACTIVE
            self._update_activity()

    def fail(self, reason: Optional[str] = None) -> None:
        """
        Mark the session as failed.

        Args:
            reason: Optional reason for failure
        """
        self.state = SessionState.FAILED
        if reason:
            self.context["failure_reason"] = reason
        self._update_activity()

    def timeout(self) -> None:
        """Mark the session as timed out."""
        self.state = SessionState.TIMEOUT
        self._update_activity()

    def get_context_for_subtask(
        self, subtask_id: str, task_manager: "TaskManager"
    ) -> Dict[str, Any]:
        """
        Get context specific to a subtask.

        Args:
            subtask_id: ID of the subtask
            task_manager: TaskManager instance for task lookup

        Returns:
            Context dictionary for the subtask
        """
        subtask = task_manager.get_subtask(self.task_id, subtask_id)
        return subtask.get_context(self.context)

    def is_complete(self, task_manager: "TaskManager") -> bool:
        """
        Check if the task associated with this session is complete.

        Args:
            task_manager: TaskManager instance for task lookup

        Returns:
            True if all subtasks are completed
        """
        task = task_manager.get_task(self.task_id)
        return task.is_complete(self.completed_subtasks)

    def get_progress_info(self, task_manager: "TaskManager") -> Dict[str, Any]:
        """
        Get progress information for this session.

        Args:
            task_manager: TaskManager instance for task lookup

        Returns:
            Dictionary with progress information
        """
        task = task_manager.get_task(self.task_id)

        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "state": self.state.value,
            "current_subtask_id": self.current_subtask_id,
            "completed_subtasks": list(self.completed_subtasks),
            "total_subtasks": len(task.subtasks),
            "completion_percentage": task.get_completion_percentage(self.completed_subtasks),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }

    def _update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.utcnow()
