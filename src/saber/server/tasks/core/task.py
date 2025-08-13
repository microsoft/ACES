"""Task implementation for task management system."""

from logging import getLogger
from typing import Any, Dict, List, Optional, Set

from ...execution.sandbox.environment_spec import EnvironmentSpec
from .subtask import SubTask

logger = getLogger(__name__)


class Task:
    """
    Represents a high-level scenario with metadata and subtasks
    """

    def __init__(
        self,
        task_id: str,
        domain: str,
        title: str,
        description: str,
        subtasks: Optional[List[SubTask]] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        environment_spec: Optional[EnvironmentSpec] = None,
    ):
        """
        Initialize a task.

        Args:
            task_id: Unique identifier for the task
            domain: Security domain this task belongs to
            title: Human-readable title
            description: Detailed description of the task
            subtasks: List of subtasks
            initial_context: Initial context provided when the task starts
            environment_spec: Environment specification for multi-container orchestration
        """
        self.task_id = task_id
        self.domain = domain
        self.title = title
        self.description = description
        self.subtasks = subtasks or []
        self.initial_context = initial_context or {}
        self.environment_spec = environment_spec
        """
        Initialize a task.

        Args:
            task_id: Unique identifier for the task
            domain: Security domain this task belongs to
            title: Human-readable title
            description: Detailed description of the task
            subtasks: List of subtasks
            initial_context: Initial context provided when the task starts
            environment_spec: Environment specification for multi-container orchestration
        """
        self.task_id = task_id
        self.domain = domain
        self.title = title
        self.description = description
        self.subtasks = subtasks or []
        self.initial_context = initial_context or {}
        self.environment_spec = environment_spec

        # Create lookup map for efficient subtask access
        self._subtask_map = {st.subtask_id: st for st in self.subtasks}

    def add_subtask(self, subtask: SubTask) -> None:
        """
        Add a subtask to this task.

        Args:
            subtask: The subtask to add
        """
        subtask.task_id = self.task_id  # Ensure consistency
        self.subtasks.append(subtask)
        self._subtask_map[subtask.subtask_id] = subtask

    def get_subtask_by_id(self, subtask_id: str) -> Optional[SubTask]:
        """
        Get a subtask by its ID.

        Args:
            subtask_id: The ID of the subtask to retrieve

        Returns:
            The subtask if found, None otherwise
        """
        return self._subtask_map.get(subtask_id)

    def get_all_subtask_ids(self) -> Set[str]:
        """
        Get all subtask IDs for this task.

        Returns:
            Set of all subtask IDs
        """
        return {subtask.subtask_id for subtask in self.subtasks}
