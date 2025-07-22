"""DomainTask implementation for task management system."""

from typing import Any, Dict, List, Optional, Set

from .exceptions import SubTaskNotFoundException
from .subtask import SubTask


class DomainTask:
    """
    Represents a high-level security scenario composed of multiple subtasks.

    A domain task maintains a collection of subtasks and provides methods
    to navigate and manage the subtask execution flow.
    """

    def __init__(
        self,
        task_id: str,
        domain: str,
        title: str,
        description: str,
        subtasks: Optional[List[SubTask]] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a domain task.

        Args:
            task_id: Unique identifier for the task
            domain: Security domain this task belongs to
            title: Human-readable title
            description: Detailed description of the task
            subtasks: List of subtasks that comprise this task
            initial_context: Initial context provided when the task starts
        """
        self.task_id = task_id
        self.domain = domain
        self.title = title
        self.description = description
        self.subtasks = subtasks or []
        self.initial_context = initial_context or {}

        # Create lookup map for efficient subtask access
        self._subtask_map = {st.subtask_id: st for st in self.subtasks}

    def add_subtask(self, subtask: SubTask) -> None:
        """
        Add a subtask to this domain task.

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

    def get_next_subtask(self, current_id: str) -> Optional[SubTask]:
        """
        Get the next subtask that can be executed after the current one.

        This considers dependency relationships and finds the next subtask
        that has all its dependencies satisfied.

        Args:
            current_id: ID of the currently completed subtask

        Returns:
            The next available subtask, or None if no subtasks are available
        """
        # For simplicity in MVP, we'll use the order in the subtasks list
        # In a more advanced implementation, this could use topological sorting
        # based on dependencies

        try:
            current_index = next(
                i for i, st in enumerate(self.subtasks) if st.subtask_id == current_id
            )
            if current_index + 1 < len(self.subtasks):
                return self.subtasks[current_index + 1]
        except StopIteration:
            pass

        return None

    def get_available_subtasks(self, completed_subtasks: Set[str]) -> List[SubTask]:
        """
        Get all subtasks that can currently be executed.

        Args:
            completed_subtasks: Set of subtask IDs that have been completed

        Returns:
            List of subtasks that have all dependencies satisfied
        """
        available = []
        for subtask in self.subtasks:
            if subtask.subtask_id not in completed_subtasks and subtask.can_execute(
                completed_subtasks
            ):
                available.append(subtask)
        return available

    def is_complete(self, completed_subtasks: Set[str]) -> bool:
        """
        Check if all subtasks in this domain task have been completed.

        Args:
            completed_subtasks: Set of subtask IDs that have been completed

        Returns:
            True if all subtasks are completed
        """
        return all(st.subtask_id in completed_subtasks for st in self.subtasks)

    def get_first_subtask(self) -> SubTask:
        """
        Get the first subtask that should be executed.

        Returns:
            The first subtask without dependencies, or the first in the list

        Raises:
            SubTaskNotFoundException: If no subtasks are available
        """
        if not self.subtasks:
            raise SubTaskNotFoundException(self.task_id, "no_subtasks")

        # Find first subtask with no dependencies
        for subtask in self.subtasks:
            if not subtask.depends_on:
                return subtask

        # If all subtasks have dependencies, return the first one
        # This might indicate a circular dependency, but we'll let the
        # validation catch that later
        return self.subtasks[0]

    def get_completion_percentage(self, completed_subtasks: Set[str]) -> float:
        """
        Calculate the completion percentage of this task.

        Args:
            completed_subtasks: Set of subtask IDs that have been completed

        Returns:
            Completion percentage as a float between 0.0 and 1.0
        """
        if not self.subtasks:
            return 1.0

        completed_count = sum(1 for st in self.subtasks if st.subtask_id in completed_subtasks)
        return completed_count / len(self.subtasks)

    def validate_dependencies(self) -> List[str]:
        """
        Validate that all subtask dependencies are valid.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        subtask_ids = {st.subtask_id for st in self.subtasks}

        for subtask in self.subtasks:
            for dep_id in subtask.depends_on:
                if dep_id not in subtask_ids:
                    errors.append(
                        f"SubTask '{subtask.subtask_id}' depends on unknown subtask '{dep_id}'"
                    )

        # TODO: Add circular dependency detection

        return errors
