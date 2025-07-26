"""Task implementation for task management system."""

from typing import Any, Dict, List, Optional, Set

from ..episodes.episode import Episode
from ..exceptions import SubTaskNotFoundException
from .subtask import SubTask


class Task:
    """
    Represents a high-level security scenario composed of multiple subtasks.

    A task maintains a collection of subtasks and provides methods
    to manage checkpoint progression during episode execution.
    Note: Renamed from DomainTask for clarity.
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
        Initialize a task.

        Args:
            task_id: Unique identifier for the task
            domain: Security domain this task belongs to
            title: Human-readable title
            description: Detailed description of the task
            subtasks: List of subtasks that comprise this task (used as checkpoints)
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

    def check_progression_criteria(self, episode: "Episode") -> Optional[SubTask]:
        """
        Check if any subtask can be progressed based on episode state.

        This replaces agent-facing methods like get_next_subtask() with
        automatic checkpoint progression logic.

        Args:
            episode: Current episode state

        Returns:
            Next subtask that should be activated, or None if no progression
        """
        # Check if any in-progress subtasks can be completed
        for subtask_id in episode.in_progress_subtasks:
            subtask = self.get_subtask_by_id(subtask_id)
            if subtask and subtask.check_exit_conditions(episode):
                # This subtask should be marked as completed
                return subtask

        # Check if any not-visited subtasks can be started
        for subtask in self.subtasks:
            if subtask.subtask_id in episode.not_visited_subtasks and subtask.check_entry_conditions(episode):
                # This subtask can now be started
                return subtask

        return None

    def get_current_checkpoint(self, episode: "Episode") -> Optional[SubTask]:
        """
        Get the current active checkpoint (subtask) for an episode.

        Args:
            episode: Current episode state

        Returns:
            Current active subtask, or None if no active checkpoint
        """
        if episode.current_subtask:
            return self.get_subtask_by_id(episode.current_subtask)
        return None

    def is_complete(self, completed_subtasks: Set[str]) -> bool:
        """
        Check if all subtasks in this task have been completed.

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
                    errors.append(f"SubTask '{subtask.subtask_id}' depends on unknown subtask '{dep_id}'")

        # TODO: Add circular dependency detection

        return errors

    def get_all_subtask_ids(self) -> Set[str]:
        """
        Get all subtask IDs for this task.

        Returns:
            Set of all subtask IDs
        """
        return {subtask.subtask_id for subtask in self.subtasks}
