"""Task implementation for task management system."""

from logging import getLogger
from typing import Any, Dict, List, Optional, Set

from ..base import Action, Step
from ..episodes.episode import Episode
from .subtask import SubTask

logger = getLogger(__name__)


class Task:
    """
    Represents a high-level security scenario composed of multiple subtasks.

    A task maintains a collection of subtasks and provides methods
    to manage checkpoint progression during episode execution.
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

    def check_episode_progression(self, episode: Episode, step: Step) -> None:
        """
        Check and handle automatic subtask progression based on DAG logic.

        Args:
            episode: Episode to check for progression
            step: Latest step that was executed
        """
        # Update step with current subtask state
        self._update_step_subtask_state(episode, step)

        # Check for completed subtasks based on command execution
        completed_subtasks = self._check_completed_subtasks(episode)

        # Check for newly available subtasks (entry conditions met)
        available_subtasks = self._check_available_subtasks(episode)

        # Update episode step state
        step.completed_subtasks = completed_subtasks
        step.in_progress_subtasks = episode.in_progress_subtasks.copy()
        step.not_visited_subtasks = episode.not_visited_subtasks.copy()

        # Move newly available subtasks from not_visited to in_progress
        for subtask_id in available_subtasks:
            if subtask_id in step.not_visited_subtasks:
                step.not_visited_subtasks.remove(subtask_id)
                step.in_progress_subtasks.add(subtask_id)

        # Move completed subtasks from in_progress to completed
        newly_completed = completed_subtasks - episode.completed_subtasks
        for subtask_id in newly_completed:
            if subtask_id in step.in_progress_subtasks:
                step.in_progress_subtasks.remove(subtask_id)

        # Log progression
        if newly_completed:
            for subtask_id in newly_completed:
                logger.info(f"Episode '{episode.episode_id}' completed subtask '{subtask_id}'")

        if available_subtasks:
            for subtask_id in available_subtasks:
                logger.info(f"Episode '{episode.episode_id}' can now access subtask '{subtask_id}'")

    def initialize_episode(self, episode: Episode) -> None:
        """
        Initialize subtask state for a new episode.

        Args:
            episode: Episode to initialize
        """
        all_subtask_ids = self.get_all_subtask_ids()

        # Find entry point subtasks (no dependencies)
        entry_subtasks = {st.subtask_id for st in self.subtasks if not st.depends_on}

        # Initialize episode step state if no steps exist yet
        if not episode.steps:
            # Create initial step to establish subtask state
            initial_action = Action(tool_name="system_init", parameters={}, command=None)

            initial_step = Step(
                step_number=0,
                action=initial_action,
                response={"status": "episode_initialized"},
                current_subtask=None,
                completed_subtasks=set(),
                in_progress_subtasks=entry_subtasks.copy(),
                not_visited_subtasks=all_subtask_ids - entry_subtasks,
                context_snapshot=episode.context.copy(),
                done=False,
            )

            episode.steps.append(initial_step)

        logger.info(
            f"Initialized episode '{episode.episode_id}' with {len(entry_subtasks)}\
                entry subtasks for task '{self.task_id}'"
        )

    def _update_step_subtask_state(self, episode: Episode, step: Step) -> None:
        """
        Update step with current subtask state from episode.

        Args:
            episode: Current episode
            step: Step to update
        """
        # Copy current state from episode or previous step
        if episode.steps:
            previous_step = episode.steps[-1]
            step.current_subtask = previous_step.current_subtask
            step.completed_subtasks = previous_step.completed_subtasks.copy()
            step.in_progress_subtasks = previous_step.in_progress_subtasks.copy()
            step.not_visited_subtasks = previous_step.not_visited_subtasks.copy()
        else:
            # Initialize from episode
            step.current_subtask = episode.current_subtask
            step.completed_subtasks = episode.completed_subtasks.copy()
            step.in_progress_subtasks = episode.in_progress_subtasks.copy()
            step.not_visited_subtasks = episode.not_visited_subtasks.copy()

    def _check_completed_subtasks(self, episode: Episode) -> Set[str]:
        """
        Check which subtasks have completed based on command execution.

        Args:
            episode: Current episode

        Returns:
            Set of completed subtask IDs
        """
        completed_subtasks = episode.completed_subtasks.copy()

        # Check each in-progress subtask for completion
        for subtask_id in episode.in_progress_subtasks:
            subtask = self.get_subtask_by_id(subtask_id)
            if subtask and subtask.check_exit_conditions(episode):
                completed_subtasks.add(subtask_id)
                logger.debug(f"Subtask '{subtask_id}' completed exit conditions")

        return completed_subtasks

    def _check_available_subtasks(self, episode: Episode) -> Set[str]:
        """
        Check which new subtasks are now available (entry conditions met).

        Args:
            episode: Current episode

        Returns:
            Set of newly available subtask IDs
        """
        available_subtasks = set()

        # Check each not-visited subtask for entry conditions
        for subtask_id in episode.not_visited_subtasks:
            subtask = self.get_subtask_by_id(subtask_id)
            if subtask and subtask.check_entry_conditions(episode):
                available_subtasks.add(subtask_id)
                logger.debug(f"Subtask '{subtask_id}' now available (entry conditions met)")

        return available_subtasks

    def get_current_subtask(self, episode: Episode) -> Optional[SubTask]:
        """
        Get the current active subtask for an episode.

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
