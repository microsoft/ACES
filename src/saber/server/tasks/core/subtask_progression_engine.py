"""SubTaskProgressionEngine for handling DAG-based progression logic."""

from logging import getLogger
from typing import Set

from ..episodes.episode import Action, Episode, Step
from .task import Task

logger = getLogger(__name__)


class SubTaskProgressionEngine:
    """
    Handles DAG-based subtask progression logic and state management.

    Responsible for:
    - Checking subtask completion conditions based on command execution
    - Managing subtask state transitions (not_visited → in_progress → completed)
    - Validating entry/exit conditions for subtasks
    - Initializing episode subtask state
    """

    def check_subtask_progression(self, task: Task, episode: Episode, step: Step) -> None:
        """
        Check and handle automatic subtask progression based on DAG logic.

        Args:
            task: Task definition containing subtasks
            episode: Episode to check for progression
            step: Latest step that was executed
        """
        # Update step with current subtask state
        self._update_step_subtask_state(episode, step)

        # Check for completed subtasks based on command execution
        completed_subtasks = self._check_completed_subtasks(task, episode)

        # Check for newly available subtasks (entry conditions met)
        available_subtasks = self._check_available_subtasks(task, episode)

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

    def initialize_episode_subtasks(self, task: Task, episode: Episode) -> None:
        """
        Initialize subtask state for a new episode.

        Args:
            task: Task definition
            episode: Episode to initialize
        """
        all_subtask_ids = {st.subtask_id for st in task.subtasks}

        # Find entry point subtasks (no dependencies)
        entry_subtasks = {st.subtask_id for st in task.subtasks if not st.depends_on}

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

        logger.info(f"Initialized episode '{episode.episode_id}' with {len(entry_subtasks)} entry subtasks")

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

    def _check_completed_subtasks(self, task: Task, episode: Episode) -> Set[str]:
        """
        Check which subtasks have completed based on command execution.

        Args:
            task: Task definition
            episode: Current episode

        Returns:
            Set of completed subtask IDs
        """
        completed_subtasks = episode.completed_subtasks.copy()

        # Check each in-progress subtask for completion
        for subtask_id in episode.in_progress_subtasks:
            subtask = task.get_subtask_by_id(subtask_id)
            if subtask and subtask.check_exit_conditions(episode):
                completed_subtasks.add(subtask_id)
                logger.debug(f"Subtask '{subtask_id}' completed exit conditions")

        return completed_subtasks

    def _check_available_subtasks(self, task: Task, episode: Episode) -> Set[str]:
        """
        Check which new subtasks are now available (entry conditions met).

        Args:
            task: Task definition
            episode: Current episode

        Returns:
            Set of newly available subtask IDs
        """
        available_subtasks = set()

        # Check each not-visited subtask for entry conditions
        for subtask_id in episode.not_visited_subtasks:
            subtask = task.get_subtask_by_id(subtask_id)
            if subtask and subtask.check_entry_conditions(episode):
                available_subtasks.add(subtask_id)
                logger.debug(f"Subtask '{subtask_id}' now available (entry conditions met)")

        return available_subtasks
