"""EpisodeManager implementation for RL-friendly task execution."""

from datetime import datetime
from logging import getLogger
from typing import Any, Dict, Optional

from ...tools.base import ToolResult
from ..base import EpisodeState, Observation
from ..core import SubTaskProgressionEngine, Task
from ..exceptions import EpisodeNotFoundException
from .episode import Action, Episode, EpisodeResult, Step, StepResult

logger = getLogger(__name__)


class EpisodeManager:
    """
    Manages RL-style episodes for task execution.

    Provides gym-compatible interfaces for episode lifecycle management,
    action-response tracking, and automatic subtask progression.
    """

    def __init__(self) -> None:
        """Initialize the EpisodeManager."""
        self.active_episodes: Dict[str, Episode] = {}  # session_id -> Episode

    def start_episode(self, session_id: str, task_id: str, initial_context: Optional[Dict[str, Any]] = None) -> Episode:
        """
        Start a new episode for a session.

        Args:
            session_id: ID of the session starting the episode
            task_id: ID of the task to execute
            initial_context: Initial context for the episode

        Returns:
            New Episode instance
        """
        logger.info(f"Starting new episode for session '{session_id}' with task '{task_id}'")

        # Create new episode
        episode = Episode(
            task_id=task_id,
            session_id=session_id,
            state=EpisodeState.ACTIVE,
            context=initial_context or {},
            metadata={"created_at": datetime.utcnow().isoformat()},
            end_time=None,
        )

        # Store as active episode
        self.active_episodes[session_id] = episode

        logger.info(f"Created episode '{episode.episode_id}' for session '{session_id}'")
        return episode

    def step(
        self,
        session_id: str,
        action: Action,
        tool_result: ToolResult,
        task_subtask_count: int = 0,
        task_description: str = "",
        current_objective: Optional[str] = None,
        all_subtask_ids: Optional[set] = None,
        task: Optional[Task] = None,
        progression_engine: Optional[SubTaskProgressionEngine] = None,
    ) -> StepResult:
        """
        Execute an RL-style step in the current episode.

        Args:
            session_id: ID of the session
            action: Action to execute
            tool_result: ToolResult from tool execution
            task_subtask_count: Total number of subtasks in the task
            task_description: Description of the current task
            current_objective: Current objective based on active subtasks
            all_subtask_ids: Set of all subtask IDs for completion checking
            task: Task instance for progression logic
            progression_engine: SubTaskProgressionEngine for progression logic

        Returns:
            StepResult with observation, done, and info

        Raises:
            EpisodeNotFoundException: If session has no active episode
        """
        episode = self.get_current_episode(session_id)
        if not episode:
            raise EpisodeNotFoundException(session_id)

        logger.debug(f"Executing step {len(episode.steps) + 1} for episode '{episode.episode_id}'")

        # Record tool execution and create step
        step = self.create_step(episode, action, tool_result)
        self.update_episode_state(episode, step)

        # Run progression logic if both task and progression_engine provided
        if task and progression_engine:
            progression_engine.check_subtask_progression(task, episode, step)

        # Build observation
        observation = self.build_observation(
            episode=episode,
            task_subtask_count=task_subtask_count,
            task_description=task_description,
            current_objective=current_objective,
            last_action={
                "tool_name": action.tool_name,
                "timestamp": action.timestamp.isoformat() if action.timestamp else None,
            },
        )

        # Check if episode is complete
        done = False
        if all_subtask_ids:
            done = self.is_episode_complete(episode, all_subtask_ids)

        info = {
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "current_subtask": episode.current_subtask,
            "total_steps": len(episode.steps),
            "completion_percentage": (
                len(episode.completed_subtasks) / task_subtask_count if task_subtask_count > 0 else 1.0
            ),
            "subtask_states": {
                "completed": list(episode.completed_subtasks),
                "in_progress": list(episode.in_progress_subtasks),
                "not_visited": list(episode.not_visited_subtasks),
            },
        }

        return StepResult(observation=observation.model_dump(), done=done, info=info)

    def end_episode(self, session_id: str, reason: str) -> EpisodeResult:
        """
        End the current episode for a session.

        Args:
            session_id: ID of the session
            reason: Reason for ending the episode

        Returns:
            EpisodeResult with completion information

        Raises:
            EpisodeNotFoundException: If session has no active episode
        """
        episode = self.get_current_episode(session_id)
        if not episode:
            raise EpisodeNotFoundException(session_id)

        logger.info(f"Ending episode '{episode.episode_id}' for session '{session_id}': {reason}")

        # Update episode state
        episode.end_time = datetime.utcnow()
        episode.state = EpisodeState.COMPLETED if "success" in reason.lower() else EpisodeState.FAILED

        # Create result
        result = EpisodeResult(
            episode_id=episode.episode_id,
            success=episode.state == EpisodeState.COMPLETED,
            total_steps=len(episode.steps),
            duration=episode.duration,
            final_reward=0.0,  # TODO: Hook in EvaluationManager to calculate this reward, leaving it zero for now
            completion_reason=reason,
            metadata=episode.metadata.copy(),
        )

        # Remove from active episodes
        del self.active_episodes[session_id]

        logger.info(f"Episode '{episode.episode_id}' completed: {result.total_steps} steps, success={result.success}")
        return result

    def reset_episode(self, session_id: str) -> Episode:
        """
        Reset the current episode (start a new attempt).

        Args:
            session_id: ID of the session

        Returns:
            New Episode instance

        Raises:
            EpisodeNotFoundException: If session has no active episode
        """
        current_episode = self.get_current_episode(session_id)
        if not current_episode:
            raise EpisodeNotFoundException(session_id)

        logger.info(f"Resetting episode for session '{session_id}'")

        # End current episode with reset reason
        self.end_episode(session_id, "reset")

        # Start new episode with same task and context
        new_episode = self.start_episode(
            session_id=session_id, task_id=current_episode.task_id, initial_context=current_episode.context.copy()
        )

        return new_episode

    def get_episode_state(self, session_id: str) -> Optional[EpisodeState]:
        """
        Get the current episode state for a session.

        Args:
            session_id: ID of the session

        Returns:
            Current episode state, or None if no active episode
        """
        episode = self.get_current_episode(session_id)
        return episode.state if episode else None

    def create_step(self, episode: Episode, action: Action, response: ToolResult) -> Step:
        """
        Create a step from an action and response.

        Args:
            episode: Episode to add the step to
            action: Action that was taken
            response: Tool execution response

        Returns:
            Created Step instance
        """
        response_dict = {
            "success": response.success,
            "data": response.data,
            "error": response.error,
            "execution_time": response.execution_time,
            "metadata": response.metadata,
        }

        # Extract command from action for completion matching
        command = self._extract_command_from_action(action)
        action.command = command

        # Create step
        step = Step(
            step_number=len(episode.steps) + 1,
            timestamp=datetime.utcnow(),
            action=action,
            response=response_dict,
            # These will be updated by the TaskManager progression logic
            current_subtask=episode.current_subtask,
            completed_subtasks=episode.completed_subtasks.copy(),
            in_progress_subtasks=episode.in_progress_subtasks.copy(),
            not_visited_subtasks=episode.not_visited_subtasks.copy(),
            context_snapshot=episode.context.copy(),
            done=False,  # Will be set by TaskManager
        )

        # Add step to episode
        episode.add_step(step)
        return step

    def update_episode_state(self, episode: Episode, step: Step) -> None:
        """
        Update episode state after a step.

        Args:
            episode: Episode to update
            step: Step that was completed
        """
        episode.context.update(step.context_snapshot)
        if step.done:
            episode.state = EpisodeState.COMPLETED

    def _extract_command_from_action(self, action: Action) -> Optional[str]:
        """
        Extract actual command executed from action parameters.

        Currently focused on DockerCLIExecutor tool only.

        Args:
            action: Action to extract command from

        Returns:
            Extracted command string, or None if not extractable
        """
        if action.tool_name == "docker_cli_executor":
            # Extract command from DockerCLIExecutor parameters
            command = action.parameters.get("command", "")
            return str(command) if command else None

        # For any other tools, just return the tool name as fallback
        return action.tool_name

    def _get_episode_progress_info(self, episode: Episode) -> Dict[str, Any]:
        """
        Get progress information for an episode.

        Args:
            episode: Episode to get progress for

        Returns:
            Dictionary with progress information
        """
        return {
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "state": episode.state.value,
            "total_steps": len(episode.steps),
            "current_subtask": episode.current_subtask,
            "completed_subtasks": list(episode.completed_subtasks),
            "in_progress_subtasks": list(episode.in_progress_subtasks),
            "not_visited_subtasks": list(episode.not_visited_subtasks),
            "duration": episode.duration,
            "start_time": episode.start_time.isoformat() if episode.start_time else None,
            "end_time": episode.end_time.isoformat() if episode.end_time else None,
        }

    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up all episodes for a session.

        Args:
            session_id: ID of the session to clean up
        """
        logger.info(f"Cleaning up episodes for session '{session_id}'")

        # End active episode if exists
        if session_id in self.active_episodes:
            self.end_episode(session_id, "session_cleanup")

    def get_current_episode(self, session_id: str) -> Optional[Episode]:
        """
        Get the current active episode for a session.

        Args:
            session_id: ID of the session

        Returns:
            Current Episode instance, or None if no active episode
        """
        return self.active_episodes.get(session_id)

    def build_observation(
        self,
        episode: Episode,
        task_subtask_count: int,
        task_description: str = "",
        current_objective: Optional[str] = None,
        last_action: Optional[Dict[str, Any]] = None,
    ) -> Observation:
        """
        Build RL observation for current episode state.

        Args:
            episode: Episode to build observation for
            task_subtask_count: Total number of subtasks in the task
            task_description: Description of the current task
            current_objective: Current objective based on active subtasks
            last_action: Information about the last action taken

        Returns:
            Observation object
        """
        return Observation(
            task_id=episode.task_id,
            episode_id=episode.episode_id,
            current_subtask=episode.current_subtask,
            completed_subtasks=list(episode.completed_subtasks),
            in_progress_subtasks=list(episode.in_progress_subtasks),
            not_visited_subtasks=list(episode.not_visited_subtasks),
            total_subtasks=task_subtask_count,
            completion_percentage=(
                len(episode.completed_subtasks) / task_subtask_count if task_subtask_count > 0 else 1.0
            ),
            total_steps=len(episode.steps),
            context=episode.context.copy(),
            task_description=task_description,
            current_objective=current_objective,
            last_action=last_action,
        )

    def is_episode_complete(self, episode: Episode, all_subtask_ids: set) -> bool:
        """
        Check if episode is complete (all subtasks completed).

        Args:
            episode: Episode to check
            all_subtask_ids: Set of all subtask IDs in the task

        Returns:
            True if episode is complete
        """
        return episode.completed_subtasks >= all_subtask_ids
