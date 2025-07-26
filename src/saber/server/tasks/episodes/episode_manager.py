"""EpisodeManager implementation for RL-friendly task execution."""

from datetime import datetime
from logging import getLogger
from typing import Any, Dict, List, Optional

from ..base import EpisodeState
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
        self.episode_history: Dict[str, List[Episode]] = {}  # session_id -> List[Episode]

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

        # Calculate attempt number
        attempt_number = 1
        if session_id in self.episode_history:
            # Count previous episodes for this task
            task_episodes = [ep for ep in self.episode_history[session_id] if ep.task_id == task_id]
            attempt_number = len(task_episodes) + 1

        # Create new episode
        episode = Episode(
            task_id=task_id,
            session_id=session_id,
            attempt_number=attempt_number,
            state=EpisodeState.ACTIVE,
            context=initial_context or {},
            metadata={"created_at": datetime.utcnow().isoformat(), "attempt_number": attempt_number},
            end_time=None,
        )

        # Store as active episode
        self.active_episodes[session_id] = episode

        # Initialize episode history if needed
        if session_id not in self.episode_history:
            self.episode_history[session_id] = []

        logger.info(f"Created episode '{episode.episode_id}' (attempt {attempt_number}) for session '{session_id}'")
        return episode

    def step(self, session_id: str, action: Action) -> StepResult:
        """
        Execute an RL-style step in the current episode.

        Args:
            session_id: ID of the session
            action: Action to execute

        Returns:
            StepResult with observation, reward, done, and info

        Raises:
            EpisodeNotFoundException: If session has no active episode
        """
        episode = self.get_current_episode(session_id)
        if not episode:
            raise EpisodeNotFoundException(session_id)

        logger.debug(f"Executing step {len(episode.steps) + 1} for episode '{episode.episode_id}'")

        # This is a placeholder - the actual tool execution and step creation
        # will be handled by the TaskManager integration
        # For now, return a basic step result
        step_result = StepResult(
            observation={"current_step": len(episode.steps) + 1},
            reward=0.0,
            done=False,
            info={"episode_id": episode.episode_id},
        )

        return step_result

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

        # Calculate final reward (sum of all step rewards)
        final_reward = sum(step.reward or 0.0 for step in episode.steps)

        # Create result
        result = EpisodeResult(
            episode_id=episode.episode_id,
            success=episode.state == EpisodeState.COMPLETED,
            total_steps=len(episode.steps),
            duration=episode.duration,
            final_reward=final_reward,
            completion_reason=reason,
            metadata=episode.metadata.copy(),
        )

        # Move to history and remove from active
        self.episode_history[session_id].append(episode)
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

    def replay_episode(self, episode_id: str) -> Optional[Episode]:
        """
        Get a specific episode for replay.

        Args:
            episode_id: ID of the episode to replay

        Returns:
            Episode instance if found, None otherwise
        """
        # Search through all episode history
        for episodes in self.episode_history.values():
            for episode in episodes:
                if episode.episode_id == episode_id:
                    logger.info(f"Found episode '{episode_id}' for replay")
                    return episode

        # Check active episodes too
        for episode in self.active_episodes.values():
            if episode.episode_id == episode_id:
                logger.info(f"Found active episode '{episode_id}' for replay")
                return episode

        logger.warning(f"Episode '{episode_id}' not found for replay")
        return None

    def get_current_episode(self, session_id: str) -> Optional[Episode]:
        """
        Get the current active episode for a session.

        Args:
            session_id: ID of the session

        Returns:
            Current Episode instance, or None if no active episode
        """
        return self.active_episodes.get(session_id)

    def create_step(self, episode: Episode, action: Action, response: Dict[str, Any]) -> Step:
        """
        Create a step from an action and response.

        Args:
            episode: Episode to add the step to
            action: Action that was taken
            response: Tool execution response

        Returns:
            Created Step instance
        """
        # Extract command from action for completion matching
        command = self._extract_command_from_action(action)
        action.command = command

        # Create step
        step = Step(
            step_number=len(episode.steps) + 1,
            timestamp=datetime.utcnow(),
            action=action,
            response=response,
            # These will be updated by the TaskManager progression logic
            current_subtask=episode.current_subtask,
            completed_subtasks=episode.completed_subtasks.copy(),
            in_progress_subtasks=episode.in_progress_subtasks.copy(),
            not_visited_subtasks=episode.not_visited_subtasks.copy(),
            context_snapshot=episode.context.copy(),
            reward=0.0,  # Will be calculated by TaskManager
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
        # Update episode context with any changes from the step
        episode.context.update(step.context_snapshot)

        # If step marked as done, update episode state
        if step.done:
            episode.state = EpisodeState.COMPLETED if step.reward and step.reward > 0 else EpisodeState.FAILED

    def _extract_command_from_action(self, action: Action) -> Optional[str]:
        """
        Extract actual command executed from action parameters.

        Args:
            action: Action to extract command from

        Returns:
            Extracted command string, or None if not extractable
        """
        if action.tool_name == "docker_cli_executor":
            # Extract command from CLI tool parameters
            command = action.parameters.get("command", "")
            return str(command) if command else None
        elif action.tool_name == "file_operation":
            # Construct command representation for file operations
            operation = action.parameters.get("operation", "")
            path = action.parameters.get("path", "")
            return f"{operation} {path}" if operation else None
        # Add other tool types as needed
        return action.tool_name  # Fallback to tool name

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
            "attempt_number": episode.attempt_number,
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

        # Optional: Clean up history (or keep for analysis)
        # del self.episode_history[session_id]

    def get_session_episodes(self, session_id: str) -> List[Episode]:
        """
        Get all episodes (active and historical) for a session.

        Args:
            session_id: ID of the session

        Returns:
            List of all episodes for the session
        """
        episodes = []

        # Add historical episodes
        if session_id in self.episode_history:
            episodes.extend(self.episode_history[session_id])

        # Add active episode
        if session_id in self.active_episodes:
            episodes.append(self.active_episodes[session_id])

        return episodes

    def build_observation(self, episode: Episode, task_subtask_count: int) -> Dict[str, Any]:
        """
        Build RL observation for current episode state.

        Args:
            episode: Episode to build observation for
            task_subtask_count: Total number of subtasks in the task

        Returns:
            Observation dictionary
        """
        return {
            "task_id": episode.task_id,
            "episode_id": episode.episode_id,
            "current_subtask": episode.current_subtask,
            "completed_subtasks": list(episode.completed_subtasks),
            "in_progress_subtasks": list(episode.in_progress_subtasks),
            "not_visited_subtasks": list(episode.not_visited_subtasks),
            "total_subtasks": task_subtask_count,
            "completion_percentage": (
                len(episode.completed_subtasks) / task_subtask_count if task_subtask_count > 0 else 1.0
            ),
            "total_steps": len(episode.steps),
            "context": episode.context.copy(),
        }

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
