"""EpisodeManager implementation for RL-friendly task execution."""

from dataclasses import asdict
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, NamedTuple, Optional

from ..base import Action, CommandResult, Episode, EpisodeState, Step
from .exceptions import EpisodeNotFoundException

logger = getLogger(__name__)


class StepResult(NamedTuple):
    """Result from executing a step, including termination information."""

    step: Step
    should_terminate: bool
    termination_reason: Optional[str] = None


class EpisodeManager:
    """
    Manages RL-style episodes for task execution.

    Provides gym-compatible interfaces for episode lifecycle management,
    action-response tracking, and automatic subtask progression.
    """

    def __init__(self) -> None:
        """Initialize the EpisodeManager."""
        self.active_episodes: Dict[str, Episode] = {}  # session_id -> Episode
        self.episode_configs: Dict[str, Dict[str, Any]] = {}  # session_id -> episode config from task

    def configure_for_task(self, session_id: str, task: Any) -> None:
        """
        Configure EpisodeManager for a specific task/session.

        Args:
            session_id: Session identifier
            task: Task object containing episode configuration
        """
        # Extract episode configuration from task
        episode_config = {}
        if task and task.episode_config:
            episode_config = task.episode_config.copy()
            logger.debug(f"Using episode config from task {task.task_id}: {episode_config}")

        # Validate required configuration values
        if "max_steps" not in episode_config:
            logger.error(f"Task {task.task_id if task else 'None'} missing required episode_config.max_steps")
            raise ValueError(
                f"Task {task.task_id if task else 'None'} episode_config missing required 'max_steps' value"
            )

        # Apply defaults for optional values
        episode_config.setdefault("step_timeout_seconds", 300)
        episode_config.setdefault("episode_timeout_minutes", 30)

        # Store configuration for this session
        self.episode_configs[session_id] = episode_config

        logger.info(f"EpisodeManager configured for session {session_id} with task {task.task_id if task else 'None'}")

    def should_terminate_episode(self, session_id: str) -> tuple[bool, str]:
        """
        Check if the current episode should be terminated based on configured limits.

        Args:
            session_id: ID of the session

        Returns:
            Tuple of (should_terminate, reason)
        """
        episode = self.get_current_episode(session_id)
        if not episode:
            return True, "no_active_episode"

        if episode.is_complete:
            return True, episode.completion_reason or "completed"

        # Get episode configuration for this session
        episode_config = self.episode_configs.get(session_id)
        if not episode_config:
            logger.error(f"No episode configuration found for session {session_id}")
            return True, f"configuration_error: No episode configuration found for session {session_id}"

        try:
            # Check max steps
            max_steps = episode_config["max_steps"]
            current_steps = len(episode.steps)

            if current_steps >= max_steps:
                return True, f"max_steps_reached ({current_steps}/{max_steps})"

            # Could add more termination conditions here:
            # - episode timeout based on episode_config["episode_timeout_minutes"]
            # - step timeout based on episode_config["step_timeout_seconds"]
            # - resource limits
            # etc.

        except Exception as e:
            logger.error(f"Failed to check episode termination conditions for session {session_id}: {e}")
            return True, f"configuration_error: {e}"

        return False, ""

    def start_episode(self, session_id: str, task_id: str, initial_context: Optional[Dict[str, Any]] = None) -> Episode:
        """
        Start a new episode for a session with the provided task.

        Args:
            session_id: ID of the session starting the episode
            task_id: Task ID to execute
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
            completion_reason=None,
        )

        # Store as active episode
        self.active_episodes[session_id] = episode

        logger.info(f"Created episode '{episode.episode_id}' for session '{session_id}'")
        return episode

    def step(
        self,
        session_id: str,
        action: Action,
        command_result: CommandResult,
    ) -> StepResult:
        """
        Execute an RL-style step in the current episode.

        Args:
            session_id: ID of the session
            action: Action to execute
            command_result: CommandResult from command execution

        Returns:
            StepResult object with step information and termination status

        Raises:
            EpisodeNotFoundException: If session has no active episode
        """
        episode = self.get_current_episode(session_id)
        if not episode:
            raise EpisodeNotFoundException(session_id)

        logger.debug(f"Executing step {len(episode.steps) + 1} for episode '{episode.episode_id}'")

        # Record tool execution and create step
        step = self.create_step(episode, action, command_result)

        # Update episode state
        self.update_episode_state(episode, step)

        # Add step to episode
        episode.add_step(step)

        # Check if episode should terminate after this step
        should_terminate, termination_reason = self.should_terminate_episode(session_id)

        return StepResult(step=step, should_terminate=should_terminate, termination_reason=termination_reason)

    def end_episode(self, session_id: str, reason: str) -> Episode:
        """
        End the current episode for a session.

        Args:
            session_id: ID of the session
            reason: Reason for ending the episode

        Returns:
            Episode with completion information

        Raises:
            EpisodeNotFoundException: If session has no active episode
        """
        logger.warning(
            f"🔥 EPISODE END: EpisodeManager.end_episode() called for session {session_id}, reason: {reason}"
        )
        episode = self.get_current_episode(session_id)
        if not episode:
            raise EpisodeNotFoundException(session_id)

        logger.info(
            f"Ending episode '{episode.episode_id}'\
                    for session '{session_id}': {reason}"
        )

        # Update episode state
        episode.end_time = datetime.utcnow()

        # Consider episode successful if:
        # 1. Reason contains "success" OR
        # 2. Agent voluntarily completed (agent_completed) OR
        # 3. Episode completed normally (completed)
        success_indicators = ["success", "agent_completed", "completed"]
        is_successful = any(indicator in reason.lower() for indicator in success_indicators)

        episode.state = EpisodeState.COMPLETED if is_successful else EpisodeState.FAILED
        episode.completion_reason = reason

        # Remove from active episodes
        logger.info(f"Removing episode {episode.episode_id} from active episodes for session {session_id}")
        del self.active_episodes[session_id]

        logger.info(
            f"Episode '{episode.episode_id}' completed: {len(episode.steps)} steps, "
            f"success={episode.state == EpisodeState.COMPLETED}"
        )
        return episode

    def reset_episode(self, session_id: str, task_id: str) -> Episode:
        """
        Reset the current episode (start a new attempt).

        Args:
            session_id: ID of the session
            task_id: Task ID to use for the new episode

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
            session_id=session_id, task_id=task_id, initial_context=current_episode.context.copy()
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

    def create_step(self, episode: Episode, action: Action, response: CommandResult) -> Step:
        """
        Create a step from an action and response.

        Args:
            episode: Episode to create the step for
            action: Action that was taken
            response: Tool execution response

        Returns:
            Created Step instance (not yet added to episode)
        """
        # Convert CommandResult to dictionary using dataclass asdict
        response_dict = asdict(response)

        # Create step without adding it to episode yet
        step = Step(
            step_number=len(episode.steps),
            timestamp=datetime.utcnow(),
            action=action,  # Pass Action object directly - it should work with Pydantic
            response=response_dict,
            context_snapshot=episode.context.copy(),
            done=False,  # Will be set by external completion logic
        )

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

    def _extract_parameters_from_action(self, action: Action) -> Optional[str]:
        """
        Extract actual parameters executed from action.

        Currently focused on DockerCLIExecutor tool only.

        Args:
            action: Action to extract parameters from

        Returns:
            Extracted parameters string, or None if not extractable
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
            "duration": episode.duration,
            "start_time": episode.start_time.isoformat() if episode.start_time else None,
            "end_time": episode.end_time.isoformat() if episode.end_time else None,
        }

    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up episode resources for a session.

        Args:
            session_id: Session identifier to clean up
        """
        logger.info(f"EpisodeManager.cleanup_session() called for session {session_id}")
        if session_id in self.active_episodes:
            logger.info(f"Ending active episode for session {session_id} during cleanup")
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

    def remove_episode_on_error(self, session_id: str, error: Exception) -> None:
        """
        Immediately remove episode tracking due to error.

        Args:
            session_id: ID of the session with the error
            error: Exception that caused the episode to be removed
        """
        logger.error(f"Removing episode tracking for session '{session_id}' due to error: {error}")

        if session_id in self.active_episodes:
            # End episode with error reason
            episode = self.active_episodes[session_id]
            episode.end_time = datetime.utcnow()
            episode.state = EpisodeState.FAILED
            episode.completion_reason = f"error: {str(error)}"

            # Remove from active tracking
            del self.active_episodes[session_id]
            logger.info(f"Episode '{episode.episode_id}' removed from tracking due to error")
