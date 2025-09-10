"""EpisodeManager implementation for RL-friendly task execution."""

from dataclasses import asdict
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, List, NamedTuple, Optional

from ..base import Action, CommandResult, Episode, EpisodeState, Step
from .constants import EpisodeTerminationReason
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
    Supports multiple concurrent episodes per session.
    """

    def __init__(self) -> None:
        """Initialize the EpisodeManager."""
        self.episodes: Dict[str, Episode] = {}  # episode_id -> Episode
        self.session_episodes: Dict[str, List[str]] = {}  # session_id -> [episode_ids]
        self.completed_episodes: Dict[str, Episode] = {}  # episode_id -> completed Episode (for history)
        self.episode_configs: Dict[str, Dict[str, Any]] = {}  # episode_id -> episode config from task

    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """Get episode by episode ID from active or completed episodes."""
        return self.episodes.get(episode_id) or self.completed_episodes.get(episode_id)

    def get_session_episodes(self, session_id: str, include_completed: bool = False) -> List[Episode]:
        """Get all episodes for a session."""
        episode_ids = self.session_episodes.get(session_id, [])
        episodes = []

        for episode_id in episode_ids:
            episode = self.episodes.get(episode_id)
            if episode:
                episodes.append(episode)
            elif include_completed:
                completed_episode = self.completed_episodes.get(episode_id)
                if completed_episode:
                    episodes.append(completed_episode)

        return episodes

    def get_active_episodes_for_session(self, session_id: str) -> List[Episode]:
        """Get only active episodes for a session."""
        return self.get_session_episodes(session_id, include_completed=False)

    def add_episode_to_session(self, session_id: str, episode: Episode) -> None:
        """Add an episode to a session's episode list."""
        if session_id not in self.session_episodes:
            self.session_episodes[session_id] = []

        self.session_episodes[session_id].append(episode.episode_id)
        self.episodes[episode.episode_id] = episode

    def complete_episode(self, episode_id: str) -> Optional[Episode]:
        """Move an episode from active to completed."""
        episode = self.episodes.pop(episode_id, None)
        if episode:
            self.completed_episodes[episode_id] = episode
        return episode

    def configure_for_task(self, episode_id: str, task: Any) -> None:
        """
        Configure EpisodeManager for a specific task/episode.

        Args:
            episode_id: Episode identifier
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

        # Store configuration for this episode
        self.episode_configs[episode_id] = episode_config

        logger.info(f"EpisodeManager configured for episode {episode_id} with task {task.task_id if task else 'None'}")

    def should_terminate_episode(self, episode_id: str) -> tuple[bool, str]:
        """
        Check if an episode should be terminated based on configured limits.

        Args:
            episode_id: ID of the episode to check

        Returns:
            Tuple of (should_terminate, reason)
        """
        episode = self.get_episode_by_id(episode_id)
        if not episode:
            return True, "episode_not_found"

        if episode.is_complete:
            return True, episode.completion_reason or EpisodeTerminationReason.COMPLETED

        # Get episode configuration for this episode
        episode_config = self.episode_configs.get(episode_id)
        if not episode_config:
            logger.error(f"No episode configuration found for episode {episode_id}")
            return True, f"configuration_error: No episode configuration found for episode {episode_id}"

        try:
            # Note: Step limit checking moved to client-side
            # Server no longer terminates episodes based on step count
            max_steps = episode_config["max_steps"]
            current_steps = len(episode.steps)

            logger.debug(
                f"� Step count info: episode={episode_id}, current_steps={current_steps}, "
                f"max_steps={max_steps} (server-side termination disabled)"
            )

            # Could add other termination conditions here:
            # - episode timeout based on episode_config["episode_timeout_minutes"]
            # - step timeout based on episode_config["step_timeout_seconds"]
            # - resource limits
            # etc.

        except Exception as e:
            logger.error(f"Failed to check episode termination conditions for episode {episode_id}: {e}")
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
            eval_submission=None,
            completion_reason=None,
            submission=None,
        )

        # Add episode to session's episode tracking
        self.add_episode_to_session(session_id, episode)

        logger.info(f"Created episode '{episode.episode_id}' for session '{session_id}'")
        return episode

    def step(
        self,
        episode_id: str,
        action: Action,
        command_result: CommandResult,
    ) -> StepResult:
        """
        Execute an RL-style step in the specified episode.

        Args:
            episode_id: ID of the episode
            action: Action to execute
            command_result: CommandResult from command execution

        Returns:
            StepResult object with step information and termination status

        Raises:
            EpisodeNotFoundException: If episode not found
        """
        episode = self.get_episode_by_id(episode_id)
        if not episode:
            raise EpisodeNotFoundException(episode_id)

        logger.debug(f"Executing step {len(episode.steps) + 1} for episode '{episode.episode_id}'")

        # Note: Step limit enforcement has been moved to client-side
        # The server no longer automatically terminates episodes based on step count
        current_steps = len(episode.steps)
        will_terminate_after_this_step = False
        termination_reason = None

        # Log step information for debugging (no automatic termination)
        episode_config = self.episode_configs.get(episode_id)
        if episode_config and "max_steps" in episode_config:
            max_steps = episode_config["max_steps"]
            logger.debug(
                f"📊 Step {current_steps + 1}/{max_steps} for episode {episode_id} (server-side limit checking disabled)"
            )

        # Record tool execution and create step
        step = self.create_step(episode, action, command_result)

        # Update episode state
        self.update_episode_state(episode, step)

        # Add step to episode
        episode.add_step(step)

        # Return result with termination info determined in advance
        return StepResult(
            step=step, should_terminate=will_terminate_after_this_step, termination_reason=termination_reason
        )

    def end_episode(self, episode_id: str, reason: str, result: Optional[str] = None) -> Episode:
        """
        End the specified episode.

        Args:
            episode_id: ID of the episode to end
            reason: Reason for ending the episode
            result: Optional result/submission from the episode (e.g., captured flag)

        Returns:
            Episode with completion information

        Raises:
            EpisodeNotFoundException: If episode not found
        """
        logger.warning(
            f"🔥 EPISODE END: EpisodeManager.end_episode() called for episode {episode_id}, reason: {reason}"
        )
        episode = self.get_episode_by_id(episode_id)
        if not episode:
            raise EpisodeNotFoundException(episode_id)

        logger.info(f"Ending episode '{episode.episode_id}': {reason}")

        # Create a final step if the agent provided a result/submission
        if result:
            logger.info(f"Creating final step for episode result: {result}")

            # Create an action representing the agent's submission
            submission_action = Action(
                tool_name="submission", parameters={"result": result, "submission": result, "episode_completed": True}
            )

            # Create a successful command result for the submission
            submission_result = CommandResult.success_result(
                data={"submission": result, "message": f"Episode completed with result: {result}"}, execution_time=0.0
            )

            # Create and add the final step
            final_step = self.create_step(episode, submission_action, submission_result)
            episode.steps.append(final_step)
            logger.info(f"Added final step {final_step.step_number} with submission: {result}")

        # Update episode state
        episode.end_time = datetime.utcnow()

        # Consider episode successful if:
        # 1. Reason contains "success" OR
        # 2. Agent voluntarily completed (agent_completed) OR
        # 3. Episode completed normally (completed)
        success_indicators = [
            EpisodeTerminationReason.SUCCESS,
            EpisodeTerminationReason.AGENT_COMPLETED,
            EpisodeTerminationReason.COMPLETED,
        ]
        is_successful = any(indicator in reason.lower() for indicator in success_indicators)

        episode.state = EpisodeState.COMPLETED if is_successful else EpisodeState.FAILED
        episode.completion_reason = reason

        # Move from active to completed episodes
        if episode_id in self.episodes:
            logger.info(f"Moving episode {episode_id} from active to completed")
            self.complete_episode(episode_id)

        # Clean up episode configuration
        if episode_id in self.episode_configs:
            del self.episode_configs[episode_id]

        logger.info(
            f"Episode '{episode.episode_id}' completed: {len(episode.steps)} steps, "
            f"success={episode.state == EpisodeState.COMPLETED}"
        )
        return episode

    def get_episode_state(self, episode_id: str) -> Optional[EpisodeState]:
        """
        Get the episode state for a specific episode.

        Args:
            episode_id: ID of the episode

        Returns:
            Episode state, or None if episode not found
        """
        episode = self.get_episode_by_id(episode_id)
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

        # End all active episodes for this session
        active_episodes = self.get_active_episodes_for_session(session_id)
        for episode in active_episodes:
            logger.info(f"Ending active episode {episode.episode_id} for session {session_id} during cleanup")
            self.end_episode(episode.episode_id, "session_cleanup")

        # Clean up session episode tracking
        if session_id in self.session_episodes:
            del self.session_episodes[session_id]

    def remove_episode_on_error(self, episode_id: str, error: Exception) -> None:
        """
        Immediately remove episode tracking due to error.

        Args:
            episode_id: ID of the episode with the error
            error: Exception that caused the episode to be removed
        """
        logger.error(f"Removing episode tracking for episode '{episode_id}' due to error: {error}")

        episode = self.get_episode_by_id(episode_id)
        if episode:
            # Mark episode as failed
            episode.end_time = datetime.utcnow()
            episode.state = EpisodeState.FAILED
            episode.completion_reason = f"error: {str(error)}"

            # Move from active to completed
            self.complete_episode(episode_id)

            # Clean up episode configuration
            if episode_id in self.episode_configs:
                del self.episode_configs[episode_id]

            logger.info(f"Episode '{episode.episode_id}' removed from tracking due to error")
