"""EpisodeManager implementation for RL-friendly task execution."""

import uuid
from dataclasses import asdict
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, Optional

from ..base import Action, CommandResult, Episode, EpisodeState, Step
from .exceptions import EpisodeNotFoundException

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
        self.cleanup_tokens: Dict[str, str] = {}  # session_id -> cleanup_token for container coordination

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

        # Generate cleanup token for container coordination
        cleanup_token = str(uuid.uuid4())

        # Create new episode
        episode = Episode(
            task_id=task_id,
            session_id=session_id,
            state=EpisodeState.ACTIVE,
            context=initial_context or {},
            metadata={"created_at": datetime.utcnow().isoformat(), "cleanup_token": cleanup_token},
            end_time=None,
            completion_reason=None,
        )

        # Store as active episode and register cleanup token
        self.active_episodes[session_id] = episode
        self.cleanup_tokens[session_id] = cleanup_token

        logger.info(f"Created episode '{episode.episode_id}' for session '{session_id}'")
        return episode

    def step(
        self,
        session_id: str,
        action: Action,
        command_result: CommandResult,
    ) -> Step:
        """
        Execute an RL-style step in the current episode.

        Args:
            session_id: ID of the session
            action: Action to execute
            command_result: CommandResult from command execution

        Returns:
            Step object with all step information

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

        return step

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
        episode = self.get_current_episode(session_id)
        if not episode:
            raise EpisodeNotFoundException(session_id)

        logger.info(
            f"Ending episode '{episode.episode_id}'\
                    for session '{session_id}': {reason}"
        )

        # Update episode state
        episode.end_time = datetime.utcnow()
        episode.state = EpisodeState.COMPLETED if "success" in reason.lower() else EpisodeState.FAILED
        episode.completion_reason = reason

        # Remove from active episodes
        del self.active_episodes[session_id]

        # Remove cleanup token (containers will detect this and self-terminate)
        if session_id in self.cleanup_tokens:
            del self.cleanup_tokens[session_id]

        logger.info(
            f"Episode '{episode.episode_id}' completed: {len(episode.steps)} steps,\
                success={episode.state == EpisodeState.COMPLETED}"
        )
        return episode

    def reset_episode(self, session_id: str, task_id: str) -> Episode:
        """
        Reset the current episode (start a new attempt).

        Args:
            session_id: ID of the session
            task_id: Task ID to use for the new episode

        Returns:
            New Episode instance (cleanup token available in episode.metadata["cleanup_token"])

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
        Clean up all episodes for a session.

        Args:
            session_id: ID of the session to clean up
        """
        logger.info(f"Cleaning up episodes for session '{session_id}'")

        # End active episode if exists
        if session_id in self.active_episodes:
            self.end_episode(session_id, "session_cleanup")

        # Clean up any remaining cleanup tokens
        if session_id in self.cleanup_tokens:
            del self.cleanup_tokens[session_id]
            logger.debug(f"Removed cleanup token for session '{session_id}'")

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
        Immediately remove episode tracking due to error, triggering container self-termination.

        This is the key insight - on any error, remove episode tracking so containers
        detect the episode is no longer active and self-terminate.

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

        # Remove cleanup token (containers will detect this and self-terminate)
        if session_id in self.cleanup_tokens:
            del self.cleanup_tokens[session_id]
            logger.info(f"Cleanup token removed for session '{session_id}' - containers will self-terminate")

    def is_episode_active(self, session_id: str, token: str) -> bool:
        """
        Check if an episode is active for container polling.

        Containers use this endpoint to determine if they should continue running.
        If this returns False, containers should self-terminate.

        Args:
            session_id: ID of the session to check
            token: Cleanup token for authentication

        Returns:
            True if episode is active and token is valid, False otherwise
        """
        # Check if cleanup token exists and matches
        if session_id not in self.cleanup_tokens:
            logger.debug(f"No cleanup token found for session '{session_id}' - episode inactive")
            return False

        if self.cleanup_tokens[session_id] != token:
            logger.warning(f"Invalid cleanup token for session '{session_id}' - potential security issue")
            return False

        # Check if episode is still active
        is_active = session_id in self.active_episodes
        logger.debug(f"Episode active check for session '{session_id}': {is_active}")
        return is_active

    def get_cleanup_token(self, session_id: str) -> Optional[str]:
        """
        Get the cleanup token for a session.

        Used by container orchestration systems to pass the token to containers.

        Args:
            session_id: ID of the session

        Returns:
            Cleanup token if session exists, None otherwise
        """
        return self.cleanup_tokens.get(session_id)
