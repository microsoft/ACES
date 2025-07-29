"""EpisodeManager implementation for RL-friendly task execution."""

from dataclasses import asdict
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, Optional

from ...tools.base import ToolResult
from ..base import Action, EpisodeState, Step
from ..exceptions import EpisodeNotFoundException
from .episode import Episode

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
        Start a new episode for a session with the provided task.

        Args:
            session_id: ID of the session starting the episode
            task: Task instance to execute
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
        tool_result: ToolResult,
        current_objective: Optional[str] = None,
    ) -> Step:
        """
        Execute an RL-style step in the current episode.

        Args:
            session_id: ID of the session
            action: Action to execute
            tool_result: ToolResult from tool execution
            current_objective: Current objective based on active subtasks

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
        step = self.create_step(episode, action, tool_result)

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
            task: Task instance to use for the new episode

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

    def create_step(self, episode: Episode, action: Action, response: ToolResult) -> Step:
        """
        Create a step from an action and response.

        Args:
            episode: Episode to create the step for
            action: Action that was taken
            response: Tool execution response

        Returns:
            Created Step instance (not yet added to episode)
        """
        # Convert ToolResult to dictionary using dataclass asdict
        response_dict = asdict(response)

        # Extract command from action for completion matching
        command = self._extract_command_from_action(action)
        action.command = command

        # Create step without adding it to episode yet
        step = Step(
            step_number=len(episode.steps),
            timestamp=datetime.utcnow(),
            action=action,
            response=response_dict,
            # These will be updated by the Task progression logic
            current_subtask=episode.current_subtask,
            completed_subtasks=episode.completed_subtasks.copy(),
            in_progress_subtasks=episode.in_progress_subtasks.copy(),
            not_visited_subtasks=episode.not_visited_subtasks.copy(),
            context_snapshot=episode.context.copy(),
            done=False,  # Will be set by Task progression logic
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
