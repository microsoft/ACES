"""EpisodeManager implementation for RL-friendly task execution.

Logging Category: EPISODE
"""

import asyncio
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional

from saber.logging_config import LogCategory, get_saber_logger

from ..base import Action, CommandResult, Episode, EpisodeState, Step
from .constants import EpisodeTerminationReason
from .exceptions import EpisodeNotFoundException

logger = get_saber_logger(LogCategory.EPISODE, __name__)


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

    def find_available_episode_for_dependency(
        self, session_id: str, target_task_id: str, dependent_task_id: str
    ) -> Optional[str]:
        """
        Find an available running episode with the target_task_id that can be attached to.

        Args:
            session_id: Session ID to search within
            target_task_id: Task ID we need to find a running episode for
            dependent_task_id: Task ID of the episode that wants to attach

        Returns:
            Episode ID of available episode, or None if none found

        Raises:
            ValueError: If a circular dependency would be created
        """
        # Prevent circular dependencies
        if target_task_id == dependent_task_id:
            raise ValueError(f"Circular dependency detected: task {dependent_task_id} cannot depend on itself")

        session_episodes = self.get_active_episodes_for_session(session_id)

        for episode in session_episodes:
            if episode.task_id == target_task_id and episode.state == EpisodeState.ACTIVE:
                if not episode.has_attached_episode_with_task(dependent_task_id, self.episodes):
                    logger.info(
                        "Dependency episode available",
                        extra={
                            "event": "episode_dependency_available",
                            "session_id": session_id,
                            "episode_id": episode.episode_id,
                            "target_task_id": target_task_id,
                            "dependent_task_id": dependent_task_id,
                        },
                    )
                    return episode.episode_id
                logger.debug(
                    "Dependency already attached",
                    extra={
                        "event": "episode_dependency_already_attached",
                        "session_id": session_id,
                        "episode_id": episode.episode_id,
                        "target_task_id": target_task_id,
                        "dependent_task_id": dependent_task_id,
                    },
                )

        logger.warning(
            "Dependency episode unavailable",
            extra={
                "event": "episode_dependency_unavailable",
                "session_id": session_id,
                "target_task_id": target_task_id,
                "dependent_task_id": dependent_task_id,
            },
        )
        return None

    async def find_available_episode_for_dependency_with_retry(
        self,
        session_id: str,
        target_task_id: str,
        dependent_task_id: str,
        max_wait_seconds: float = 10.0,
        retry_interval: float = 0.5,
        max_retry_interval: float = 2.0,
    ) -> Optional[str]:
        """
        Find an available running episode with retry logic to handle race conditions.

        Args:
            session_id: Session ID to search within
            target_task_id: Task ID we need to find a running episode for
            dependent_task_id: Task ID of the episode that wants to attach
            max_wait_seconds: Maximum total time to wait for dependency (default 10 seconds)
            retry_interval: Initial retry interval in seconds (default 0.5 seconds)
            max_retry_interval: Maximum retry interval for exponential backoff (default 2 seconds)

        Returns:
            Episode ID of available episode, or None if none found within timeout

        Raises:
            ValueError: If a circular dependency would be created
        """
        start_time = time.time()
        current_retry_interval = retry_interval

        logger.info(
            "Dependency search started",
            extra={
                "event": "episode_dependency_search_started",
                "session_id": session_id,
                "target_task_id": target_task_id,
                "dependent_task_id": dependent_task_id,
                "max_wait_seconds": max_wait_seconds,
                "initial_retry_interval": retry_interval,
                "max_retry_interval": max_retry_interval,
            },
        )

        while True:
            # Try to find the dependency using the existing synchronous method
            try:
                episode_id = self.find_available_episode_for_dependency(
                    session_id=session_id, target_task_id=target_task_id, dependent_task_id=dependent_task_id
                )

                if episode_id:
                    elapsed = time.time() - start_time
                    logger.info(
                        "Dependency resolved with retry",
                        extra={
                            "event": "episode_dependency_found_with_retry",
                            "session_id": session_id,
                            "target_task_id": target_task_id,
                            "dependent_task_id": dependent_task_id,
                            "episode_id": episode_id,
                            "elapsed_seconds": round(elapsed, 2),
                        },
                    )
                    return episode_id

            except ValueError as e:
                logger.error(
                    "Dependency validation failed",
                    extra={
                        "event": "episode_dependency_validation_failed",
                        "session_id": session_id,
                        "target_task_id": target_task_id,
                        "dependent_task_id": dependent_task_id,
                        "error": str(e),
                    },
                )
                raise

            # Check if we've exceeded the maximum wait time
            elapsed = time.time() - start_time
            if elapsed >= max_wait_seconds:
                logger.warning(
                    "Dependency search timed out",
                    extra={
                        "event": "episode_dependency_search_timeout",
                        "session_id": session_id,
                        "target_task_id": target_task_id,
                        "dependent_task_id": dependent_task_id,
                        "elapsed_seconds": round(elapsed, 2),
                        "max_wait_seconds": max_wait_seconds,
                    },
                )
                return None

            # Wait before retrying (exponential backoff)
            logger.debug(
                "Dependency retry scheduled",
                extra={
                    "event": "episode_dependency_retry_scheduled",
                    "session_id": session_id,
                    "target_task_id": target_task_id,
                    "dependent_task_id": dependent_task_id,
                    "elapsed_seconds": round(elapsed, 2),
                    "retry_interval_seconds": round(current_retry_interval, 2),
                },
            )
            await asyncio.sleep(current_retry_interval)

            # Exponential backoff with maximum
            current_retry_interval = min(current_retry_interval * 1.5, max_retry_interval)

    def attach_episode_to_episode(self, dependent_episode_id: str, target_episode_id: str) -> None:
        """
        Attach one episode to another, updating both episodes' tracking fields.

        Args:
            dependent_episode_id: ID of episode being attached
            target_episode_id: ID of episode being attached to
        """
        dependent_episode = self.episodes.get(dependent_episode_id)
        target_episode = self.episodes.get(target_episode_id)

        if not dependent_episode or not target_episode:
            raise ValueError(f"Episode not found: dependent={dependent_episode_id}, target={target_episode_id}")

        # Update attachment tracking
        dependent_episode.attach_to_episode(target_episode_id)
        target_episode.add_attached_episode(dependent_episode_id)

        logger.info(
            "Episode attached to dependency",
            extra={
                "event": "episode_attached_to_dependency",
                "dependent_episode_id": dependent_episode_id,
                "target_episode_id": target_episode_id,
                "session_id": dependent_episode.session_id,
            },
        )

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
            logger.debug(
                "Episode config loaded from task",
                extra={
                    "event": "episode_config_loaded_from_task",
                    "episode_id": episode_id,
                    "task_id": getattr(task, "task_id", None),
                    "config_keys": sorted(episode_config.keys()),
                },
            )

        # Validate required configuration values
        if "max_steps" not in episode_config:
            logger.error(
                "Episode configuration missing max_steps",
                extra={
                    "event": "episode_config_missing_required_value",
                    "episode_id": episode_id,
                    "task_id": getattr(task, "task_id", None),
                },
            )
            raise ValueError(
                f"Task {task.task_id if task else 'None'} episode_config missing required 'max_steps' value"
            )

        # Apply defaults for optional values
        episode_config.setdefault("step_timeout_seconds", 300)
        episode_config.setdefault("episode_timeout_minutes", 30)

        # Store configuration for this episode
        self.episode_configs[episode_id] = episode_config

        logger.info(
            "Episode configuration applied",
            extra={
                "event": "episode_config_applied",
                "episode_id": episode_id,
                "task_id": getattr(task, "task_id", None),
                "max_steps": episode_config["max_steps"],
                "step_timeout_seconds": episode_config.get("step_timeout_seconds"),
                "episode_timeout_minutes": episode_config.get("episode_timeout_minutes"),
            },
        )

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
            logger.error(
                "Episode configuration missing",
                extra={
                    "event": "episode_config_missing",
                    "episode_id": episode_id,
                },
            )
            return True, f"configuration_error: No episode configuration found for episode {episode_id}"

        try:
            # Note: Step limit checking moved to client-side
            # Server no longer terminates episodes based on step count
            max_steps = episode_config["max_steps"]
            current_steps = len(episode.steps)

            logger.debug(
                "Episode step count evaluated",
                extra={
                    "event": "episode_step_count_evaluated",
                    "episode_id": episode_id,
                    "current_steps": current_steps,
                    "max_steps": max_steps,
                },
            )

            # Could add other termination conditions here:
            # - episode timeout based on episode_config["episode_timeout_minutes"]
            # - step timeout based on episode_config["step_timeout_seconds"]
            # - resource limits
            # etc.

        except Exception as e:
            logger.error(
                "Episode termination evaluation failed",
                extra={
                    "event": "episode_termination_evaluation_failed",
                    "episode_id": episode_id,
                    "error": str(e),
                },
            )
            return True, f"configuration_error: {e}"

        return False, ""

    def start_episode(
        self,
        session_id: str,
        task_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        task: Optional[Any] = None,
    ) -> Episode:
        """
        Start a new episode for a session with the provided task.

        Args:
            session_id: ID of the session starting the episode
            task_id: Task ID to execute
            initial_context: Initial context for the episode
            task: Optional task object for dependency tracking

        Returns:
            New Episode instance
        """
        logger.info(
            "Episode start requested",
            extra={
                "event": "episode_start_requested",
                "session_id": session_id,
                "task_id": task_id,
            },
        )

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
            depends_on_task_id=task.depends_on_task_id if task else None,
            attached_to_episode_id=None,  # Will be set later if episode is attached
        )

        # Add episode to session's episode tracking
        self.add_episode_to_session(session_id, episode)

        logger.info(
            "Episode created",
            extra={
                "event": "episode_created",
                "session_id": session_id,
                "episode_id": episode.episode_id,
                "task_id": task_id,
                "depends_on_task_id": getattr(task, "depends_on_task_id", None),
            },
        )
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

        logger.debug(
            "Episode step executing",
            extra={
                "event": "episode_step_executing",
                "episode_id": episode.episode_id,
                "session_id": episode.session_id,
                "step_number": len(episode.steps) + 1,
                "action_tool": action.tool_name,
            },
        )

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
                "Episode step count tracked",
                extra={
                    "event": "episode_step_count_tracked",
                    "episode_id": episode_id,
                    "current_step": current_steps + 1,
                    "max_steps": max_steps,
                },
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
        logger.info(
            "Episode end requested",
            extra={
                "event": "episode_end_requested",
                "episode_id": episode_id,
                "reason": reason,
            },
        )
        episode = self.get_episode_by_id(episode_id)
        if not episode:
            raise EpisodeNotFoundException(episode_id)

        logger.info(
            "Episode ending",
            extra={
                "event": "episode_ending",
                "episode_id": episode.episode_id,
                "reason": reason,
            },
        )

        # Create a final step if the agent provided a result/submission
        if result:
            logger.info(
                "Episode result submission recorded",
                extra={
                    "event": "episode_result_submission_recorded",
                    "episode_id": episode.episode_id,
                    "result": result,
                },
            )

            # Create an action representing the agent's submission
            submission_action = Action(
                tool_name="submission",
                parameters={"result": result, "submission": result, "episode_completed": True},
                reasoning=None,
                assistant_message=None,
            )

            # Create a successful command result for the submission
            submission_result = CommandResult.success_result(
                data={"submission": result, "message": f"Episode completed with result: {result}"}, execution_time=0.0
            )

            # Create and add the final step
            final_step = self.create_step(episode, submission_action, submission_result)
            episode.steps.append(final_step)
            logger.info(
                "Episode final step added",
                extra={
                    "event": "episode_final_step_added",
                    "episode_id": episode.episode_id,
                    "step_number": final_step.step_number,
                    "result": result,
                },
            )

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
            logger.info(
                "Episode moved to completed",
                extra={
                    "event": "episode_moved_to_completed",
                    "episode_id": episode_id,
                    "session_id": episode.session_id,
                },
            )
            self.complete_episode(episode_id)

        # Clean up episode configuration
        if episode_id in self.episode_configs:
            del self.episode_configs[episode_id]

        logger.info(
            "Episode completed",
            extra={
                "event": "episode_completed",
                "episode_id": episode.episode_id,
                "session_id": episode.session_id,
                "total_steps": len(episode.steps),
                "success": episode.state == EpisodeState.COMPLETED,
                "completion_reason": episode.completion_reason,
            },
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

        Currently focused on DockerBashExecutor tool only.

        Args:
            action: Action to extract parameters from

        Returns:
            Extracted parameters string, or None if not extractable
        """
        if action.tool_name == "docker_cli_executor":
            # Extract command from DockerBashExecutor parameters
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
        active_episodes = self.get_active_episodes_for_session(session_id)
        logger.info(
            "Session cleanup started",
            extra={
                "event": "episode_session_cleanup_started",
                "session_id": session_id,
                "active_episode_count": len(active_episodes),
            },
        )

        # End all active episodes for this session
        for episode in active_episodes:
            logger.info(
                "Active episode ended during cleanup",
                extra={
                    "event": "episode_cleanup_ending_active_episode",
                    "session_id": session_id,
                    "episode_id": episode.episode_id,
                },
            )
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
        logger.error(
            "Episode removed due to error",
            extra={
                "event": "episode_removed_due_to_error",
                "episode_id": episode_id,
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

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

            logger.info(
                "Episode failure recorded",
                extra={
                    "event": "episode_failure_recorded",
                    "episode_id": episode.episode_id,
                    "session_id": episode.session_id,
                },
            )
