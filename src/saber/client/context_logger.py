"""
SABER Client - Context-aware logging wrapper.

Provides a clean interface for structured logging with automatic context injection.
"""

import logging
from typing import Any, Dict, Optional

RESPONSE_OUTPUT_LENGTH = 100


class ContextLogger:
    """
    Context-aware logger wrapper that automatically injects session context.

    This wrapper reduces boilerplate by automatically including session_id,
    episode_id, and step_number in all log calls when available.
    """

    def __init__(
        self,
        logger: logging.Logger,
        session_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        step_number: Optional[int] = None,
    ):
        """
        Initialize the context logger.

        Args:
            logger: The underlying Python logger
            session_id: Current session ID (optional)
            episode_id: Current episode ID (optional)
            step_number: Current step number (optional)
        """
        self.logger = logger
        self.session_id = session_id
        self.episode_id = episode_id
        self.step_number = step_number

    def update_context(
        self, session_id: Optional[str] = None, episode_id: Optional[str] = None, step_number: Optional[int] = None
    ) -> None:
        """Update the context fields."""
        if session_id is not None:
            self.session_id = session_id
        if episode_id is not None:
            self.episode_id = episode_id
        if step_number is not None:
            self.step_number = step_number

    def _build_extra(self, event_type: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Build the extra context dictionary for logging."""
        extra: Dict[str, Any] = {}

        # Add session context if available
        if self.session_id:
            extra["session_id"] = self.session_id
        if self.episode_id:
            extra["episode_id"] = self.episode_id
        if self.step_number:
            extra["step_number"] = self.step_number

        # Add event type if provided
        if event_type:
            extra["event_type"] = event_type

        # Add any additional fields
        extra.update(kwargs)

        return extra

    def info(self, message: str, event_type: Optional[str] = None, **kwargs: Any) -> None:
        """Log an info message with context."""
        extra = self._build_extra(event_type, **kwargs)
        self.logger.info(message, extra=extra)

    def warning(self, message: str, event_type: Optional[str] = None, **kwargs: Any) -> None:
        """Log a warning message with context."""
        extra = self._build_extra(event_type, **kwargs)
        self.logger.warning(message, extra=extra)

    def error(self, message: str, event_type: Optional[str] = None, **kwargs: Any) -> None:
        """Log an error message with context."""
        extra = self._build_extra(event_type, **kwargs)
        self.logger.error(message, extra=extra)

    def debug(self, message: str, event_type: Optional[str] = None, **kwargs: Any) -> None:
        """Log a debug message with context."""
        extra = self._build_extra(event_type, **kwargs)
        self.logger.debug(message, extra=extra)

    # Convenience methods for common event types
    def log_step_start(self, step_number: int) -> None:
        """Log the start of a step."""
        self.update_context(step_number=step_number)
        self.info(f"Step {step_number}", event_type="step_start")

    def log_agent_response(self, command: str) -> None:
        """Log an agent response/command."""
        self.info(f"Agent command: {command}", event_type="agent_response", agent_command=command)

    def log_step_response(self, success: bool, done: bool, output: str = "") -> None:
        """Log a step execution response."""
        # Truncate output for logging
        output_preview = output[:RESPONSE_OUTPUT_LENGTH] + "..." if len(output) > RESPONSE_OUTPUT_LENGTH else output
        self.info(
            f"Step success: {success}",
            event_type="step_response",
            step_success=success,
            step_done=done,
            output_preview=output_preview,
        )

    def log_step_error(self, error_message: str) -> None:
        """Log a step error."""
        self.warning(f"Step error: {error_message}", event_type="step_error", step_error=error_message)

    def log_agent_error(self, error: str) -> None:
        """Log an agent error."""
        self.error(f"Agent error: {error}", event_type="agent_error", error=error)

    def log_server_error(self, error: str) -> None:
        """Log a server communication error."""
        self.error(f"Server communication error: {error}", event_type="server_error", error=error)

    def log_episode_start(self, episode_id: str, task_id: Optional[str] = None) -> None:
        """Log the start of an episode."""
        self.update_context(episode_id=episode_id)
        extra = {"task_id": task_id} if task_id else {}
        self.info(f"Started episode: {episode_id}", event_type="episode_start", **extra)

    def log_empty_response(self) -> None:
        """Log when agent returns empty response."""
        self.warning("Agent returned empty response", event_type="agent_error")

    def log_episode_completed(self) -> None:
        """Log episode completion with context."""
        self.info("Episode completed", event_type="episode_completed")

    def log_final_result(self, completion_prompt: str) -> None:
        """Log final result with context."""
        self.info("Final result:", event_type="final_result")
        self.info(completion_prompt, event_type="completion_prompt")

    def log_test_completed(self, results: Dict[str, Any]) -> None:
        """Log test completion with results and context."""
        self.info(f"Test completed: {results}", event_type="test_completed", test_results=results)

    def log_test_failed(self, error: str) -> None:
        """Log test failure with context."""
        self.error(f"Test execution failed: {error}", event_type="test_failed", error=error)

    def log_task_info(self, title: str, available_commands: list) -> None:
        """Log task information with context."""
        self.info(f"Task: {title}", event_type="task_info", task_title=title)
        self.info(
            f"Available commands: {available_commands}", event_type="policy_info", available_commands=available_commands
        )

    def log_session_created(self, session_id: str) -> None:
        """Log session creation."""
        self.info(f"Created session: {session_id}", event_type="session_created", session_id=session_id)

    def log_episode_starting(self, task_id: Optional[str] = None) -> None:
        """Log episode start preparation."""
        if task_id:
            self.info(f"Starting episode with task_id: {task_id}", event_type="episode_starting", task_id=task_id)
        else:
            self.info("Starting episode with default task", event_type="episode_starting")
