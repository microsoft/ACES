"""Model wrapper for transcript synchronization.

This module provides a transparent wrapper around inspect_ai Model objects that
automatically pushes transcripts to the SABER server after each generate() call.

The wrapper intercepts model.generate() calls at the lowest level, ensuring that
transcripts are synchronized BEFORE tools execute, solving the race condition where
the server needs assistant messages for context extraction.

Key features:
- Works with ANY agent type (react, basic_agent, custom agents, etc.)
- Transparent delegation - behaves exactly like the wrapped model
- Differential sync - only pushes new messages
- Graceful error handling - failures don't crash agent execution
"""

from typing import Any, Optional

from inspect_ai.model import ChatMessage, Model, ModelOutput

from ...logging_config import LogCategory, get_saber_logger
from .transcript_sync import _push_single_message

logger = get_saber_logger(LogCategory.AGENT, __name__)


class TranscriptSyncingModelWrapper:
    """Wraps a Model to push transcripts after each generate() call.

    This wrapper intercepts at the model level, making it compatible with all
    agent implementations without modification. The agent calls get_model().generate(),
    which now goes through this wrapper, ensuring the server has fresh context
    before any tool executions occur.

    Attributes:
        base_model: The original Model being wrapped
        session_id: SABER session identifier
        episode_id: SABER episode identifier
        rest_url: Base URL of SABER REST API

    Example:
        >>> original_model = get_model()
        >>> wrapped = TranscriptSyncingModelWrapper(
        ...     base_model=original_model,
        ...     session_id="session_123",
        ...     episode_id="episode_456",
        ...     rest_url="http://localhost:8000"
        ... )
        >>> active_model_context_var.set(wrapped)
        >>> # Now any agent will use wrapped model automatically
    """

    def __init__(
        self,
        base_model: Model,
        session_id: str,
        episode_id: str,
        rest_url: str,
    ):
        """Initialize the model wrapper.

        Args:
            base_model: The Model to wrap
            session_id: SABER session identifier
            episode_id: SABER episode identifier
            rest_url: Base URL of SABER REST API (e.g., "http://localhost:8000")
        """
        self._base_model = base_model
        self._session_id = session_id
        self._episode_id = episode_id
        self._rest_url = rest_url

        logger.debug(
            "Created TranscriptSyncingModelWrapper",
            extra={
                "session_id": session_id,
                "episode_id": episode_id,
                "base_model_type": type(base_model).__name__,
            },
        )

    async def generate(
        self,
        input: str | list[ChatMessage],
        tools: Optional[list] = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Generate model output and push transcript.

        This method:
        1. Calls the base model's generate() method
        2. Pushes the assistant message to SABER server (append mode)
        3. Returns the original output

        The transcript push happens AFTER generation but BEFORE the agent
        processes tool calls, ensuring the server has context.

        Args:
            input: Messages or text input to the model
            tools: Optional list of tools available to the model
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput from the base model

        Raises:
            Any exceptions from the base model's generate() method.
            Transcript push failures are logged but not raised.
        """
        # Call original model generate
        output = await self._base_model.generate(input, tools, **kwargs)

        # Push the assistant message that was just generated
        # This happens BEFORE tools execute, solving the race condition
        try:
            await _push_single_message(
                message=output.message,
                session_id=self._session_id,
                episode_id=self._episode_id,
                rest_url=self._rest_url,
            )

            logger.debug(
                "Pushed assistant message after generate",
                extra={
                    "episode_id": self._episode_id,
                    "has_tool_calls": bool(output.message.tool_calls),
                },
            )

        except Exception as e:
            # Log but don't crash - graceful degradation
            logger.warning(
                "Failed to push transcript after model.generate(), continuing execution",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                },
            )

        return output

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the base model.

        This makes the wrapper transparent - it behaves exactly like the
        wrapped model for all attributes/methods except generate().

        Args:
            name: Attribute name being accessed

        Returns:
            The attribute from the base model
        """
        return getattr(self._base_model, name)

    def __repr__(self) -> str:
        """Return string representation showing wrapped model."""
        return f"TranscriptSyncingModelWrapper({self._base_model!r})"


__all__ = ["TranscriptSyncingModelWrapper"]
