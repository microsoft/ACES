"""Copilot SDK session event handling.

This module provides a clean handler-based approach for processing
events emitted by the GitHub Copilot SDK during session execution.

Event types handled:
- session.idle: Turn completion signal
- session.start: Session initialization with session ID
- session.error: Error conditions
- assistant.usage: Token usage metrics
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from inspect_ai.model._model_output import ModelUsage

from .....logging_config import LogCategory, get_saber_logger

if TYPE_CHECKING:
    import asyncio

logger = get_saber_logger(LogCategory.AGENT, __name__)


# =============================================================================
# Event Types
# =============================================================================


class SessionEventType(str, Enum):
    """Event types emitted by the Copilot SDK session.

    These correspond to the event.type values from the SDK.
    See: .archive/copilot-sdk/python/copilot/generated/session_events.py
    """

    SESSION_IDLE = "session.idle"
    SESSION_START = "session.start"
    SESSION_ERROR = "session.error"
    ASSISTANT_USAGE = "assistant.usage"


class CopilotSDKEvent(Protocol):
    """Protocol for Copilot SDK events.

    The SDK event has a `.type` attribute with a `.value` property,
    and optionally a `.data` attribute with event-specific data.
    """

    @property
    def type(self) -> object:
        """Event type with .value property."""
        ...

    @property
    def data(self) -> object | None:
        """Event data (optional, varies by event type)."""
        ...


# =============================================================================
# Data Extraction Utilities
# =============================================================================


def _get_attr_or_key(obj: Any, *names: str, default: Any = None) -> Any:
    """Extract a value from an object by trying multiple attribute/key names.

    Handles both SDK dataclass objects (attribute access) and dicts (key access).

    Args:
        obj: Object to extract from (can be dataclass, dict, or other)
        *names: Attribute/key names to try in order
        default: Value to return if none found

    Returns:
        First found value, or default if none found
    """
    if obj is None:
        return default

    for name in names:
        # Try attribute access first (SDK objects)
        value = getattr(obj, name, None)
        if value is not None:
            return value

        # Try dict access
        if isinstance(obj, dict):
            value = obj.get(name)
            if value is not None:
                return value

    return default


# =============================================================================
# Token Usage Tracking
# =============================================================================


@dataclass
class TokenUsage:
    """Accumulated token usage from assistant.usage events."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    def add(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Add token counts to the accumulator."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_write_tokens += cache_write_tokens

    def to_model_usage(self) -> ModelUsage:
        """Convert to Inspect AI ModelUsage object."""
        return ModelUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
        )

    def has_usage(self) -> bool:
        """Check if any tokens have been recorded."""
        return self.input_tokens > 0 or self.output_tokens > 0


# =============================================================================
# Event Handlers
# =============================================================================


class SessionEventHandler:
    """Handler for a specific session event type.

    Subclass this to create handlers for specific event types.
    """

    event_type: SessionEventType

    def handle(self, event: CopilotSDKEvent, context: SessionEventContext) -> None:
        """Handle the event.

        Args:
            event: The SDK event object
            context: Shared context for event handling
        """
        raise NotImplementedError


class SessionIdleHandler(SessionEventHandler):
    """Handler for session.idle events."""

    event_type = SessionEventType.SESSION_IDLE

    def handle(self, event: CopilotSDKEvent, context: SessionEventContext) -> None:
        """Signal that the session is idle (turn complete)."""
        context.idle_event.set()


class SessionStartHandler(SessionEventHandler):
    """Handler for session.start events."""

    event_type = SessionEventType.SESSION_START

    def handle(self, event: CopilotSDKEvent, context: SessionEventContext) -> None:
        """Extract and store session ID from session.start event."""
        data = getattr(event, "data", None)
        logger.debug(f"session.start data: {data}, type: {type(data)}")

        if data:
            session_id = _get_attr_or_key(data, "session_id", "sessionId")
            if session_id:
                context.session_id = session_id
                logger.info(f"Session started with ID: {session_id}")
            else:
                logger.warning(f"Could not extract session_id from data: {data}")


class SessionErrorHandler(SessionEventHandler):
    """Handler for session.error events."""

    event_type = SessionEventType.SESSION_ERROR

    def handle(self, event: CopilotSDKEvent, context: SessionEventContext) -> None:
        """Store error and signal idle to unblock waiters."""
        data = getattr(event, "data", None)
        error_msg = _get_attr_or_key(data, "message", default=str(data)) if data else "Unknown error"
        context.set_error(Exception(f"Session error: {error_msg}"))
        context.idle_event.set()


class AssistantUsageHandler(SessionEventHandler):
    """Handler for assistant.usage events.

    Extracts token usage metrics from the SDK event and accumulates them.
    """

    event_type = SessionEventType.ASSISTANT_USAGE

    def handle(self, event: CopilotSDKEvent, context: SessionEventContext) -> None:
        """Extract and accumulate token usage from assistant.usage event."""
        data = getattr(event, "data", None)
        if not data:
            return

        # Extract token counts (SDK uses camelCase, we also check snake_case)
        input_tokens = int(_get_attr_or_key(data, "input_tokens", "inputTokens", default=0))
        output_tokens = int(_get_attr_or_key(data, "output_tokens", "outputTokens", default=0))
        cache_read_tokens = int(_get_attr_or_key(data, "cache_read_tokens", "cacheReadTokens", default=0))
        cache_write_tokens = int(_get_attr_or_key(data, "cache_write_tokens", "cacheWriteTokens", default=0))

        context.token_usage.add(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

        # Log current turn's tokens and running totals (info level for debugging, will reduce later)
        logger.info(
            "Token usage update",
            extra={
                "turn_input": input_tokens,
                "turn_output": output_tokens,
                "turn_cache_read": cache_read_tokens,
                "turn_cache_write": cache_write_tokens,
                "total_input": context.token_usage.input_tokens,
                "total_output": context.token_usage.output_tokens,
                "total_tokens": context.token_usage.total_tokens,
            },
        )


# =============================================================================
# Event Context & Dispatcher
# =============================================================================


@dataclass
class SessionEventContext:
    """Shared context for session event handling.

    This holds the state that event handlers can read from and write to.
    """

    idle_event: asyncio.Event
    session_id: str | None = None
    _error: Exception | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    def set_error(self, error: Exception) -> None:
        """Set an error condition."""
        self._error = error

    def get_error(self) -> Exception | None:
        """Get any stored error."""
        return self._error

    def clear_error(self) -> None:
        """Clear any stored error."""
        self._error = None


class SessionEventDispatcher:
    """Dispatches Copilot SDK events to registered handlers.

    Uses a handler registry pattern for clean, extensible event handling.
    """

    def __init__(self) -> None:
        """Initialize with default handlers."""
        self._handlers: dict[str, SessionEventHandler] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register the built-in event handlers."""
        self.register(SessionIdleHandler())
        self.register(SessionStartHandler())
        self.register(SessionErrorHandler())
        self.register(AssistantUsageHandler())

    def register(self, handler: SessionEventHandler) -> None:
        """Register a handler for an event type.

        Args:
            handler: Handler instance to register
        """
        self._handlers[handler.event_type.value] = handler
        logger.debug(f"Registered handler for {handler.event_type.value}")

    def dispatch(self, event: CopilotSDKEvent | None, context: SessionEventContext) -> None:
        """Dispatch an event to the appropriate handler.

        Args:
            event: SDK event to handle (may be None)
            context: Shared context for handlers
        """
        if not event:
            return

        # Extract event type string
        event_type = str(event.type.value) if hasattr(event.type, "value") else str(event.type)

        # Find and invoke handler
        handler = self._handlers.get(event_type)
        if handler:
            handler.handle(event, context)


# Default dispatcher instance
_default_dispatcher = SessionEventDispatcher()


def get_default_dispatcher() -> SessionEventDispatcher:
    """Get the default event dispatcher with all standard handlers."""
    return _default_dispatcher


# =============================================================================
# Convenience Functions
# =============================================================================


def create_event_callback(
    context: SessionEventContext,
    dispatcher: SessionEventDispatcher | None = None,
) -> Callable[[CopilotSDKEvent | None], None]:
    """Create an event callback function for use with session.on().

    Args:
        context: Event context to use
        dispatcher: Optional custom dispatcher (uses default if not provided)

    Returns:
        Callback function compatible with Copilot SDK session.on()
    """
    disp = dispatcher or get_default_dispatcher()

    def callback(event: CopilotSDKEvent | None) -> None:
        disp.dispatch(event, context)

    return callback


__all__ = [
    "SessionEventType",
    "CopilotSDKEvent",
    "TokenUsage",
    "SessionEventContext",
    "SessionEventHandler",
    "SessionEventDispatcher",
    "SessionIdleHandler",
    "SessionStartHandler",
    "SessionErrorHandler",
    "AssistantUsageHandler",
    "get_default_dispatcher",
    "create_event_callback",
]
