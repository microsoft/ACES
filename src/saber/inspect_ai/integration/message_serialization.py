"""Message serialization utilities for transcript synchronization.

This module provides utilities for serializing and deserializing Inspect AI
ChatMessage objects to/from JSON-safe dictionaries for WebSocket transmission.

Uses Pydantic's native model_dump()/model_validate() for complete field preservation,
including metadata, source, model, and other fields that may be added in future
versions of inspect_ai.

Logging category: AGENT.
"""

from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MessageRole

logger = get_saber_logger(LogCategory.AGENT, __name__)


# Mapping from role string to ChatMessage subclass for deserialization
_ROLE_TO_MESSAGE_CLASS: dict[str, type[ChatMessage]] = {
    MessageRole.SYSTEM.value: ChatMessageSystem,
    MessageRole.USER.value: ChatMessageUser,
    MessageRole.ASSISTANT.value: ChatMessageAssistant,
    MessageRole.TOOL.value: ChatMessageTool,
}


def is_websocket_closed(websocket: Any) -> bool:
    """Check if a WebSocket connection is closed.

    Compatible with both websockets <15.0 (.closed attribute)
    and websockets >=15.0 (.state attribute with State enum).

    Args:
        websocket: WebSocket connection object

    Returns:
        True if the connection is closed or closing, False if open
    """
    if websocket is None:
        return True

    # Try to import WebSocketState for websockets >=15.0
    try:
        from websockets.protocol import State as WebSocketState

        # websockets >=15.0 uses .state attribute with State enum
        if hasattr(websocket, "state"):
            return websocket.state in (WebSocketState.CLOSED, WebSocketState.CLOSING)
    except ImportError:
        pass

    # websockets <15.0 uses .closed attribute
    if hasattr(websocket, "closed"):
        return bool(websocket.closed)

    # Default to closed if we can't determine
    return True


def serialize_message(msg: ChatMessage) -> dict[str, Any]:
    """Convert ChatMessage to JSON-safe dict using Pydantic's native serialization.

    Uses model_dump() to preserve ALL fields including metadata, source, model,
    id, and any future fields added to ChatMessage types.

    Args:
        msg: Inspect AI ChatMessage object

    Returns:
        Dictionary representation of the message with all fields preserved.

    Raises:
        ValueError: If message type is unknown or unsupported
    """
    if not isinstance(msg, (ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool)):
        raise ValueError(f"Unknown message type: {type(msg)}")

    result: dict[str, Any] = msg.model_dump()
    return result


def deserialize_message(msg_data: dict[str, Any]) -> ChatMessage:
    """Convert message dictionary to ChatMessage object using Pydantic's native deserialization.

    Uses model_validate() to restore ALL fields including metadata, source, model,
    id, and any future fields added to ChatMessage types.

    Args:
        msg_data: Dictionary representation of a ChatMessage (from serialize_message or JSON)

    Returns:
        ChatMessage object (ChatMessageUser, ChatMessageAssistant, etc.)

    Raises:
        ValueError: If role is unknown or message structure is invalid
    """
    role_str = msg_data.get("role")

    if role_str not in _ROLE_TO_MESSAGE_CLASS:
        raise ValueError(f"Unknown message role: {role_str}")

    message_class = _ROLE_TO_MESSAGE_CLASS[role_str]

    try:
        return message_class.model_validate(msg_data)
    except Exception as e:
        raise ValueError(f"Failed to deserialize {role_str} message: {e}") from e


__all__ = [
    "is_websocket_closed",
    "serialize_message",
    "deserialize_message",
]
