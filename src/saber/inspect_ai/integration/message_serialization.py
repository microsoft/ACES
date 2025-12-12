"""Message serialization utilities for transcript synchronization.

This module provides utilities for serializing and deserializing Inspect AI
ChatMessage objects to/from JSON-safe dictionaries for WebSocket transmission.

Logging category: AGENT.
"""

from typing import Any, Dict

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


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


def serialize_message(msg: ChatMessage) -> Dict[str, Any]:
    """Convert ChatMessage to JSON-safe dict.

    Args:
        msg: Inspect AI ChatMessage object

    Returns:
        Dictionary with keys: role, content, tool_calls (optional), etc.

    Raises:
        ValueError: If message type is unknown or unsupported
    """
    result: Dict[str, Any] = {}

    # Extract role
    if isinstance(msg, ChatMessageSystem):
        result["role"] = "system"
    elif isinstance(msg, ChatMessageUser):
        result["role"] = "user"
    elif isinstance(msg, ChatMessageAssistant):
        result["role"] = "assistant"
    elif isinstance(msg, ChatMessageTool):
        result["role"] = "tool"
    else:
        raise ValueError(f"Unknown message type: {type(msg)}")

    # Extract content - handle both string and list of Content objects
    if isinstance(msg.content, str):
        result["content"] = msg.content
    elif isinstance(msg.content, list):
        # Extract text content parts and concatenate
        text_parts = []
        reasoning_text = None

        for content_item in msg.content:
            if isinstance(content_item, ContentText):
                text_parts.append(content_item.text)
            elif isinstance(content_item, ContentReasoning):
                reasoning_text = content_item.reasoning

        result["content"] = "".join(text_parts)

        # Add reasoning if present (for assistant messages)
        if reasoning_text:
            result["reasoning"] = reasoning_text
    else:
        result["content"] = ""

    # Handle assistant-specific fields
    if isinstance(msg, ChatMessageAssistant):
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": tc.function,
                    "arguments": tc.arguments,
                }
                for tc in msg.tool_calls
            ]

    # Handle tool message-specific fields
    if isinstance(msg, ChatMessageTool):
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        if msg.function:
            result["name"] = msg.function

    return result


def deserialize_message(msg_data: Dict[str, Any]) -> ChatMessage:
    """Convert message dictionary to ChatMessage object.

    Inverse of serialize_message() - converts JSON-safe dict back to
    Inspect AI ChatMessage objects.

    Args:
        msg_data: Dictionary with keys: role, content, tool_calls (optional), etc.

    Returns:
        ChatMessage object (ChatMessageUser, ChatMessageAssistant, etc.)

    Raises:
        ValueError: If role is unknown or message structure is invalid
    """
    role = msg_data.get("role")
    content = msg_data.get("content", "")

    if role == "system":
        return ChatMessageSystem(content=content)

    elif role == "user":
        return ChatMessageUser(content=content)

    elif role == "assistant":
        # Assistant messages may have tool calls
        tool_calls_data = msg_data.get("tool_calls")
        if tool_calls_data:
            # Convert tool call dicts to ToolCall objects
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    function=tc["function"],
                    arguments=tc["arguments"],
                    type="function",
                )
                for tc in tool_calls_data
            ]
            return ChatMessageAssistant(content=content, tool_calls=tool_calls)
        else:
            return ChatMessageAssistant(content=content)

    elif role == "tool":
        # Tool messages need tool_call_id and function name
        tool_call_id = msg_data.get("tool_call_id")
        function_name = msg_data.get("name")
        if not tool_call_id or not function_name:
            raise ValueError(f"Tool message missing tool_call_id or name: {msg_data}")
        return ChatMessageTool(
            content=content,
            tool_call_id=tool_call_id,
            function=function_name,
        )

    else:
        raise ValueError(f"Unknown message role: {role}")


__all__ = [
    "is_websocket_closed",
    "serialize_message",
    "deserialize_message",
]
