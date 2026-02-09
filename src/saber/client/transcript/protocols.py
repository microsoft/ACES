"""Protocol definitions for harness-agnostic transcript synchronization.

This module defines the MessageSerializer protocol that harnesses must implement
to use the generic transcript sync client.

The protocol uses generics to allow type-safe usage with any message type,
while the client code remains harness-agnostic.
"""

from typing import Any, Protocol, TypeVar, runtime_checkable

# Generic message type - could be ChatMessage (inspect_ai), dict, or any other type
M = TypeVar("M")


@runtime_checkable
class MessageSerializer(Protocol[M]):
    """Protocol for serializing/deserializing harness-specific message types.

    Implementations provide the bridge between harness-specific message types
    (e.g., inspect_ai.ChatMessage) and the generic dict format used by the
    SABER server WebSocket protocol.

    Type Parameter:
        M: The harness-specific message type (e.g., ChatMessage)

    Example implementation for a custom harness:

        class MyMessageSerializer(MessageSerializer[MyMessage]):
            def serialize(self, msg: MyMessage) -> dict[str, Any]:
                return {"role": msg.role, "content": msg.text}

            def deserialize(self, data: dict[str, Any]) -> MyMessage:
                return MyMessage(role=data["role"], text=data["content"])

            def get_role(self, msg: MyMessage) -> str:
                return msg.role

            def has_tool_calls(self, msg: MyMessage) -> bool:
                return hasattr(msg, "tool_calls") and bool(msg.tool_calls)

            def get_tool_call_ids(self, msg: MyMessage) -> list[str]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    return [tc.id for tc in msg.tool_calls if tc.id]
                return []

            def get_tool_call_id(self, msg: MyMessage) -> str | None:
                return getattr(msg, "tool_call_id", None)
    """

    def serialize(self, message: M) -> dict[str, Any]:
        """Convert a harness message to a dict for WebSocket transmission.

        The returned dict must be JSON-serializable and should contain at minimum
        a 'role' key. The SABER server expects message dicts in a format compatible
        with the ChatMessage schema (role, content, tool_calls, etc.).

        Args:
            message: Harness-specific message object

        Returns:
            Dictionary representation of the message

        Raises:
            ValueError: If the message cannot be serialized
        """
        ...

    def deserialize(self, data: dict[str, Any]) -> M:
        """Convert a dict from WebSocket to a harness message.

        Args:
            data: Dictionary received from server (contains role, content, etc.)

        Returns:
            Harness-specific message object

        Raises:
            ValueError: If the data cannot be deserialized
        """
        ...

    def get_role(self, message: M) -> str:
        """Get the role of a message (system, user, assistant, tool).

        Args:
            message: Harness-specific message object

        Returns:
            Role string (one of: "system", "user", "assistant", "tool")
        """
        ...

    def has_tool_calls(self, message: M) -> bool:
        """Check if an assistant message has tool calls.

        Args:
            message: Harness-specific message object

        Returns:
            True if message has tool_calls, False otherwise
        """
        ...

    def get_tool_call_ids(self, message: M) -> list[str]:
        """Get tool call IDs from an assistant message.

        Used to match assistant tool_calls with tool response messages.

        Args:
            message: Harness-specific message object (typically assistant role)

        Returns:
            List of tool_call IDs (empty if none or not an assistant message)
        """
        ...

    def get_tool_call_id(self, message: M) -> str | None:
        """Get the tool_call_id for a tool message.

        Tool messages reference the tool_call they're responding to via this ID.

        Args:
            message: Harness-specific tool message

        Returns:
            tool_call_id if present, None otherwise
        """
        ...


__all__ = ["MessageSerializer", "M"]
