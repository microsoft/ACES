"""Tests for message serialization utilities.

Tests cover:
- Serialization of all message types (system, user, assistant, tool)
- Deserialization of all message types
- Content list handling (ContentText, ContentReasoning)
- Tool call serialization
- Error handling for invalid messages
- Roundtrip serialization consistency
"""

import pytest

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall

from saber.inspect_ai.integration.message_serialization import (
    deserialize_message,
    is_websocket_closed,
    serialize_message,
)


class TestSerializeMessage:
    """Tests for serialize_message function."""

    def test_serialize_system_message(self):
        """Test serialization of system messages."""
        msg = ChatMessageSystem(content="You are a helpful assistant")
        serialized = serialize_message(msg)

        assert serialized["role"] == "system"
        assert serialized["content"] == "You are a helpful assistant"

    def test_serialize_user_message(self):
        """Test serialization of user messages."""
        msg = ChatMessageUser(content="Hello, how are you?")
        serialized = serialize_message(msg)

        assert serialized["role"] == "user"
        assert serialized["content"] == "Hello, how are you?"

    def test_serialize_assistant_message_simple(self):
        """Test serialization of simple assistant messages without tool calls."""
        msg = ChatMessageAssistant(content="I'm doing great!")
        serialized = serialize_message(msg)

        assert serialized["role"] == "assistant"
        assert serialized["content"] == "I'm doing great!"
        assert "tool_calls" not in serialized

    def test_serialize_assistant_message_with_tool_calls(self):
        """Test serialization of assistant messages with tool calls."""
        tool_calls = [
            ToolCall(
                id="call_123",
                function="get_weather",
                arguments={"location": "San Francisco"},
                type="function"
            )
        ]
        msg = ChatMessageAssistant(content="Let me check the weather", tool_calls=tool_calls)
        serialized = serialize_message(msg)

        assert serialized["role"] == "assistant"
        assert serialized["content"] == "Let me check the weather"
        assert len(serialized["tool_calls"]) == 1
        assert serialized["tool_calls"][0]["id"] == "call_123"
        assert serialized["tool_calls"][0]["function"] == "get_weather"
        assert serialized["tool_calls"][0]["arguments"] == {"location": "San Francisco"}

    def test_serialize_tool_message(self):
        """Test serialization of tool result messages."""
        msg = ChatMessageTool(
            content="Temperature is 72°F",
            tool_call_id="call_123",
            function="get_weather"
        )
        serialized = serialize_message(msg)

        assert serialized["role"] == "tool"
        assert serialized["content"] == "Temperature is 72°F"
        assert serialized["tool_call_id"] == "call_123"
        assert serialized["name"] == "get_weather"

    def test_serialize_message_with_content_list(self):
        """Test serialization of messages with list content (ContentText, ContentReasoning)."""
        msg = ChatMessageAssistant(content=[
            ContentText(text="First part"),
            ContentText(text=" Second part"),
            ContentReasoning(reasoning="My reasoning here")
        ])
        serialized = serialize_message(msg)

        assert serialized["role"] == "assistant"
        assert serialized["content"] == "First part Second part"
        assert serialized["reasoning"] == "My reasoning here"

    def test_serialize_unknown_message_type(self):
        """Test serialization of unknown message type raises error."""
        class UnknownMessage:
            def __init__(self):
                self.content = "test"

        with pytest.raises(ValueError, match="Unknown message type"):
            serialize_message(UnknownMessage())


class TestDeserializeMessage:
    """Tests for deserialize_message function."""

    def test_deserialize_system_message(self):
        """Test deserialization of system messages."""
        msg_data = {"role": "system", "content": "You are helpful"}
        msg = deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageSystem)
        assert msg.content == "You are helpful"

    def test_deserialize_user_message(self):
        """Test deserialization of user messages."""
        msg_data = {"role": "user", "content": "Hello"}
        msg = deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageUser)
        assert msg.content == "Hello"

    def test_deserialize_assistant_message_simple(self):
        """Test deserialization of simple assistant messages."""
        msg_data = {"role": "assistant", "content": "Hi there"}
        msg = deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageAssistant)
        assert msg.content == "Hi there"
        assert msg.tool_calls is None or len(msg.tool_calls) == 0

    def test_deserialize_assistant_message_with_tool_calls(self):
        """Test deserialization of assistant messages with tool calls."""
        msg_data = {
            "role": "assistant",
            "content": "Calling tool",
            "tool_calls": [
                {
                    "id": "call_456",
                    "function": "search",
                    "arguments": {"query": "test"}
                }
            ]
        }
        msg = deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageAssistant)
        assert msg.content == "Calling tool"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_456"
        assert msg.tool_calls[0].function == "search"

    def test_deserialize_tool_message(self):
        """Test deserialization of tool messages."""
        msg_data = {
            "role": "tool",
            "content": "Result data",
            "tool_call_id": "call_789",
            "name": "search"
        }
        msg = deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageTool)
        assert msg.content == "Result data"
        assert msg.tool_call_id == "call_789"
        assert msg.function == "search"

    def test_deserialize_tool_message_missing_fields(self):
        """Test deserialization of tool messages with missing required fields raises error."""
        # Missing tool_call_id
        msg_data = {"role": "tool", "content": "Result", "name": "search"}
        with pytest.raises(ValueError, match="missing tool_call_id or name"):
            deserialize_message(msg_data)

        # Missing name
        msg_data = {"role": "tool", "content": "Result", "tool_call_id": "call_123"}
        with pytest.raises(ValueError, match="missing tool_call_id or name"):
            deserialize_message(msg_data)

    def test_deserialize_unknown_role(self):
        """Test deserialization with unknown role raises error."""
        msg_data = {"role": "unknown_role", "content": "Test"}
        with pytest.raises(ValueError, match="Unknown message role"):
            deserialize_message(msg_data)


class TestRoundtripSerialization:
    """Tests for roundtrip serialize -> deserialize consistency."""

    def test_roundtrip_system_message(self):
        """Test roundtrip for system messages."""
        original = ChatMessageSystem(content="You are helpful")
        serialized = serialize_message(original)
        deserialized = deserialize_message(serialized)

        assert isinstance(deserialized, ChatMessageSystem)
        assert deserialized.content == original.content

    def test_roundtrip_user_message(self):
        """Test roundtrip for user messages."""
        original = ChatMessageUser(content="Hello world")
        serialized = serialize_message(original)
        deserialized = deserialize_message(serialized)

        assert isinstance(deserialized, ChatMessageUser)
        assert deserialized.content == original.content

    def test_roundtrip_assistant_with_tool_calls(self):
        """Test roundtrip for assistant messages with tool calls."""
        original = ChatMessageAssistant(
            content="Test message",
            tool_calls=[
                ToolCall(id="tc1", function="func1", arguments={"arg": "val"}, type="function")
            ]
        )

        serialized = serialize_message(original)
        deserialized = deserialize_message(serialized)

        assert isinstance(deserialized, ChatMessageAssistant)
        assert deserialized.content == original.content
        assert len(deserialized.tool_calls) == len(original.tool_calls)
        assert deserialized.tool_calls[0].id == original.tool_calls[0].id

    def test_roundtrip_tool_message(self):
        """Test roundtrip for tool messages."""
        original = ChatMessageTool(
            content="Tool result",
            tool_call_id="call_abc",
            function="my_func"
        )
        serialized = serialize_message(original)
        deserialized = deserialize_message(serialized)

        assert isinstance(deserialized, ChatMessageTool)
        assert deserialized.content == original.content
        assert deserialized.tool_call_id == original.tool_call_id
        assert deserialized.function == original.function


class TestIsWebsocketClosed:
    """Tests for is_websocket_closed utility."""

    def test_none_websocket_is_closed(self):
        """Test that None WebSocket is considered closed."""
        assert is_websocket_closed(None) is True

    def test_websocket_with_closed_attribute_true(self):
        """Test WebSocket with closed=True is closed."""
        class MockWebSocket:
            closed = True

        assert is_websocket_closed(MockWebSocket()) is True

    def test_websocket_with_closed_attribute_false(self):
        """Test WebSocket with closed=False is not closed."""
        class MockWebSocket:
            closed = False

        assert is_websocket_closed(MockWebSocket()) is False

    def test_unknown_websocket_is_closed(self):
        """Test that unknown WebSocket type defaults to closed."""
        class UnknownWebSocket:
            pass

        assert is_websocket_closed(UnknownWebSocket()) is True
