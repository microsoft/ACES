"""Tests for message serialization utilities.

Tests cover:
- Serialization of all message types (system, user, assistant, tool)
- Deserialization of all message types
- Metadata preservation (the key feature of Pydantic-native serialization)
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
from saber.models.constants import MessageRole


class TestSerializeMessage:
    """Tests for serialize_message function using Pydantic's model_dump()."""

    def test_serialize_system_message(self):
        """Test serialization of system messages."""
        msg = ChatMessageSystem(content="You are a helpful assistant")
        serialized = serialize_message(msg)

        assert serialized["role"] == MessageRole.SYSTEM.value
        assert serialized["content"] == "You are a helpful assistant"

    def test_serialize_user_message(self):
        """Test serialization of user messages."""
        msg = ChatMessageUser(content="Hello, how are you?")
        serialized = serialize_message(msg)

        assert serialized["role"] == MessageRole.USER.value
        assert serialized["content"] == "Hello, how are you?"

    def test_serialize_assistant_message_simple(self):
        """Test serialization of simple assistant messages without tool calls."""
        msg = ChatMessageAssistant(content="I'm doing great!")
        serialized = serialize_message(msg)

        assert serialized["role"] == MessageRole.ASSISTANT.value
        assert serialized["content"] == "I'm doing great!"
        # Pydantic includes all fields, tool_calls will be None
        assert serialized["tool_calls"] is None

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

        assert serialized["role"] == MessageRole.ASSISTANT.value
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

        assert serialized["role"] == MessageRole.TOOL.value
        assert serialized["content"] == "Temperature is 72°F"
        assert serialized["tool_call_id"] == "call_123"
        # Pydantic uses 'function' field, not 'name'
        assert serialized["function"] == "get_weather"

    def test_serialize_message_with_content_list(self):
        """Test serialization of messages with list content preserves structure."""
        msg = ChatMessageAssistant(content=[
            ContentText(text="First part"),
            ContentText(text=" Second part"),
            ContentReasoning(reasoning="My reasoning here")
        ])
        serialized = serialize_message(msg)

        assert serialized["role"] == MessageRole.ASSISTANT.value
        # Pydantic preserves content as list of dicts
        assert isinstance(serialized["content"], list)
        assert len(serialized["content"]) == 3
        assert serialized["content"][0]["text"] == "First part"
        assert serialized["content"][1]["text"] == " Second part"
        assert serialized["content"][2]["reasoning"] == "My reasoning here"

    def test_serialize_message_with_metadata(self):
        """Test that metadata is preserved during serialization (key feature)."""
        metadata = {"custom_key": "custom_value", "score": 0.95, "nested": {"a": 1}}
        msg = ChatMessageAssistant(
            content="Response with metadata",
            metadata=metadata
        )
        serialized = serialize_message(msg)

        assert serialized["metadata"] == metadata

    def test_serialize_message_preserves_id(self):
        """Test that message id is preserved during serialization."""
        msg = ChatMessageAssistant(content="Test")
        serialized = serialize_message(msg)

        # Message should have an auto-generated id
        assert "id" in serialized
        assert serialized["id"] is not None

    def test_serialize_message_preserves_source(self):
        """Test that source field is preserved during serialization."""
        msg = ChatMessageAssistant(content="Test", source="generate")
        serialized = serialize_message(msg)

        assert serialized["source"] == "generate"

    def test_serialize_message_preserves_model(self):
        """Test that model field is preserved for assistant messages."""
        msg = ChatMessageAssistant(content="Test", model="gpt-4")
        serialized = serialize_message(msg)

        assert serialized["model"] == "gpt-4"

    def test_serialize_unknown_message_type(self):
        """Test serialization of unknown message type raises error."""
        class UnknownMessage:
            def __init__(self):
                self.content = "test"

        with pytest.raises(ValueError, match="Unknown message type"):
            serialize_message(UnknownMessage())


class TestDeserializeMessage:
    """Tests for deserialize_message function using Pydantic's model_validate()."""

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
        assert msg.tool_calls is None

    def test_deserialize_assistant_message_with_tool_calls(self):
        """Test deserialization of assistant messages with tool calls."""
        msg_data = {
            "role": "assistant",
            "content": "Calling tool",
            "tool_calls": [
                {
                    "id": "call_456",
                    "function": "search",
                    "arguments": {"query": "test"},
                    "type": "function"
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
            "function": "search"  # Pydantic uses 'function', not 'name'
        }
        msg = deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageTool)
        assert msg.content == "Result data"
        assert msg.tool_call_id == "call_789"
        assert msg.function == "search"

    def test_deserialize_assistant_with_metadata(self):
        """Test that metadata is preserved during deserialization (key feature)."""
        metadata = {"custom_key": "custom_value", "score": 0.95}
        msg_data = {
            "role": "assistant",
            "content": "Response",
            "metadata": metadata
        }
        msg = deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageAssistant)
        assert msg.metadata == metadata

    def test_deserialize_with_id(self):
        """Test that id is preserved during deserialization."""
        msg_data = {
            "role": "assistant",
            "content": "Test",
            "id": "my-custom-id"
        }
        msg = deserialize_message(msg_data)

        assert msg.id == "my-custom-id"

    def test_deserialize_with_source(self):
        """Test that source is preserved during deserialization."""
        msg_data = {
            "role": "assistant",
            "content": "Test",
            "source": "generate"
        }
        msg = deserialize_message(msg_data)

        assert msg.source == "generate"

    def test_deserialize_with_model(self):
        """Test that model is preserved during deserialization."""
        msg_data = {
            "role": "assistant",
            "content": "Test",
            "model": "gpt-4"
        }
        msg = deserialize_message(msg_data)

        assert msg.model == "gpt-4"

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
        assert deserialized.id == original.id

    def test_roundtrip_user_message(self):
        """Test roundtrip for user messages."""
        original = ChatMessageUser(content="Hello world")
        serialized = serialize_message(original)
        deserialized = deserialize_message(serialized)

        assert isinstance(deserialized, ChatMessageUser)
        assert deserialized.content == original.content
        assert deserialized.id == original.id

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
        assert deserialized.id == original.id
        assert len(deserialized.tool_calls) == len(original.tool_calls)
        assert deserialized.tool_calls[0].id == original.tool_calls[0].id
        assert deserialized.tool_calls[0].function == original.tool_calls[0].function

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
        assert deserialized.id == original.id
        assert deserialized.tool_call_id == original.tool_call_id
        assert deserialized.function == original.function

    def test_roundtrip_preserves_metadata(self):
        """Test that metadata survives roundtrip serialization (key test for bug fix)."""
        metadata = {
            "custom_key": "custom_value",
            "score": 0.95,
            "nested": {"a": 1, "b": [1, 2, 3]}
        }
        original = ChatMessageAssistant(
            content="Response with metadata",
            metadata=metadata
        )

        serialized = serialize_message(original)
        deserialized = deserialize_message(serialized)

        assert deserialized.metadata == original.metadata
        assert deserialized.metadata == metadata

    def test_roundtrip_preserves_all_fields(self):
        """Test that all fields survive roundtrip serialization."""
        original = ChatMessageAssistant(
            content="Full message",
            metadata={"key": "value"},
            source="generate",
            model="gpt-4",
            tool_calls=[
                ToolCall(id="tc1", function="func1", arguments={"x": 1}, type="function")
            ]
        )

        serialized = serialize_message(original)
        deserialized = deserialize_message(serialized)

        assert deserialized.id == original.id
        assert deserialized.content == original.content
        assert deserialized.metadata == original.metadata
        assert deserialized.source == original.source
        assert deserialized.model == original.model
        assert len(deserialized.tool_calls) == len(original.tool_calls)


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
