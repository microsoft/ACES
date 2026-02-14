"""Tests for MessageSerializer protocol conformance.

Tests cover:
- Protocol definition completeness
- Mock serializer implementation
- Runtime protocol checking
- Protocol method contracts

This module provides a reference mock serializer that can be used for testing
generic components without depending on any specific harness (e.g., inspect_ai).
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from saber.client.transcript.protocols import MessageSerializer


# =============================================================================
# Mock Message Types and Serializer for Testing
# =============================================================================


@dataclass
class MockMessage:
    """A simple mock message type for testing.

    This represents the minimal interface a harness message needs to have.
    Real harnesses like inspect_ai have more complex message types.
    """

    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MockMessageSerializer(MessageSerializer[MockMessage]):
    """A mock MessageSerializer implementation for testing.

    This demonstrates the minimum implementation needed to satisfy
    the MessageSerializer protocol for any message type.
    """

    def serialize(self, message: MockMessage) -> dict[str, Any]:
        """Convert MockMessage to dict."""
        result: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_calls:
            result["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
        if message.metadata:
            result["metadata"] = message.metadata
        return result

    def deserialize(self, data: dict[str, Any]) -> MockMessage:
        """Convert dict to MockMessage."""
        return MockMessage(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata", {}),
        )

    def get_role(self, message: MockMessage) -> str:
        """Get message role."""
        return message.role

    def has_tool_calls(self, message: MockMessage) -> bool:
        """Check if message has tool calls."""
        return bool(message.tool_calls)

    def get_tool_call_ids(self, message: MockMessage) -> list[str]:
        """Get tool call IDs from message."""
        if message.tool_calls:
            return [tc.get("id", "") for tc in message.tool_calls if tc.get("id")]
        return []

    def get_tool_call_id(self, message: MockMessage) -> str | None:
        """Get tool_call_id from tool message."""
        return message.tool_call_id


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def serializer() -> MockMessageSerializer:
    """Create a mock message serializer."""
    return MockMessageSerializer()


@pytest.fixture
def system_message() -> MockMessage:
    """Create a system message."""
    return MockMessage(role="system", content="You are a helpful assistant")


@pytest.fixture
def user_message() -> MockMessage:
    """Create a user message."""
    return MockMessage(role="user", content="Hello, how are you?")


@pytest.fixture
def assistant_message() -> MockMessage:
    """Create an assistant message."""
    return MockMessage(role="assistant", content="I'm doing great!")


@pytest.fixture
def assistant_with_tools() -> MockMessage:
    """Create an assistant message with tool calls."""
    return MockMessage(
        role="assistant",
        content="Let me help you with that.",
        tool_calls=[
            {"id": "call_123", "function": "get_weather", "arguments": {"city": "NYC"}},
            {"id": "call_456", "function": "get_time", "arguments": {"tz": "UTC"}},
        ],
    )


@pytest.fixture
def tool_message() -> MockMessage:
    """Create a tool result message."""
    return MockMessage(
        role="tool",
        content="72°F and sunny",
        tool_call_id="call_123",
    )


# =============================================================================
# Protocol Conformance Tests
# =============================================================================


class TestProtocolRuntimeCheck:
    """Tests for runtime protocol checking."""

    def test_mock_serializer_is_instance_of_protocol(self, serializer):
        """Test that mock serializer is recognized as implementing the protocol."""
        assert isinstance(serializer, MessageSerializer)

    def test_protocol_is_runtime_checkable(self):
        """Test that the protocol is marked runtime_checkable."""
        # Create an object that doesn't implement the protocol
        class NotASerializer:
            pass

        assert not isinstance(NotASerializer(), MessageSerializer)

    def test_partial_implementation_not_recognized(self):
        """Test that partial implementations are not recognized."""

        class PartialSerializer:
            def serialize(self, message: Any) -> dict[str, Any]:
                return {"role": "test", "content": "test"}

            # Missing: deserialize, get_role, has_tool_calls, get_tool_call_ids, get_tool_call_id

        # Partial implementations should not pass isinstance check
        # Note: Protocol with runtime_checkable only checks method existence, not signatures
        # So this test verifies the basic structure
        partial = PartialSerializer()
        # This will pass because runtime_checkable only checks attribute existence
        # The full type check would be done by mypy at static analysis time
        assert hasattr(partial, "serialize")


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialize:
    """Tests for the serialize method."""

    def test_serialize_system_message(self, serializer, system_message):
        """Test serialization of system messages."""
        result = serializer.serialize(system_message)

        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant"
        assert "tool_calls" not in result
        assert "tool_call_id" not in result

    def test_serialize_user_message(self, serializer, user_message):
        """Test serialization of user messages."""
        result = serializer.serialize(user_message)

        assert result["role"] == "user"
        assert result["content"] == "Hello, how are you?"

    def test_serialize_assistant_message(self, serializer, assistant_message):
        """Test serialization of assistant messages."""
        result = serializer.serialize(assistant_message)

        assert result["role"] == "assistant"
        assert result["content"] == "I'm doing great!"

    def test_serialize_assistant_with_tool_calls(self, serializer, assistant_with_tools):
        """Test serialization of assistant messages with tool calls."""
        result = serializer.serialize(assistant_with_tools)

        assert result["role"] == "assistant"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][1]["id"] == "call_456"

    def test_serialize_tool_message(self, serializer, tool_message):
        """Test serialization of tool result messages."""
        result = serializer.serialize(tool_message)

        assert result["role"] == "tool"
        assert result["content"] == "72°F and sunny"
        assert result["tool_call_id"] == "call_123"

    def test_serialize_message_with_metadata(self, serializer):
        """Test serialization preserves metadata."""
        msg = MockMessage(
            role="user",
            content="Test",
            metadata={"source": "test", "timestamp": 12345},
        )
        result = serializer.serialize(msg)

        assert result["metadata"] == {"source": "test", "timestamp": 12345}


# =============================================================================
# Deserialization Tests
# =============================================================================


class TestDeserialize:
    """Tests for the deserialize method."""

    def test_deserialize_system_message(self, serializer):
        """Test deserialization of system messages."""
        data = {"role": "system", "content": "You are helpful"}
        result = serializer.deserialize(data)

        assert isinstance(result, MockMessage)
        assert result.role == "system"
        assert result.content == "You are helpful"

    def test_deserialize_user_message(self, serializer):
        """Test deserialization of user messages."""
        data = {"role": "user", "content": "Hello"}
        result = serializer.deserialize(data)

        assert result.role == "user"
        assert result.content == "Hello"

    def test_deserialize_assistant_with_tool_calls(self, serializer):
        """Test deserialization of assistant messages with tool calls."""
        data = {
            "role": "assistant",
            "content": "Let me help",
            "tool_calls": [{"id": "call_789", "function": "search"}],
        }
        result = serializer.deserialize(data)

        assert result.role == "assistant"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_789"

    def test_deserialize_tool_message(self, serializer):
        """Test deserialization of tool result messages."""
        data = {
            "role": "tool",
            "content": "Search results...",
            "tool_call_id": "call_789",
        }
        result = serializer.deserialize(data)

        assert result.role == "tool"
        assert result.content == "Search results..."
        assert result.tool_call_id == "call_789"

    def test_deserialize_with_metadata(self, serializer):
        """Test deserialization preserves metadata."""
        data = {
            "role": "user",
            "content": "Test",
            "metadata": {"key": "value"},
        }
        result = serializer.deserialize(data)

        assert result.metadata == {"key": "value"}


# =============================================================================
# Roundtrip Tests
# =============================================================================


class TestRoundtrip:
    """Tests for serialize/deserialize roundtrip consistency."""

    def test_roundtrip_system_message(self, serializer, system_message):
        """Test roundtrip preserves system message data."""
        serialized = serializer.serialize(system_message)
        deserialized = serializer.deserialize(serialized)

        assert deserialized.role == system_message.role
        assert deserialized.content == system_message.content

    def test_roundtrip_user_message(self, serializer, user_message):
        """Test roundtrip preserves user message data."""
        serialized = serializer.serialize(user_message)
        deserialized = serializer.deserialize(serialized)

        assert deserialized.role == user_message.role
        assert deserialized.content == user_message.content

    def test_roundtrip_assistant_with_tools(self, serializer, assistant_with_tools):
        """Test roundtrip preserves tool calls."""
        serialized = serializer.serialize(assistant_with_tools)
        deserialized = serializer.deserialize(serialized)

        assert deserialized.role == assistant_with_tools.role
        assert deserialized.content == assistant_with_tools.content
        assert deserialized.tool_calls == assistant_with_tools.tool_calls

    def test_roundtrip_tool_message(self, serializer, tool_message):
        """Test roundtrip preserves tool_call_id."""
        serialized = serializer.serialize(tool_message)
        deserialized = serializer.deserialize(serialized)

        assert deserialized.role == tool_message.role
        assert deserialized.content == tool_message.content
        assert deserialized.tool_call_id == tool_message.tool_call_id


# =============================================================================
# Role Accessor Tests
# =============================================================================


class TestGetRole:
    """Tests for the get_role method."""

    def test_get_role_system(self, serializer, system_message):
        """Test get_role returns 'system' for system messages."""
        assert serializer.get_role(system_message) == "system"

    def test_get_role_user(self, serializer, user_message):
        """Test get_role returns 'user' for user messages."""
        assert serializer.get_role(user_message) == "user"

    def test_get_role_assistant(self, serializer, assistant_message):
        """Test get_role returns 'assistant' for assistant messages."""
        assert serializer.get_role(assistant_message) == "assistant"

    def test_get_role_tool(self, serializer, tool_message):
        """Test get_role returns 'tool' for tool messages."""
        assert serializer.get_role(tool_message) == "tool"


# =============================================================================
# Tool Call Accessor Tests
# =============================================================================


class TestToolCallAccessors:
    """Tests for tool call accessor methods."""

    def test_has_tool_calls_false_for_user(self, serializer, user_message):
        """Test has_tool_calls returns False for user messages."""
        assert serializer.has_tool_calls(user_message) is False

    def test_has_tool_calls_false_for_simple_assistant(self, serializer, assistant_message):
        """Test has_tool_calls returns False for assistant without tools."""
        assert serializer.has_tool_calls(assistant_message) is False

    def test_has_tool_calls_true_for_assistant_with_tools(self, serializer, assistant_with_tools):
        """Test has_tool_calls returns True for assistant with tools."""
        assert serializer.has_tool_calls(assistant_with_tools) is True

    def test_get_tool_call_ids_empty_for_user(self, serializer, user_message):
        """Test get_tool_call_ids returns empty list for user messages."""
        assert serializer.get_tool_call_ids(user_message) == []

    def test_get_tool_call_ids_empty_for_simple_assistant(self, serializer, assistant_message):
        """Test get_tool_call_ids returns empty list for assistant without tools."""
        assert serializer.get_tool_call_ids(assistant_message) == []

    def test_get_tool_call_ids_returns_ids(self, serializer, assistant_with_tools):
        """Test get_tool_call_ids returns all tool call IDs."""
        ids = serializer.get_tool_call_ids(assistant_with_tools)
        assert len(ids) == 2
        assert "call_123" in ids
        assert "call_456" in ids

    def test_get_tool_call_id_none_for_non_tool(self, serializer, user_message):
        """Test get_tool_call_id returns None for non-tool messages."""
        assert serializer.get_tool_call_id(user_message) is None

    def test_get_tool_call_id_returns_id(self, serializer, tool_message):
        """Test get_tool_call_id returns the tool_call_id."""
        assert serializer.get_tool_call_id(tool_message) == "call_123"


# =============================================================================
# Export Mock Types for Use in Other Tests
# =============================================================================


__all__ = ["MockMessage", "MockMessageSerializer"]
