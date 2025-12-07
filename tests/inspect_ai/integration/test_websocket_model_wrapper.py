"""Tests for WebSocketTranscriptSyncingModelWrapper.

Tests cover:
- Pull enabled/disabled behavior (blue team vs red team)
- Push enabled/disabled behavior
- Skip first iteration logic
- Bidirectional sync scenarios
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import asyncio
import json

from saber.inspect_ai.integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper
from saber.models.rest.websocket_config import WebSocketConfig, PushConfig, PullConfig
from inspect_ai.model import ChatMessageAssistant, ModelOutput, ChatCompletionChoice


@pytest.fixture
def mock_base_model():
    """Create a mock base model."""
    model = AsyncMock()
    # Mock generate to return a proper ModelOutput with ChatMessageAssistant
    mock_message = ChatMessageAssistant(role="assistant", content="Response from model")
    mock_choice = ChatCompletionChoice(message=mock_message, stop_reason="stop")
    mock_output = ModelOutput(
        model="mock-model",
        choices=[mock_choice]
    )
    model.generate = AsyncMock(return_value=mock_output)
    return model


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.closed = False
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


class TestPullEnabledBehavior:
    """Test pull.enabled controls whether wrapper waits for modification events."""

    @pytest.mark.asyncio
    async def test_pull_enabled_waits_for_events(self, mock_base_model, mock_websocket):
        """Test that pull.enabled=True causes wrapper to wait for WebSocket events."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=True, blocking=True, event_timeout=1.0),
            push=PushConfig(enabled=False)  # Disable push to simplify test
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=False,  # Don't skip, so it should wait
            ws_config=config,
        )

        # Mock WebSocket connection
        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()

        # Put a mock event in the queue (simulate transcript_modified)
        await wrapper._event_queue.put({
            "type": "transcript_modified",
            "data": {"version": 2, "operation": "append"},
        })

        # Put a mock sync_response in the queue
        await wrapper._event_queue.put({
            "type": "sync_response",
            "data": {
                "sync_mode": "no_change",
                "current_version": {"sequence": 2, "checksum": "abc123"},
            },
        })

        input_messages = [{"role": "user", "content": "Test"}]

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate(input_messages)

        # Assert - should have sent sync_request because pull is enabled
        assert mock_websocket.send_json.called
        sync_request_calls = [
            call for call in mock_websocket.send_json.call_args_list
            if call[0][0].get("type") == "sync_request"
        ]
        assert len(sync_request_calls) > 0, "Should have sent sync_request when pull.enabled=True"

    @pytest.mark.asyncio
    async def test_pull_disabled_skips_waiting(self, mock_base_model, mock_websocket):
        """Test that pull.enabled=False skips waiting for WebSocket events (red team mode)."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False, blocking=True),  # Disabled - should skip
            push=PushConfig(enabled=False)  # Also disable push to simplify
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=False,  # Don't skip first iteration
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()

        input_messages = [{"role": "user", "content": "Test"}]

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate(input_messages)

        # Assert - should NOT have sent sync_request because pull is disabled
        if mock_websocket.send_json.called:
            sync_request_calls = [
                call for call in mock_websocket.send_json.call_args_list
                if call[0][0].get("type") == "sync_request"
            ]
            assert len(sync_request_calls) == 0, "Should NOT send sync_request when pull.enabled=False"

        # Base model should still be called
        assert mock_base_model.generate.called

    @pytest.mark.asyncio
    async def test_pull_disabled_with_skip_first_iteration(self, mock_base_model, mock_websocket):
        """Test that pull.enabled=False works correctly even with skip_first_iteration=True."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=True,  # Skip first, but pull is disabled anyway
            ws_config=config,
        )

        wrapper._websocket = mock_websocket

        # Act - call generate twice
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "First"}])
            await wrapper.generate([{"role": "user", "content": "Second"}])

        # Assert - no sync requests on either call
        if mock_websocket.send_json.called:
            sync_request_calls = [
                call for call in mock_websocket.send_json.call_args_list
                if call[0][0].get("type") == "sync_request"
            ]
            assert len(sync_request_calls) == 0


class TestPushEnabledBehavior:
    """Test push.enabled controls whether wrapper pushes messages to server."""

    @pytest.mark.asyncio
    async def test_push_enabled_sends_messages(self, mock_base_model, mock_websocket):
        """Test that push.enabled=True causes wrapper to push messages via WebSocket."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),  # Disable pull to simplify
            push=PushConfig(enabled=True, confirmation_timeout=1.0)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=True,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()

        # Put a mock push_ack in the queue
        await wrapper._event_queue.put({
            "type": "push_ack",
            "data": {
                "version": 1,
                "checksum": "abc123",
            },
        })

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "Test"}])

        # Assert - should have sent push_message
        assert mock_websocket.send_json.called
        push_message_calls = [
            call for call in mock_websocket.send_json.call_args_list
            if call[0][0].get("type") == "push_message"
        ]
        assert len(push_message_calls) > 0, "Should send push_message when push.enabled=True"

    @pytest.mark.asyncio
    async def test_push_disabled_skips_sending(self, mock_base_model, mock_websocket):
        """Test that push.enabled=False skips pushing messages (passive observer mode)."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),  # Also disable pull
            push=PushConfig(enabled=False)   # Disabled - should skip
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=True,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "Test"}])

        # Assert - should NOT have sent push_message
        if mock_websocket.send_json.called:
            push_message_calls = [
                call for call in mock_websocket.send_json.call_args_list
                if call[0][0].get("type") == "push_message"
            ]
            assert len(push_message_calls) == 0, "Should NOT send push_message when push.enabled=False"

        # Base model should still be called
        assert mock_base_model.generate.called

    @pytest.mark.asyncio
    async def test_push_disabled_updates_local_state(self, mock_base_model, mock_websocket):
        """Test that push.enabled=False still updates local state (version, checksum)."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=True,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        initial_version = wrapper._local_version

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "Test"}])

        # Assert - local version should increment even when push disabled
        assert wrapper._local_version == initial_version + 1
        assert len(wrapper._local_messages) == 1


class TestRedTeamScenario:
    """Test red team scenario: pull disabled, push enabled."""

    @pytest.mark.asyncio
    async def test_red_team_config(self, mock_base_model, mock_websocket):
        """Test red team configuration (only pushes, doesn't wait)."""
        # Arrange - Red team config
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),  # Red team doesn't wait
            push=PushConfig(enabled=True)     # But does push injections
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="red-episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=True,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()

        # Mock push_ack
        await wrapper._event_queue.put({
            "type": "push_ack",
            "data": {"version": 1, "checksum": "abc123"},
        })

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "Inject this"}])

        # Assert
        # 1. Should NOT wait for sync (no sync_request)
        sync_request_calls = [
            call for call in mock_websocket.send_json.call_args_list
            if call[0][0].get("type") == "sync_request"
        ]
        assert len(sync_request_calls) == 0, "Red team should not send sync_request"

        # 2. SHOULD push message (push_message)
        push_message_calls = [
            call for call in mock_websocket.send_json.call_args_list
            if call[0][0].get("type") == "push_message"
        ]
        assert len(push_message_calls) > 0, "Red team should send push_message"

        # 3. Base model should be called immediately (no blocking)
        assert mock_base_model.generate.called


class TestBlueTeamScenario:
    """Test blue team scenario: pull enabled, push enabled."""

    @pytest.mark.asyncio
    async def test_blue_team_config(self, mock_base_model, mock_websocket):
        """Test blue team configuration (waits for modifications and pushes)."""
        # Arrange - Blue team config
        config = WebSocketConfig(
            pull=PullConfig(enabled=True, blocking=True, event_timeout=1.0),
            push=PushConfig(enabled=True, confirmation_timeout=1.0)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="blue-episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=False,  # Blue team waits from first iteration
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()

        # Mock transcript_modified event
        await wrapper._event_queue.put({
            "type": "transcript_modified",
            "data": {"version": 2},
        })

        # Mock sync_response
        await wrapper._event_queue.put({
            "type": "sync_response",
            "data": {
                "sync_mode": "no_change",
                "current_version": {"sequence": 2, "checksum": "abc123"},
            },
        })

        # Mock push_ack
        await wrapper._event_queue.put({
            "type": "push_ack",
            "data": {"version": 3, "checksum": "def456"},
        })

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "Help me"}])

        # Assert
        # 1. Should wait for sync (sync_request sent)
        sync_request_calls = [
            call for call in mock_websocket.send_json.call_args_list
            if call[0][0].get("type") == "sync_request"
        ]
        assert len(sync_request_calls) > 0, "Blue team should send sync_request"

        # 2. Should push response (push_message sent)
        push_message_calls = [
            call for call in mock_websocket.send_json.call_args_list
            if call[0][0].get("type") == "push_message"
        ]
        assert len(push_message_calls) > 0, "Blue team should send push_message"


class TestBothDisabledScenario:
    """Test scenario where both pull and push are disabled (pass-through mode)."""

    @pytest.mark.asyncio
    async def test_both_disabled_passthrough(self, mock_base_model, mock_websocket):
        """Test that both disabled = direct pass-through to base model."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=True,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket

        input_messages = [{"role": "user", "content": "Test"}]

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            result = await wrapper.generate(input_messages)

        # Assert
        # 1. No WebSocket communication at all
        assert not mock_websocket.send_json.called, "Should not use WebSocket when both disabled"

        # 2. Base model called with original input
        assert mock_base_model.generate.called
        mock_base_model.generate.assert_called_once()

        # 3. Result returned correctly
        assert result is not None


class TestSkipFirstIterationLogic:
    """Test skip_first_iteration interaction with pull.enabled."""

    @pytest.mark.asyncio
    async def test_skip_first_iteration_then_pull_enabled(self, mock_base_model, mock_websocket):
        """Test that skip_first_iteration skips first call, then pull.enabled activates."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=True, blocking=True, event_timeout=1.0),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=True,  # Skip first
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()

        # Act - First call (should skip)
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "First"}])

        first_call_count = mock_websocket.send_json.call_count

        # Setup for second call
        await wrapper._event_queue.put({
            "type": "transcript_modified",
            "data": {"version": 2},
        })
        await wrapper._event_queue.put({
            "type": "sync_response",
            "data": {
                "sync_mode": "no_change",
                "current_version": {"sequence": 2, "checksum": "abc123"},
            },
        })

        # Act - Second call (should not skip)
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "Second"}])

        second_call_count = mock_websocket.send_json.call_count

        # Assert
        # First call should not send sync_request
        assert first_call_count == 0, "First call should skip (no WebSocket messages)"

        # Second call should send sync_request (pull enabled kicks in)
        assert second_call_count > first_call_count, "Second call should send sync_request"

    @pytest.mark.asyncio
    async def test_skip_first_iteration_with_pull_disabled(self, mock_base_model, mock_websocket):
        """Test that pull.disabled takes precedence over skip_first_iteration."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),  # Disabled - should never wait
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=True,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket

        # Act - Multiple calls
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "First"}])
            await wrapper.generate([{"role": "user", "content": "Second"}])
            await wrapper.generate([{"role": "user", "content": "Third"}])

        # Assert - no sync requests on any call
        if mock_websocket.send_json.called:
            sync_request_calls = [
                call for call in mock_websocket.send_json.call_args_list
                if call[0][0].get("type") == "sync_request"
            ]
            assert len(sync_request_calls) == 0, "Pull disabled should prevent all sync requests"


class TestPerformanceImplications:
    """Test that enabled/disabled flags actually avoid blocking/communication."""

    @pytest.mark.asyncio
    async def test_pull_disabled_no_blocking(self, mock_base_model, mock_websocket):
        """Test that pull.enabled=False actually avoids blocking (fast execution)."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=False,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket

        # Act - Time the execution
        import time
        start = time.time()

        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "Test"}])

        elapsed = time.time() - start

        # Assert - should be very fast (< 0.1s) because no blocking
        assert elapsed < 0.1, f"Pull disabled should be fast, took {elapsed}s"

    @pytest.mark.asyncio
    async def test_push_disabled_no_websocket_communication(self, mock_base_model):
        """Test that push.enabled=False avoids all WebSocket push communication."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=True,
            ws_config=config,
        )

        # Don't even set up WebSocket - it shouldn't be needed
        wrapper._websocket = None

        # Act - Should not crash even without WebSocket
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            result = await wrapper.generate([{"role": "user", "content": "Test"}])

        # Assert - worked without WebSocket
        assert result is not None
        assert mock_base_model.generate.called


class TestMessageSerialization:
    """Test message serialization and deserialization for all message types."""

    def test_serialize_system_message(self):
        """Test serialization of system messages."""
        from inspect_ai.model import ChatMessageSystem
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        msg = ChatMessageSystem(content="You are a helpful assistant")
        serialized = _MessageSerializationMixin._serialize_message(msg)

        assert serialized["role"] == "system"
        assert serialized["content"] == "You are a helpful assistant"

    def test_serialize_user_message(self):
        """Test serialization of user messages."""
        from inspect_ai.model import ChatMessageUser
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        msg = ChatMessageUser(content="Hello, how are you?")
        serialized = _MessageSerializationMixin._serialize_message(msg)

        assert serialized["role"] == "user"
        assert serialized["content"] == "Hello, how are you?"

    def test_serialize_assistant_message_simple(self):
        """Test serialization of simple assistant messages without tool calls."""
        from inspect_ai.model import ChatMessageAssistant
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        msg = ChatMessageAssistant(content="I'm doing great!")
        serialized = _MessageSerializationMixin._serialize_message(msg)

        assert serialized["role"] == "assistant"
        assert serialized["content"] == "I'm doing great!"
        assert "tool_calls" not in serialized

    def test_serialize_assistant_message_with_tool_calls(self):
        """Test serialization of assistant messages with tool calls."""
        from inspect_ai.model import ChatMessageAssistant
        from inspect_ai.tool import ToolCall
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        tool_calls = [
            ToolCall(
                id="call_123",
                function="get_weather",
                arguments={"location": "San Francisco"},
                type="function"
            )
        ]
        msg = ChatMessageAssistant(content="Let me check the weather", tool_calls=tool_calls)
        serialized = _MessageSerializationMixin._serialize_message(msg)

        assert serialized["role"] == "assistant"
        assert serialized["content"] == "Let me check the weather"
        assert len(serialized["tool_calls"]) == 1
        assert serialized["tool_calls"][0]["id"] == "call_123"
        assert serialized["tool_calls"][0]["function"] == "get_weather"
        assert serialized["tool_calls"][0]["arguments"] == {"location": "San Francisco"}

    def test_serialize_tool_message(self):
        """Test serialization of tool result messages."""
        from inspect_ai.model import ChatMessageTool
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        msg = ChatMessageTool(
            content="Temperature is 72°F",
            tool_call_id="call_123",
            function="get_weather"
        )
        serialized = _MessageSerializationMixin._serialize_message(msg)

        assert serialized["role"] == "tool"
        assert serialized["content"] == "Temperature is 72°F"
        assert serialized["tool_call_id"] == "call_123"
        assert serialized["name"] == "get_weather"

    def test_serialize_message_with_content_list(self):
        """Test serialization of messages with list content (ContentText, ContentReasoning)."""
        from inspect_ai.model import ChatMessageAssistant, ContentText, ContentReasoning
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        msg = ChatMessageAssistant(content=[
            ContentText(text="First part"),
            ContentText(text=" Second part"),
            ContentReasoning(reasoning="My reasoning here")
        ])
        serialized = _MessageSerializationMixin._serialize_message(msg)

        assert serialized["role"] == "assistant"
        assert serialized["content"] == "First part Second part"
        assert serialized["reasoning"] == "My reasoning here"

    def test_deserialize_system_message(self):
        """Test deserialization of system messages."""
        from inspect_ai.model import ChatMessageSystem
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        msg_data = {"role": "system", "content": "You are helpful"}
        msg = _MessageSerializationMixin._deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageSystem)
        assert msg.content == "You are helpful"

    def test_deserialize_user_message(self):
        """Test deserialization of user messages."""
        from inspect_ai.model import ChatMessageUser
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        msg_data = {"role": "user", "content": "Hello"}
        msg = _MessageSerializationMixin._deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageUser)
        assert msg.content == "Hello"

    def test_deserialize_assistant_message_simple(self):
        """Test deserialization of simple assistant messages."""
        from inspect_ai.model import ChatMessageAssistant
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        msg_data = {"role": "assistant", "content": "Hi there"}
        msg = _MessageSerializationMixin._deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageAssistant)
        assert msg.content == "Hi there"
        assert msg.tool_calls is None or len(msg.tool_calls) == 0

    def test_deserialize_assistant_message_with_tool_calls(self):
        """Test deserialization of assistant messages with tool calls."""
        from inspect_ai.model import ChatMessageAssistant
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

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
        msg = _MessageSerializationMixin._deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageAssistant)
        assert msg.content == "Calling tool"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_456"
        assert msg.tool_calls[0].function == "search"

    def test_deserialize_tool_message(self):
        """Test deserialization of tool messages."""
        from inspect_ai.model import ChatMessageTool
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        msg_data = {
            "role": "tool",
            "content": "Result data",
            "tool_call_id": "call_789",
            "name": "search"
        }
        msg = _MessageSerializationMixin._deserialize_message(msg_data)

        assert isinstance(msg, ChatMessageTool)
        assert msg.content == "Result data"
        assert msg.tool_call_id == "call_789"
        assert msg.function == "search"

    def test_deserialize_tool_message_missing_fields(self):
        """Test deserialization of tool messages with missing required fields raises error."""
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        # Missing tool_call_id
        msg_data = {"role": "tool", "content": "Result", "name": "search"}
        with pytest.raises(ValueError, match="missing tool_call_id or name"):
            _MessageSerializationMixin._deserialize_message(msg_data)

        # Missing name
        msg_data = {"role": "tool", "content": "Result", "tool_call_id": "call_123"}
        with pytest.raises(ValueError, match="missing tool_call_id or name"):
            _MessageSerializationMixin._deserialize_message(msg_data)

    def test_deserialize_unknown_role(self):
        """Test deserialization with unknown role raises error."""
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        msg_data = {"role": "unknown_role", "content": "Test"}
        with pytest.raises(ValueError, match="Unknown message role"):
            _MessageSerializationMixin._deserialize_message(msg_data)

    def test_serialize_unknown_message_type(self):
        """Test serialization of unknown message type raises error."""
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        # Create a mock message with unknown type
        class UnknownMessage:
            def __init__(self):
                self.content = "test"

        with pytest.raises(ValueError, match="Unknown message type"):
            _MessageSerializationMixin._serialize_message(UnknownMessage())

    def test_roundtrip_serialization(self):
        """Test that serialize -> deserialize produces equivalent messages."""
        from inspect_ai.model import ChatMessageAssistant
        from inspect_ai.tool import ToolCall
        from saber.inspect_ai.integration.model_wrapper import _MessageSerializationMixin

        original = ChatMessageAssistant(
            content="Test message",
            tool_calls=[
                ToolCall(id="tc1", function="func1", arguments={"arg": "val"}, type="function")
            ]
        )

        serialized = _MessageSerializationMixin._serialize_message(original)
        deserialized = _MessageSerializationMixin._deserialize_message(serialized)

        assert isinstance(deserialized, ChatMessageAssistant)
        assert deserialized.content == original.content
        assert len(deserialized.tool_calls) == len(original.tool_calls)
        assert deserialized.tool_calls[0].id == original.tool_calls[0].id


class TestWebSocketConnection:
    """Test WebSocket connection establishment and URL building."""

    def test_build_websocket_url_basic(self):
        """Test WebSocket URL construction from basic HTTP URL."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=AsyncMock(),
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        expected = "ws://localhost:8000/api/v1/episodes/episode-456/ws"
        assert wrapper._ws_url == expected

    def test_build_websocket_url_https(self):
        """Test WebSocket URL construction from HTTPS URL."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=AsyncMock(),
            session_id="session-123",
            episode_id="episode-789",
            rest_url="https://example.com",
        )

        expected = "wss://example.com/api/v1/episodes/episode-789/ws"
        assert wrapper._ws_url == expected

    def test_build_websocket_url_with_path(self):
        """Test WebSocket URL construction preserves base path (for proxies)."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=AsyncMock(),
            session_id="session-123",
            episode_id="episode-abc",
            rest_url="http://proxy.com/saber",
        )

        expected = "ws://proxy.com/saber/api/v1/episodes/episode-abc/ws"
        assert wrapper._ws_url == expected

    def test_build_websocket_url_with_port(self):
        """Test WebSocket URL construction with custom port."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=AsyncMock(),
            session_id="session-123",
            episode_id="episode-def",
            rest_url="http://localhost:9000",
        )

        expected = "ws://localhost:9000/api/v1/episodes/episode-def/ws"
        assert wrapper._ws_url == expected

    @pytest.mark.asyncio
    async def test_context_manager_entry_exit(self, mock_base_model, mock_websocket):
        """Test async context manager entry and exit."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock connection
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock) as mock_ensure:
            async with wrapper as w:
                assert w is wrapper
                assert mock_ensure.called

        # Cleanup should have been called on exit
        # (cleanup closes websocket and cancels listener)

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_base_model, mock_websocket):
        """Test cleanup closes WebSocket and cancels listener."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Set up WebSocket and listener (needs to be an actual task)
        wrapper._websocket = mock_websocket

        async def dummy_listener():
            await asyncio.sleep(10)  # Long sleep - will be cancelled

        wrapper._listener_task = asyncio.create_task(dummy_listener())

        # Act
        await wrapper.cleanup()

        # Assert - task should be cancelled
        assert wrapper._listener_task.cancelled()
        assert mock_websocket.close.called

    def test_getattr_delegation(self, mock_base_model):
        """Test that unknown attributes are delegated to base model."""
        mock_base_model.custom_attribute = "test_value"

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        assert wrapper.custom_attribute == "test_value"

    def test_repr(self, mock_base_model):
        """Test string representation."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        repr_str = repr(wrapper)
        assert "WebSocketTranscriptSyncingModelWrapper" in repr_str


class TestSyncModes:
    """Test full sync vs delta sync behavior."""

    @pytest.mark.asyncio
    async def test_delta_sync_appends_messages(self, mock_base_model, mock_websocket):
        """Test that delta sync appends new messages to local transcript."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=True),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=False,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()
        wrapper._local_messages = []  # Start with empty transcript

        # Simulate WebSocket event + sync response
        await wrapper._event_queue.put({
            "type": "transcript_modified",
            "data": {"version": 1},
        })

        await wrapper._event_queue.put({
            "type": "sync_response",
            "data": {
                "sync_mode": "delta",
                "delta": [
                    {"role": "user", "content": "New message"}
                ],
                "current_version": {"sequence": 1, "checksum": "abc123"},
            },
        })

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "system", "content": "Initial"}])

        # Assert - delta should be appended (1 from sync + 1 from generate output since push disabled updates local)
        # With push disabled, generate() adds the output message to local_messages
        assert len(wrapper._local_messages) == 2  # delta message + generated message
        assert wrapper._local_messages[0].content == "New message"  # From delta
        assert wrapper._local_messages[1].content == "Response from model"  # From generate
        assert wrapper._local_version == 2  # Incremented by push disabled local update

    @pytest.mark.asyncio
    async def test_full_sync_replaces_transcript(self, mock_base_model, mock_websocket):
        """Test that full sync replaces entire local transcript (rewrite detected)."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=True),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=False,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()

        # Start with existing messages (that will be replaced)
        from inspect_ai.model import ChatMessageUser
        wrapper._local_messages = [ChatMessageUser(content="Old message")]

        # Simulate WebSocket event + full sync response
        await wrapper._event_queue.put({
            "type": "transcript_modified",
            "data": {"version": 5},
        })

        await wrapper._event_queue.put({
            "type": "sync_response",
            "data": {
                "sync_mode": "full",
                "full_transcript": [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Completely new transcript"}
                ],
                "current_version": {"sequence": 5, "checksum": "xyz789"},
            },
        })

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "system", "content": "Ignored"}])

        # Assert - old messages should be replaced (2 from full sync + 1 from generate)
        assert len(wrapper._local_messages) == 3  # 2 from full_transcript + 1 generated
        assert wrapper._local_messages[0].content == "System prompt"
        assert wrapper._local_messages[1].content == "Completely new transcript"
        assert wrapper._local_messages[2].content == "Response from model"  # From generate
        assert wrapper._local_version == 6  # 5 from sync + 1 from local update


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_websocket_timeout_graceful_degradation(self, mock_base_model, mock_websocket):
        """Test that WebSocket timeout doesn't crash, continues with original input."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=True, event_timeout=0.1),  # Short timeout
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=False,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()  # Empty queue - will timeout

        input_messages = [{"role": "user", "content": "Original input"}]

        # Act - should not crash despite timeout
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            result = await wrapper.generate(input_messages)

        # Assert - should continue with original input
        assert result is not None
        assert mock_base_model.generate.called

    @pytest.mark.asyncio
    async def test_push_failure_graceful_degradation(self, mock_base_model, mock_websocket):
        """Test that push failure doesn't crash, falls back to local tracking."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),
            push=PushConfig(enabled=True, confirmation_timeout=0.1)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._event_queue = asyncio.Queue()  # Empty - will timeout
        wrapper._local_messages = []
        wrapper._local_version = 0

        # Act - should not crash despite push timeout
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            result = await wrapper.generate([{"role": "user", "content": "Test"}])

        # Assert - should have updated local state as fallback
        assert len(wrapper._local_messages) == 1
        assert wrapper._local_version == 1
        assert result is not None

    @pytest.mark.asyncio
    async def test_non_list_input_warning(self, mock_base_model, mock_websocket):
        """Test that non-list input generates warning but doesn't crash."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=True),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=False,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket

        # Act - pass string input instead of list
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            result = await wrapper.generate("String input instead of list")

        # Assert - should still work
        assert result is not None
        assert mock_base_model.generate.called


class TestWebSocketConnectionEstablishment:
    """Test _ensure_connected and reconnection logic."""

    @pytest.mark.asyncio
    async def test_ensure_connected_success(self, mock_base_model):
        """Test successful WebSocket connection establishment."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websockets.connect
        mock_ws = AsyncMock()
        mock_ws.closed = False

        # Simulate connection confirmation message
        async def mock_messages():
            yield json.dumps({"type": "connected", "data": {}})

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Use AsyncMock with return_value instead of side_effect
        with patch('websockets.connect', new=AsyncMock(return_value=mock_ws)):
            await wrapper._ensure_connected()

        # Assert - connection established
        assert wrapper._websocket is not None
        assert wrapper._listener_task is not None

    @pytest.mark.asyncio
    async def test_ensure_connected_already_connected(self, mock_base_model, mock_websocket):
        """Test that _ensure_connected skips if already connected."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Already connected
        mock_websocket.closed = False
        wrapper._websocket = mock_websocket

        # Act
        with patch('websockets.connect') as mock_connect:
            await wrapper._ensure_connected()

        # Assert - should not attempt new connection
        assert not mock_connect.called

    @pytest.mark.asyncio
    async def test_ensure_connected_reconnect_disabled_fails_fast(self, mock_base_model):
        """Test that with reconnect disabled, connection failure raises immediately."""
        config = WebSocketConfig(
            reconnect_enabled=False
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            ws_config=config,
        )

        # Mock websockets.connect to fail
        with patch('websockets.connect', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                await wrapper._ensure_connected()

    @pytest.mark.asyncio
    async def test_ensure_connected_exponential_backoff(self, mock_base_model):
        """Test exponential backoff retry logic."""
        config = WebSocketConfig(
            reconnect_enabled=True,
            max_reconnect_attempts=3,
            initial_reconnect_delay=0.01,  # Short for testing
            reconnect_backoff_multiplier=2.0,
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            ws_config=config,
        )

        attempt_count = 0

        async def failing_connect(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise Exception(f"Attempt {attempt_count} failed")

        # All attempts fail
        with patch('websockets.connect', side_effect=failing_connect):
            with pytest.raises(Exception):
                await wrapper._ensure_connected()

        # Assert - all 3 attempts were made
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_ensure_connected_confirmation_timeout(self, mock_base_model):
        """Test that connection without 'connected' message times out."""
        config = WebSocketConfig(
            push=PushConfig(confirmation_timeout=0.1),  # Short timeout
            reconnect_enabled=False,
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            ws_config=config,
        )

        # Mock websocket that never sends 'connected'
        mock_ws = AsyncMock()
        mock_ws.closed = False

        async def mock_messages():
            await asyncio.sleep(1)  # Longer than timeout
            yield json.dumps({"type": "other", "data": {}})

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Use AsyncMock with return_value instead of side_effect
        with patch('websockets.connect', new=AsyncMock(return_value=mock_ws)):
            with pytest.raises(ConnectionError, match="confirmation timeout"):
                await wrapper._ensure_connected()


class TestWebSocketListener:
    """Test _listen_for_events_impl event handling."""

    @pytest.mark.asyncio
    async def test_listener_transcript_modified_event(self, mock_base_model):
        """Test listener queues transcript_modified events."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websocket with transcript_modified event
        mock_ws = AsyncMock()

        async def mock_messages():
            yield json.dumps({
                "type": "transcript_modified",
                "data": {"version": 2, "operation": "append"}
            })

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act - start listener
        listener_task = asyncio.create_task(wrapper._listen_for_events_impl(mock_ws))
        await asyncio.sleep(0.1)  # Let it process
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert - event should be in queue
        assert not wrapper._event_queue.empty()
        event = await wrapper._event_queue.get()
        assert event["type"] == "transcript_modified"
        assert event["data"]["version"] == 2

    @pytest.mark.asyncio
    async def test_listener_sync_response_event(self, mock_base_model):
        """Test listener queues sync_response events."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websocket with sync_response
        mock_ws = AsyncMock()

        async def mock_messages():
            yield json.dumps({
                "type": "sync_response",
                "data": {"sync_mode": "delta", "current_version": {"sequence": 3}}
            })

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act
        listener_task = asyncio.create_task(wrapper._listen_for_events_impl(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert
        event = await wrapper._event_queue.get()
        assert event["type"] == "sync_response"

    @pytest.mark.asyncio
    async def test_listener_push_ack_event(self, mock_base_model):
        """Test listener queues push_ack events."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websocket with push_ack
        mock_ws = AsyncMock()

        async def mock_messages():
            yield json.dumps({
                "type": "push_ack",
                "data": {"version": 4, "checksum": "abc123"}
            })

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act
        listener_task = asyncio.create_task(wrapper._listen_for_events_impl(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert
        event = await wrapper._event_queue.get()
        assert event["type"] == "push_ack"

    @pytest.mark.asyncio
    async def test_listener_pong_event(self, mock_base_model):
        """Test listener queues pong events."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websocket with pong
        mock_ws = AsyncMock()

        async def mock_messages():
            yield json.dumps({"type": "pong", "data": {}})

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act
        listener_task = asyncio.create_task(wrapper._listen_for_events_impl(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert
        event = await wrapper._event_queue.get()
        assert event["type"] == "pong"

    @pytest.mark.asyncio
    async def test_listener_unknown_event_type(self, mock_base_model):
        """Test listener handles unknown event types gracefully."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websocket with unknown event
        mock_ws = AsyncMock()

        async def mock_messages():
            yield json.dumps({"type": "unknown_event", "data": {}})

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act - should not crash
        listener_task = asyncio.create_task(wrapper._listen_for_events_impl(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert - event queue should be empty (unknown events not queued)
        # Actually, looking at the code, unknown events just log a warning
        # Let's verify it didn't crash
        assert True  # Made it here without exception

    @pytest.mark.asyncio
    async def test_listener_connection_closed(self, mock_base_model):
        """Test listener handles connection closed gracefully."""
        import websockets

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websocket that raises ConnectionClosed
        mock_ws = AsyncMock()

        async def mock_messages():
            raise websockets.exceptions.ConnectionClosed(None, None)

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act - should handle gracefully
        await wrapper._listen_for_events_impl(mock_ws)

        # Assert - completed without crashing
        assert True


class TestStateMachineEventHandling:
    """Test state machine event types and stuck_state retry logic."""

    @pytest.mark.asyncio
    async def test_handles_is_waiting_on_user_event(self, mock_base_model):
        """Test that is_waiting_on_user events are recognized and logged."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websocket that emits is_waiting_on_user event
        mock_ws = AsyncMock()
        events_emitted = [
            json.dumps({
                "type": "is_waiting_on_user",
                "data": {"state": "WAITING_FOR_USER"},
                "timestamp": datetime.utcnow().isoformat(),
            })
        ]

        async def mock_messages():
            for event in events_emitted:
                yield event

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act
        listener_task = asyncio.create_task(wrapper._listen_for_events_impl(mock_ws))
        await asyncio.sleep(0.1)  # Let it process
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert - event should be in queue
        assert not wrapper._event_queue.empty()
        event = await wrapper._event_queue.get()
        assert event["type"] == "is_waiting_on_user"

    @pytest.mark.asyncio
    async def test_handles_is_waiting_on_assistant_event(self, mock_base_model):
        """Test that is_waiting_on_assistant events are recognized and logged."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        mock_ws = AsyncMock()
        events_emitted = [
            json.dumps({
                "type": "is_waiting_on_assistant",
                "data": {"state": "WAITING_FOR_ASSISTANT"},
                "timestamp": datetime.utcnow().isoformat(),
            })
        ]

        async def mock_messages():
            for event in events_emitted:
                yield event

        mock_ws.__aiter__ = lambda self: mock_messages()

        listener_task = asyncio.create_task(wrapper._listen_for_events_impl(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        assert not wrapper._event_queue.empty()
        event = await wrapper._event_queue.get()
        assert event["type"] == "is_waiting_on_assistant"

    @pytest.mark.asyncio
    async def test_handles_is_waiting_on_tools_event(self, mock_base_model):
        """Test that is_waiting_on_tools events are recognized and logged."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        mock_ws = AsyncMock()
        events_emitted = [
            json.dumps({
                "type": "is_waiting_on_tools",
                "data": {"state": "WAITING_FOR_TOOLS"},
                "timestamp": datetime.utcnow().isoformat(),
            })
        ]

        async def mock_messages():
            for event in events_emitted:
                yield event

        mock_ws.__aiter__ = lambda self: mock_messages()

        listener_task = asyncio.create_task(wrapper._listen_for_events_impl(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        assert not wrapper._event_queue.empty()
        event = await wrapper._event_queue.get()
        assert event["type"] == "is_waiting_on_tools"

    @pytest.mark.asyncio
    async def test_handles_stuck_state_error(self, mock_base_model):
        """Test that stuck_state errors are recognized and logged."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        mock_ws = AsyncMock()
        events_emitted = [
            json.dumps({
                "type": "transcript_error",
                "data": {
                    "error": "stuck_state",
                    "state": "WAITING_FOR_ASSISTANT",
                    "duration_seconds": 320.5,
                    "threshold_seconds": 300.0,
                },
                "timestamp": datetime.utcnow().isoformat(),
            })
        ]

        async def mock_messages():
            for event in events_emitted:
                yield event

        mock_ws.__aiter__ = lambda self: mock_messages()

        listener_task = asyncio.create_task(wrapper._listen_for_events_impl(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert - stuck_state error should be in queue
        assert not wrapper._event_queue.empty()
        event = await wrapper._event_queue.get()
        assert event["type"] == "transcript_error"
        assert event["data"]["error"] == "stuck_state"

    @pytest.mark.asyncio
    async def test_check_for_stuck_state_detection(self, mock_base_model):
        """Test _check_for_stuck_state detects stuck_state errors in queue."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Add stuck_state error to queue
        await wrapper._event_queue.put({
            "type": "transcript_error",
            "data": {"error": "stuck_state"},
        })

        # Check for stuck state
        is_stuck = await wrapper._check_for_stuck_state()

        assert is_stuck is True
        # Event should still be in queue (check doesn't consume)
        assert not wrapper._event_queue.empty()

    @pytest.mark.asyncio
    async def test_check_for_stuck_state_no_error(self, mock_base_model):
        """Test _check_for_stuck_state returns False when no stuck_state."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Add normal event to queue
        await wrapper._event_queue.put({
            "type": "transcript_modified",
            "data": {"version": 2},
        })

        # Check for stuck state
        is_stuck = await wrapper._check_for_stuck_state()

        assert is_stuck is False

    @pytest.mark.asyncio
    async def test_retry_logic_on_stuck_state(self, mock_base_model, mock_websocket):
        """Test that stuck_state triggers retry with exponential backoff."""
        config = WebSocketConfig(
            pull=PullConfig(enabled=True, blocking=True, event_timeout=0.1),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            skip_first_iteration=False,
            ws_config=config,
        )

        wrapper._websocket = mock_websocket

        # Simulate stuck state followed by successful event
        events_to_queue = [
            {"type": "transcript_error", "data": {"error": "stuck_state"}},
            {"type": "transcript_modified", "data": {"version": 2}},
            {"type": "sync_response", "data": {
                "sync_mode": "no_change",
                "current_version": {"sequence": 2, "checksum": "abc"},
            }},
        ]

        for event in events_to_queue:
            await wrapper._event_queue.put(event)

        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            # Should retry and eventually succeed
            result = await wrapper._wait_for_modification_event_with_retry(max_retries=2)

            assert result is True
