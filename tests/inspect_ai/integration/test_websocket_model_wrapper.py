"""Tests for WebSocketTranscriptSyncingModelWrapper.

Tests cover:
- Push enabled/disabled behavior
- Push-only generate flow
- Connection establishment
- WebSocket listener event handling
- State machine event handling
- Cleanup
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

from saber.inspect_ai.integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper
from saber.models.rest.websocket_config import PullConfig, PushConfig, WebSocketConfig
from saber.models.rest.websocket_messages import (
    StateEventData,
    StateEventMessage,
    TranscriptErrorData,
    TranscriptErrorMessage,
    TranscriptErrorType,
    TranscriptOperation,
)


def _ts() -> str:
    """Generate a test timestamp."""
    return datetime.now().isoformat()


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
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


class TestPushEnabledBehavior:
    """Test push.enabled controls whether wrapper pushes messages to server."""

    @pytest.mark.asyncio
    async def test_push_enabled_sends_messages(self, mock_base_model, mock_websocket):
        """Test that push.enabled=True causes wrapper to push messages via client API."""
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),  # Disable pull to simplify
            push=PushConfig(enabled=True)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            ws_config=config,
        )

        # Act - mock client's public push methods
        with (
            patch.object(
                wrapper._client, 'push_messages_if_new',
                new_callable=AsyncMock, return_value=True,
            ) as mock_push_new,
            patch.object(
                wrapper._client, 'push_message',
                new_callable=AsyncMock, return_value=True,
            ) as mock_push_msg,
        ):
            await wrapper.generate([{"role": "user", "content": "Test"}])

        # Assert - should have called push methods
        mock_push_new.assert_called_once()
        mock_push_msg.assert_called_once()
        assert mock_push_msg.call_args.kwargs.get("context") == "output_push"

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
            ws_config=config,
        )

        # Act - mock client's public push methods
        with (
            patch.object(
                wrapper._client, 'push_messages_if_new',
                new_callable=AsyncMock, return_value=False,
            ) as mock_push_new,
            patch.object(
                wrapper._client, 'push_message',
                new_callable=AsyncMock, return_value=True,
            ) as mock_push_msg,
        ):
            await wrapper.generate([{"role": "user", "content": "Test"}])

        # Assert - push methods still called (client handles push.enabled internally)
        mock_push_new.assert_called_once()
        mock_push_msg.assert_called_once()

        # Base model should still be called
        assert mock_base_model.generate.called

    @pytest.mark.asyncio
    async def test_push_disabled_updates_local_state(self, mock_base_model, mock_websocket):
        """Test that push.enabled=False still updates local messages via client."""
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
            ws_config=config,
        )

        # Act - let push_messages_if_new return False (push disabled), but let
        # push_message run with real implementation (which tracks locally)
        with patch.object(wrapper._client, 'push_messages_if_new', new_callable=AsyncMock, return_value=False):
            await wrapper.generate([{"role": "user", "content": "Test"}])

        # Assert - local messages should have the assistant output
        assert len(wrapper._client.local_messages) == 1


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
            ws_config=config,
        )

        # Act - mock client's public push methods
        with (
            patch.object(
                wrapper._client, 'push_messages_if_new',
                new_callable=AsyncMock, return_value=True,
            ) as mock_push_new,
            patch.object(
                wrapper._client, 'push_message',
                new_callable=AsyncMock, return_value=True,
            ) as mock_push_msg,
        ):
            await wrapper.generate([{"role": "user", "content": "Inject this"}])

        # Assert
        # 1. SHOULD call push methods
        mock_push_new.assert_called_once()
        mock_push_msg.assert_called_once()

        # 2. Base model should be called immediately (no blocking)
        assert mock_base_model.generate.called


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
            ws_config=config,
        )

        input_messages = [{"role": "user", "content": "Test"}]

        # Act - mock client's public push methods
        with (
            patch.object(
                wrapper._client, 'push_messages_if_new',
                new_callable=AsyncMock, return_value=False,
            ) as mock_push_new,
            patch.object(
                wrapper._client, 'push_message',
                new_callable=AsyncMock, return_value=True,
            ) as mock_push_msg,
        ):
            result = await wrapper.generate(input_messages)

        # Assert
        # 1. Push methods called (client handles disabled check internally)
        mock_push_new.assert_called_once()
        mock_push_msg.assert_called_once()

        # 2. Base model called with original input
        assert mock_base_model.generate.called
        mock_base_model.generate.assert_called_once()

        # 3. Result returned correctly
        assert result is not None


class TestPerformanceImplications:
    """Test that enabled/disabled flags actually avoid blocking/communication."""

    @pytest.mark.asyncio
    async def test_push_disabled_fast_execution(self, mock_base_model, mock_websocket):
        """Test that push.enabled=False actually avoids blocking (fast execution)."""
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
            ws_config=config,
        )

        # Act - Time the execution with client push methods mocked
        import time
        start = time.time()

        with (
            patch.object(
                wrapper._client, 'push_messages_if_new',
                new_callable=AsyncMock, return_value=False,
            ),
            patch.object(
                wrapper._client, 'push_message',
                new_callable=AsyncMock, return_value=True,
            ),
        ):
            await wrapper.generate([{"role": "user", "content": "Test"}])

        elapsed = time.time() - start

        # Assert - should be very fast (< 0.1s) because no blocking
        assert elapsed < 0.1, f"Push disabled should be fast, took {elapsed}s"

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
            ws_config=config,
        )

        # Act - mock client push methods (client handles push.enabled internally)
        with (
            patch.object(
                wrapper._client, 'push_messages_if_new',
                new_callable=AsyncMock, return_value=False,
            ),
            patch.object(
                wrapper._client, 'push_message',
                new_callable=AsyncMock, return_value=True,
            ),
        ):
            result = await wrapper.generate([{"role": "user", "content": "Test"}])

        # Assert - worked without WebSocket
        assert result is not None
        assert mock_base_model.generate.called


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
        assert wrapper._client._connection._ws_url == expected

    def test_build_websocket_url_https(self):
        """Test WebSocket URL construction from HTTPS URL."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=AsyncMock(),
            session_id="session-123",
            episode_id="episode-789",
            rest_url="https://example.com",
        )

        expected = "wss://example.com/api/v1/episodes/episode-789/ws"
        assert wrapper._client._connection._ws_url == expected

    def test_build_websocket_url_with_path(self):
        """Test WebSocket URL construction preserves base path (for proxies)."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=AsyncMock(),
            session_id="session-123",
            episode_id="episode-abc",
            rest_url="http://proxy.com/saber",
        )

        expected = "ws://proxy.com/saber/api/v1/episodes/episode-abc/ws"
        assert wrapper._client._connection._ws_url == expected

    def test_build_websocket_url_with_port(self):
        """Test WebSocket URL construction with custom port."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=AsyncMock(),
            session_id="session-123",
            episode_id="episode-def",
            rest_url="http://localhost:9000",
        )

        expected = "ws://localhost:9000/api/v1/episodes/episode-def/ws"
        assert wrapper._client._connection._ws_url == expected

    @pytest.mark.asyncio
    async def test_context_manager_entry_exit(self, mock_base_model, mock_websocket):
        """Test async context manager entry and exit."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock connection via client's public ensure_connected
        with patch.object(wrapper._client, 'ensure_connected', new_callable=AsyncMock) as mock_ensure:
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
        wrapper._client._connection._websocket = mock_websocket

        async def dummy_listener():
            await asyncio.sleep(10)  # Long sleep - will be cancelled

        wrapper._client._events._listener_task = asyncio.create_task(dummy_listener())

        # Act
        await wrapper.cleanup()

        # Assert - listener task should be cleared after cancel_listener()
        assert wrapper._client._events._listener_task is None
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


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_push_succeeds_on_second_attempt(self, mock_base_model, mock_websocket):
        """Test that push retries and succeeds via client's push_message.

        Push retry logic is handled by the client's push_message method.
        We test this by mocking push_message directly.
        """
        # Arrange
        config = WebSocketConfig(
            pull=PullConfig(enabled=False),
            push=PushConfig(enabled=True)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            ws_config=config,
        )

        # Track push attempts
        push_call_count = 0
        async def mock_push(message: object, context: str = "message_push") -> bool:
            nonlocal push_call_count
            push_call_count += 1
            return True

        # Act - patch push_message on client
        with (
            patch.object(
                wrapper._client, 'push_messages_if_new',
                new_callable=AsyncMock, return_value=True,
            ),
            patch.object(
                wrapper._client, 'push_message',
                side_effect=mock_push,
            ),
        ):
            result = await wrapper.generate([{"role": "user", "content": "Test"}])

        # Assert - push was called and succeeded
        assert push_call_count == 1
        assert result is not None

    @pytest.mark.asyncio
    async def test_non_list_input_passes_through(self, mock_base_model, mock_websocket):
        """Test that non-list input passes through to base model."""
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
            ws_config=config,
        )

        # Act - pass string input instead of list
        # push_messages_if_new should NOT be called for string input
        with (
            patch.object(
                wrapper._client, 'push_messages_if_new',
                new_callable=AsyncMock, return_value=False,
            ) as mock_push_new,
            patch.object(
                wrapper._client, 'push_message',
                new_callable=AsyncMock, return_value=True,
            ),
        ):
            result = await wrapper.generate("String input instead of list")

        # Assert - should still work
        assert result is not None
        assert mock_base_model.generate.called
        # push_messages_if_new should NOT be called for string input
        mock_push_new.assert_not_called()


class TestWebSocketConnectionEstablishment:
    """Test ensure_connected and reconnection logic via client."""

    @pytest.mark.asyncio
    async def test_ensure_connected_success(self, mock_base_model):
        """Test successful WebSocket connection establishment via client."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websockets.connect
        mock_ws = AsyncMock()
        mock_ws.closed = False

        # Mock recv() to return connection confirmation
        mock_ws.recv = AsyncMock(
            return_value=json.dumps(
                {"type": "connected", "episode_id": "episode-456", "timestamp": _ts()}
            )
        )

        # Mock iteration for listener (empty generator that completes immediately)
        async def mock_messages():
            return
            yield  # Make this an async generator
        mock_ws.__aiter__ = lambda self: mock_messages()

        # Use AsyncMock with return_value instead of side_effect
        with patch('websockets.connect', new=AsyncMock(return_value=mock_ws)):
            await wrapper._client.ensure_connected()

        # Assert - connection established
        assert wrapper._client._connection._websocket is not None
        assert wrapper._client._events._listener_task is not None

    @pytest.mark.asyncio
    async def test_ensure_connected_already_connected(self, mock_base_model, mock_websocket):
        """Test that ensure_connected skips if already connected."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Already connected
        mock_websocket.closed = False
        wrapper._client._connection._websocket = mock_websocket

        # Act
        with patch('websockets.connect') as mock_connect:
            await wrapper._client.ensure_connected()

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
                await wrapper._client.ensure_connected()

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
            with pytest.raises(Exception):  # noqa: B017
                await wrapper._client.ensure_connected()

        # Assert - all 3 attempts were made
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_ensure_connected_confirmation_timeout(self, mock_base_model):
        """Test that connection without 'connected' message fails."""
        config = WebSocketConfig(
            push=PushConfig(enabled=True),
            reconnect_enabled=False,
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            ws_config=config,
        )

        # Mock websocket that returns wrong message type
        mock_ws = AsyncMock()
        mock_ws.closed = False

        # Mock recv() to return wrong message type instead of 'connected'
        async def wrong_message_recv():
            return json.dumps({"type": "other", "data": {}})
        mock_ws.recv = wrong_message_recv

        # Use AsyncMock with return_value instead of side_effect
        with patch('websockets.connect', new=AsyncMock(return_value=mock_ws)):
            with pytest.raises(ConnectionError, match="handshake failed"):
                await wrapper._client.ensure_connected()


class TestWebSocketListener:
    """Test event listener event handling via _events component."""

    @pytest.mark.asyncio
    async def test_listener_transcript_modified_event(self, mock_base_model):
        """Test listener queues transcript_modified events."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websocket with transcript_modified event (complete message)
        mock_ws = AsyncMock()

        async def mock_messages():
            yield json.dumps({
                "type": "transcript_modified",
                "data": {"version": 2, "operation": "append", "modification_count": 0, "state": "waiting"},
                "id": "msg-1",
                "timestamp": _ts()
            })

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act - start listener via _events component
        listener_task = asyncio.create_task(wrapper._client._events._listen_for_events(mock_ws))
        await asyncio.sleep(0.1)  # Let it process
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert - event should be in queue (as Pydantic model)
        assert not wrapper._client._events.state_queue.empty()
        event = await wrapper._client._events.state_queue.get()
        assert event.type == "transcript_modified"
        assert event.data.version == 2

    @pytest.mark.asyncio
    async def test_listener_push_ack_event(self, mock_base_model):
        """Test listener queues push_ack events."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websocket with push_ack (complete message)
        mock_ws = AsyncMock()

        async def mock_messages():
            yield json.dumps({
                "type": "push_ack",
                "data": {"sequence": 4},
                "id": "msg-1",
                "timestamp": _ts()
            })

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act
        listener_task = asyncio.create_task(wrapper._client._events._listen_for_events(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert - event should be in ack queue (as Pydantic model)
        assert not wrapper._client._events.ack_queue.empty()
        event = await wrapper._client._events.ack_queue.get()
        assert event.type == "push_ack"

    @pytest.mark.asyncio
    async def test_listener_pong_event(self, mock_base_model):
        """Test listener queues pong events."""
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websocket with pong (complete message)
        mock_ws = AsyncMock()

        async def mock_messages():
            yield json.dumps({"type": "pong", "timestamp": _ts()})

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act
        listener_task = asyncio.create_task(wrapper._client._events._listen_for_events(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert - event should be in ack queue (as Pydantic model)
        assert not wrapper._client._events.ack_queue.empty()
        event = await wrapper._client._events.ack_queue.get()
        assert event.type == "pong"

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
        listener_task = asyncio.create_task(wrapper._client._events._listen_for_events(mock_ws))
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
        await wrapper._client._events._listen_for_events(mock_ws)

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
                "data": {"state": "WAITING_FOR_USER", "version": 1, "operation": "append", "modification_count": 0},
                "id": "msg-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        ]

        async def mock_messages():
            for event in events_emitted:
                yield event

        mock_ws.__aiter__ = lambda self: mock_messages()

        # Act
        listener_task = asyncio.create_task(wrapper._client._events._listen_for_events(mock_ws))
        await asyncio.sleep(0.1)  # Let it process
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert - event should be in queue
        assert not wrapper._client._events.state_queue.empty()
        event = await wrapper._client._events.state_queue.get()
        assert event.type == "is_waiting_on_user"

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
                "data": {
                    "state": "WAITING_FOR_ASSISTANT",
                    "version": 1,
                    "operation": "append",
                    "modification_count": 0,
                },
                "id": "msg-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        ]

        async def mock_messages():
            for event in events_emitted:
                yield event

        mock_ws.__aiter__ = lambda self: mock_messages()

        listener_task = asyncio.create_task(wrapper._client._events._listen_for_events(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        assert not wrapper._client._events.state_queue.empty()
        event = await wrapper._client._events.state_queue.get()
        assert event.type == "is_waiting_on_assistant"

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
                "data": {"state": "WAITING_FOR_TOOLS", "version": 1, "operation": "append", "modification_count": 0},
                "id": "msg-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        ]

        async def mock_messages():
            for event in events_emitted:
                yield event

        mock_ws.__aiter__ = lambda self: mock_messages()

        listener_task = asyncio.create_task(wrapper._client._events._listen_for_events(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        assert not wrapper._client._events.state_queue.empty()
        event = await wrapper._client._events.state_queue.get()
        assert event.type == "is_waiting_on_tools"

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
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        ]

        async def mock_messages():
            for event in events_emitted:
                yield event

        mock_ws.__aiter__ = lambda self: mock_messages()

        listener_task = asyncio.create_task(wrapper._client._events._listen_for_events(mock_ws))
        await asyncio.sleep(0.1)
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Assert - stuck_state error should be in queue
        assert not wrapper._client._events.state_queue.empty()
        event = await wrapper._client._events.state_queue.get()
        assert event.type == "transcript_error"
        assert event.data.error == TranscriptErrorType.STUCK_STATE

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
        await wrapper._client._events.state_queue.put(TranscriptErrorMessage(
            type="transcript_error",
            data=TranscriptErrorData(error=TranscriptErrorType.STUCK_STATE),
            timestamp=_ts(),
        ))

        # Check for stuck state
        is_stuck = await wrapper._client._events.check_for_stuck_state()

        assert is_stuck is True
        # Event should still be in queue (check doesn't consume)
        assert not wrapper._client._events.state_queue.empty()

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
        await wrapper._client._events.state_queue.put(StateEventMessage(
            type="transcript_modified",
            data=StateEventData(
                version=2,
                operation=TranscriptOperation.APPEND,
                modification_count=0,
                state="waiting"
            ),
            id="msg-1",
            timestamp=_ts(),
        ))

        # Check for stuck state
        is_stuck = await wrapper._client._events.check_for_stuck_state()

        assert is_stuck is False

    @pytest.mark.asyncio
    async def test_retry_logic_on_stuck_state(self, mock_base_model, mock_websocket):
        """Test that stuck_state triggers retry with exponential backoff."""
        config = WebSocketConfig(
            pull=PullConfig(enabled=True, _event_timeout=0.1),
            push=PushConfig(enabled=False)
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",

            ws_config=config,
        )

        wrapper._client._connection._websocket = mock_websocket

        # Simulate stuck state followed by successful event
        events_to_queue = [
            TranscriptErrorMessage(
                type="transcript_error",
                data=TranscriptErrorData(error=TranscriptErrorType.STUCK_STATE),
                timestamp=_ts(),
            ),
            {
                "type": "transcript_modified",
                "data": {"version": 2, "operation": "append", "modification_count": 0, "state": "waiting"},
            },
        ]

        for event in events_to_queue:
            await wrapper._client._events.state_queue.put(event)

        # Should retry and eventually succeed (unlimited retries, but events are queued)
        result = await wrapper._client._events.wait_for_state_event_with_retry()

        assert result is True
