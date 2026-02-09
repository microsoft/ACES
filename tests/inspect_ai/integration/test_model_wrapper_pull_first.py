"""Tests for client pull-first logic in WebSocketTranscriptSyncingModelWrapper.

Tests the implementation of server-authoritative transcript initialization where
the client pulls the initial transcript from the server on first generate() call.

Design Doc: SERVER_SIDE_TRANSCRIPT_INIT_DESIGN.md
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from datetime import datetime

from saber.inspect_ai.integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper
from saber.models.rest.websocket_config import WebSocketConfig, PushConfig, PullConfig
from saber.models.rest.websocket_messages import (
    PushAckData,
    PushAckMessage,
    StateEventData,
    StateEventMessage,
    SyncMode,
    SyncResponseData,
    SyncResponseMessage,
    TranscriptVersion,
)
from saber.models.transcript import compute_checksum
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ModelOutput, ChatCompletionChoice


def _ts() -> str:
    """Generate a test timestamp."""
    return datetime.now().isoformat()


@pytest.fixture
def mock_base_model():
    """Create a mock base model."""
    model = AsyncMock()
    mock_message = ChatMessageAssistant(role="assistant", content="Response from model")
    mock_choice = ChatCompletionChoice(message=mock_message, stop_reason="stop")
    mock_output = ModelOutput(model="mock-model", choices=[mock_choice])
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


class TestClientPullFirst:
    """Test client pulls initial transcript on first generate() call."""

    @pytest.mark.asyncio
    async def test_first_call_pulls_initial_transcript(self, mock_base_model, mock_websocket):
        """Client waits for is_waiting_on_assistant event, then pulls transcript (unified flow)."""
        # Arrange
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        wrapper._websocket = mock_websocket
        wrapper._first_call = True

        # Mock server transcript (initialized by server)
        initial_transcript = [
            {"role": "system", "content": "You are an agent\n\n---\n\nUse tools\n\n---\n\nSubmit with /submit"},
            {"role": "user", "content": "Solve this CTF"}
        ]

        # Mock event queue: first returns is_waiting_on_assistant, then sync_response, then push_ack
        responses = [
            # State event from server (transcript ends with user → WAITING_FOR_ASSISTANT)
            StateEventMessage(
                type="is_waiting_on_assistant",
                data=StateEventData(
                    version=0,
                    operation="init",
                    modification_count=0,
                    state="WAITING_FOR_ASSISTANT",
                ),
                id="msg-1",
                timestamp=_ts(),
            ),
            # sync_response after client requests sync
            SyncResponseMessage(
                type="sync_response",
                data=SyncResponseData(
                    sync_mode=SyncMode.FULL,
                    full_transcript=initial_transcript,
                    current_version=TranscriptVersion(
                        sequence=0,
                        checksum=compute_checksum(initial_transcript),
                    ),
                ),
                id="msg-2",
                timestamp=_ts(),
            ),
            # push_ack after client pushes response
            PushAckMessage(
                type="push_ack",
                data=PushAckData(version=1, checksum="new_checksum"),
                id="msg-3",
                timestamp=_ts(),
            ),
        ]

        async def mock_event_queue_get():
            return responses.pop(0)

        wrapper._event_queue = AsyncMock()
        wrapper._event_queue.get = AsyncMock(side_effect=mock_event_queue_get)

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "ignored"}])

        # Assert - should have pulled transcript after state event
        # _local_messages includes the pulled transcript PLUS the assistant response pushed after generate()
        assert len(wrapper._local_messages) == 3  # system, user (pulled), assistant (pushed)
        # Compare role and content (local_messages are ChatMessage objects, initial_transcript are dicts)
        assert wrapper._local_messages[0].role == initial_transcript[0]["role"]
        assert wrapper._local_messages[0].content == initial_transcript[0]["content"]
        assert wrapper._local_messages[1].role == initial_transcript[1]["role"]
        assert wrapper._local_messages[1].content == initial_transcript[1]["content"]
        assert wrapper._local_version == 1  # After push
        assert wrapper._first_call == False

    @pytest.mark.asyncio
    async def test_event_driven_waits_for_event(self, mock_base_model, mock_websocket):
        """Client waits for state event with unlimited retries (sample invalid without sync)."""
        # Arrange - create wrapper with short timeout for faster test
        # Disable push to simplify test - we're testing the pull/wait behavior
        config = WebSocketConfig(
            push=PushConfig(enabled=False),
            pull=PullConfig(enabled=True, _event_timeout=0.1)  # Short timeout per attempt
        )

        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            ws_config=config,
        )

        wrapper._websocket = mock_websocket
        wrapper._first_call = False  # Past first call

        # Track that wait_for_state_event_with_retry was called
        retry_called = False
        async def mock_wait_for_state_event_with_retry():
            nonlocal retry_called
            retry_called = True
            return True  # Event received

        # Act - should call wait_for_state_event_with_retry and eventually succeed
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            with patch.object(wrapper._events, 'wait_for_state_event_with_retry', side_effect=mock_wait_for_state_event_with_retry):
                with patch.object(wrapper._sync, 'request_sync', new_callable=AsyncMock, return_value=SyncResponseData(
                    sync_mode=SyncMode.NO_CHANGE,
                    current_version=TranscriptVersion(sequence=1, checksum="abc"),
                )):
                    client_input = [{"role": "user", "content": "test"}]
                    result = await wrapper.generate(client_input)

        # Assert - should have called wait_for_state_event_with_retry and base model
        assert retry_called
        mock_base_model.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_first_call_overrides_client_input(self, mock_base_model, mock_websocket):
        """Client uses server transcript after receiving state event on first call."""
        # Arrange
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        wrapper._websocket = mock_websocket
        wrapper._first_call = True

        # Server transcript (initialized by server)
        server_transcript = [
            {"role": "system", "content": "Server system message"},
            {"role": "user", "content": "Server user message"}
        ]

        responses = [
            StateEventMessage(
                type="is_waiting_on_assistant",
                data=StateEventData(version=0, operation="init", modification_count=0, state="WAITING_FOR_ASSISTANT"),
                id="msg-1",
                timestamp=_ts(),
            ),
            SyncResponseMessage(
                type="sync_response",
                data=SyncResponseData(
                    sync_mode=SyncMode.FULL,
                    full_transcript=server_transcript,
                    current_version=TranscriptVersion(sequence=0, checksum=compute_checksum(server_transcript)),
                ),
                id="msg-2",
                timestamp=_ts(),
            ),
            PushAckMessage(
                type="push_ack",
                data=PushAckData(version=1, checksum="new"),
                id="msg-3",
                timestamp=_ts(),
            ),
        ]

        async def mock_response():
            return responses.pop(0)

        wrapper._event_queue = AsyncMock()
        wrapper._event_queue.get = AsyncMock(side_effect=mock_response)

        # Client input (should be overridden by server transcript)
        client_input = [{"role": "user", "content": "Client message - should be ignored"}]

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate(client_input)

        # Assert - base model should receive SERVER transcript, not client input
        mock_base_model.generate.assert_called_once()
        actual_input = mock_base_model.generate.call_args[0][0]
        # actual_input are ChatMessage objects, server_transcript are dicts - compare by role/content
        assert len(actual_input) == len(server_transcript)
        for actual, expected in zip(actual_input, server_transcript):
            assert actual.role == expected["role"]
            assert actual.content == expected["content"]
        # Make sure it's not the client input
        assert actual_input[0].content != client_input[0]["content"]

    @pytest.mark.asyncio
    async def test_subsequent_calls_use_normal_flow(self, mock_base_model, mock_websocket):
        """After first call, subsequent calls use same event-driven flow."""
        # Arrange
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        wrapper._websocket = mock_websocket
        wrapper._first_call = False  # Past first call
        wrapper._local_messages = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="User")
        ]
        wrapper._ws_config.pull.enabled = False  # Disable pull for this test
        wrapper._ws_config.push.enabled = False  # Disable push for this test

        client_input = [ChatMessageUser(content="Second call message")]

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate(client_input)

        # Assert - should use client input when pull disabled
        mock_base_model.generate.assert_called_once()
        actual_input = mock_base_model.generate.call_args[0][0]
        assert actual_input == client_input

    @pytest.mark.asyncio
    async def test_first_call_establishes_connection(self, mock_base_model, mock_websocket):
        """WebSocket connection is established on first call."""
        # Arrange
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        wrapper._first_call = True

        # Track that ensure_connected was called
        ensure_connected_called = False

        async def track_ensure_connected():
            nonlocal ensure_connected_called
            ensure_connected_called = True
            wrapper._websocket = mock_websocket

        # Mock responses for event-driven flow
        responses = [
            StateEventMessage(
                type="is_waiting_on_assistant",
                data=StateEventData(version=0, operation="init", modification_count=0, state="WAITING_FOR_ASSISTANT"),
                id="msg-1",
                timestamp=_ts(),
            ),
            SyncResponseMessage(
                type="sync_response",
                data=SyncResponseData(
                    sync_mode=SyncMode.FULL,
                    full_transcript=[{"role": "system", "content": "test"}],
                    current_version=TranscriptVersion(sequence=0, checksum="abc"),
                ),
                id="msg-2",
                timestamp=_ts(),
            ),
            PushAckMessage(
                type="push_ack",
                data=PushAckData(version=1, checksum="xyz"),
                id="msg-3",
                timestamp=_ts(),
            ),
        ]

        wrapper._event_queue = AsyncMock()
        wrapper._event_queue.get = AsyncMock(side_effect=lambda: responses.pop(0))

        # Act
        with patch.object(wrapper, '_ensure_connected', side_effect=track_ensure_connected):
            await wrapper.generate([{"role": "user", "content": "test"}])

        # Assert - ensure_connected must be called on first call
        assert ensure_connected_called == True
        assert wrapper._first_call == False

    @pytest.mark.asyncio
    async def test_validates_checksum(self, mock_base_model, mock_websocket):
        """Client validates and stores checksum from server."""
        # Arrange
        wrapper = WebSocketTranscriptSyncingModelWrapper(
            base_model=mock_base_model,
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        wrapper._websocket = mock_websocket
        wrapper._first_call = True

        initial_transcript = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"}
        ]

        correct_checksum = compute_checksum(initial_transcript)

        responses = [
            StateEventMessage(
                type="is_waiting_on_assistant",
                data=StateEventData(version=0, operation="init", modification_count=0, state="WAITING_FOR_ASSISTANT"),
                id="msg-1",
                timestamp=_ts(),
            ),
            SyncResponseMessage(
                type="sync_response",
                data=SyncResponseData(
                    sync_mode=SyncMode.FULL,
                    full_transcript=initial_transcript,
                    current_version=TranscriptVersion(sequence=0, checksum=correct_checksum),
                ),
                id="msg-2",
                timestamp=_ts(),
            ),
            PushAckMessage(
                type="push_ack",
                data=PushAckData(version=1, checksum="updated"),
                id="msg-3",
                timestamp=_ts(),
            ),
        ]

        wrapper._event_queue = AsyncMock()
        wrapper._event_queue.get = AsyncMock(side_effect=lambda: responses.pop(0))

        # Act
        with patch.object(wrapper, '_ensure_connected', new_callable=AsyncMock):
            await wrapper.generate([{"role": "user", "content": "test"}])

        # Assert - checksum should be stored correctly after pull (before push updates it)
        assert wrapper._local_checksum == "updated"  # Push updated the checksum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
