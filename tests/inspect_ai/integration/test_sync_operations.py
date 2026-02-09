"""Tests for transcript sync operations.

Tests cover:
- Push message with retry
- Tool result pushing
- Sync request/response handling
- Local state tracking
- Full sync vs delta sync
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser

from saber.client.transcript import GenericTranscriptSyncOperations, WebSocketEventProcessor
from saber.inspect_ai.integration.model_wrapper import InspectAIMessageSerializer
from saber.models.rest.websocket_config import PullConfig, PushConfig, WebSocketConfig
from saber.models.rest.websocket_messages import (
    PushAckData,
    PushAckMessage,
    SyncMode,
    SyncResponseData,
    SyncResponseMessage,
    TranscriptVersion,
)


def _ts() -> str:
    """Generate a test timestamp."""
    return datetime.now().isoformat()


@pytest.fixture
def event_processor():
    """Create an event processor."""
    return WebSocketEventProcessor(
        episode_id="episode-456",
    )


@pytest.fixture
def serializer():
    """Create an InspectAI message serializer."""
    return InspectAIMessageSerializer()


@pytest.fixture
def sync_ops(event_processor, serializer):
    """Create sync operations with event processor and serializer."""
    return GenericTranscriptSyncOperations(
        episode_id="episode-456",
        event_processor=event_processor,
        serializer=serializer,
    )


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = AsyncMock()
    ws.closed = False
    ws.send = AsyncMock()
    return ws


class TestLocalStateTracking:
    """Tests for local state tracking."""

    def test_initial_state(self, sync_ops):
        """Test initial local state."""
        assert sync_ops.local_version == 0
        assert sync_ops.local_messages == []
        assert sync_ops.local_checksum is not None

    def test_local_version_setter(self, sync_ops):
        """Test setting local version."""
        sync_ops.local_version = 5
        assert sync_ops.local_version == 5

    def test_local_messages_setter(self, sync_ops):
        """Test setting local messages."""
        messages = [ChatMessageUser(content="Hello")]
        sync_ops.local_messages = messages
        assert sync_ops.local_messages == messages

    def test_clear_state(self, sync_ops):
        """Test clear_state resets all state."""
        sync_ops.local_version = 10
        sync_ops.local_messages = [ChatMessageUser(content="Test")]
        sync_ops.local_checksum = "test_checksum"

        sync_ops.clear_state()

        assert sync_ops.local_version == 0
        assert sync_ops.local_messages == []


class TestPushMessageWithRetry:
    """Tests for push_message_with_retry method."""

    @pytest.mark.asyncio
    async def test_push_message_success(self, sync_ops, event_processor, mock_websocket):
        """Test successful message push."""
        msg = ChatMessageAssistant(content="Hello from assistant")

        # Put push_ack in queue
        await event_processor.event_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(version=1, checksum="new_checksum"),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await sync_ops.push_message_with_retry(
            websocket=mock_websocket,
            msg=msg,
            context="test_push",
        )

        assert result is True
        assert sync_ops.local_version == 1
        assert sync_ops.local_checksum == "new_checksum"
        assert len(sync_ops.local_messages) == 1
        assert mock_websocket.send.called

    @pytest.mark.asyncio
    async def test_push_message_updates_local_state(self, sync_ops, event_processor, mock_websocket):
        """Test that push updates local state correctly."""
        # Push first message
        msg1 = ChatMessageAssistant(content="First")
        await event_processor.event_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(version=1, checksum="checksum1"),
            id="msg-1",
            timestamp=_ts(),
        ))
        await sync_ops.push_message_with_retry(mock_websocket, msg1)

        # Push second message
        msg2 = ChatMessageAssistant(content="Second")
        await event_processor.event_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(version=2, checksum="checksum2"),
            id="msg-2",
            timestamp=_ts(),
        ))
        await sync_ops.push_message_with_retry(mock_websocket, msg2)

        assert sync_ops.local_version == 2
        assert len(sync_ops.local_messages) == 2


class TestPushToolResultsIfNeeded:
    """Tests for push_tool_results_if_needed method."""

    @pytest.mark.asyncio
    async def test_no_push_when_local_empty(self, sync_ops, mock_websocket):
        """Test no push when local messages is empty."""
        input_messages = [ChatMessageUser(content="Test")]

        result = await sync_ops.push_tool_results_if_needed(
            websocket=mock_websocket,
            input_messages=input_messages,
        )

        assert result is False
        assert not mock_websocket.send.called

    @pytest.mark.asyncio
    async def test_no_push_when_no_new_messages(self, sync_ops, mock_websocket):
        """Test no push when input equals local messages."""
        msg = ChatMessageUser(content="Test")
        sync_ops.local_messages = [msg]

        result = await sync_ops.push_tool_results_if_needed(
            websocket=mock_websocket,
            input_messages=[msg],
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_push_tool_results(self, sync_ops, event_processor, mock_websocket):
        """Test pushing tool results."""
        # Set up existing local messages
        user_msg = ChatMessageUser(content="Query")
        assistant_msg = ChatMessageAssistant(content="Let me help", tool_calls=[])
        sync_ops.local_messages = [user_msg, assistant_msg]
        sync_ops.local_version = 2

        # Input has tool result added by Inspect AI
        tool_result = ChatMessageTool(
            content="Tool output",
            tool_call_id="call_123",
            function="my_func"
        )
        input_messages = [user_msg, assistant_msg, tool_result]

        # Put push_ack in queue for tool result
        await event_processor.event_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(version=3, checksum="new_checksum"),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await sync_ops.push_tool_results_if_needed(
            websocket=mock_websocket,
            input_messages=input_messages,
        )

        assert result is True
        assert sync_ops.local_version == 3


class TestRequestSync:
    """Tests for request_sync method."""

    @pytest.mark.asyncio
    async def test_request_sync_full(self, sync_ops, event_processor, mock_websocket):
        """Test requesting full sync."""
        # Put sync_response in queue
        await event_processor.event_queue.put(SyncResponseMessage(
            type="sync_response",
            data=SyncResponseData(
                sync_mode=SyncMode.FULL,
                full_transcript=[
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ],
                current_version=TranscriptVersion(sequence=2, checksum="server_checksum"),
            ),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await sync_ops.request_sync(mock_websocket)

        assert result is not None
        assert result.sync_mode == SyncMode.FULL
        assert len(result.full_transcript) == 2
        assert mock_websocket.send.called

    @pytest.mark.asyncio
    async def test_request_sync_delta(self, sync_ops, event_processor, mock_websocket):
        """Test requesting delta sync."""
        sync_ops.local_version = 1

        await event_processor.event_queue.put(SyncResponseMessage(
            type="sync_response",
            data=SyncResponseData(
                sync_mode=SyncMode.DELTA,
                delta=[{"role": "user", "content": "New message"}],
                current_version=TranscriptVersion(sequence=2, checksum="new_checksum"),
            ),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await sync_ops.request_sync(mock_websocket)

        assert result is not None
        assert result.sync_mode == SyncMode.DELTA
        assert len(result.delta) == 1


class TestApplySyncResponse:
    """Tests for apply_sync_response method."""

    def test_apply_full_sync(self, sync_ops):
        """Test applying full sync replaces transcript."""
        # Start with existing messages
        sync_ops.local_messages = [ChatMessageUser(content="Old")]
        sync_ops.local_version = 1

        sync_data = SyncResponseData(
            sync_mode=SyncMode.FULL,
            full_transcript=[
                {"role": "system", "content": "New system"},
                {"role": "user", "content": "New user"},
            ],
            current_version=TranscriptVersion(sequence=5, checksum="new_checksum"),
        )

        result = sync_ops.apply_sync_response(sync_data)

        assert len(result) == 2
        assert sync_ops.local_version == 5
        assert sync_ops.local_checksum == "new_checksum"

    def test_apply_delta_sync(self, sync_ops):
        """Test applying delta sync appends messages."""
        # Start with existing messages
        sync_ops.local_messages = [ChatMessageUser(content="Existing")]
        sync_ops.local_version = 1

        sync_data = SyncResponseData(
            sync_mode=SyncMode.DELTA,
            delta=[{"role": "assistant", "content": "New response"}],
            current_version=TranscriptVersion(sequence=2, checksum="new_checksum"),
        )

        result = sync_ops.apply_sync_response(sync_data)

        assert len(result) == 2
        assert result[0].content == "Existing"
        assert result[1].content == "New response"
        assert sync_ops.local_version == 2

    def test_apply_no_change_sync(self, sync_ops):
        """Test applying no_change sync keeps messages."""
        existing_msg = ChatMessageUser(content="Keep me")
        sync_ops.local_messages = [existing_msg]
        sync_ops.local_version = 1

        sync_data = SyncResponseData(
            sync_mode=SyncMode.NO_CHANGE,
            current_version=TranscriptVersion(sequence=1, checksum="same_checksum"),
        )

        result = sync_ops.apply_sync_response(sync_data)

        assert len(result) == 1
        assert result[0].content == "Keep me"


class TestUpdateLocalStateWithoutPush:
    """Tests for update_local_state_without_push method."""

    def test_updates_state(self, sync_ops):
        """Test local state update without push."""
        msg = ChatMessageAssistant(content="Response")

        sync_ops.update_local_state_without_push(msg)

        assert len(sync_ops.local_messages) == 1
        assert sync_ops.local_version == 1
        assert sync_ops.local_messages[0].content == "Response"

    def test_increments_version(self, sync_ops):
        """Test version increments with each update."""
        sync_ops.update_local_state_without_push(ChatMessageAssistant(content="First"))
        sync_ops.update_local_state_without_push(ChatMessageAssistant(content="Second"))

        assert sync_ops.local_version == 2
        assert len(sync_ops.local_messages) == 2
