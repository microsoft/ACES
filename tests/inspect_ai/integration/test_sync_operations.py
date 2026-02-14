"""Tests for transcript push operations.

Tests cover:
- Push message with retry
- Tool result pushing
- Local state tracking
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall

from saber.client.transcript import GenericTranscriptSyncOperations, WebSocketEventProcessor
from saber.inspect_ai.integration.model_wrapper import InspectAIMessageSerializer
from saber.models.rest.websocket_messages import (
    PushAckData,
    PushAckMessage,
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
        assert sync_ops.local_messages == []

    def test_local_messages_setter(self, sync_ops):
        """Test setting local messages."""
        messages = [ChatMessageUser(content="Hello")]
        sync_ops.local_messages = messages
        assert sync_ops.local_messages == messages

    def test_clear_state(self, sync_ops):
        """Test clear_state resets all state."""
        sync_ops.local_messages = [ChatMessageUser(content="Test")]

        sync_ops.clear_state()

        assert sync_ops.local_messages == []


class TestPushMessageWithRetry:
    """Tests for push_message_with_retry method."""

    @pytest.mark.asyncio
    async def test_push_message_success(self, sync_ops, event_processor, mock_websocket):
        """Test successful message push."""
        msg = ChatMessageAssistant(content="Hello from assistant")

        # Put push_ack in queue
        await event_processor.ack_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(sequence=1),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await sync_ops.push_message_with_retry(
            websocket=mock_websocket,
            msg=msg,
            context="test_push",
        )

        assert result is True
        assert len(sync_ops.local_messages) == 1
        assert mock_websocket.send.called

    @pytest.mark.asyncio
    async def test_push_message_updates_local_state(self, sync_ops, event_processor, mock_websocket):
        """Test that push updates local state correctly."""
        # Push first message
        msg1 = ChatMessageAssistant(content="First")
        await event_processor.ack_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(sequence=1),
            id="msg-1",
            timestamp=_ts(),
        ))
        await sync_ops.push_message_with_retry(mock_websocket, msg1)

        # Push second message
        msg2 = ChatMessageAssistant(content="Second")
        await event_processor.ack_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(sequence=2),
            id="msg-2",
            timestamp=_ts(),
        ))
        await sync_ops.push_message_with_retry(mock_websocket, msg2)

        assert len(sync_ops.local_messages) == 2


class TestPushToolResultsIfNeeded:
    """Tests for push_tool_results_if_needed method."""

    @pytest.mark.asyncio
    async def test_no_push_when_local_empty(self, sync_ops, mock_websocket):
        """Test no push when local messages is empty — seeds local tracking."""
        input_messages = [ChatMessageUser(content="Test")]

        result = await sync_ops.push_tool_results_if_needed(
            websocket=mock_websocket,
            input_messages=input_messages,
        )

        assert result is False
        assert not mock_websocket.send.called
        # First call seeds _local_messages with input so future diffs are correct
        assert len(sync_ops.local_messages) == 1

    @pytest.mark.asyncio
    async def test_seeded_local_prevents_false_diff(self, sync_ops, event_processor, mock_websocket):
        """After seeding on first call, second call correctly diffs only new messages.

        Reproduces the production warning:
          'New messages in input include non-tool messages, unexpected'
        which was caused by _local_messages only tracking pushed messages (assistant)
        while input_messages contained the full conversation (system + user + assistant + tool).
        """
        # First generate: input = [system, user], _local_messages = []
        system_msg = ChatMessageSystem(content="You are helpful")
        user_msg = ChatMessageUser(content="Hello")
        initial_input = [system_msg, user_msg]

        result = await sync_ops.push_tool_results_if_needed(
            websocket=mock_websocket,
            input_messages=initial_input,
        )
        assert result is False
        assert len(sync_ops.local_messages) == 2  # Seeded with initial input

        # Simulate: push_message adds assistant to local_messages
        assistant_msg = ChatMessageAssistant(
            content="Let me check",
            tool_calls=[ToolCall(id="call_1", function="bash", arguments={"command": "ls"}, type="function")],
        )
        sync_ops.local_messages.append(assistant_msg)
        assert len(sync_ops.local_messages) == 3

        # Second generate: input = [system, user, assistant, tool_result]
        tool_result = ChatMessageTool(content="file.txt", tool_call_id="call_1", function="bash")

        # Put push_ack in queue for tool result
        await event_processor.ack_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(sequence=4),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await sync_ops.push_tool_results_if_needed(
            websocket=mock_websocket,
            input_messages=[system_msg, user_msg, assistant_msg, tool_result],
        )
        assert result is True
        # Only the tool result was pushed (1 send call)
        assert mock_websocket.send.call_count == 1
        assert len(sync_ops.local_messages) == 4

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

        # Input has tool result added by Inspect AI
        tool_result = ChatMessageTool(
            content="Tool output",
            tool_call_id="call_123",
            function="my_func"
        )
        input_messages = [user_msg, assistant_msg, tool_result]

        # Put push_ack in queue for tool result
        await event_processor.ack_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(sequence=3),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await sync_ops.push_tool_results_if_needed(
            websocket=mock_websocket,
            input_messages=input_messages,
        )

        assert result is True
        assert len(sync_ops.local_messages) == 3
