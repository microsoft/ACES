"""Tests for GenericTranscriptSyncOperations (harness-agnostic).

Tests cover:
- Push message with retry (success, retry on failure, reconnect on closed WS)
- Tool result pushing (seeding, diff detection, no-op, non-tool warning)
- State clearing

These tests use MockMessage / MockMessageSerializer from test_protocols.py
so they do NOT depend on inspect_ai or any specific harness.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from saber.client.transcript import GenericTranscriptSyncOperations, WebSocketEventProcessor
from saber.models.rest.websocket_config import PushConfig, WebSocketConfig
from saber.models.rest.websocket_messages import (
    PushAckData,
    PushAckMessage,
)

from .test_protocols import MockMessage, MockMessageSerializer


def _ts() -> str:
    """Generate a UTC ISO-format timestamp for test messages."""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def serializer() -> MockMessageSerializer:
    """Create a mock message serializer."""
    return MockMessageSerializer()


@pytest.fixture
def event_processor() -> WebSocketEventProcessor:
    """Create an event processor with default config."""
    return WebSocketEventProcessor(episode_id="test-episode")


@pytest.fixture
def sync_ops(
    event_processor: WebSocketEventProcessor,
    serializer: MockMessageSerializer,
) -> GenericTranscriptSyncOperations[MockMessage]:
    """Create sync operations wired to the event processor and mock serializer."""
    return GenericTranscriptSyncOperations(
        episode_id="test-episode",
        event_processor=event_processor,
        serializer=serializer,
    )


@pytest.fixture
def mock_websocket() -> AsyncMock:
    """Create a mock WebSocket that appears open."""
    ws = AsyncMock()
    ws.closed = False
    ws.send = AsyncMock()
    return ws


def _make_ack(sequence: int = 1) -> PushAckMessage:
    """Build a PushAckMessage for queuing."""
    return PushAckMessage(
        type="push_ack",
        data=PushAckData(sequence=sequence),
        id=f"ack-{sequence}",
        timestamp=_ts(),
    )


# =============================================================================
# TestPushMessageWithRetry
# =============================================================================


class TestPushMessageWithRetry:
    """Tests for push_message_with_retry method."""

    @pytest.mark.asyncio
    async def test_push_message_success(
        self,
        sync_ops: GenericTranscriptSyncOperations[MockMessage],
        event_processor: WebSocketEventProcessor,
        mock_websocket: AsyncMock,
    ) -> None:
        """Happy path: sends message, receives ack, returns True, updates local_messages."""
        msg = MockMessage(role="assistant", content="Hello from assistant")

        await event_processor.ack_queue.put(_make_ack(sequence=1))

        result = await sync_ops.push_message_with_retry(
            websocket=mock_websocket,
            msg=msg,
            context="test_push",
        )

        assert result is True
        assert len(sync_ops.local_messages) == 1
        assert sync_ops.local_messages[0] is msg
        mock_websocket.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_message_retries_on_failure(
        self,
        event_processor: WebSocketEventProcessor,
        serializer: MockMessageSerializer,
        mock_websocket: AsyncMock,
    ) -> None:
        """First send raises, second attempt succeeds after retry."""
        # Use very fast retry config so the test doesn't sleep long
        ws_config = WebSocketConfig(
            push=PushConfig(
                enabled=True,
                _confirmation_timeout=0.1,
                _retry_backoff_multiplier=1.0,
            ),
        )
        sync_ops = GenericTranscriptSyncOperations(
            episode_id="test-episode",
            event_processor=event_processor,
            serializer=serializer,
            ws_config=ws_config,
        )

        msg = MockMessage(role="assistant", content="Retry me")

        # First send raises, second send succeeds
        mock_websocket.send.side_effect = [ConnectionError("boom"), AsyncMock()]

        # The ack will be consumed on the successful second attempt
        await event_processor.ack_queue.put(_make_ack(sequence=1))

        result = await sync_ops.push_message_with_retry(
            websocket=mock_websocket,
            msg=msg,
            context="test_retry",
        )

        assert result is True
        assert len(sync_ops.local_messages) == 1
        assert mock_websocket.send.call_count == 2

    @pytest.mark.asyncio
    async def test_push_message_reconnects_on_closed_websocket(
        self,
        event_processor: WebSocketEventProcessor,
        serializer: MockMessageSerializer,
    ) -> None:
        """WebSocket is closed; reconnect callback is invoked and push succeeds on new connection."""
        ws_config = WebSocketConfig(
            push=PushConfig(
                enabled=True,
                _confirmation_timeout=0.5,
                _retry_backoff_multiplier=1.0,
            ),
        )
        sync_ops = GenericTranscriptSyncOperations(
            episode_id="test-episode",
            event_processor=event_processor,
            serializer=serializer,
            ws_config=ws_config,
        )

        msg = MockMessage(role="assistant", content="Reconnect me")

        # Original websocket reports as closed
        closed_ws = AsyncMock()
        closed_ws.closed = True

        # New websocket returned by reconnect callback
        new_ws = AsyncMock()
        new_ws.closed = False
        new_ws.send = AsyncMock()

        reconnect_cb = AsyncMock(return_value=new_ws)

        await event_processor.ack_queue.put(_make_ack(sequence=1))

        # Patch is_websocket_closed so the closed mock is recognised
        with patch(
            "saber.client.transcript.sync_operations.is_websocket_closed",
            side_effect=lambda ws: ws is closed_ws,
        ):
            result = await sync_ops.push_message_with_retry(
                websocket=closed_ws,
                msg=msg,
                context="test_reconnect",
                reconnect_callback=reconnect_cb,
            )

        assert result is True
        reconnect_cb.assert_awaited_once()
        new_ws.send.assert_called_once()
        assert len(sync_ops.local_messages) == 1


# =============================================================================
# TestPushToolResultsIfNeeded
# =============================================================================


class TestPushToolResultsIfNeeded:
    """Tests for push_tool_results_if_needed method."""

    @pytest.mark.asyncio
    async def test_seeds_local_messages_on_first_call(
        self,
        sync_ops: GenericTranscriptSyncOperations[MockMessage],
        mock_websocket: AsyncMock,
    ) -> None:
        """When local_messages is empty, copies input_messages as seed without pushing."""
        input_messages = [
            MockMessage(role="system", content="System prompt"),
            MockMessage(role="user", content="Hello"),
        ]

        result = await sync_ops.push_tool_results_if_needed(
            websocket=mock_websocket,
            input_messages=input_messages,
        )

        assert result is False
        mock_websocket.send.assert_not_called()
        # Seeded with a copy of input_messages
        assert len(sync_ops.local_messages) == 2
        assert sync_ops.local_messages[0].role == "system"

    @pytest.mark.asyncio
    async def test_pushes_new_tool_messages(
        self,
        sync_ops: GenericTranscriptSyncOperations[MockMessage],
        event_processor: WebSocketEventProcessor,
        mock_websocket: AsyncMock,
    ) -> None:
        """When input has more messages than local, pushes the new tool results."""
        user_msg = MockMessage(role="user", content="Run a tool")
        assistant_msg = MockMessage(
            role="assistant",
            content="Sure",
            tool_calls=[{"id": "call_1", "function": "bash", "arguments": {}}],
        )
        tool_msg = MockMessage(role="tool", content="output.txt", tool_call_id="call_1")

        # Pre-seed local state (simulating a prior generate pass)
        sync_ops.local_messages = [user_msg, assistant_msg]

        # Queue ack for the tool result push
        await event_processor.ack_queue.put(_make_ack(sequence=3))

        result = await sync_ops.push_tool_results_if_needed(
            websocket=mock_websocket,
            input_messages=[user_msg, assistant_msg, tool_msg],
        )

        assert result is True
        mock_websocket.send.assert_called_once()
        assert len(sync_ops.local_messages) == 3
        assert sync_ops.local_messages[-1] is tool_msg

    @pytest.mark.asyncio
    async def test_no_push_when_no_new_messages(
        self,
        sync_ops: GenericTranscriptSyncOperations[MockMessage],
        mock_websocket: AsyncMock,
    ) -> None:
        """When input_len <= local_len, returns False and does not push."""
        msg = MockMessage(role="user", content="Hello")
        sync_ops.local_messages = [msg]

        result = await sync_ops.push_tool_results_if_needed(
            websocket=mock_websocket,
            input_messages=[msg],
        )

        assert result is False
        mock_websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_warns_on_non_tool_messages(
        self,
        sync_ops: GenericTranscriptSyncOperations[MockMessage],
        event_processor: WebSocketEventProcessor,
        mock_websocket: AsyncMock,
    ) -> None:
        """New messages that aren't role 'tool' log a warning but are still pushed."""
        user_msg = MockMessage(role="user", content="Hello")
        sync_ops.local_messages = [user_msg]

        non_tool_msg = MockMessage(role="assistant", content="Surprise!")

        await event_processor.ack_queue.put(_make_ack(sequence=2))

        with patch(
            "saber.client.transcript.sync_operations.logger"
        ) as mock_logger:
            result = await sync_ops.push_tool_results_if_needed(
                websocket=mock_websocket,
                input_messages=[user_msg, non_tool_msg],
            )

        assert result is True
        mock_logger.warning.assert_called_once()
        assert "non-tool" in mock_logger.warning.call_args[0][0].lower()
        assert len(sync_ops.local_messages) == 2


# =============================================================================
# TestClearState
# =============================================================================


class TestClearState:
    """Tests for clear_state method."""

    def test_clear_state_empties_local_messages(
        self,
        sync_ops: GenericTranscriptSyncOperations[MockMessage],
    ) -> None:
        """clear_state resets local_messages to empty."""
        sync_ops.local_messages = [
            MockMessage(role="user", content="A"),
            MockMessage(role="assistant", content="B"),
        ]

        sync_ops.clear_state()

        assert sync_ops.local_messages == []
