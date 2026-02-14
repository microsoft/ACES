"""Tests for WebSocket event processing.

Tests cover:
- Event listener background task
- Event queue management
- State event waiting and filtering
- Stuck state detection
- Retry logic with exponential backoff

These tests are harness-agnostic - they test the generic WebSocketEventProcessor
that can be used with any evaluation harness.
"""

import asyncio
from datetime import datetime

import pytest

from saber.client.transcript import WebSocketEventProcessor
from saber.models.rest.websocket_config import PullConfig, WebSocketConfig
from saber.models.rest.websocket_messages import (
    PushAckData,
    PushAckMessage,
    StateEventData,
    StateEventMessage,
    TranscriptErrorData,
    TranscriptErrorMessage,
    TranscriptErrorType,
    TranscriptOperation,
    WebSocketMessageType,
)


def _ts() -> str:
    """Generate a test timestamp."""
    return datetime.now().isoformat()


@pytest.fixture
def event_processor():
    """Create an event processor with default config."""
    return WebSocketEventProcessor(
        episode_id="episode-456",
    )


@pytest.fixture
def event_processor_short_timeout():
    """Create an event processor with short timeout for tests."""
    config = WebSocketConfig(
        pull=PullConfig(enabled=True, _event_timeout=0.1)
    )
    return WebSocketEventProcessor(
        episode_id="episode-456",
        ws_config=config,
    )


class TestEventQueueAccess:
    """Tests for event queue access."""

    def test_ack_queue_exists(self, event_processor):
        """Test that ack queue is created."""
        assert event_processor.ack_queue is not None
        assert isinstance(event_processor.ack_queue, asyncio.Queue)

    def test_state_queue_exists(self, event_processor):
        """Test that state queue is created."""
        assert event_processor.state_queue is not None
        assert isinstance(event_processor.state_queue, asyncio.Queue)

    def test_queues_are_empty_initially(self, event_processor):
        """Test that both queues are empty on creation."""
        assert event_processor.ack_queue.empty()
        assert event_processor.state_queue.empty()


class TestWaitForStateEvent:
    """Tests for wait_for_state_event method."""

    @pytest.mark.asyncio
    async def test_wait_for_state_event_returns_true_on_state_event(self, event_processor_short_timeout):
        """Test that state events are accepted."""
        processor = event_processor_short_timeout

        # Put a state event in the state queue
        await processor.state_queue.put(StateEventMessage(
            type="is_waiting_on_assistant",
            data=StateEventData(
                version=1,
                operation=TranscriptOperation.APPEND,
                modification_count=0,
                state="WAITING_FOR_ASSISTANT"
            ),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await processor.wait_for_state_event()

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_state_event_accepts_transcript_modified(self, event_processor_short_timeout):
        """Test that transcript_modified events are accepted."""
        processor = event_processor_short_timeout

        await processor.state_queue.put(StateEventMessage(
            type="transcript_modified",
            data=StateEventData(
                version=2,
                operation=TranscriptOperation.APPEND,
                modification_count=1,
                state="WAITING_FOR_USER"
            ),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await processor.wait_for_state_event()

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_state_event_timeout_returns_false(self, event_processor_short_timeout):
        """Test that timeout returns False."""
        processor = event_processor_short_timeout

        # Empty queue - will timeout
        result = await processor.wait_for_state_event()

        assert result is False


class TestWaitForMessageType:
    """Tests for wait_for_message_type method."""

    @pytest.mark.asyncio
    async def test_wait_for_push_ack(self, event_processor_short_timeout):
        """Test waiting for specific message type."""
        processor = event_processor_short_timeout

        await processor.ack_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(sequence=1),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await processor.wait_for_message_type(
            expected_type=WebSocketMessageType.PUSH_ACK,
            timeout=1.0,
        )

        assert result is not None
        assert result.type == "push_ack"

    @pytest.mark.asyncio
    async def test_wait_for_message_type_timeout(self, event_processor_short_timeout):
        """Test timeout returns None."""
        processor = event_processor_short_timeout

        result = await processor.wait_for_message_type(
            expected_type=WebSocketMessageType.PUSH_ACK,
            timeout=0.05,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_message_type_discards_non_matching(self, event_processor_short_timeout):
        """Test that non-matching ack events are discarded."""
        processor = event_processor_short_timeout

        # Put pong then push_ack — pong should be discarded
        await processor.ack_queue.put({"type": "pong", "id": "pong-1"})
        await processor.ack_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(sequence=1),
            id="ack-1",
            timestamp=_ts(),
        ))

        result = await processor.wait_for_message_type(
            expected_type=WebSocketMessageType.PUSH_ACK,
            timeout=1.0,
        )

        assert result is not None
        assert result.type == "push_ack"

    @pytest.mark.asyncio
    async def test_wait_for_message_type_max_iterations_exceeded(self, event_processor_short_timeout):
        """Test that max_iterations limit is enforced."""
        processor = event_processor_short_timeout

        # Put 3 pong events — should exhaust iterations
        for i in range(3):
            await processor.ack_queue.put({"type": "pong", "id": f"pong-{i}"})

        result = await processor.wait_for_message_type(
            expected_type=WebSocketMessageType.PUSH_ACK,
            timeout=1.0,
            max_iterations=3,
        )

        assert result is None


class TestCheckForStuckState:
    """Tests for check_for_stuck_state method."""

    @pytest.mark.asyncio
    async def test_check_for_stuck_state_detects_stuck(self, event_processor):
        """Test stuck state detection."""
        processor = event_processor

        # Put a stuck_state error event
        await processor.state_queue.put(TranscriptErrorMessage(
            type="transcript_error",
            data=TranscriptErrorData(
                error=TranscriptErrorType.STUCK_STATE,
                message="Episode stuck",
                state="WAITING_FOR_ASSISTANT",
                duration_seconds=300,
                threshold_seconds=120,
            ),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await processor.check_for_stuck_state()

        assert result is True

        # Event should be re-queued
        assert not processor.state_queue.empty()

    @pytest.mark.asyncio
    async def test_check_for_stuck_state_no_stuck(self, event_processor):
        """Test no stuck state in queue."""
        processor = event_processor

        # Put a normal state event
        await processor.state_queue.put(StateEventMessage(
            type="is_waiting_on_assistant",
            data=StateEventData(
                version=1,
                operation=TranscriptOperation.APPEND,
                modification_count=0,
                state="WAITING_FOR_ASSISTANT"
            ),
            id="msg-1",
            timestamp=_ts(),
        ))

        result = await processor.check_for_stuck_state()

        assert result is False

        # Event should be re-queued
        assert not processor.state_queue.empty()

    @pytest.mark.asyncio
    async def test_check_for_stuck_state_empty_queue(self, event_processor):
        """Test empty queue returns False."""
        processor = event_processor

        result = await processor.check_for_stuck_state()

        assert result is False


class TestDrainQueue:
    """Tests for drain_queue method."""

    @pytest.mark.asyncio
    async def test_drain_queue_empties_queue(self, event_processor):
        """Test drain_queue removes all events from both queues."""
        processor = event_processor

        # Add state events
        for i in range(3):
            await processor.state_queue.put(StateEventMessage(
                type="is_waiting_on_assistant",
                data=StateEventData(
                    version=i,
                    operation=TranscriptOperation.APPEND,
                    modification_count=0,
                    state="WAITING_FOR_ASSISTANT"
                ),
                id=f"state-{i}",
                timestamp=_ts(),
            ))

        # Add ack events
        for i in range(2):
            await processor.ack_queue.put(PushAckMessage(
                type="push_ack",
                data=PushAckData(sequence=i),
                id=f"ack-{i}",
                timestamp=_ts(),
            ))

        assert not processor.state_queue.empty()
        assert not processor.ack_queue.empty()

        processor.drain_queue()

        assert processor.state_queue.empty()
        assert processor.ack_queue.empty()

    def test_drain_queue_safe_on_empty(self, event_processor):
        """Test drain_queue is safe on empty queue."""
        processor = event_processor

        # Should not raise
        processor.drain_queue()

        assert processor.state_queue.empty()
        assert processor.ack_queue.empty()


class TestWaitForStateEventWithRetry:
    """Tests for wait_for_state_event_with_retry method."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, event_processor_short_timeout):
        """Test retry behavior on timeout - event arrives after first timeout."""
        processor = event_processor_short_timeout

        # Schedule a state event to arrive after first timeout
        async def delayed_event():
            await asyncio.sleep(0.15)  # After first timeout (0.1s)
            await processor.state_queue.put(StateEventMessage(
                type="is_waiting_on_assistant",
                data=StateEventData(
                    version=1,
                    operation=TranscriptOperation.APPEND,
                    modification_count=0,
                    state="WAITING_FOR_ASSISTANT"
                ),
                id="msg-1",
                timestamp=_ts(),
            ))

        asyncio.create_task(delayed_event())

        # Should succeed on retry (unlimited retries until event arrives)
        result = await processor.wait_for_state_event_with_retry()

        assert result is True


class TestConfig:
    """Tests for configuration handling."""

    def test_default_config_used(self):
        """Test default config is created if not provided."""
        processor = WebSocketEventProcessor(episode_id="episode-456")

        assert processor.config is not None
        assert isinstance(processor.config, WebSocketConfig)

    def test_custom_config_used(self):
        """Test custom config is used when provided."""
        custom_config = WebSocketConfig(
            pull=PullConfig(enabled=True, _event_timeout=5.0)
        )

        processor = WebSocketEventProcessor(
            episode_id="episode-456",
            ws_config=custom_config,
        )

        assert processor.config is custom_config
        assert processor.config.pull.event_timeout == 5.0
