"""Tests for TranscriptSyncClient orchestrator.

Tests cover:
- Initialization and configuration
- Connection lifecycle (context manager)
- Push operations
- State event waiting
- Cleanup behavior

These tests use the MockMessageSerializer from test_protocols.py to test
the generic client without depending on any specific harness.
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from saber.client.transcript import TranscriptSyncClient
from saber.models.rest.websocket_config import PullConfig, PushConfig, WebSocketConfig
from saber.models.rest.websocket_messages import (
    PushAckData,
    PushAckMessage,
    StateEventData,
    StateEventMessage,
    TranscriptOperation,
)

from .test_protocols import MockMessage, MockMessageSerializer


def _ts() -> str:
    """Generate a test timestamp."""
    return datetime.now().isoformat()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def serializer() -> MockMessageSerializer:
    """Create a mock message serializer."""
    return MockMessageSerializer()


@pytest.fixture
def client(serializer) -> TranscriptSyncClient[MockMessage]:
    """Create a transcript sync client with default config."""
    client = TranscriptSyncClient(
        episode_id="episode-test-123",
        rest_url="http://localhost:8000",
        serializer=serializer,
    )
    yield client


@pytest.fixture
def client_push_disabled(serializer) -> TranscriptSyncClient[MockMessage]:
    """Create a transcript sync client with push disabled."""
    config = WebSocketConfig(
        push=PushConfig(enabled=False),
    )
    client = TranscriptSyncClient(
        episode_id="episode-push-disabled",
        rest_url="http://localhost:8000",
        serializer=serializer,
        ws_config=config,
    )
    yield client


@pytest.fixture
def client_pull_disabled(serializer) -> TranscriptSyncClient[MockMessage]:
    """Create a transcript sync client with pull disabled."""
    config = WebSocketConfig(
        pull=PullConfig(enabled=False),
    )
    client = TranscriptSyncClient(
        episode_id="episode-pull-disabled",
        rest_url="http://localhost:8000",
        serializer=serializer,
        ws_config=config,
    )
    yield client


@pytest.fixture
def client_short_timeout(serializer) -> TranscriptSyncClient[MockMessage]:
    """Create a client with short timeouts for testing."""
    config = WebSocketConfig(
        pull=PullConfig(enabled=True, _event_timeout=0.1),
        push=PushConfig(enabled=True, _confirmation_timeout=0.1),
    )
    client = TranscriptSyncClient(
        episode_id="episode-short-timeout",
        rest_url="http://localhost:8000",
        serializer=serializer,
        ws_config=config,
    )
    yield client


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for client initialization."""

    def test_creates_with_required_params(self, serializer):
        """Test client creation with required parameters."""
        client = TranscriptSyncClient(
            episode_id="ep-123",
            rest_url="http://localhost:8000",
            serializer=serializer,
        )

        assert client.episode_id == "ep-123"
        assert client.config is not None

    def test_creates_with_custom_config(self, serializer):
        """Test client creation with custom configuration."""
        custom_config = WebSocketConfig(
            connection_timeout=30.0,
            pull=PullConfig(enabled=True),
            push=PushConfig(enabled=False),
        )

        client = TranscriptSyncClient(
            episode_id="ep-456",
            rest_url="http://localhost:8000",
            serializer=serializer,
            ws_config=custom_config,
        )

        assert client.config is custom_config
        assert client.config.connection_timeout == 30.0
        assert client.push_enabled is False


# =============================================================================
# Property Tests
# =============================================================================


class TestProperties:
    """Tests for client properties."""

    def test_episode_id(self, client):
        """Test episode_id property."""
        assert client.episode_id == "episode-test-123"

    def test_config(self, client):
        """Test config property."""
        assert client.config is not None
        assert isinstance(client.config, WebSocketConfig)

    def test_pull_enabled(self, client, client_pull_disabled):
        """Test pull_enabled property."""
        assert client.pull_enabled is True
        assert client_pull_disabled.pull_enabled is False

    def test_push_enabled(self, client, client_push_disabled):
        """Test push_enabled property."""
        assert client.push_enabled is True
        assert client_push_disabled.push_enabled is False

    def test_local_messages(self, client):
        """Test local_messages property."""
        assert client.local_messages == []

    def test_is_connected_initially_false(self, client):
        """Test is_connected is False initially."""
        assert client.is_connected is False


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, serializer):
        """Test context manager calls cleanup on exit."""
        client = TranscriptSyncClient(
            episode_id="ep-ctx-test",
            rest_url="http://localhost:8000",
            serializer=serializer,
        )

        # Mock cleanup
        cleanup_called = False
        original_cleanup = client.cleanup

        async def mock_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
            await original_cleanup()

        client.cleanup = mock_cleanup

        # Mock ensure_connected to avoid actual connection
        client.ensure_connected = AsyncMock()

        async with client:
            pass

        assert cleanup_called


# =============================================================================
# Push Operation Tests
# =============================================================================


class TestPushMessage:
    """Tests for push_message method."""

    @pytest.mark.asyncio
    async def test_push_disabled_updates_local_only(self, client_push_disabled):
        """Test push_message with push disabled updates local state only."""
        msg = MockMessage(role="assistant", content="Hello")

        result = await client_push_disabled.push_message(msg)

        assert result is True
        assert len(client_push_disabled.local_messages) == 1

    @pytest.mark.asyncio
    async def test_push_message_sends_to_server(self, client_short_timeout):
        """Test push_message sends message to server."""
        client = client_short_timeout

        # Mock the connection
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send = AsyncMock()
        client._connection._websocket = mock_ws

        # Put push_ack in ack queue (consumed by wait_for_message_type)
        await client._events.ack_queue.put(PushAckMessage(
            type="push_ack",
            data=PushAckData(sequence=1),
            id="msg-1",
            timestamp=_ts(),
        ))

        msg = MockMessage(role="assistant", content="Hello from assistant")
        result = await client.push_message(msg)

        assert result is True
        assert mock_ws.send.called


# =============================================================================
# Sync Operation Tests
# =============================================================================


class TestWaitForStateEvent:
    """Tests for wait_for_state_event method."""

    @pytest.mark.asyncio
    async def test_wait_for_state_event_delegates_to_events(self, client_short_timeout):
        """Test wait_for_state_event delegates to event processor."""
        client = client_short_timeout

        # Put state event in queue
        await client._events.state_queue.put(StateEventMessage(
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

        result = await client.wait_for_state_event()

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_state_event_timeout(self, client_short_timeout):
        """Test wait_for_state_event returns False on timeout."""
        client = client_short_timeout

        # Empty queue - will timeout
        result = await client.wait_for_state_event()

        assert result is False


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanup:
    """Tests for cleanup method."""

    @pytest.mark.asyncio
    async def test_cleanup_closes_connection(self, serializer):
        """Test cleanup closes the WebSocket connection."""
        client = TranscriptSyncClient(
            episode_id="ep-cleanup-test",
            rest_url="http://localhost:8000",
            serializer=serializer,
        )

        # Mock the connection close
        client._connection.close = AsyncMock()

        await client.cleanup()

        client._connection.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_signals_shutdown(self, serializer):
        """Test cleanup signals shutdown to event processor."""
        client = TranscriptSyncClient(
            episode_id="ep-shutdown-test",
            rest_url="http://localhost:8000",
            serializer=serializer,
        )

        # Mock the shutdown signal
        shutdown_signaled = False
        original_signal = client._events.signal_shutdown

        def mock_signal():
            nonlocal shutdown_signaled
            shutdown_signaled = True
            original_signal()

        client._events.signal_shutdown = mock_signal
        client._connection.close = AsyncMock()

        await client.cleanup()

        assert shutdown_signaled

    @pytest.mark.asyncio
    async def test_cleanup_clears_sync_state(self, serializer):
        """Test cleanup clears sync state."""
        client = TranscriptSyncClient(
            episode_id="ep-clear-test",
            rest_url="http://localhost:8000",
            serializer=serializer,
        )

        # Set some state
        client._sync.local_messages = [MockMessage(role="user", content="Test")]

        # Mock connection close
        client._connection.close = AsyncMock()

        await client.cleanup()

        assert client._sync.local_messages == []
