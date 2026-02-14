"""Tests for AgentTranscriptSync - manual transcript sync for custom agents.

Tests cover:
- Initialization with valid/invalid wrapper
- Push messages to server
- Differential sync (only new messages)
- Reset sync state
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser

from saber.inspect_ai.constants import InspectStoreKeys
from saber.inspect_ai.integration.agent_transcript_sync import AgentTranscriptSync
from saber.inspect_ai.integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper


class MockStore:
    """Mock TaskState store."""

    def __init__(self):
        self._store: dict = {}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def set(self, key, value):
        self._store[key] = value


class MockTaskState:
    """Mock TaskState for testing."""

    def __init__(self):
        self.store = MockStore()
        self.messages = []
        self.metadata = {}
        self.output = MagicMock()


class MockClient:
    """Mock TranscriptSyncClient."""

    def __init__(self, push_return: bool = True):
        self.push_message = AsyncMock(return_value=push_return)
        self.local_messages: list = []
        self.ensure_connected = AsyncMock()


def create_mock_wrapper(client: MockClient | None = None) -> MagicMock:
    """Create a mock that passes isinstance check for WebSocketTranscriptSyncingModelWrapper."""
    mock = MagicMock(spec=WebSocketTranscriptSyncingModelWrapper)
    mock._client = client or MockClient()
    mock._episode_id = "test-episode-123"
    mock._ensure_connected = AsyncMock()
    return mock


class TestAgentTranscriptSyncInit:
    """Tests for AgentTranscriptSync initialization."""

    def test_init_sets_state(self):
        """Test that init stores the state reference."""
        state = MockTaskState()
        sync = AgentTranscriptSync(state)

        assert sync._state is state
        assert sync._wrapper is None
        assert sync._last_synced_count == 0
        assert sync._enabled is False

    def test_is_enabled_initially_false(self):
        """Test that is_enabled is False before initialization."""
        state = MockTaskState()
        sync = AgentTranscriptSync(state)

        assert sync.is_enabled is False

    def test_synced_message_count_initially_zero(self):
        """Test that synced_message_count is 0 before any sync."""
        state = MockTaskState()
        sync = AgentTranscriptSync(state)

        assert sync.synced_message_count == 0


class TestAgentTranscriptSyncInitialize:
    """Tests for the initialize() method."""

    @pytest.mark.asyncio
    async def test_initialize_with_valid_wrapper(self):
        """Test successful initialization with valid wrapper in store."""
        state = MockTaskState()
        mock_wrapper = create_mock_wrapper()
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        result = await sync.initialize()

        assert result is True
        assert sync.is_enabled is True
        assert sync._wrapper is mock_wrapper
        mock_wrapper._client.ensure_connected.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_without_wrapper(self):
        """Test initialization fails when no wrapper in store."""
        state = MockTaskState()

        sync = AgentTranscriptSync(state)
        result = await sync.initialize()

        assert result is False
        assert sync.is_enabled is False
        assert sync._wrapper is None

    @pytest.mark.asyncio
    async def test_initialize_with_wrong_wrapper_type(self):
        """Test initialization fails when wrapper is wrong type."""
        state = MockTaskState()
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, "not a wrapper")

        sync = AgentTranscriptSync(state)
        result = await sync.initialize()

        assert result is False
        assert sync.is_enabled is False

    @pytest.mark.asyncio
    async def test_initialize_connection_error(self):
        """Test initialization fails gracefully on connection error."""
        state = MockTaskState()
        mock_wrapper = create_mock_wrapper()
        mock_wrapper._client.ensure_connected = AsyncMock(side_effect=Exception("Connection failed"))
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        result = await sync.initialize()

        assert result is False
        assert sync.is_enabled is False


class TestAgentTranscriptSyncPushMessages:
    """Tests for the push_messages() method."""

    @pytest.mark.asyncio
    async def test_push_messages_when_not_enabled(self):
        """Test push_messages returns False when not enabled."""
        state = MockTaskState()
        sync = AgentTranscriptSync(state)

        messages = [ChatMessageUser(content="Hello")]
        result = await sync.push_messages(messages)

        assert result is False

    @pytest.mark.asyncio
    async def test_push_messages_empty_list(self):
        """Test push_messages with empty list returns True."""
        state = MockTaskState()
        mock_wrapper = create_mock_wrapper()
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True

        result = await sync.push_messages([])

        assert result is True
        mock_wrapper._client.push_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_push_single_message(self):
        """Test pushing a single message."""
        state = MockTaskState()
        mock_wrapper = create_mock_wrapper()
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True

        messages = [ChatMessageUser(content="Hello")]
        result = await sync.push_messages(messages)

        assert result is True
        mock_wrapper._client.push_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_multiple_messages(self):
        """Test pushing multiple messages."""
        state = MockTaskState()
        mock_wrapper = create_mock_wrapper()
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True

        messages = [
            ChatMessageSystem(content="System prompt"),
            ChatMessageUser(content="Hello"),
            ChatMessageAssistant(content="Hi there!"),
        ]
        result = await sync.push_messages(messages)

        assert result is True
        assert mock_wrapper._client.push_message.call_count == 3

    @pytest.mark.asyncio
    async def test_push_message_failure(self):
        """Test push_messages returns False when client.push_message fails."""
        state = MockTaskState()
        mock_client = MockClient(push_return=False)
        mock_wrapper = create_mock_wrapper(client=mock_client)
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True

        messages = [ChatMessageUser(content="Hello")]
        result = await sync.push_messages(messages)

        assert result is False

    @pytest.mark.asyncio
    async def test_push_message_partial_failure(self):
        """Test push_messages returns False on partial failure."""
        state = MockTaskState()
        mock_client = MockClient()
        # First call succeeds, second fails
        mock_client.push_message = AsyncMock(side_effect=[True, False])
        mock_wrapper = create_mock_wrapper(client=mock_client)
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True

        messages = [
            ChatMessageUser(content="Hello"),
            ChatMessageAssistant(content="Response"),
        ]
        result = await sync.push_messages(messages)

        assert result is False

    @pytest.mark.asyncio
    async def test_push_message_exception_handling(self):
        """Test push_messages handles exceptions gracefully."""
        state = MockTaskState()
        mock_client = MockClient()
        mock_client.push_message = AsyncMock(side_effect=Exception("Network error"))
        mock_wrapper = create_mock_wrapper(client=mock_client)
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True

        messages = [ChatMessageUser(content="Hello")]
        result = await sync.push_messages(messages)

        assert result is False


class TestAgentTranscriptSyncSyncStateMessages:
    """Tests for the sync_state_messages() method."""

    @pytest.mark.asyncio
    async def test_sync_state_when_not_enabled(self):
        """Test sync_state_messages returns False when not enabled."""
        state = MockTaskState()
        state.messages = [ChatMessageUser(content="Hello")]

        sync = AgentTranscriptSync(state)
        result = await sync.sync_state_messages(state)

        assert result is False

    @pytest.mark.asyncio
    async def test_sync_state_no_new_messages(self):
        """Test sync_state_messages returns True when no new messages."""
        state = MockTaskState()
        mock_wrapper = create_mock_wrapper()
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True
        sync._last_synced_count = 0

        # No messages to sync
        state.messages = []
        result = await sync.sync_state_messages(state)

        assert result is True
        mock_wrapper._client.push_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_state_differential_sync(self):
        """Test that sync_state_messages only syncs new messages."""
        state = MockTaskState()
        mock_wrapper = create_mock_wrapper()
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True

        # Add initial messages
        state.messages = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="Hello"),
        ]

        # Sync - should push both messages
        result = await sync.sync_state_messages(state)
        assert result is True
        assert mock_wrapper._client.push_message.call_count == 2
        assert sync.synced_message_count == 2

        # Reset mock
        mock_wrapper._client.push_message.reset_mock()

        # Add one more message
        state.messages.append(ChatMessageAssistant(content="Response"))

        # Sync again - should only push the new message
        result = await sync.sync_state_messages(state)
        assert result is True
        assert mock_wrapper._client.push_message.call_count == 1
        assert sync.synced_message_count == 3

    @pytest.mark.asyncio
    async def test_sync_state_updates_count_on_success(self):
        """Test that last_synced_count is updated on success."""
        state = MockTaskState()
        mock_wrapper = create_mock_wrapper()
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True

        state.messages = [
            ChatMessageUser(content="Hello"),
            ChatMessageAssistant(content="Hi"),
        ]

        await sync.sync_state_messages(state)

        assert sync._last_synced_count == 2
        assert sync.synced_message_count == 2

    @pytest.mark.asyncio
    async def test_sync_state_does_not_update_count_on_failure(self):
        """Test that last_synced_count is not updated on failure."""
        state = MockTaskState()
        mock_client = MockClient(push_return=False)
        mock_wrapper = create_mock_wrapper(client=mock_client)
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True

        state.messages = [ChatMessageUser(content="Hello")]

        result = await sync.sync_state_messages(state)

        assert result is False
        assert sync._last_synced_count == 0


class TestAgentTranscriptSyncResetState:
    """Tests for the reset_sync_state() method."""

    def test_reset_sync_state(self):
        """Test that reset_sync_state resets the counter."""
        state = MockTaskState()
        sync = AgentTranscriptSync(state)
        sync._last_synced_count = 10

        sync.reset_sync_state()

        assert sync._last_synced_count == 0
        assert sync.synced_message_count == 0

    @pytest.mark.asyncio
    async def test_reset_allows_resync(self):
        """Test that reset allows re-syncing all messages."""
        state = MockTaskState()
        mock_wrapper = create_mock_wrapper()
        state.store.set(InspectStoreKeys.MODEL_WRAPPER, mock_wrapper)

        sync = AgentTranscriptSync(state)
        sync._wrapper = mock_wrapper
        sync._enabled = True

        state.messages = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="Hello"),
        ]

        # First sync
        await sync.sync_state_messages(state)
        assert mock_wrapper._client.push_message.call_count == 2

        # Reset mock
        mock_wrapper._client.push_message.reset_mock()

        # Try to sync again - should do nothing since messages already synced
        await sync.sync_state_messages(state)
        assert mock_wrapper._client.push_message.call_count == 0

        # Reset sync state
        sync.reset_sync_state()

        # Now syncing should push all messages again
        await sync.sync_state_messages(state)
        assert mock_wrapper._client.push_message.call_count == 2


class TestAgentTranscriptSyncProperties:
    """Tests for property accessors."""

    def test_is_enabled_property(self):
        """Test is_enabled property returns correct value."""
        state = MockTaskState()
        sync = AgentTranscriptSync(state)

        assert sync.is_enabled is False

        sync._enabled = True
        assert sync.is_enabled is True

    def test_synced_message_count_property(self):
        """Test synced_message_count property returns correct value."""
        state = MockTaskState()
        sync = AgentTranscriptSync(state)

        assert sync.synced_message_count == 0

        sync._last_synced_count = 5
        assert sync.synced_message_count == 5
