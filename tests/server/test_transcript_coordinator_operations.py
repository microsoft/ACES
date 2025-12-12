"""Unit tests for TranscriptCoordinator push operations.

Tests for operation types:
- append (default, blue team)
- restart (red team - reset to initial transcript then append)
"""

import pytest
from saber.models.constants import MetadataKeys
from saber.models.transcript import TranscriptSyncRequest
from saber.server.base import Episode, EpisodeState
from saber.server.episodes.transcript_coordinator import TranscriptCoordinator


class MockConnectionManager:
    """Mock ConnectionManager for testing."""

    def __init__(self):
        self.broadcasts = []

    async def broadcast_to_episode(self, episode_id: str, message: dict):
        """Record broadcasts for verification."""
        self.broadcasts.append({"episode_id": episode_id, "message": message})

    async def cleanup_episode(self, episode_id: str):
        """Mock cleanup."""
        pass


class MockEpisodeManager:
    """Mock EpisodeManager for testing."""

    def __init__(self):
        self.episodes = {}

    def get_episode_by_id(self, episode_id: str):
        return self.episodes.get(episode_id)

    def start_episode(self, session_id: str, task_id: str, **kwargs):
        """Mock start_episode to support lifecycle hooks."""
        episode_id = f"{session_id}-{task_id}"
        episode = Episode(
            episode_id=episode_id,
            session_id=session_id,
            task_id=task_id,
        )
        self.episodes[episode_id] = episode
        return episode

    def end_episode(self, episode_id: str, **kwargs):
        """Mock end_episode."""
        if episode_id in self.episodes:
            del self.episodes[episode_id]


@pytest.fixture
def sample_transcript():
    """Sample transcript with 3 messages (system->user->assistant)."""
    return [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]


@pytest.fixture
def initial_transcript():
    """Initial transcript state captured after first assistant response."""
    return [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]


@pytest.fixture
def episode_with_transcript(sample_transcript, initial_transcript):
    """Episode with existing transcript and initial transcript."""
    return Episode(
        episode_id="ep-test-123",
        task_id="task-1",
        session_id="session-1",
        state=EpisodeState.ACTIVE,
        context={
            MetadataKeys.CLIENT_TRANSCRIPT: sample_transcript.copy(),
            MetadataKeys.INITIAL_TRANSCRIPT: initial_transcript.copy(),
            MetadataKeys.TRANSCRIPT_VERSION: 3,
            MetadataKeys.TRANSCRIPT_LAST_OPERATION: "append",
        },
    )


@pytest.fixture
def coordinator(episode_with_transcript):
    """Coordinator with test episode."""
    episode_manager = MockEpisodeManager()
    episode_manager.episodes[episode_with_transcript.episode_id] = episode_with_transcript

    connection_manager = MockConnectionManager()
    coordinator = TranscriptCoordinator(episode_manager, connection_manager)

    return coordinator, episode_with_transcript


class TestPushMessagesAppend:
    """Test _push_messages with append operation (default blue team behavior)."""

    @pytest.mark.asyncio
    async def test_append_single_message(self, coordinator):
        """Test appending a single message to transcript."""
        coord, episode = coordinator

        new_message = {"role": "user", "content": "New message"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="append"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 4  # 3 original + 1 new
        assert transcript[-1] == new_message
        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == 4
        assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == "append"

    @pytest.mark.asyncio
    async def test_append_multiple_messages(self, coordinator):
        """Test appending multiple messages."""
        coord, episode = coordinator

        new_messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Reply 1"},
        ]

        await coord._push_messages(
            episode=episode,
            messages=new_messages,
            operation="append"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 5  # 3 original + 2 new
        assert transcript[-2:] == new_messages
        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == 4

    @pytest.mark.asyncio
    async def test_append_default_operation(self, coordinator):
        """Test append is default operation when not specified."""
        coord, episode = coordinator

        new_message = {"role": "user", "content": "Default append"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message]
            # operation not specified - should default to "append"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 4
        assert transcript[-1] == new_message

    @pytest.mark.asyncio
    async def test_append_empty_messages(self, coordinator):
        """Test appending empty list doesn't change transcript."""
        coord, episode = coordinator

        original_len = len(episode.context[MetadataKeys.CLIENT_TRANSCRIPT])
        original_version = episode.context[MetadataKeys.TRANSCRIPT_VERSION]

        await coord._push_messages(
            episode=episode,
            messages=[],
            operation="append"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == original_len
        # Version still increments even for empty push
        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == original_version + 1


class TestPushMessagesRestart:
    """Test _push_messages with restart operation (red team).

    Restart resets the transcript to the initial state (system->user->assistant)
    that was captured after the blue team's first response, then appends the
    new message. This allows the red team to start fresh conversations while
    preserving the blue team's opening response.
    """

    @pytest.mark.asyncio
    async def test_restart_resets_to_initial_and_appends(self, coordinator):
        """Test restart resets to INITIAL_TRANSCRIPT and appends new message."""
        coord, episode = coordinator

        # Add some extra conversation after initial
        episode.context[MetadataKeys.CLIENT_TRANSCRIPT].extend([
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I don't have weather data."},
        ])

        new_message = {"role": "user", "content": "Fresh start attack"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="restart"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Should have: initial transcript (3 messages) + new message
        assert len(transcript) == 4
        assert transcript[0]["role"] == "system"
        assert transcript[0]["content"] == "You are helpful"
        assert transcript[1]["role"] == "user"
        assert transcript[1]["content"] == "Hello"
        assert transcript[2]["role"] == "assistant"
        assert transcript[2]["content"] == "Hi! How can I help?"  # Blue's first response preserved!
        assert transcript[3] == new_message
        assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == "restart"

    @pytest.mark.asyncio
    async def test_restart_preserves_blue_team_opening_response(self, coordinator):
        """Test restart preserves the blue team's first assistant response."""
        coord, episode = coordinator

        # Simulate extended conversation
        episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Tell me about security"},
            {"role": "assistant", "content": "Security is important..."},
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "content": "...and you should use encryption."},
        ]

        # Red team does restart
        await coord._push_messages(
            episode=episode,
            messages=[{"role": "user", "content": "New attack vector"}],
            operation="restart"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Initial transcript (3) + new attack message (1)
        assert len(transcript) == 4
        # Blue team's first response is preserved
        assert transcript[2]["content"] == "Hi! How can I help?"

    @pytest.mark.asyncio
    async def test_restart_without_initial_transcript_falls_back_to_append(self, coordinator):
        """Test restart falls back to append if no INITIAL_TRANSCRIPT exists."""
        coord, episode = coordinator

        # Remove INITIAL_TRANSCRIPT
        del episode.context[MetadataKeys.INITIAL_TRANSCRIPT]

        original_transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT].copy()
        new_message = {"role": "user", "content": "Fallback append"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="restart"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Should fall back to append behavior
        assert len(transcript) == len(original_transcript) + 1
        assert transcript[-1] == new_message

    @pytest.mark.asyncio
    async def test_restart_with_empty_initial_transcript_falls_back(self, coordinator):
        """Test restart with empty INITIAL_TRANSCRIPT falls back to append."""
        coord, episode = coordinator

        # Set empty INITIAL_TRANSCRIPT
        episode.context[MetadataKeys.INITIAL_TRANSCRIPT] = []

        original_len = len(episode.context[MetadataKeys.CLIENT_TRANSCRIPT])
        new_message = {"role": "user", "content": "Empty fallback"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="restart"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Falls back to append
        assert len(transcript) == original_len + 1

    @pytest.mark.asyncio
    async def test_multiple_restarts_always_use_same_initial(self, coordinator):
        """Test multiple restarts always reset to the same INITIAL_TRANSCRIPT."""
        coord, episode = coordinator

        initial = episode.context[MetadataKeys.INITIAL_TRANSCRIPT].copy()

        # First restart
        await coord._push_messages(
            episode=episode,
            messages=[{"role": "user", "content": "Attack 1"}],
            operation="restart"
        )

        # Simulate some conversation after restart
        await coord._push_messages(
            episode=episode,
            messages=[
                {"role": "assistant", "content": "Response to attack 1"},
                {"role": "user", "content": "Follow up"},
            ],
            operation="append"
        )

        # Second restart
        await coord._push_messages(
            episode=episode,
            messages=[{"role": "user", "content": "Attack 2"}],
            operation="restart"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Should reset to original initial + new attack
        assert len(transcript) == len(initial) + 1
        assert transcript[:3] == initial  # Original initial preserved
        assert transcript[3]["content"] == "Attack 2"


class TestPushMessagesVersioning:
    """Test version incrementing for all operations."""

    @pytest.mark.asyncio
    async def test_version_increments_on_append(self, coordinator):
        """Test version increments for append."""
        coord, episode = coordinator

        initial_version = episode.context[MetadataKeys.TRANSCRIPT_VERSION]

        await coord._push_messages(
            episode=episode,
            messages=[{"role": "user", "content": "Test"}],
            operation="append"
        )

        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == initial_version + 1

    @pytest.mark.asyncio
    async def test_version_increments_on_restart(self, coordinator):
        """Test version increments for restart."""
        coord, episode = coordinator

        initial_version = episode.context[MetadataKeys.TRANSCRIPT_VERSION]

        await coord._push_messages(
            episode=episode,
            messages=[{"role": "user", "content": "Test"}],
            operation="restart"
        )

        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == initial_version + 1

    @pytest.mark.asyncio
    async def test_last_operation_tracked(self, coordinator):
        """Test last_operation is tracked for each operation."""
        coord, episode = coordinator

        await coord._push_messages(
            episode=episode,
            messages=[{"role": "user", "content": "test"}],
            operation="append"
        )
        assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == "append"

        await coord._push_messages(
            episode=episode,
            messages=[{"role": "user", "content": "test"}],
            operation="restart"
        )
        assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == "restart"


class TestUnknownOperationFallback:
    """Test handling of unknown operations."""

    @pytest.mark.asyncio
    async def test_unknown_operation_falls_back_to_append(self, coordinator):
        """Test unknown operation falls back to append behavior."""
        coord, episode = coordinator

        original_len = len(episode.context[MetadataKeys.CLIENT_TRANSCRIPT])
        new_message = {"role": "user", "content": "Unknown op"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="some_unknown_operation"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Should fall back to append
        assert len(transcript) == original_len + 1
        assert transcript[-1] == new_message
