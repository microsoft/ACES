"""Unit tests for TranscriptCoordinator push operations (Phase 2).

Tests for operation types:
- append (default, blue team)
- rewind (red team)
- rewrite (red team)
- insert (red team)
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


@pytest.fixture
def sample_transcript():
    """Sample transcript with 3 messages."""
    return [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]


@pytest.fixture
def episode_with_transcript(sample_transcript):
    """Episode with existing transcript."""
    return Episode(
        episode_id="ep-test-123",
        task_id="task-1",
        session_id="session-1",
        state=EpisodeState.ACTIVE,
        context={
            MetadataKeys.CLIENT_TRANSCRIPT: sample_transcript.copy(),
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


class TestPushMessagesRewind:
    """Test _push_messages with rewind operation (red team)."""

    @pytest.mark.asyncio
    async def test_rewind_single_message(self, coordinator):
        """Test rewinding 1 message and appending new one."""
        coord, episode = coordinator

        new_message = {"role": "assistant", "content": "Injected response"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="rewind",
            rewind_count=1
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Should have: [system, user, new_message] (removed last assistant message)
        assert len(transcript) == 3
        assert transcript[-1] == new_message
        assert transcript[0]["role"] == "system"
        assert transcript[1]["role"] == "user"
        assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == "rewind"

    @pytest.mark.asyncio
    async def test_rewind_multiple_messages(self, coordinator):
        """Test rewinding multiple messages."""
        coord, episode = coordinator

        new_message = {"role": "user", "content": "Replacement"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="rewind",
            rewind_count=2  # Remove last 2 messages
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Should have: [system, new_message]
        assert len(transcript) == 2
        assert transcript[0]["role"] == "system"
        assert transcript[1] == new_message

    @pytest.mark.asyncio
    async def test_rewind_more_than_exists(self, coordinator):
        """Test rewinding more messages than exist (should rewind all)."""
        coord, episode = coordinator

        new_message = {"role": "system", "content": "New start"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="rewind",
            rewind_count=100  # More than 3 existing messages
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        # Should have only new message (all existing removed)
        assert len(transcript) == 1
        assert transcript[0] == new_message

    @pytest.mark.asyncio
    async def test_rewind_zero_messages(self, coordinator):
        """Test rewind with count=0 (equivalent to append)."""
        coord, episode = coordinator

        new_message = {"role": "user", "content": "Appended"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="rewind",
            rewind_count=0
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 4  # Same as append
        assert transcript[-1] == new_message


class TestPushMessagesRewrite:
    """Test _push_messages with rewrite operation (red team)."""

    @pytest.mark.asyncio
    async def test_rewrite_entire_transcript(self, coordinator):
        """Test replacing entire transcript."""
        coord, episode = coordinator

        new_transcript = [
            {"role": "system", "content": "Completely new system"},
            {"role": "user", "content": "New conversation"},
        ]

        await coord._push_messages(
            episode=episode,
            messages=new_transcript,
            operation="rewrite"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert transcript == new_transcript
        assert len(transcript) == 2
        assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == "rewrite"

    @pytest.mark.asyncio
    async def test_rewrite_with_empty_transcript(self, coordinator):
        """Test rewriting to empty transcript."""
        coord, episode = coordinator

        await coord._push_messages(
            episode=episode,
            messages=[],
            operation="rewrite"
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert transcript == []
        assert len(transcript) == 0


class TestPushMessagesInsert:
    """Test _push_messages with insert operation (red team)."""

    @pytest.mark.asyncio
    async def test_insert_at_beginning(self, coordinator):
        """Test inserting message at position 0."""
        coord, episode = coordinator

        new_message = {"role": "system", "content": "Inserted at start"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="insert",
            insert_position=0
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 4
        assert transcript[0] == new_message
        assert transcript[1]["content"] == "You are helpful"  # Original system message shifted

    @pytest.mark.asyncio
    async def test_insert_in_middle(self, coordinator):
        """Test inserting message in the middle."""
        coord, episode = coordinator

        new_message = {"role": "system", "content": "Inserted in middle"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="insert",
            insert_position=2  # After system and user
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 4
        assert transcript[2] == new_message
        assert transcript[0]["role"] == "system"
        assert transcript[1]["role"] == "user"
        assert transcript[3]["role"] == "assistant"  # Original last message shifted

    @pytest.mark.asyncio
    async def test_insert_at_end(self, coordinator):
        """Test inserting at end (equivalent to append)."""
        coord, episode = coordinator

        new_message = {"role": "user", "content": "Inserted at end"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="insert",
            insert_position=3  # At end of 3-message transcript
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 4
        assert transcript[-1] == new_message

    @pytest.mark.asyncio
    async def test_insert_beyond_end(self, coordinator):
        """Test insert position beyond transcript length (should insert at end)."""
        coord, episode = coordinator

        new_message = {"role": "user", "content": "Beyond end"}

        await coord._push_messages(
            episode=episode,
            messages=[new_message],
            operation="insert",
            insert_position=100  # Way beyond length
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 4
        assert transcript[-1] == new_message

    @pytest.mark.asyncio
    async def test_insert_multiple_messages(self, coordinator):
        """Test inserting multiple messages at once."""
        coord, episode = coordinator

        new_messages = [
            {"role": "system", "content": "Insert 1"},
            {"role": "system", "content": "Insert 2"},
        ]

        await coord._push_messages(
            episode=episode,
            messages=new_messages,
            operation="insert",
            insert_position=1
        )

        transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 5
        assert transcript[1:3] == new_messages


class TestPushMessagesVersioning:
    """Test version incrementing for all operations."""

    @pytest.mark.asyncio
    async def test_version_increments_on_all_operations(self, coordinator):
        """Test version increments for each operation type."""
        coord, episode = coordinator

        initial_version = episode.context[MetadataKeys.TRANSCRIPT_VERSION]

        # Test append
        await coord._push_messages(episode, [{"role": "user", "content": "1"}], "append")
        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == initial_version + 1

        # Test rewind
        await coord._push_messages(episode, [{"role": "user", "content": "2"}], "rewind", rewind_count=1)
        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == initial_version + 2

        # Test insert
        await coord._push_messages(episode, [{"role": "user", "content": "3"}], "insert", insert_position=0)
        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == initial_version + 3

        # Test rewrite
        await coord._push_messages(episode, [{"role": "user", "content": "4"}], "rewrite")
        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == initial_version + 4

    @pytest.mark.asyncio
    async def test_last_operation_tracked(self, coordinator):
        """Test last_operation is tracked for each operation."""
        coord, episode = coordinator

        await coord._push_messages(episode, [{"role": "user", "content": "test"}], "rewind")
        assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == "rewind"

        await coord._push_messages(episode, [{"role": "user", "content": "test"}], "insert")
        assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == "insert"

        await coord._push_messages(episode, [{"role": "user", "content": "test"}], "rewrite")
        assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == "rewrite"
