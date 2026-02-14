"""Unit tests for TranscriptRepository with mocked EpisodeEventRepository.

Tests cover:
- append_message: delegates to event_repo.insert_message with correct params
- append_message: extracts role from message dict
- append_message: extracts tool_call_id from message dict
- get_transcript: returns all messages when since_sequence=0
- get_transcript: filters by since_sequence
- get_message_count: delegates to event_repo.get_transcript_count
- get_latest_sequence: delegates to event_repo.get_latest_sequence
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from saber.server.episodes.transcript.repository import (
    TranscriptRepository,
)

TEST_EPISODE_ID = "12345678-1234-1234-1234-123456789abc"
TEST_SESSION_ID = "session-001"


@pytest.fixture
def mock_event_repo() -> MagicMock:
    """Create a mock EpisodeEventRepository."""
    repo = MagicMock()
    repo.insert_message = AsyncMock(return_value=1)
    repo.get_transcript_messages = AsyncMock(return_value=[])
    repo.get_episode_events = AsyncMock(return_value=[])
    repo.get_transcript_count = AsyncMock(return_value=0)
    repo.get_latest_sequence = AsyncMock(return_value=-1)
    return repo


@pytest.fixture
def repo(mock_event_repo: MagicMock) -> TranscriptRepository:
    """Create TranscriptRepository with mocked event repo."""
    return TranscriptRepository(mock_event_repo)


@pytest.mark.asyncio
class TestAppendMessage:
    """Tests for append_message()."""

    async def test_delegates_to_event_repo(self, repo: TranscriptRepository, mock_event_repo: MagicMock) -> None:
        """append_message delegates to event_repo.insert_message."""
        mock_event_repo.insert_message = AsyncMock(return_value=5)

        seq = await repo.append_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            message={"role": "user", "content": "Hello"},
        )

        assert seq == 5
        mock_event_repo.insert_message.assert_called_once_with(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="user",
            content={"role": "user", "content": "Hello"},
            tool_call_id=None,
        )

    async def test_extracts_role(self, repo: TranscriptRepository, mock_event_repo: MagicMock) -> None:
        """append_message extracts role from message dict."""
        mock_event_repo.insert_message = AsyncMock(return_value=1)

        await repo.append_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            message={"role": "assistant", "content": "Sure"},
        )

        call_kwargs = mock_event_repo.insert_message.call_args[1]
        assert call_kwargs["role"] == "assistant"

    async def test_extracts_tool_call_id(self, repo: TranscriptRepository, mock_event_repo: MagicMock) -> None:
        """append_message extracts tool_call_id from message dict."""
        mock_event_repo.insert_message = AsyncMock(return_value=4)

        await repo.append_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            message={"role": "tool", "content": "output", "tool_call_id": "tc-42"},
        )

        call_kwargs = mock_event_repo.insert_message.call_args[1]
        assert call_kwargs["tool_call_id"] == "tc-42"

    async def test_defaults_role_to_user(self, repo: TranscriptRepository, mock_event_repo: MagicMock) -> None:
        """append_message defaults role to 'user' when not specified."""
        mock_event_repo.insert_message = AsyncMock(return_value=1)

        await repo.append_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            message={"content": "no role"},
        )

        call_kwargs = mock_event_repo.insert_message.call_args[1]
        assert call_kwargs["role"] == "user"


@pytest.mark.asyncio
class TestGetTranscript:
    """Tests for get_transcript()."""

    async def test_returns_all_messages(self, repo: TranscriptRepository, mock_event_repo: MagicMock) -> None:
        """get_transcript returns all messages when since_sequence=0."""
        mock_event_repo.get_transcript_messages = AsyncMock(return_value=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ])

        messages = await repo.get_transcript(TEST_EPISODE_ID)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    async def test_filters_by_since_sequence(self, repo: TranscriptRepository, mock_event_repo: MagicMock) -> None:
        """get_transcript filters messages after since_sequence without calling get_transcript_messages."""
        mock_event_repo.get_episode_events = AsyncMock(return_value=[
            {"sequence_num": 1, "content": {"role": "user", "content": "Hello"}},
            {"sequence_num": 2, "content": {"role": "assistant", "content": "Hi"}},
            {"sequence_num": 3, "content": {"role": "user", "content": "More"}},
        ])

        messages = await repo.get_transcript(TEST_EPISODE_ID, since_sequence=1)

        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"
        # get_transcript_messages should NOT be called when since_sequence > 0
        mock_event_repo.get_transcript_messages.assert_not_called()

    async def test_filters_skips_none_content(self, repo: TranscriptRepository, mock_event_repo: MagicMock) -> None:
        """get_transcript skips events with None content in filtered results."""
        mock_event_repo.get_episode_events = AsyncMock(return_value=[
            {"sequence_num": 2, "content": None},
            {"sequence_num": 3, "content": {"role": "user", "content": "test"}},
        ])

        messages = await repo.get_transcript(TEST_EPISODE_ID, since_sequence=1)

        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "test"}

    async def test_returns_empty(self, repo: TranscriptRepository, mock_event_repo: MagicMock) -> None:
        """get_transcript returns [] when empty."""
        mock_event_repo.get_transcript_messages = AsyncMock(return_value=[])

        messages = await repo.get_transcript(TEST_EPISODE_ID)

        assert messages == []


@pytest.mark.asyncio
class TestGetMessageCount:
    """Tests for get_message_count()."""

    async def test_delegates_to_event_repo(self, repo: TranscriptRepository, mock_event_repo: MagicMock) -> None:
        """get_message_count delegates to event_repo.get_transcript_count."""
        mock_event_repo.get_transcript_count = AsyncMock(return_value=10)

        count = await repo.get_message_count(TEST_EPISODE_ID)

        assert count == 10
        mock_event_repo.get_transcript_count.assert_called_once_with(TEST_EPISODE_ID)


@pytest.mark.asyncio
class TestGetLatestSequence:
    """Tests for get_latest_sequence()."""

    async def test_delegates_to_event_repo(self, repo: TranscriptRepository, mock_event_repo: MagicMock) -> None:
        """get_latest_sequence delegates to event_repo.get_latest_sequence."""
        mock_event_repo.get_latest_sequence = AsyncMock(return_value=42)

        seq = await repo.get_latest_sequence(TEST_EPISODE_ID)

        assert seq == 42
        mock_event_repo.get_latest_sequence.assert_called_once_with(TEST_EPISODE_ID)
