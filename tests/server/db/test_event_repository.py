"""Behavioral tests for EpisodeEventRepository backed by fakeredis.

Tests cover:
- insert_message: returns sequence, stores correct data, handles tool_call_id
- insert_action_start: returns sequence, stores tool_name and parameters
- insert_action_end: returns sequence, stores result/metadata, increments step count
- get_episode_events: ordered by sequence, filters by event_type, empty for unknown
- get_transcript_messages: returns message content only, empty when none
- get_transcript_count: returns message count, zero when empty
- get_step_count: returns action_end count, zero when empty
- get_latest_sequence: returns latest, -1 when empty
- delete_episode_events: returns count and removes data, zero for empty
- concurrent inserts get unique sequences
"""

import asyncio

import fakeredis.aioredis
import pytest

from saber.server.db.event_repository import EpisodeEventRepository

TEST_EPISODE_ID = "12345678-1234-1234-1234-123456789abc"
TEST_SESSION_ID = "session-001"


@pytest.fixture
async def redis_client() -> fakeredis.aioredis.FakeRedis:
    """Create a fresh fakeredis client per test."""
    client = fakeredis.aioredis.FakeRedis()
    yield client  # type: ignore[misc]
    await client.aclose()


@pytest.fixture
def repo(redis_client: fakeredis.aioredis.FakeRedis) -> EpisodeEventRepository:
    """Create repository with fakeredis client."""
    return EpisodeEventRepository(redis_client)


@pytest.mark.asyncio
class TestInsertMessage:
    """Tests for insert_message()."""

    async def test_insert_returns_sequence_1_for_first(self, repo: EpisodeEventRepository) -> None:
        """First insert returns sequence 1."""
        seq = await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="user",
            content={"content": "Hello"},
        )
        assert seq == 1

    async def test_insert_returns_incrementing_sequences(self, repo: EpisodeEventRepository) -> None:
        """Successive inserts return incrementing sequences."""
        seq1 = await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="user",
            content={"content": "First"},
        )
        seq2 = await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="assistant",
            content={"content": "Second"},
        )
        assert seq1 == 1
        assert seq2 == 2

    async def test_insert_stores_correct_data(self, repo: EpisodeEventRepository) -> None:
        """Inserted message is retrievable via get_episode_events."""
        await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="user",
            content={"content": "Hello"},
        )

        events = await repo.get_episode_events(TEST_EPISODE_ID)
        assert len(events) == 1
        assert events[0]["event_type"] == "message"
        assert events[0]["role"] == "user"
        assert events[0]["content"] == {"content": "Hello"}
        assert events[0]["session_id"] == TEST_SESSION_ID

    async def test_insert_with_tool_call_id(self, repo: EpisodeEventRepository) -> None:
        """Message with tool_call_id stores it correctly."""
        await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="tool",
            content={"content": "result"},
            tool_call_id="tc-42",
        )

        events = await repo.get_episode_events(TEST_EPISODE_ID)
        assert events[0]["tool_call_id"] == "tc-42"


@pytest.mark.asyncio
class TestInsertActionStart:
    """Tests for insert_action_start()."""

    async def test_insert_returns_sequence(self, repo: EpisodeEventRepository) -> None:
        """insert_action_start returns a sequence number."""
        seq = await repo.insert_action_start(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            tool_name="bash",
            parameters={"command": "ls"},
        )
        assert seq == 1

    async def test_insert_stores_tool_name_and_parameters(self, repo: EpisodeEventRepository) -> None:
        """action_start stores tool_name and parameters in content."""
        await repo.insert_action_start(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            tool_name="python",
            parameters={"code": "print(1)"},
        )

        events = await repo.get_episode_events(TEST_EPISODE_ID)
        assert events[0]["event_type"] == "action_start"
        assert events[0]["content"] == {
            "tool_name": "python",
            "parameters": {"code": "print(1)"},
        }


@pytest.mark.asyncio
class TestInsertActionEnd:
    """Tests for insert_action_end()."""

    async def test_insert_returns_sequence(self, repo: EpisodeEventRepository) -> None:
        """insert_action_end returns a sequence number."""
        seq = await repo.insert_action_end(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            result={"stdout": "ok", "exit_code": 0},
            execution_time_ms=150,
            success=True,
        )
        assert seq == 1

    async def test_insert_stores_result_and_execution_metadata(self, repo: EpisodeEventRepository) -> None:
        """action_end stores result, execution_time_ms, and success."""
        await repo.insert_action_end(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            result={"error": "timeout"},
            execution_time_ms=30000,
            success=False,
        )

        events = await repo.get_episode_events(TEST_EPISODE_ID)
        assert events[0]["event_type"] == "action_end"
        assert events[0]["content"] == {"error": "timeout"}
        assert events[0]["execution_time_ms"] == 30000
        assert events[0]["success"] is False

    async def test_insert_increments_step_count(self, repo: EpisodeEventRepository) -> None:
        """Each action_end increments the step counter."""
        await repo.insert_action_end(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            result={"stdout": "a"},
            execution_time_ms=10,
            success=True,
        )
        await repo.insert_action_end(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-2",
            result={"stdout": "b"},
            execution_time_ms=20,
            success=True,
        )
        assert await repo.get_step_count(TEST_EPISODE_ID) == 2


@pytest.mark.asyncio
class TestGetEpisodeEvents:
    """Tests for get_episode_events()."""

    async def test_returns_events_ordered_by_sequence(self, repo: EpisodeEventRepository) -> None:
        """Events are returned in insertion (sequence) order."""
        await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="user",
            content={"content": "first"},
        )
        await repo.insert_action_start(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            tool_name="bash",
            parameters={"command": "ls"},
        )
        await repo.insert_action_end(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            result={"stdout": "ok"},
            execution_time_ms=50,
            success=True,
        )

        events = await repo.get_episode_events(TEST_EPISODE_ID)
        assert len(events) == 3
        assert events[0]["sequence_num"] == 1
        assert events[1]["sequence_num"] == 2
        assert events[2]["sequence_num"] == 3

    async def test_filters_by_event_type(self, repo: EpisodeEventRepository) -> None:
        """Filtering by event_type returns only matching events."""
        await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="user",
            content={"content": "hello"},
        )
        await repo.insert_action_start(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            tool_name="bash",
            parameters={"command": "ls"},
        )

        events = await repo.get_episode_events(TEST_EPISODE_ID, event_types=["message"])
        assert len(events) == 1
        assert events[0]["event_type"] == "message"

    async def test_returns_empty_for_unknown_episode(self, repo: EpisodeEventRepository) -> None:
        """Unknown episode returns empty list."""
        events = await repo.get_episode_events("nonexistent-episode-id")
        assert events == []


@pytest.mark.asyncio
class TestGetTranscriptMessages:
    """Tests for get_transcript_messages()."""

    async def test_returns_message_content_only(self, repo: EpisodeEventRepository) -> None:
        """Returns only message content dicts, excluding action events."""
        await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="user",
            content={"content": "Hi"},
        )
        await repo.insert_action_start(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            tool_name="bash",
            parameters={"command": "ls"},
        )

        messages = await repo.get_transcript_messages(TEST_EPISODE_ID)
        assert len(messages) == 1
        assert messages[0] == {"content": "Hi"}

    async def test_returns_empty_for_no_messages(self, repo: EpisodeEventRepository) -> None:
        """Returns empty list when no messages exist."""
        messages = await repo.get_transcript_messages("nonexistent")
        assert messages == []


@pytest.mark.asyncio
class TestGetTranscriptCount:
    """Tests for get_transcript_count()."""

    async def test_returns_message_count(self, repo: EpisodeEventRepository) -> None:
        """Returns count of messages inserted."""
        await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="user",
            content={"content": "a"},
        )
        await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="assistant",
            content={"content": "b"},
        )
        assert await repo.get_transcript_count(TEST_EPISODE_ID) == 2

    async def test_returns_zero_for_empty(self, repo: EpisodeEventRepository) -> None:
        """Returns 0 for episode with no messages."""
        assert await repo.get_transcript_count("nonexistent") == 0


@pytest.mark.asyncio
class TestGetStepCount:
    """Tests for get_step_count()."""

    async def test_returns_action_end_count(self, repo: EpisodeEventRepository) -> None:
        """Returns count of action_end events."""
        await repo.insert_action_end(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            result={"stdout": "ok"},
            execution_time_ms=10,
            success=True,
        )
        assert await repo.get_step_count(TEST_EPISODE_ID) == 1

    async def test_returns_zero_for_empty(self, repo: EpisodeEventRepository) -> None:
        """Returns 0 for episode with no actions."""
        assert await repo.get_step_count("nonexistent") == 0


@pytest.mark.asyncio
class TestGetLatestSequence:
    """Tests for get_latest_sequence()."""

    async def test_returns_latest_sequence(self, repo: EpisodeEventRepository) -> None:
        """Returns the current sequence counter value."""
        await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="user",
            content={"content": "a"},
        )
        await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="assistant",
            content={"content": "b"},
        )
        assert await repo.get_latest_sequence(TEST_EPISODE_ID) == 2

    async def test_returns_negative_one_for_empty(self, repo: EpisodeEventRepository) -> None:
        """Returns -1 for episode with no events."""
        assert await repo.get_latest_sequence("nonexistent") == -1


@pytest.mark.asyncio
class TestDeleteEpisodeEvents:
    """Tests for delete_episode_events()."""

    async def test_returns_count_and_removes_all_data(self, repo: EpisodeEventRepository) -> None:
        """delete_episode_events returns event count and clears all keys."""
        await repo.insert_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            role="user",
            content={"content": "a"},
        )
        await repo.insert_action_end(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            tool_call_id="tc-1",
            result={"stdout": "ok"},
            execution_time_ms=10,
            success=True,
        )

        count = await repo.delete_episode_events(TEST_EPISODE_ID)
        assert count == 2

        # Verify all data is gone
        assert await repo.get_episode_events(TEST_EPISODE_ID) == []
        assert await repo.get_latest_sequence(TEST_EPISODE_ID) == -1
        assert await repo.get_transcript_count(TEST_EPISODE_ID) == 0
        assert await repo.get_step_count(TEST_EPISODE_ID) == 0

    async def test_returns_zero_for_empty_episode(self, repo: EpisodeEventRepository) -> None:
        """Returns 0 when no events to delete."""
        count = await repo.delete_episode_events("nonexistent")
        assert count == 0


@pytest.mark.asyncio
class TestSequenceAtomicity:
    """Tests for sequence number atomicity under concurrency."""

    async def test_concurrent_inserts_get_unique_sequences(self, repo: EpisodeEventRepository) -> None:
        """Concurrent inserts all receive unique sequence numbers."""

        async def _insert(i: int) -> int:
            return await repo.insert_message(
                episode_id=TEST_EPISODE_ID,
                session_id=TEST_SESSION_ID,
                role="user",
                content={"content": f"msg-{i}"},
            )

        seqs = await asyncio.gather(*[_insert(i) for i in range(10)])
        assert len(set(seqs)) == 10  # all unique
        assert sorted(seqs) == list(range(1, 11))
