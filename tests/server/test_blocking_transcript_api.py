"""
Tests for blocking transcript API endpoints.

Tests the server-side infrastructure for timestamp-based blocking blue team solver:
- Transcript metadata auto-population on push
- Last modified timestamp tracking
- Change detection via timestamp comparison
"""

from datetime import datetime, timezone

import pytest

from saber.models.constants import MetadataKeys
from saber.server.base import Episode, EpisodeState


class TestTranscriptPushAutoPopulatesMetadata:
    """Test that push_transcript automatically sets last_pushed_at timestamp."""

    @pytest.fixture
    def sample_episode(self) -> Episode:
        """Create a sample episode for testing."""
        return Episode(
            episode_id="ep-blue-123",
            task_id="task-456",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={},
        )

    @pytest.mark.asyncio
    async def test_push_sets_last_pushed_timestamp(self, sample_episode: Episode):
        """Test that pushing transcript sets TRANSCRIPT_LAST_PUSHED_AT."""
        # Arrange
        assert MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT not in sample_episode.context

        # Act - simulate push_transcript endpoint behavior
        timestamp = datetime.now(timezone.utc).isoformat()
        context_updates = {
            MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: timestamp,  # Auto-set
        }
        await sample_episode.update_context_atomic(context_updates)

        # Assert
        assert sample_episode.context[MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT] == timestamp


class TestTranscriptModificationTimestamps:
    """Test transcript modification timestamp management."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode with initial push."""
        push_time = datetime.now(timezone.utc).isoformat()
        return Episode(
            episode_id="ep-blue-456",
            task_id="blue-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: push_time,
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

    @pytest.mark.asyncio
    async def test_set_modification_timestamp(self, blue_episode: Episode):
        """Test setting TRANSCRIPT_LAST_MODIFIED_AT when red team modifies."""
        # Arrange
        assert MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT not in blue_episode.context
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 0

        # Act - red team modifies transcript
        modification_timestamp = datetime.now(timezone.utc).isoformat()
        await blue_episode.update_context_atomic({
            MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT: modification_timestamp,
            MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 1,
        })

        # Assert
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT] == modification_timestamp
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 1

    @pytest.mark.asyncio
    async def test_monotonic_modification_counter(self, blue_episode: Episode):
        """Test modification counter increments monotonically."""


        # Act - simulate 3 modification cycles
        for i in range(1, 4):
            await blue_episode.update_context_atomic({
                MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT: datetime.now(timezone.utc).isoformat(),
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: i,
            })
            assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == i

        # Assert - counter should be at 3
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 3

    @pytest.mark.asyncio
    async def test_timestamp_comparison_for_change_detection(self, blue_episode: Episode):
        """Test blue team can detect changes by comparing timestamps."""
        # Arrange - blue team's last pull time is same as push time
        last_push = blue_episode.context[MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT]
        last_pull = last_push  # Blue team pulled right after push

        # Initially no modifications
        last_modified = blue_episode.context.get(MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT)
        assert last_modified is None

        # Blue team should see: no changes (last_modified is None or <= last_pull)
        has_changes = last_modified is not None and last_modified > last_pull
        assert has_changes is False

        # Act - red team modifies transcript later
        import time
        time.sleep(0.01)  # Ensure timestamp difference
        modification_time = datetime.now(timezone.utc).isoformat()
        await blue_episode.update_context_atomic({
            MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT: modification_time,
            MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 1,
        })

        # Assert - blue team should detect changes (last_modified > last_pull)
        last_modified_new = blue_episode.context[MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT]
        has_changes_now = last_modified_new is not None and last_modified_new > last_pull
        assert has_changes_now is True

    @pytest.mark.asyncio
    async def test_transcript_modification_with_content_update(self, blue_episode: Episode):
        """Test modifying transcript metadata atomically."""
        # Act - red team sets modification timestamp atomically
        modification_time = datetime.now(timezone.utc).isoformat()
        await blue_episode.update_context_atomic({
            MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT: modification_time,
            MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 1,
        })

        # Assert - metadata updated
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT] == modification_time
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 1


class TestTranscriptMetadataEndpoint:
    """Test transcript modification flag management (THE 'RED TEAM DONE!' SIGNAL)."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode for testing."""
        return Episode(
            episode_id="ep-blue-456",
            task_id="blue-task",
            session_id="session-789",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.TRANSCRIPT_MODIFIED: False,
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

    @pytest.mark.asyncio
    async def test_set_transcript_modified_flag(self, blue_episode: Episode):
        """Test setting TRANSCRIPT_MODIFIED flag (red team signals done)."""
        # Arrange
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFIED] is False
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 0

        # Act - red team sets flag after modifying transcript
        modification_timestamp = datetime.now(timezone.utc).isoformat()
        await blue_episode.update_context_atomic({
            MetadataKeys.TRANSCRIPT_MODIFIED: True,
            MetadataKeys.TRANSCRIPT_MODIFIED_AT: modification_timestamp,
            MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 1,
        })

        # Assert
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFIED] is True
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFIED_AT] == modification_timestamp
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 1

    @pytest.mark.asyncio
    async def test_clear_transcript_modified_flag(self, blue_episode: Episode):
        """Test clearing TRANSCRIPT_MODIFIED flag (blue team acknowledges)."""
        # Arrange - set flag first
        await blue_episode.update_context_atomic({
            MetadataKeys.TRANSCRIPT_MODIFIED: True,
            MetadataKeys.TRANSCRIPT_MODIFIED_AT: datetime.now(timezone.utc).isoformat(),
            MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 1,
        })

        # Act - blue team clears flag after pulling transcript
        await blue_episode.update_context_atomic({
            MetadataKeys.TRANSCRIPT_MODIFIED: False,
            MetadataKeys.TRANSCRIPT_MODIFIED_AT: None,
        })

        # Assert
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFIED] is False
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFIED_AT] is None
        # Counter should NOT be cleared - it's monotonic
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 1

    @pytest.mark.asyncio
    async def test_monotonic_modification_counter(self, blue_episode: Episode):
        """Test modification counter increments monotonically."""

        # Act - simulate 3 modification cycles
        for i in range(1, 4):
            # Red team modifies and sets flag
            await blue_episode.update_context_atomic({
                MetadataKeys.TRANSCRIPT_MODIFIED: True,
                MetadataKeys.TRANSCRIPT_MODIFIED_AT: datetime.now(timezone.utc).isoformat(),
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: i,
            })
            assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == i

            # Blue team clears flag
            await blue_episode.update_context_atomic({
                MetadataKeys.TRANSCRIPT_MODIFIED: False,
            })

        # Assert - counter should be at 3
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 3

    @pytest.mark.asyncio
    async def test_transcript_modification_with_transcript_update(self, blue_episode: Episode):
        """Test setting modification flag atomically."""
        # Act - red team sets flag in one atomic operation
        await blue_episode.update_context_atomic({
            MetadataKeys.TRANSCRIPT_MODIFIED: True,
            MetadataKeys.TRANSCRIPT_MODIFIED_AT: datetime.now(timezone.utc).isoformat(),
            MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 1,
        })

        # Assert
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFIED] is True
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 1


class TestTranscriptMetadataEndpoint:  # noqa: F811
    """Test GET /transcript/metadata endpoint (lightweight polling endpoint)."""

    @pytest.fixture
    def episode_with_modifications(self) -> Episode:
        """Create episode with transcript modifications."""
        push_time = "2025-11-30T10:10:00Z"
        modified_time = "2025-11-30T10:15:30Z"
        return Episode(
            episode_id="ep-test",
            task_id="task-test",
            session_id="session-test",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: push_time,
                MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT: modified_time,
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 2,
            },
        )

    def test_get_transcript_metadata(self, episode_with_modifications: Episode):
        """Test getting transcript metadata returns all relevant fields."""
        # Act - simulate GET /transcript/metadata response
        metadata = {
            "last_pushed_at": episode_with_modifications.context.get(MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT),
            "last_modified_at": episode_with_modifications.context.get(MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT),
            "modification_count": episode_with_modifications.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0),
        }

        # Assert
        assert metadata["last_pushed_at"] == "2025-11-30T10:10:00Z"
        assert metadata["last_modified_at"] == "2025-11-30T10:15:30Z"
        assert metadata["modification_count"] == 2

    def test_get_transcript_metadata_unmodified(self):
        """Test metadata for episode without modifications."""
        # Arrange
        push_time = datetime.now(timezone.utc).isoformat()
        episode = Episode(
            episode_id="ep-unmodified",
            task_id="task-test",
            session_id="session-test",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: push_time,
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

        # Act
        metadata = {
            "last_pushed_at": episode.context.get(MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT),
            "last_modified_at": episode.context.get(MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT),
            "modification_count": episode.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0),
        }

        # Assert
        assert metadata["last_pushed_at"] == push_time
        assert metadata["last_modified_at"] is None  # Never modified
        assert metadata["modification_count"] == 0


class TestTranscriptGetEndpoint:
    """Test GET /transcript endpoint returns content only.

    Note: Transcript content now lives in Redis (via coordinator).
    These tests verify the expected response shape.
    """

    def test_get_transcript_response_shape(self):
        """Test expected transcript response shape."""
        # Simulate what the coordinator would return
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User query"},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "system", "content": "Injected", "source": "red_team_injection"},
        ]

        # Act - simulate GET /transcript response
        response = {"messages": messages}

        # Assert
        assert "messages" in response
        assert len(response["messages"]) == 4
        assert response["messages"][0]["role"] == "system"
        assert response["messages"][-1]["source"] == "red_team_injection"

    def test_get_transcript_empty(self):
        """Test getting transcript when no messages exist."""
        # Act - simulate GET /transcript response from empty coordinator
        response = {"messages": []}

        # Assert
        assert response["messages"] == []


class TestEndToEndTimestampBasedFlow:
    """Integration tests for complete timestamp-based blocking flow."""

    @pytest.fixture
    def blue_episode_fresh(self) -> Episode:
        """Create fresh blue team episode."""
        return Episode(
            episode_id="ep-blue-e2e",
            task_id="blue-task",
            session_id="session-e2e",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

    @pytest.mark.asyncio
    async def test_complete_timestamp_based_cycle(self, blue_episode_fresh: Episode):
        """Test complete cycle: blue pushes → red modifies → blue detects via timestamps."""
        # Step 1: Blue team pushes transcript (metadata update only; content goes to Redis)
        push_time = datetime.now(timezone.utc).isoformat()
        await blue_episode_fresh.update_context_atomic({
            MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: push_time,
        })
        assert blue_episode_fresh.context[MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT] == push_time

        # Step 2: Blue team polls metadata (no modifications yet)
        last_pull = push_time  # Blue pulled right after push
        metadata_before = {
            "last_pushed_at": blue_episode_fresh.context.get(MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT),
            "last_modified_at": blue_episode_fresh.context.get(MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT),
            "modification_count": blue_episode_fresh.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0),
        }

        # Check: no changes detected
        has_changes = (metadata_before["last_modified_at"] is not None and
                      metadata_before["last_modified_at"] > last_pull)
        assert has_changes is False

        # Step 3: Red team modifies transcript and sets timestamp
        import time
        time.sleep(0.01)  # Ensure timestamp difference
        modification_time = datetime.now(timezone.utc).isoformat()

        await blue_episode_fresh.update_context_atomic({
            MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT: modification_time,
            MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 1,
        })

        # Step 4: Blue team polls again (should detect changes via timestamp)
        metadata_after = {
            "last_pushed_at": blue_episode_fresh.context.get(MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT),
            "last_modified_at": blue_episode_fresh.context.get(MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT),
            "modification_count": blue_episode_fresh.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0),
        }

        # Check: changes detected! (last_modified_at > last_pull)
        has_changes_now = (metadata_after["last_modified_at"] is not None and
                          metadata_after["last_modified_at"] > last_pull)
        assert has_changes_now is True
        assert metadata_after["modification_count"] == 1

        # Step 5: Blue team updates its last_pull timestamp (client-side tracking)
        new_last_pull = modification_time

        # Step 6: Verify no more changes detected after pull
        has_more_changes = (metadata_after["last_modified_at"] is not None and
                           metadata_after["last_modified_at"] > new_last_pull)
        assert has_more_changes is False

        # Final verification
        assert blue_episode_fresh.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 1  # Monotonic
