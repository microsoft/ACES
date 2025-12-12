"""
Integration tests for blocking transcript REST API endpoints.

Tests the complete REST API flow for timestamp-based blocking blue team solver.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import time

from saber.models.constants import MetadataKeys
from saber.server.base import Episode, EpisodeState
from saber.server.session_manager import SessionManager
from saber.server.api.session_rest_api import SessionRestAPI


@pytest.fixture
def mock_session_manager(mocker):
    """Create a mock SessionManager for testing."""
    mock_manager = mocker.MagicMock(spec=SessionManager)
    mock_manager.domain_name = "test_domain"
    return mock_manager


@pytest.fixture
def rest_api(mock_session_manager):
    """Create SessionRestAPI instance with mock manager."""
    api = SessionRestAPI(session_manager=mock_session_manager, host="localhost", port=8000)
    return api


@pytest.fixture
def test_client(rest_api):
    """Create FastAPI test client."""
    return TestClient(rest_api.app)


@pytest.fixture
def sample_episode():
    """Create a sample blue team episode."""
    push_time = datetime.utcnow().isoformat()
    return Episode(
        episode_id="ep-blue-rest-test",
        task_id="task-123",
        session_id="session-456",
        state=EpisodeState.ACTIVE,
        context={
            MetadataKeys.CLIENT_TRANSCRIPT: [
                {"role": "system", "content": "You are helpful..."},
                {"role": "assistant", "content": "Hello!"},
            ],
            MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: push_time,
            MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
        },
    )


class TestTranscriptMetadataEndpoint:
    """Test GET /transcript/metadata endpoint."""

    def test_get_metadata_unmodified(self, test_client, mock_session_manager, sample_episode):
        """Test GET /transcript/metadata returns correct timestamps for unmodified transcript."""
        # Arrange
        mock_session_manager.get_episode_by_id.return_value = sample_episode

        # Act
        response = test_client.get(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript/metadata"
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["last_pushed_at"] == sample_episode.context[MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT]
        assert data["last_modified_at"] is None  # Never modified
        assert data["modification_count"] == 0
        assert data["message_count"] == 2

    def test_get_metadata_modified(self, test_client, mock_session_manager, sample_episode):
        """Test GET /transcript/metadata returns correct timestamps after modification."""
        # Arrange - set modification timestamp
        modification_time = datetime.utcnow().isoformat()
        sample_episode.context[MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT] = modification_time
        sample_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] = 3
        mock_session_manager.get_episode_by_id.return_value = sample_episode

        # Act
        response = test_client.get(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript/metadata"
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["last_modified_at"] == modification_time
        assert data["modification_count"] == 3

    def test_get_metadata_episode_not_found(self, test_client, mock_session_manager):
        """Test GET /transcript/metadata returns 404 for non-existent episode."""
        # Arrange
        mock_session_manager.get_episode_by_id.return_value = None

        # Act
        response = test_client.get("/api/v1/session/session-123/episodes/ep-nonexistent/transcript/metadata")

        # Assert
        assert response.status_code == 404


class TestEndToEndTimestampBasedFlowREST:
    """Integration test for complete timestamp-based blocking flow via REST API."""

    def test_complete_timestamp_based_cycle_via_rest(self, test_client, mock_session_manager, sample_episode):
        """Test complete cycle: blue pushes → metadata shows push time → red modifies → metadata shows modification time."""
        mock_session_manager.get_episode_by_id.return_value = sample_episode

        # Step 1: Blue team's initial push already happened (in fixture)
        # Verify metadata shows last_pushed_at
        response = test_client.get(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript/metadata"
        )
        assert response.status_code == 200
        data = response.json()
        last_push = data["last_pushed_at"]
        assert last_push is not None
        assert data["last_modified_at"] is None  # No modifications yet

        # Step 2: Blue team polls metadata (client-side tracking of last_pull)
        last_pull = last_push  # Blue pulled right after push

        # Check: no changes (last_modified_at is None)
        has_changes = data["last_modified_at"] is not None and data["last_modified_at"] > last_pull
        assert has_changes is False

        # Step 3: Simulate red team modifying transcript and setting timestamp
        time.sleep(0.01)  # Ensure timestamp difference
        injected_message = {"role": "system", "content": "Malicious", "source": "red_team"}
        sample_episode.context[MetadataKeys.CLIENT_TRANSCRIPT].append(injected_message)
        modification_time = datetime.utcnow().isoformat()
        sample_episode.context[MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT] = modification_time
        sample_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] = 1

        # Step 4: Blue team polls again (should detect changes via timestamp)
        response = test_client.get(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript/metadata"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["last_modified_at"] == modification_time
        assert data["modification_count"] == 1
        assert data["message_count"] == 3  # Original 2 + injected 1

        # Check: changes detected! (last_modified_at > last_pull)
        has_changes_now = data["last_modified_at"] is not None and data["last_modified_at"] > last_pull
        assert has_changes_now is True

        # Step 5: Verify transcript content directly from episode context
        # (GET /transcript endpoint was removed - transcript retrieval now uses WebSocket sync_request)
        transcript = sample_episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(transcript) == 3
        assert transcript[-1]["content"] == "Malicious"

        # Step 6: Blue team updates client-side last_pull timestamp
        new_last_pull = modification_time

        # Step 7: Verify no more changes after pull
        has_more_changes = data["last_modified_at"] is not None and data["last_modified_at"] > new_last_pull
        assert has_more_changes is False

        # Final verification - modification count is monotonic
        assert sample_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 1


class TestPollingPerformance:
    """Test that metadata endpoint is lightweight for polling."""

    def test_metadata_endpoint_does_not_return_full_transcript(self, test_client, mock_session_manager):
        """Verify metadata endpoint only returns timestamps/counts, not full messages."""
        # Arrange - create episode with large transcript
        large_transcript = [{"role": "user", "content": f"Message {i}"} for i in range(1000)]
        push_time = datetime.utcnow().isoformat()
        episode = Episode(
            episode_id="ep-large",
            task_id="task-large",
            session_id="session-large",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: large_transcript,
                MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: push_time,
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )
        mock_session_manager.get_episode_by_id.return_value = episode

        # Act
        response = test_client.get(
            f"/api/v1/session/{episode.session_id}/episodes/{episode.episode_id}/transcript/metadata"
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should only have metadata fields, not full messages
        assert "message_count" in data
        assert "messages" not in data
        assert data["message_count"] == 1000

        # Response should be small (< 1KB for metadata only)
        import json
        response_size = len(json.dumps(data).encode('utf-8'))
        assert response_size < 1024  # Less than 1KB


class TestTimestampComparison:
    """Test timestamp comparison logic for change detection."""

    def test_change_detection_via_timestamp_comparison(self, test_client, mock_session_manager):
        """Test that blue team can detect changes by comparing timestamps."""
        # Arrange - episode with push but no modifications
        push_time = "2025-12-01T10:00:00.000000"
        episode = Episode(
            episode_id="ep-test",
            task_id="task-test",
            session_id="session-test",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [{"role": "system", "content": "Test"}],
                MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: push_time,
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )
        mock_session_manager.get_episode_by_id.return_value = episode

        # Act - get metadata
        response = test_client.get(
            f"/api/v1/session/{episode.session_id}/episodes/{episode.episode_id}/transcript/metadata"
        )
        data = response.json()

        # Blue team logic: compare last_modified_at with last_pull
        last_pull = push_time
        has_changes = data["last_modified_at"] is not None and data["last_modified_at"] > last_pull
        assert has_changes is False  # No modifications yet

        # Simulate modification
        modification_time = "2025-12-01T10:05:00.000000"  # 5 minutes later
        episode.context[MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT] = modification_time
        episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] = 1

        # Poll again
        response = test_client.get(
            f"/api/v1/session/{episode.session_id}/episodes/{episode.episode_id}/transcript/metadata"
        )
        data = response.json()

        # Now should detect changes (modification_time > last_pull)
        has_changes_now = data["last_modified_at"] is not None and data["last_modified_at"] > last_pull
        assert has_changes_now is True
