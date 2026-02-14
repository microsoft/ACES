"""
Unit tests for transcript synchronization API endpoints.

Tests the POST and GET endpoints for transcript push/pull between client and server.
Transcript data is stored in Redis via TranscriptCoordinator.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from saber.server.base import Episode, EpisodeState
from saber.server.session_manager import SessionManager


class TestTranscriptAPI:
    """Test transcript push/pull REST API endpoints."""

    @pytest.fixture
    def mock_transcript_coordinator(self):
        """Create a mock TranscriptCoordinator."""
        coordinator = MagicMock()
        coordinator.push_message = AsyncMock(return_value=1)
        coordinator.get_message_count = AsyncMock(return_value=0)
        coordinator.get_transcript = AsyncMock(return_value=[])
        return coordinator

    @pytest.fixture
    def session_manager_app(self, mock_transcript_coordinator):
        """Create SessionManager with test client."""
        mock_task_manager = MagicMock()
        mock_config_loader = MagicMock()
        mock_config_loader.get_permanent_environment = MagicMock(return_value=None)
        mock_task_manager.config_loader = mock_config_loader

        mock_execution_manager = MagicMock()
        mock_execution_manager.step = AsyncMock()
        mock_execution_manager.initialize_permanent_environment_manager = MagicMock()
        mock_execution_manager._permanent_environment_manager = None

        mock_policy_manager = MagicMock()
        mock_evaluation_manager = MagicMock()
        mock_evaluation_manager.log_session_start = AsyncMock()
        mock_evaluation_manager.log_session_end = AsyncMock()
        mock_evaluation_manager.log_episode_start = AsyncMock()
        mock_evaluation_manager.log_episode_end = AsyncMock()
        mock_evaluation_manager.log_action = AsyncMock()

        mock_episode_manager = MagicMock()
        mock_episode_manager.start_episode = MagicMock()
        mock_episode_manager.end_episode = MagicMock()
        mock_episode_manager.get_episode = MagicMock()
        mock_episode_manager.transcript_coordinator = mock_transcript_coordinator

        with (
            patch("saber.server.session_manager.BenchmarkManager", return_value=mock_task_manager),
            patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
            patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
            patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
            patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8003)
            return manager, TestClient(manager.rest_api.app)

    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode for testing."""
        return Episode(
            episode_id="test_episode_123",
            task_id="test_task_456",
            session_id="test_session_789",
            state=EpisodeState.ACTIVE,
            context={}
        )

    @pytest.fixture
    def sample_transcript(self):
        """Create a sample transcript for testing."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "List all files in the current directory"
                },
                {
                    "role": "assistant",
                    "content": "I'll use the ls command to list files.",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "function": "bash",
                            "arguments": {"command": "ls -la"}
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "total 48\ndrwxr-xr-x 2 user user 4096...",
                    "name": "bash"
                }
            ],
            "metadata": {
                "step_number": 3,
                "timestamp": "2025-11-26T10:30:00Z",
                "source": "inspect_ai"
            }
        }

    # ========================================================================
    # POST /api/v1/session/{sid}/episodes/{eid}/transcript - Push Transcript
    # ========================================================================

    def test_push_transcript_success(
        self, session_manager_app, sample_episode, sample_transcript, mock_transcript_coordinator
    ):
        """Test successful transcript push calls coordinator.push_message."""
        manager, client = session_manager_app

        session_id = sample_episode.session_id
        episode_id = sample_episode.episode_id

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[session_id] = {"episodes": {episode_id: sample_episode}}
        mock_transcript_coordinator.get_message_count.return_value = 4

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/transcript",
            json=sample_transcript
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["episode_id"] == episode_id
        assert data["message_count"] == 4
        assert "stored_at" in data

        # Verify coordinator.push_message was called for each message
        assert mock_transcript_coordinator.push_message.call_count == 4

    def test_push_transcript_episode_not_found(self, session_manager_app, sample_transcript):
        """Test pushing transcript to non-existent episode returns 404."""
        manager, client = session_manager_app

        manager.get_episode_by_id = MagicMock(return_value=None)

        response = client.post(
            "/api/v1/session/fake_session/episodes/fake_episode/transcript",
            json=sample_transcript
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_push_transcript_episode_terminated(
        self, session_manager_app, sample_episode, sample_transcript, mock_transcript_coordinator
    ):
        """Test pushing transcript to terminated episode is allowed (for audit trail)."""
        manager, client = session_manager_app

        sample_episode.state = EpisodeState.COMPLETED
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }
        mock_transcript_coordinator.get_message_count.return_value = 4

        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=sample_transcript
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message_count"] == 4

    def test_push_transcript_payload_too_large(self, session_manager_app, sample_episode):
        """Test pushing oversized transcript returns 413 Payload Too Large."""
        manager, client = session_manager_app

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        large_content = "x" * (11 * 1024 * 1024)
        large_transcript = {
            "messages": [
                {
                    "role": "user",
                    "content": large_content
                }
            ],
            "metadata": {}
        }

        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=large_transcript
        )

        assert response.status_code == 413
        data = response.json()
        assert "detail" in data
        assert "too large" in data["detail"].lower() or "size" in data["detail"].lower()

    def test_push_transcript_empty_messages(
        self, session_manager_app, sample_episode, mock_transcript_coordinator
    ):
        """Test pushing empty transcript is allowed."""
        manager, client = session_manager_app

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }
        mock_transcript_coordinator.get_message_count.return_value = 0

        empty_transcript = {
            "messages": [],
            "metadata": {}
        }

        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=empty_transcript
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message_count"] == 0

        # No push_message calls for empty messages
        mock_transcript_coordinator.push_message.assert_not_called()

    def test_push_transcript_idempotent(
        self, session_manager_app, sample_episode, sample_transcript, mock_transcript_coordinator
    ):
        """Test pushing same transcript twice succeeds both times."""
        manager, client = session_manager_app

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }
        mock_transcript_coordinator.get_message_count.return_value = 4

        response1 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=sample_transcript
        )

        modified_transcript = sample_transcript.copy()
        modified_transcript["messages"] = list(sample_transcript["messages"]) + [
            {"role": "assistant", "content": "New message"}
        ]
        mock_transcript_coordinator.get_message_count.return_value = 9

        response2 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=modified_transcript
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Total push_message calls = 4 (first push) + 5 (second push)
        assert mock_transcript_coordinator.push_message.call_count == 9

    # ========================================================================
    # Error Handling & Edge Cases
    # ========================================================================

    def test_push_transcript_malformed_json(self, session_manager_app, sample_episode):
        """Test pushing malformed JSON returns 422 Unprocessable Entity."""
        manager, client = session_manager_app

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        malformed = {"not_messages": []}

        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=malformed
        )

        assert response.status_code == 422

    def test_push_transcript_invalid_message_format(self, session_manager_app, sample_episode):
        """Test pushing transcript with invalid message format returns 422."""
        manager, client = session_manager_app

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        invalid_transcript = {
            "messages": [
                {"content": "Missing role field"}
            ],
            "metadata": {}
        }

        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=invalid_transcript
        )

        assert response.status_code == 422

    def test_concurrent_transcript_pushes(
        self, session_manager_app, sample_episode, sample_transcript, mock_transcript_coordinator
    ):
        """Test concurrent pushes to same episode both succeed."""
        manager, client = session_manager_app

        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        transcript1 = sample_transcript.copy()
        transcript2 = sample_transcript.copy()
        transcript2["messages"] = list(sample_transcript["messages"]) + [
            {"role": "assistant", "content": "Extra message"}
        ]
        mock_transcript_coordinator.get_message_count.return_value = 5

        response1 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=transcript1
        )
        response2 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=transcript2
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Total calls: 4 (first) + 5 (second) = 9
        assert mock_transcript_coordinator.push_message.call_count == 9
