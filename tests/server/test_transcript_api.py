"""
Unit tests for transcript synchronization API endpoints.

Tests the POST and GET endpoints for transcript push/pull between client and server.
Following TDD: Write tests first, then implement.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from saber.models.constants import MetadataKeys
from saber.server.base import Episode, EpisodeState
from saber.server.session_manager import SessionManager


class TestTranscriptAPI:
    """Test transcript push/pull REST API endpoints."""

    @pytest.fixture
    def session_manager_app(self):
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

    def test_push_transcript_success(self, session_manager_app, sample_episode, sample_transcript):
        """Test successful transcript push stores data in Episode.context."""
        manager, client = session_manager_app

        # Setup: Create session and episode
        session_id = sample_episode.session_id
        episode_id = sample_episode.episode_id

        # Mock get_episode_by_id to return our sample episode
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[session_id] = {"episodes": {episode_id: sample_episode}}

        # Execute: Push transcript
        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/transcript",
            json=sample_transcript
        )

        # Assert: Success response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["episode_id"] == episode_id
        assert data["message_count"] == 4
        assert "stored_at" in data

        # Assert: Transcript stored in Episode.context
        stored_transcript = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert stored_transcript is not None
        assert len(stored_transcript) == 4
        assert stored_transcript[0]["role"] == "system"
        assert stored_transcript[1]["role"] == "user"
        assert stored_transcript[2]["role"] == "assistant"
        assert stored_transcript[3]["role"] == "tool"

    def test_push_transcript_episode_not_found(self, session_manager_app, sample_transcript):
        """Test pushing transcript to non-existent episode returns 404."""
        manager, client = session_manager_app

        # Mock episode manager to return None (episode not found)
        manager.get_episode_by_id = MagicMock(return_value=None)

        # Execute: Push transcript to non-existent episode
        response = client.post(
            "/api/v1/session/fake_session/episodes/fake_episode/transcript",
            json=sample_transcript
        )

        # Assert: 404 Not Found
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_push_transcript_episode_terminated(self, session_manager_app, sample_episode, sample_transcript):
        """Test pushing transcript to terminated episode is allowed (for audit trail)."""
        manager, client = session_manager_app

        # Setup: Episode is completed (terminated)
        sample_episode.state = EpisodeState.COMPLETED
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        # Execute: Push transcript to terminated episode
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=sample_transcript
        )

        # Assert: Success (200) - post-completion pushes allowed for audit trail
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message_count"] == 4

    def test_push_transcript_payload_too_large(self, session_manager_app, sample_episode):
        """Test pushing oversized transcript returns 413 Payload Too Large."""
        manager, client = session_manager_app

        # Setup: Episode exists
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        # Create large transcript (>10MB)
        large_content = "x" * (11 * 1024 * 1024)  # 11 MB
        large_transcript = {
            "messages": [
                {
                    "role": "user",
                    "content": large_content
                }
            ],
            "metadata": {}
        }

        # Execute: Push oversized transcript
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=large_transcript
        )

        # Assert: 413 Payload Too Large
        assert response.status_code == 413
        data = response.json()
        assert "detail" in data
        assert "too large" in data["detail"].lower() or "size" in data["detail"].lower()

    def test_push_transcript_empty_messages(self, session_manager_app, sample_episode):
        """Test pushing empty transcript is allowed (clear transcript)."""
        manager, client = session_manager_app

        # Setup: Episode exists
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        empty_transcript = {
            "messages": [],
            "metadata": {}
        }

        # Execute: Push empty transcript
        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=empty_transcript
        )

        # Assert: Success with 0 messages
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message_count"] == 0

        # Assert: Empty transcript stored
        stored_transcript = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert stored_transcript == []

    def test_push_transcript_idempotent(self, session_manager_app, sample_episode, sample_transcript):
        """Test pushing same transcript twice is idempotent (last write wins)."""
        manager, client = session_manager_app

        # Setup: Episode exists
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        # Execute: Push transcript twice
        response1 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=sample_transcript
        )

        # Modify transcript
        modified_transcript = sample_transcript.copy()
        modified_transcript["messages"].append({
            "role": "assistant",
            "content": "New message"
        })

        response2 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=modified_transcript
        )

        # Assert: Both succeed
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Assert: Last write wins (5 messages from second push)
        stored_transcript = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored_transcript) == 5
        assert stored_transcript[-1]["content"] == "New message"

    # ========================================================================
    # GET /api/v1/session/{sid}/episodes/{eid}/transcript - Get Transcript
    # ========================================================================

    def test_get_transcript_success(self, session_manager_app, sample_episode, sample_transcript):
        """Test successful transcript retrieval."""
        manager, client = session_manager_app

        # Setup: Episode with stored transcript
        sample_episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = sample_transcript["messages"]
        sample_episode.context[MetadataKeys.TRANSCRIPT_UPDATED_AT] = "2025-11-26T10:30:00Z"
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        # Execute: Get transcript
        response = client.get(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript"
        )

        # Assert: Success response
        if response.status_code != 200:
            print(f"Error response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert data["episode_id"] == sample_episode.episode_id
        assert data["message_count"] == 4
        assert "messages" in data
        assert len(data["messages"]) == 4
        assert data["messages"][0]["role"] == "system"
        assert "last_updated" in data

    def test_get_transcript_not_found(self, session_manager_app):
        """Test getting transcript from non-existent episode returns 404."""
        manager, client = session_manager_app

        # Mock episode manager to return None
        manager.get_episode_by_id = MagicMock(return_value=None)

        # Execute: Get transcript from non-existent episode
        response = client.get(
            "/api/v1/session/fake_session/episodes/fake_episode/transcript"
        )

        # Assert: 404 Not Found
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_get_transcript_empty(self, session_manager_app, sample_episode):
        """Test getting transcript when no transcript has been pushed."""
        manager, client = session_manager_app

        # Setup: Episode without transcript
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        # Execute: Get transcript
        response = client.get(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript"
        )

        # Assert: Success with empty messages
        assert response.status_code == 200
        data = response.json()
        assert data["message_count"] == 0
        assert data["messages"] == []

    def test_get_transcript_with_metadata(self, session_manager_app, sample_episode, sample_transcript):
        """Test getting transcript includes metadata."""
        manager, client = session_manager_app

        # Setup: Episode with transcript and metadata
        sample_episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = sample_transcript["messages"]
        sample_episode.context[MetadataKeys.TRANSCRIPT_UPDATED_AT] = "2025-11-26T10:30:00Z"
        sample_episode.context[MetadataKeys.TRANSCRIPT_METADATA] = sample_transcript["metadata"]
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        # Execute: Get transcript
        response = client.get(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript"
        )

        # Assert: Success with metadata
        assert response.status_code == 200
        data = response.json()
        assert "metadata" in data
        assert data["metadata"]["step_number"] == 3
        assert data["metadata"]["source"] == "inspect_ai"

    # ========================================================================
    # Error Handling & Edge Cases
    # ========================================================================

    def test_push_transcript_malformed_json(self, session_manager_app, sample_episode):
        """Test pushing malformed JSON returns 422 Unprocessable Entity."""
        manager, client = session_manager_app

        # Setup: Episode exists
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        # Execute: Push malformed transcript (missing required fields)
        malformed = {"not_messages": []}

        response = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=malformed
        )

        # Assert: 422 Unprocessable Entity for validation errors
        assert response.status_code == 422

    def test_push_transcript_invalid_message_format(self, session_manager_app, sample_episode):
        """Test pushing transcript with invalid message format returns 422."""
        manager, client = session_manager_app

        # Setup: Episode exists
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        # Execute: Push transcript with invalid message (missing role)
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

        # Assert: 422 Unprocessable Entity for invalid message structure
        assert response.status_code == 422

    def test_concurrent_transcript_pushes(self, session_manager_app, sample_episode, sample_transcript):
        """Test concurrent pushes to same episode don't corrupt data."""
        manager, client = session_manager_app

        # Setup: Episode exists
        manager.get_episode_by_id = MagicMock(return_value=sample_episode)
        manager.active_sessions[sample_episode.session_id] = {
            "episodes": {sample_episode.episode_id: sample_episode}
        }

        # Execute: Push two different transcripts
        transcript1 = sample_transcript.copy()
        transcript2 = sample_transcript.copy()
        transcript2["messages"].append({"role": "assistant", "content": "Extra message"})

        response1 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=transcript1
        )
        response2 = client.post(
            f"/api/v1/session/{sample_episode.session_id}/episodes/{sample_episode.episode_id}/transcript",
            json=transcript2
        )

        # Assert: Both succeed
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Assert: Last write wins (no corruption)
        stored_transcript = sample_episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT)
        assert len(stored_transcript) == 5  # transcript2 has 5 messages
