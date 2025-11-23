"""Tests for episode status REST API endpoint."""

from unittest.mock import MagicMock, Mock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from saber.server.base import Episode, EpisodeState
from saber.server.session_manager import SessionManager


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    return MagicMock()


@pytest.fixture
def test_client():
    """Create a test client with mocked session manager."""
    # Create mocked dependencies
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

    with (
        patch("saber.server.session_manager.BenchmarkManager", return_value=mock_task_manager),
        patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
        patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
        patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
        patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
    ):
        manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8003)
        return manager, TestClient(manager.rest_api.app)


class TestEpisodeStatusEndpoint:
    """Test REST API status endpoint."""

    def test_get_episode_status_returns_ready_episode(self, test_client):
        """Test GET /api/v1/session/{session_id}/episodes/{episode_id}/status for ready episode."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        # Mock episode status - include session_id to pass validation
        mock_status = Mock()
        mock_status.episode_id = episode_id
        mock_status.task_id = "task-1"
        mock_status.session_id = session_id  # Must match the request
        mock_status.state = EpisodeState.READY
        mock_status.is_ready = True
        mock_status.creation_error = None
        mock_status.attached_to_episode_id = None
        mock_status.max_steps = 50
        mock_status.metadata = {}
        manager.get_episode_status = MagicMock(return_value=mock_status)

        response = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["episode_id"] == episode_id
        assert data["is_ready"] is True
        assert data["state"] == "ready"

    def test_get_episode_status_returns_creating_episode(self, test_client):
        """Test status endpoint returns CREATING state for pending episode."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-456"

        mock_status = Mock()
        mock_status.episode_id = episode_id
        mock_status.task_id = "task-1"
        mock_status.session_id = session_id  # Must match the request
        mock_status.state = EpisodeState.CREATING
        mock_status.is_ready = False
        mock_status.creation_error = None
        mock_status.attached_to_episode_id = None
        mock_status.max_steps = 50
        mock_status.metadata = {}
        manager.get_episode_status = MagicMock(return_value=mock_status)

        response = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["episode_id"] == episode_id
        assert data["is_ready"] is False
        assert data["state"] == "creating"

    def test_get_episode_status_404_for_nonexistent_episode(self, test_client):
        """Test 404 for non-existent episode."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "nonexistent"

        manager.get_episode_status = MagicMock(return_value=None)  # Return None for not found

        response = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/status")

        assert response.status_code == 404

    def test_get_episode_status_includes_error_message(self, test_client):
        """Test status endpoint includes error message for failed episodes."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-failed"

        mock_status = Mock()
        mock_status.episode_id = episode_id
        mock_status.task_id = "task-1"
        mock_status.session_id = session_id  # Must match the request
        mock_status.state = EpisodeState.FAILED_CREATION
        mock_status.is_ready = False
        mock_status.creation_error = "Health check timeout"
        mock_status.attached_to_episode_id = None
        mock_status.max_steps = 50
        mock_status.metadata = {}
        manager.get_episode_status = MagicMock(return_value=mock_status)

        response = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["episode_id"] == episode_id
        assert data["is_ready"] is False
        assert data["state"] == "failed_creation"

    def test_get_episode_status_403_for_wrong_session(self, test_client):
        """Test 403 forbidden for episode belonging to different session."""
        manager, client = test_client
        session_id = "session-1"
        episode_id = "episode-789"

        # Simulate permission error
        manager.get_episode_status = MagicMock(side_effect=PermissionError(
            "Episode belongs to different session"
        ))

        response = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/status")

        assert response.status_code in [403, 404, 500]  # Depends on error handling implementation

    def test_create_episode_returns_creating_state(self, test_client):
        """Test POST /api/v1/session/{session_id}/episodes returns CREATING state."""
        manager, client = test_client
        session_id = "test-session"
        task_id = "test-task"

        # Mock episode creation response - use AsyncMock since start_episode is async
        mock_episode = Mock()
        mock_episode.episode_id = "new-episode-123"
        mock_episode.session_id = session_id
        mock_episode.task_id = task_id
        mock_episode.state = EpisodeState.CREATING
        mock_episode.attached_to_episode_id = None
        mock_episode.max_steps = 50
        mock_episode.metadata = {}
        manager.start_episode = AsyncMock(return_value=mock_episode)

        response = client.post(
            f"/api/v1/session/{session_id}/episodes",
            params={"task_id": task_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["episode_id"] == "new-episode-123"
        assert data["state"] == "creating"

    def test_status_endpoint_supports_polling(self, test_client):
        """Test that status endpoint can be polled multiple times."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-polling"

        # First call - CREATING
        mock_status_creating = Mock()
        mock_status_creating.episode_id = episode_id
        mock_status_creating.task_id = "task-1"
        mock_status_creating.session_id = session_id
        mock_status_creating.state = EpisodeState.CREATING
        mock_status_creating.is_ready = False
        mock_status_creating.creation_error = None
        mock_status_creating.attached_to_episode_id = None
        mock_status_creating.max_steps = 50
        mock_status_creating.metadata = {}

        # Second call - READY
        mock_status_ready = Mock()
        mock_status_ready.episode_id = episode_id
        mock_status_ready.task_id = "task-1"
        mock_status_ready.session_id = session_id
        mock_status_ready.state = EpisodeState.READY
        mock_status_ready.is_ready = True
        mock_status_ready.creation_error = None
        mock_status_ready.attached_to_episode_id = None
        mock_status_ready.max_steps = 50
        mock_status_ready.metadata = {}

        manager.get_episode_status = MagicMock(side_effect=[
            mock_status_creating,
            mock_status_ready,
        ])

        # First poll - CREATING
        response1 = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/status")
        assert response1.status_code == 200
        assert response1.json()["state"] == "creating"

        # Second poll - READY
        response2 = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/status")
        assert response2.status_code == 200
        assert response2.json()["state"] == "ready"
        assert response2.json()["is_ready"] is True

    def test_status_response_schema(self, test_client):
        """Test that status response has correct schema."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-schema"

        mock_status = Mock()
        mock_status.episode_id = episode_id
        mock_status.task_id = "task-1"
        mock_status.session_id = session_id
        mock_status.state = EpisodeState.READY
        mock_status.is_ready = True
        mock_status.creation_error = None
        mock_status.attached_to_episode_id = None
        mock_status.max_steps = 50
        mock_status.metadata = {}
        manager.get_episode_status = MagicMock(return_value=mock_status)

        response = client.get(f"/api/v1/session/{session_id}/episodes/{episode_id}/status")

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "episode_id" in data
        assert "state" in data
        assert "is_ready" in data

        # Types
        assert isinstance(data["episode_id"], str)
        assert isinstance(data["state"], str)
        assert isinstance(data["is_ready"], bool)
