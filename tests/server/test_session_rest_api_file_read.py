"""Tests for file read REST API endpoint.

Tests the GET endpoint for reading files from episode sandbox containers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from saber.models.rest.file_read import FileReadResponse
from saber.server.execution.sandbox.file_copier import FileReadResult


class TestFileReadEndpoint:
    """Test cases for the file read REST endpoint."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager with all required dependencies."""
        mock_sm = MagicMock()
        mock_sm.domain_name = "test_domain"

        # Mock episode
        mock_episode = MagicMock()
        mock_episode.is_ready = True
        mock_episode.state = MagicMock()
        mock_episode.state.value = "ready"
        mock_sm.get_episode_by_id.return_value = mock_episode

        # Mock execution manager
        mock_execution_manager = MagicMock()
        mock_sm.execution_manager = mock_execution_manager

        return mock_sm

    @pytest.fixture
    def client(self, mock_session_manager):
        """Create a test client with the REST API."""
        from saber.server.api.session_rest_api import SessionRestAPI

        api = SessionRestAPI(mock_session_manager)
        return TestClient(api.app)

    def test_read_file_success(self, client, mock_session_manager) -> None:
        """Test successful file read."""
        mock_session_manager.execution_manager.read_file_from_episode = AsyncMock(
            return_value=FileReadResult(
                success=True,
                file_path="/tmp/test.json",
                content='{"key": "value"}',
                bytes_read=16,
                error_message=None,
            )
        )

        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
            params={"file_path": "/tmp/test.json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "File read successfully"
        assert data["session_id"] == "sess-123"
        assert data["episode_id"] == "ep-456"
        assert data["file_path"] == "/tmp/test.json"
        assert data["content"] == '{"key": "value"}'
        assert data["bytes_read"] == 16
        assert data["encoding"] == "utf-8"

        mock_session_manager.execution_manager.read_file_from_episode.assert_called_once_with(
            episode_id="ep-456",
            file_path="/tmp/test.json",
            container_name=None,
            encoding="utf-8",
        )

    def test_read_file_with_container_name(self, client, mock_session_manager) -> None:
        """Test file read with explicit container name."""
        mock_session_manager.execution_manager.read_file_from_episode = AsyncMock(
            return_value=FileReadResult(
                success=True,
                file_path="/app/config.json",
                content="{}",
                bytes_read=2,
                error_message=None,
            )
        )

        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
            params={"file_path": "/app/config.json", "container_name": "custom-container"},
        )

        assert response.status_code == 200
        mock_session_manager.execution_manager.read_file_from_episode.assert_called_once_with(
            episode_id="ep-456",
            file_path="/app/config.json",
            container_name="custom-container",
            encoding="utf-8",
        )

    def test_read_file_with_custom_encoding(self, client, mock_session_manager) -> None:
        """Test file read with custom encoding."""
        mock_session_manager.execution_manager.read_file_from_episode = AsyncMock(
            return_value=FileReadResult(
                success=True,
                file_path="/tmp/test.txt",
                content="Héllo",
                bytes_read=5,
                error_message=None,
            )
        )

        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
            params={"file_path": "/tmp/test.txt", "encoding": "latin-1"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["encoding"] == "latin-1"

    def test_read_file_episode_not_found(self, client, mock_session_manager) -> None:
        """Test 404 when episode is not found."""
        mock_session_manager.get_episode_by_id.return_value = None

        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
            params={"file_path": "/tmp/test.json"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_read_file_episode_not_ready(self, client, mock_session_manager) -> None:
        """Test 422 when episode is not ready."""
        mock_episode = MagicMock()
        mock_episode.is_ready = False
        mock_episode.state = MagicMock()
        mock_episode.state.value = "creating"
        mock_session_manager.get_episode_by_id.return_value = mock_episode

        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
            params={"file_path": "/tmp/test.json"},
        )

        assert response.status_code == 422
        assert "not ready" in response.json()["detail"].lower()

    def test_read_file_file_not_found(self, client, mock_session_manager) -> None:
        """Test 404 when file is not found in container."""
        mock_session_manager.execution_manager.read_file_from_episode = AsyncMock(
            return_value=FileReadResult(
                success=False,
                file_path="/tmp/missing.json",
                content="",
                bytes_read=0,
                error_message="File not found: /tmp/missing.json",
            )
        )

        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
            params={"file_path": "/tmp/missing.json"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_read_file_container_not_found(self, client, mock_session_manager) -> None:
        """Test 404 when container is not found."""
        from docker.errors import NotFound

        mock_session_manager.execution_manager.read_file_from_episode = AsyncMock(
            side_effect=NotFound("Container not found")
        )

        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
            params={"file_path": "/tmp/test.json"},
        )

        assert response.status_code == 404
        assert "container" in response.json()["detail"].lower()

    def test_read_file_invalid_path(self, client, mock_session_manager) -> None:
        """Test 422 when file path is invalid."""
        mock_session_manager.execution_manager.read_file_from_episode = AsyncMock(
            side_effect=ValueError("file_path must be an absolute path")
        )

        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
            params={"file_path": "relative/path.json"},
        )

        assert response.status_code == 422
        assert "absolute" in response.json()["detail"].lower()

    def test_read_file_size_exceeded(self, client, mock_session_manager) -> None:
        """Test 413 when file size exceeds limit."""
        mock_session_manager.execution_manager.read_file_from_episode = AsyncMock(
            return_value=FileReadResult(
                success=False,
                file_path="/tmp/large.bin",
                content="",
                bytes_read=0,
                error_message="File size (20971520 bytes) exceeds maximum allowed (10485760 bytes)",
            )
        )

        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
            params={"file_path": "/tmp/large.bin"},
        )

        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"].lower()

    def test_read_file_missing_file_path_param(self, client, mock_session_manager) -> None:
        """Test 422 when file_path query param is missing."""
        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
        )

        # FastAPI returns 422 for missing required query params
        assert response.status_code == 422

    def test_read_file_internal_error(self, client, mock_session_manager) -> None:
        """Test 500 when an internal error occurs."""
        mock_session_manager.execution_manager.read_file_from_episode = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        response = client.get(
            "/api/v1/session/sess-123/episodes/ep-456/sandbox/files",
            params={"file_path": "/tmp/test.json"},
        )

        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()
