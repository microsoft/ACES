"""Tests for file upload REST API endpoint.

TDD tests for POST /api/v1/session/{session_id}/episodes/{episode_id}/files

The endpoint accepts base64-encoded tar archives and extracts them to
the episode container at the specified destination path.
"""

import base64
import io
import tarfile
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from docker.errors import NotFound as DockerNotFound
from fastapi.testclient import TestClient

from saber.server.base import Episode, EpisodeState
from saber.server.execution.sandbox.file_copier import FileUploadResult
from saber.server.session_manager import SessionManager


def create_test_tar(files: dict[str, bytes]) -> str:
    """Create a base64-encoded tar archive with the given files.

    Args:
        files: Dict mapping filename to file content bytes.

    Returns:
        Base64-encoded tar archive string.
    """
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for filename, content in files.items():
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content))
    tar_buffer.seek(0)
    return base64.b64encode(tar_buffer.read()).decode("utf-8")


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
    mock_execution_manager.upload_tar_to_episode = AsyncMock()

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


class TestFileUploadEndpoint:
    """Test REST API file upload endpoint."""

    def test_successful_tar_upload(self, test_client):
        """Test POST /api/v1/session/{session_id}/episodes/{episode_id}/files returns success."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        # Create test tar archive
        tar_data = create_test_tar({"test.txt": b"hello world"})

        # Mock episode exists and is ready
        mock_episode = Mock(spec=Episode)
        mock_episode.episode_id = episode_id
        mock_episode.task_id = "task-1"
        mock_episode.session_id = session_id
        mock_episode.state = EpisodeState.READY
        mock_episode.is_ready = True
        manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        # Mock execution manager upload_tar_to_episode returns success
        mock_result = FileUploadResult(
            success=True,
            source_path="<tar_data>",
            destination_path="/root",
            bytes_copied=1024,
            is_directory=True,
            error_message=None,
        )
        manager.execution_manager.upload_tar_to_episode = AsyncMock(return_value=mock_result)

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/root",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "File uploaded successfully"
        assert data["session_id"] == session_id
        assert data["episode_id"] == episode_id
        assert data["destination_path"] == "/root"
        assert data["bytes_copied"] == 1024

    def test_successful_directory_upload(self, test_client):
        """Test uploading a tar with multiple files."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        # Create test tar archive with directory structure
        tar_data = create_test_tar({
            "mydir/file1.txt": b"content1",
            "mydir/file2.txt": b"content2",
            "mydir/subdir/file3.txt": b"content3",
        })

        # Mock episode exists and is ready
        mock_episode = Mock(spec=Episode)
        mock_episode.episode_id = episode_id
        mock_episode.task_id = "task-1"
        mock_episode.session_id = session_id
        mock_episode.state = EpisodeState.READY
        mock_episode.is_ready = True
        manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        # Mock execution manager upload_tar_to_episode returns success
        mock_result = FileUploadResult(
            success=True,
            source_path="<tar_data>",
            destination_path="/workspace",
            bytes_copied=5120,
            is_directory=True,
            error_message=None,
        )
        manager.execution_manager.upload_tar_to_episode = AsyncMock(return_value=mock_result)

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/workspace",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["bytes_copied"] == 5120

    def test_file_upload_with_container_override(self, test_client):
        """Test file upload with explicit container_name parameter."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        # Mock episode exists and is ready
        mock_episode = Mock(spec=Episode)
        mock_episode.episode_id = episode_id
        mock_episode.task_id = "task-1"
        mock_episode.session_id = session_id
        mock_episode.state = EpisodeState.READY
        mock_episode.is_ready = True
        manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        mock_result = FileUploadResult(
            success=True,
            source_path="<tar_data>",
            destination_path="/root",
            bytes_copied=512,
            is_directory=True,
            error_message=None,
        )
        manager.execution_manager.upload_tar_to_episode = AsyncMock(return_value=mock_result)

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/root",
                "container_name": "custom-container",
            },
        )

        assert response.status_code == 200
        # Verify the container_name was passed to the execution manager
        call_kwargs = manager.execution_manager.upload_tar_to_episode.call_args.kwargs
        assert call_kwargs["episode_id"] == episode_id
        assert call_kwargs["destination_path"] == "/root"
        assert call_kwargs["container_name"] == "custom-container"
        # Verify tar_data was decoded
        assert isinstance(call_kwargs["tar_data"], bytes)

    def test_episode_not_found_returns_404(self, test_client):
        """Test 404 when episode does not exist."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "nonexistent-episode"

        tar_data = create_test_tar({"test.txt": b"content"})

        # Mock episode not found
        manager.get_episode_by_id = MagicMock(return_value=None)

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/root",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_episode_not_ready_returns_422(self, test_client):
        """Test 422 when episode is not in ready state."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        # Mock episode exists but is not ready (still creating)
        mock_episode = Mock(spec=Episode)
        mock_episode.episode_id = episode_id
        mock_episode.task_id = "task-1"
        mock_episode.session_id = session_id
        mock_episode.state = EpisodeState.CREATING
        mock_episode.is_ready = False
        manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/root",
            },
        )

        assert response.status_code == 422
        assert "not ready" in response.json()["detail"].lower()

    def test_container_not_found_returns_404(self, test_client):
        """Test 404 when Docker container is not found."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        # Mock episode exists and is ready
        mock_episode = Mock(spec=Episode)
        mock_episode.episode_id = episode_id
        mock_episode.task_id = "task-1"
        mock_episode.session_id = session_id
        mock_episode.state = EpisodeState.READY
        mock_episode.is_ready = True
        manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        # Mock execution manager raises docker NotFound
        manager.execution_manager.upload_tar_to_episode = AsyncMock(
            side_effect=DockerNotFound("Container not found")
        )

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/root",
            },
        )

        assert response.status_code == 404
        assert "container" in response.json()["detail"].lower()

    def test_invalid_base64_returns_422(self, test_client):
        """Test 422 when tar_data is not valid base64."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": "not-valid-base64!!!",
                "destination_path": "/root",
            },
        )

        assert response.status_code == 422
        # Pydantic validation error

    def test_invalid_destination_path_returns_422(self, test_client):
        """Test 422 when destination_path is relative (validation error)."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "relative/path",  # Should be absolute
            },
        )

        assert response.status_code == 422
        # Pydantic validation error

    def test_path_traversal_in_destination_returns_422(self, test_client):
        """Test 422 when destination_path contains path traversal."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/root/../etc/passwd",
            },
        )

        assert response.status_code == 422
        # Pydantic validation error for path traversal

    def test_empty_tar_data_returns_422(self, test_client):
        """Test 422 when tar_data is empty."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": "",
                "destination_path": "/root",
            },
        )

        assert response.status_code == 422

    def test_value_error_from_copier_returns_422(self, test_client):
        """Test 422 when file copier raises ValueError (e.g., invalid path)."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        # Mock episode exists and is ready
        mock_episode = Mock(spec=Episode)
        mock_episode.episode_id = episode_id
        mock_episode.task_id = "task-1"
        mock_episode.session_id = session_id
        mock_episode.state = EpisodeState.READY
        mock_episode.is_ready = True
        manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        # Mock execution manager raises ValueError
        manager.execution_manager.upload_tar_to_episode = AsyncMock(
            side_effect=ValueError("Invalid destination path")
        )

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/root",
            },
        )

        assert response.status_code == 422

    def test_runtime_error_from_copier_returns_500(self, test_client):
        """Test 500 when file copier raises RuntimeError (e.g., copier not initialized)."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        # Mock episode exists and is ready
        mock_episode = Mock(spec=Episode)
        mock_episode.episode_id = episode_id
        mock_episode.task_id = "task-1"
        mock_episode.session_id = session_id
        mock_episode.state = EpisodeState.READY
        mock_episode.is_ready = True
        manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        # Mock execution manager raises RuntimeError
        manager.execution_manager.upload_tar_to_episode = AsyncMock(
            side_effect=RuntimeError("File copier not initialized")
        )

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/root",
            },
        )

        assert response.status_code == 500

    def test_docker_api_error_returns_500(self, test_client):
        """Test 500 when Docker API raises an error."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        # Mock episode exists and is ready
        mock_episode = Mock(spec=Episode)
        mock_episode.episode_id = episode_id
        mock_episode.task_id = "task-1"
        mock_episode.session_id = session_id
        mock_episode.state = EpisodeState.READY
        mock_episode.is_ready = True
        manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        # Mock execution manager raises Docker APIError
        from docker.errors import APIError

        manager.execution_manager.upload_tar_to_episode = AsyncMock(
            side_effect=APIError("Docker API error")
        )

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/root",
            },
        )

        assert response.status_code == 500

    def test_upload_result_with_error_message_returns_500(self, test_client):
        """Test 500 when FileUploadResult indicates failure."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        # Mock episode exists and is ready
        mock_episode = Mock(spec=Episode)
        mock_episode.episode_id = episode_id
        mock_episode.task_id = "task-1"
        mock_episode.session_id = session_id
        mock_episode.state = EpisodeState.READY
        mock_episode.is_ready = True
        manager.get_episode_by_id = MagicMock(return_value=mock_episode)

        # Mock execution manager returns failure result
        mock_result = FileUploadResult(
            success=False,
            source_path="<tar_data>",
            destination_path="/root",
            bytes_copied=0,
            is_directory=True,
            error_message="Permission denied in container",
        )
        manager.execution_manager.upload_tar_to_episode = AsyncMock(return_value=mock_result)

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/root",
            },
        )

        assert response.status_code == 500
        assert "permission" in response.json()["detail"].lower()


class TestFileUploadRequestValidation:
    """Test Pydantic validation for FileUploadRequest."""

    def test_missing_tar_data_returns_422(self, test_client):
        """Test 422 when tar_data is missing."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "destination_path": "/root",
            },
        )

        assert response.status_code == 422

    def test_missing_destination_path_returns_422(self, test_client):
        """Test 422 when destination_path is missing."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
            },
        )

        assert response.status_code == 422

    def test_destination_root_returns_422(self, test_client):
        """Test 422 when destination_path is container root /."""
        manager, client = test_client
        session_id = "test-session"
        episode_id = "episode-123"

        tar_data = create_test_tar({"test.txt": b"content"})

        response = client.post(
            f"/api/v1/session/{session_id}/episodes/{episode_id}/files",
            json={
                "tar_data": tar_data,
                "destination_path": "/",
            },
        )

        assert response.status_code == 422
