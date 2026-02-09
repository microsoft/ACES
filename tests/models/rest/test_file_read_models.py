"""Tests for file read REST API models.

Tests validation logic and Pydantic model behavior for file read requests/responses.
The API reads files from episode sandbox containers.
"""

import pytest
from pydantic import ValidationError

from saber.models.rest.file_read import FileReadRequest, FileReadResponse, MAX_FILE_READ_BYTES


class TestFileReadRequest:
    """Test cases for FileReadRequest model."""

    def test_valid_request_minimal(self) -> None:
        """Test creating a valid request with required fields only."""
        request = FileReadRequest(
            file_path="/tmp/test.json",
        )
        assert request.file_path == "/tmp/test.json"
        assert request.container_name is None

    def test_valid_request_with_container_name(self) -> None:
        """Test creating a valid request with optional container_name."""
        request = FileReadRequest(
            file_path="/app/config.json",
            container_name="custom-container",
        )
        assert request.file_path == "/app/config.json"
        assert request.container_name == "custom-container"

    def test_file_path_must_be_absolute(self) -> None:
        """Test that file_path must start with /."""
        with pytest.raises(ValidationError) as exc_info:
            FileReadRequest(
                file_path="relative/path.txt",
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "file_path must be an absolute path" in str(errors[0]["msg"])

    def test_file_path_cannot_be_root(self) -> None:
        """Test that file_path cannot be just /."""
        with pytest.raises(ValidationError) as exc_info:
            FileReadRequest(
                file_path="/",
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "cannot be container root" in str(errors[0]["msg"])

    def test_file_path_no_traversal(self) -> None:
        """Test that file_path cannot contain path traversal."""
        with pytest.raises(ValidationError) as exc_info:
            FileReadRequest(
                file_path="/root/../etc/passwd",
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "path traversal" in str(errors[0]["msg"]).lower()

    def test_file_path_cannot_be_empty(self) -> None:
        """Test that file_path cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            FileReadRequest(
                file_path="",
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "file_path cannot be empty" in str(errors[0]["msg"])

    def test_file_path_cannot_be_whitespace(self) -> None:
        """Test that file_path cannot be just whitespace."""
        with pytest.raises(ValidationError) as exc_info:
            FileReadRequest(
                file_path="   ",
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        # Whitespace-only gets stripped, failing the empty check
        assert "file_path cannot be empty" in str(errors[0]["msg"])

    def test_model_is_frozen(self) -> None:
        """Test that the model is immutable (frozen=True)."""
        request = FileReadRequest(
            file_path="/tmp/test.json",
        )
        with pytest.raises(ValidationError):
            request.file_path = "/changed/path"

    def test_various_valid_absolute_paths(self) -> None:
        """Test various valid absolute paths."""
        valid_paths = [
            "/tmp/test.json",
            "/root/.config/settings.yaml",
            "/workspace/data/file with spaces.txt",
            "/a/b/c/d/e/f/deeply_nested.json",
        ]
        for path in valid_paths:
            request = FileReadRequest(file_path=path)
            assert request.file_path == path


class TestFileReadResponse:
    """Test cases for FileReadResponse model."""

    def test_valid_response_minimal(self) -> None:
        """Test creating a valid response with required fields."""
        response = FileReadResponse(
            message="File read successfully",
            session_id="session-123",
            episode_id="episode-456",
            file_path="/tmp/test.json",
            content='{"key": "value"}',
            bytes_read=16,
        )
        assert response.message == "File read successfully"
        assert response.session_id == "session-123"
        assert response.episode_id == "episode-456"
        assert response.file_path == "/tmp/test.json"
        assert response.content == '{"key": "value"}'
        assert response.bytes_read == 16
        assert response.encoding == "utf-8"  # default

    def test_valid_response_with_encoding(self) -> None:
        """Test creating a valid response with custom encoding."""
        response = FileReadResponse(
            message="File read successfully",
            session_id="session-123",
            episode_id="episode-456",
            file_path="/tmp/test.txt",
            content="Hello World",
            bytes_read=11,
            encoding="latin-1",
        )
        assert response.encoding == "latin-1"

    def test_bytes_read_must_be_non_negative(self) -> None:
        """Test that bytes_read cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            FileReadResponse(
                message="File read complete",
                session_id="session-123",
                episode_id="episode-456",
                file_path="/tmp/test.json",
                content="content",
                bytes_read=-1,
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "bytes_read must be non-negative" in str(errors[0]["msg"])

    def test_bytes_read_zero_is_valid(self) -> None:
        """Test that bytes_read=0 is valid (empty file)."""
        response = FileReadResponse(
            message="File read successfully",
            session_id="session-123",
            episode_id="episode-456",
            file_path="/tmp/empty.txt",
            content="",
            bytes_read=0,
        )
        assert response.bytes_read == 0
        assert response.content == ""

    def test_model_is_frozen(self) -> None:
        """Test that the model is immutable (frozen=True)."""
        response = FileReadResponse(
            message="File read successfully",
            session_id="session-123",
            episode_id="episode-456",
            file_path="/tmp/test.json",
            content="content",
            bytes_read=7,
        )
        with pytest.raises(ValidationError):
            response.content = "changed"


class TestMaxFileReadBytes:
    """Test cases for MAX_FILE_READ_BYTES constant."""

    def test_max_file_read_bytes_value(self) -> None:
        """Test that MAX_FILE_READ_BYTES is 10 MiB."""
        assert MAX_FILE_READ_BYTES == 10 * 1024 * 1024
        assert MAX_FILE_READ_BYTES == 10485760
