"""Tests for file upload REST API models.

Tests validation logic and Pydantic model behavior for tar-based file upload requests/responses.
The API accepts base64-encoded tar archives for upload to episode containers.
"""

import base64
import pytest
from pydantic import ValidationError

from saber.models.rest.file_upload import FileUploadRequest, FileUploadResponse, MAX_TAR_SIZE_BYTES


# Test data: a simple tar header (512 bytes minimum for tar format)
VALID_TAR_BYTES = b"\x00" * 512  # Minimal tar format
VALID_TAR_BASE64 = base64.b64encode(VALID_TAR_BYTES).decode()


class TestFileUploadRequest:
    """Test cases for FileUploadRequest model."""

    def test_valid_request_minimal(self) -> None:
        """Test creating a valid request with required fields only."""
        request = FileUploadRequest(
            tar_data=VALID_TAR_BASE64,
            destination_path="/workspace",
        )
        assert request.tar_data == VALID_TAR_BASE64
        assert request.destination_path == "/workspace"
        assert request.container_name is None

    def test_valid_request_with_container_name(self) -> None:
        """Test creating a valid request with optional container_name."""
        request = FileUploadRequest(
            tar_data=VALID_TAR_BASE64,
            destination_path="/app/config",
            container_name="custom-container",
        )
        assert request.tar_data == VALID_TAR_BASE64
        assert request.destination_path == "/app/config"
        assert request.container_name == "custom-container"

    def test_destination_path_must_be_absolute(self) -> None:
        """Test that destination_path must start with /."""
        with pytest.raises(ValidationError) as exc_info:
            FileUploadRequest(
                tar_data=VALID_TAR_BASE64,
                destination_path="relative/path",
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "destination_path must be an absolute path" in str(errors[0]["msg"])

    def test_destination_path_cannot_be_root(self) -> None:
        """Test that destination_path cannot be just /."""
        with pytest.raises(ValidationError) as exc_info:
            FileUploadRequest(
                tar_data=VALID_TAR_BASE64,
                destination_path="/",
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "cannot be container root" in str(errors[0]["msg"])

    def test_destination_path_no_traversal(self) -> None:
        """Test that destination_path cannot contain path traversal."""
        with pytest.raises(ValidationError) as exc_info:
            FileUploadRequest(
                tar_data=VALID_TAR_BASE64,
                destination_path="/root/../etc/passwd",
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "path traversal" in str(errors[0]["msg"]).lower()

    def test_tar_data_cannot_be_empty(self) -> None:
        """Test that tar_data cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            FileUploadRequest(
                tar_data="",
                destination_path="/workspace",
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "tar_data cannot be empty" in str(errors[0]["msg"])

    def test_tar_data_must_be_valid_base64(self) -> None:
        """Test that tar_data must be valid base64."""
        with pytest.raises(ValidationError) as exc_info:
            FileUploadRequest(
                tar_data="not-valid-base64!!!",
                destination_path="/workspace",
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "must be valid base64" in str(errors[0]["msg"])

    def test_model_is_frozen(self) -> None:
        """Test that the model is immutable (frozen=True)."""
        request = FileUploadRequest(
            tar_data=VALID_TAR_BASE64,
            destination_path="/workspace",
        )
        with pytest.raises(ValidationError):
            request.tar_data = "changed"

    def test_get_tar_bytes(self) -> None:
        """Test that get_tar_bytes() decodes the tar data correctly."""
        request = FileUploadRequest(
            tar_data=VALID_TAR_BASE64,
            destination_path="/workspace",
        )
        assert request.get_tar_bytes() == VALID_TAR_BYTES


class TestFileUploadResponse:
    """Test cases for FileUploadResponse model."""

    def test_valid_response(self) -> None:
        """Test creating a valid response."""
        response = FileUploadResponse(
            message="Tar archive uploaded successfully",
            session_id="session-123",
            episode_id="episode-456",
            destination_path="/workspace",
            bytes_copied=1024,
        )
        assert response.message == "Tar archive uploaded successfully"
        assert response.session_id == "session-123"
        assert response.episode_id == "episode-456"
        assert response.destination_path == "/workspace"
        assert response.bytes_copied == 1024

    def test_bytes_copied_must_be_non_negative(self) -> None:
        """Test that bytes_copied cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            FileUploadResponse(
                message="Upload complete",
                session_id="session-123",
                episode_id="episode-456",
                destination_path="/workspace",
                bytes_copied=-1,
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "bytes_copied must be non-negative" in str(errors[0]["msg"])

    def test_model_is_frozen(self) -> None:
        """Test that the model is immutable (frozen=True)."""
        response = FileUploadResponse(
            message="Upload complete",
            session_id="session-123",
            episode_id="episode-456",
            destination_path="/workspace",
            bytes_copied=100,
        )
        with pytest.raises(ValidationError):
            response.message = "Changed message"


class TestMaxTarSize:
    """Test cases for tar size limit validation."""

    def test_max_tar_size_constant(self) -> None:
        """Test that MAX_TAR_SIZE_BYTES is 50 MiB."""
        assert MAX_TAR_SIZE_BYTES == 50 * 1024 * 1024

    def test_tar_data_size_within_limit(self) -> None:
        """Test that tar data within size limit is accepted."""
        # Create a 1 KB tar
        small_tar = b"\x00" * 1024
        small_tar_b64 = base64.b64encode(small_tar).decode()

        request = FileUploadRequest(
            tar_data=small_tar_b64,
            destination_path="/workspace",
        )
        assert len(request.get_tar_bytes()) == 1024
