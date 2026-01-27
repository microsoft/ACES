"""File upload REST API models.

Request and response models for the file upload endpoint that copies files
to a running episode's Docker container.

The endpoint accepts tar archive content directly (base64 encoded), allowing
clients to upload files from any location without requiring server filesystem access.
"""

import base64
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, field_validator

# Maximum tar archive size: 50 MiB (base64 encoded will be ~67 MiB)
MAX_TAR_SIZE_BYTES = 50 * 1024 * 1024


class FileUploadRequest(BaseModel, frozen=True):
    """Request model for file upload to episode container.

    The client creates a tar archive of the content to upload, base64 encodes it,
    and sends it in the request body. The server decodes and extracts it to the
    specified destination path in the container.

    Validates that:
    - tar_data is non-empty and valid base64
    - destination_path is an absolute path with no traversal
    """

    tar_data: str = Field(description="Base64-encoded tar archive containing the file(s) to upload")
    destination_path: str = Field(description="Absolute path inside the container where the tar will be extracted")
    container_name: str | None = Field(
        default=None,
        description="Optional container name override (defaults to episode container)",
    )

    @field_validator("tar_data")
    @classmethod
    def validate_tar_data(cls, v: str) -> str:
        """Validate tar_data is non-empty and valid base64."""
        if not v or not v.strip():
            raise ValueError("tar_data cannot be empty")

        # Check base64 is valid by attempting decode
        try:
            decoded = base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"tar_data must be valid base64: {e}") from e

        # Check size limit
        if len(decoded) > MAX_TAR_SIZE_BYTES:
            raise ValueError(f"tar_data exceeds maximum size of {MAX_TAR_SIZE_BYTES // (1024*1024)} MiB")

        return v

    @field_validator("destination_path")
    @classmethod
    def validate_destination_path(cls, v: str) -> str:
        """Validate destination_path is absolute and safe."""
        if not v or not v.strip():
            raise ValueError("destination_path cannot be empty")

        if not v.startswith("/"):
            raise ValueError("destination_path must be an absolute path (start with /)")

        container_path = PurePosixPath(v)

        if container_path == PurePosixPath("/"):
            raise ValueError("destination_path cannot be container root (/)")

        # Check for path traversal attempts
        if ".." in container_path.parts:
            raise ValueError("destination_path contains path traversal (..)")

        return v

    def get_tar_bytes(self) -> bytes:
        """Decode and return the tar archive bytes."""
        return base64.b64decode(self.tar_data)


class FileUploadResponse(BaseModel, frozen=True):
    """Response model for file upload operation.

    Contains details about the completed upload including bytes transferred.
    """

    message: str = Field(description="Operation result message")
    session_id: str = Field(description="Session identifier")
    episode_id: str = Field(description="Episode identifier")
    destination_path: str = Field(description="Destination path in container")
    bytes_copied: int = Field(description="Number of bytes copied (tar archive size)")

    @field_validator("bytes_copied")
    @classmethod
    def validate_bytes_copied(cls, v: int) -> int:
        """Validate bytes_copied is non-negative."""
        if v < 0:
            raise ValueError("bytes_copied must be non-negative")
        return v


__all__ = ["FileUploadRequest", "FileUploadResponse", "MAX_TAR_SIZE_BYTES"]
