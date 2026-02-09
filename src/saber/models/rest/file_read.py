"""File read REST API models.

Request and response models for the file read endpoint that retrieves files
from a running episode's Docker container.
"""

from pathlib import PurePosixPath

from pydantic import BaseModel, Field, field_validator

# Maximum file size to read: 10 MiB (prevents memory issues)
MAX_FILE_READ_BYTES = 10 * 1024 * 1024


class FileReadRequest(BaseModel, frozen=True):
    """Request model for reading a file from episode container.

    Validates that:
    - file_path is an absolute path with no traversal
    - container_name is optional (defaults to execution container)
    """

    file_path: str = Field(description="Absolute path to the file inside the container to read")
    container_name: str | None = Field(
        default=None,
        description="Optional container name override (defaults to execution container)",
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file_path is absolute and safe."""
        if not v or not v.strip():
            raise ValueError("file_path cannot be empty")

        if not v.startswith("/"):
            raise ValueError("file_path must be an absolute path (start with /)")

        container_path = PurePosixPath(v)

        if container_path == PurePosixPath("/"):
            raise ValueError("file_path cannot be container root (/)")

        # Check for path traversal attempts
        if ".." in container_path.parts:
            raise ValueError("file_path contains path traversal (..)")

        return v


class FileReadResponse(BaseModel, frozen=True):
    """Response model for file read operation.

    Contains the file contents and metadata about the read operation.
    """

    message: str = Field(description="Operation result message")
    session_id: str = Field(description="Session identifier")
    episode_id: str = Field(description="Episode identifier")
    file_path: str = Field(description="Path of the file that was read")
    content: str = Field(description="File contents as string")
    bytes_read: int = Field(description="Number of bytes read")
    encoding: str = Field(default="utf-8", description="Encoding used to decode content")

    @field_validator("bytes_read")
    @classmethod
    def validate_bytes_read(cls, v: int) -> int:
        """Validate bytes_read is non-negative."""
        if v < 0:
            raise ValueError("bytes_read must be non-negative")
        return v


__all__ = ["FileReadRequest", "FileReadResponse", "MAX_FILE_READ_BYTES"]
