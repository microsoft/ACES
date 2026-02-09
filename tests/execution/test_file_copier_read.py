"""Tests for SandboxFileCopier file read functionality.

Tests the read_file_from_container method that retrieves files from Docker containers
using the Docker SDK's get_archive API.
"""

import io
import tarfile
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import NotFound

from saber.server.execution.sandbox.file_copier import FileReadResult, SandboxFileCopier


class TestFileReadResult:
    """Test cases for FileReadResult dataclass."""

    def test_successful_result(self) -> None:
        """Test creating a successful file read result."""
        result = FileReadResult(
            success=True,
            file_path="/tmp/test.json",
            content='{"key": "value"}',
            bytes_read=16,
            error_message=None,
        )
        assert result.success is True
        assert result.file_path == "/tmp/test.json"
        assert result.content == '{"key": "value"}'
        assert result.bytes_read == 16
        assert result.error_message is None

    def test_failed_result(self) -> None:
        """Test creating a failed file read result."""
        result = FileReadResult(
            success=False,
            file_path="/tmp/missing.json",
            content="",
            bytes_read=0,
            error_message="File not found",
        )
        assert result.success is False
        assert result.file_path == "/tmp/missing.json"
        assert result.content == ""
        assert result.bytes_read == 0
        assert result.error_message == "File not found"

    def test_result_is_frozen(self) -> None:
        """Test that FileReadResult is immutable (frozen=True)."""
        result = FileReadResult(
            success=True,
            file_path="/tmp/test.json",
            content="content",
            bytes_read=7,
            error_message=None,
        )
        with pytest.raises(AttributeError):
            result.content = "changed"


class TestSandboxFileCopierReadFile:
    """Test cases for SandboxFileCopier.read_file_from_container."""

    @pytest.fixture
    def temp_base_dir(self, tmp_path):
        """Create a temporary base directory."""
        base_dir = tmp_path / "server"
        base_dir.mkdir()
        return base_dir

    @pytest.fixture
    def file_copier(self, temp_base_dir):
        """Create a SandboxFileCopier instance."""
        return SandboxFileCopier(base_dir=temp_base_dir)

    def _create_tar_with_file(self, filename: str, content: bytes) -> bytes:
        """Create a tar archive containing a single file."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            file_info = tarfile.TarInfo(name=filename)
            file_info.size = len(content)
            tar.addfile(file_info, io.BytesIO(content))
        tar_buffer.seek(0)
        return tar_buffer.read()

    @pytest.mark.asyncio
    async def test_read_file_success(self, file_copier) -> None:
        """Test successful file read from container."""
        file_content = b'{"state": "active"}'
        tar_data = self._create_tar_with_file("test.json", file_content)

        mock_container = MagicMock()
        mock_container.id = "container123"
        mock_container.get_archive.return_value = (iter([tar_data]), {"size": len(tar_data)})

        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        file_copier._docker_client = mock_client

        result = await file_copier.read_file_from_container(
            episode_id="ep-123",
            file_path="/tmp/test.json",
            container_name="test-container",
        )

        assert result.success is True
        assert result.file_path == "/tmp/test.json"
        assert result.content == '{"state": "active"}'
        assert result.bytes_read == len(file_content)
        assert result.error_message is None

        mock_client.containers.get.assert_called_once_with("test-container")
        mock_container.get_archive.assert_called_once_with("/tmp/test.json")

    @pytest.mark.asyncio
    async def test_read_file_container_not_found(self, file_copier) -> None:
        """Test handling when container is not found."""
        mock_client = MagicMock()
        mock_client.containers.get.side_effect = NotFound("Container not found")
        file_copier._docker_client = mock_client

        with pytest.raises(NotFound, match="test-container"):
            await file_copier.read_file_from_container(
                episode_id="ep-123",
                file_path="/tmp/test.json",
                container_name="test-container",
            )

    @pytest.mark.asyncio
    async def test_read_file_file_not_found(self, file_copier) -> None:
        """Test handling when file is not found in container."""
        mock_container = MagicMock()
        mock_container.id = "container123"
        mock_container.get_archive.side_effect = NotFound("File not found")

        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        file_copier._docker_client = mock_client

        result = await file_copier.read_file_from_container(
            episode_id="ep-123",
            file_path="/tmp/missing.json",
            container_name="test-container",
        )

        assert result.success is False
        assert result.file_path == "/tmp/missing.json"
        assert result.content == ""
        assert result.bytes_read == 0
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_read_file_custom_encoding(self, file_copier) -> None:
        """Test reading file with custom encoding."""
        # Use latin-1 encodable content
        file_content = "Héllo Wörld".encode("latin-1")
        tar_data = self._create_tar_with_file("test.txt", file_content)

        mock_container = MagicMock()
        mock_container.id = "container123"
        mock_container.get_archive.return_value = (iter([tar_data]), {"size": len(tar_data)})

        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        file_copier._docker_client = mock_client

        result = await file_copier.read_file_from_container(
            episode_id="ep-123",
            file_path="/tmp/test.txt",
            container_name="test-container",
            encoding="latin-1",
        )

        assert result.success is True
        assert result.content == "Héllo Wörld"

    @pytest.mark.asyncio
    async def test_read_file_invalid_encoding(self, file_copier) -> None:
        """Test handling decode errors with wrong encoding."""
        # UTF-8 encoded content with characters that aren't valid ASCII
        file_content = "日本語テスト".encode("utf-8")
        tar_data = self._create_tar_with_file("test.txt", file_content)

        mock_container = MagicMock()
        mock_container.id = "container123"
        mock_container.get_archive.return_value = (iter([tar_data]), {"size": len(tar_data)})

        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        file_copier._docker_client = mock_client

        result = await file_copier.read_file_from_container(
            episode_id="ep-123",
            file_path="/tmp/test.txt",
            container_name="test-container",
            encoding="ascii",  # Wrong encoding
        )

        assert result.success is False
        assert "decode" in result.error_message.lower() or "codec" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_read_file_path_validation_failure(self, file_copier) -> None:
        """Test that invalid file paths are rejected."""
        mock_client = MagicMock()
        file_copier._docker_client = mock_client

        # Relative path should fail
        with pytest.raises(ValueError, match="absolute"):
            await file_copier.read_file_from_container(
                episode_id="ep-123",
                file_path="relative/path.txt",
                container_name="test-container",
            )

        # Path traversal should fail
        with pytest.raises(ValueError, match="traversal"):
            await file_copier.read_file_from_container(
                episode_id="ep-123",
                file_path="/root/../etc/passwd",
                container_name="test-container",
            )

    @pytest.mark.asyncio
    async def test_read_file_size_limit_exceeded(self, file_copier) -> None:
        """Test handling of files exceeding size limit."""
        # Create a tar with file info indicating large size
        from saber.models.rest.file_read import MAX_FILE_READ_BYTES

        # Create a file that exceeds the limit
        large_content = b"x" * (MAX_FILE_READ_BYTES + 1)
        tar_data = self._create_tar_with_file("large.txt", large_content)

        mock_container = MagicMock()
        mock_container.id = "container123"
        mock_container.get_archive.return_value = (iter([tar_data]), {"size": len(tar_data)})

        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        file_copier._docker_client = mock_client

        result = await file_copier.read_file_from_container(
            episode_id="ep-123",
            file_path="/tmp/large.txt",
            container_name="test-container",
        )

        assert result.success is False
        assert "size" in result.error_message.lower() or "exceeds" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_read_empty_file(self, file_copier) -> None:
        """Test reading an empty file."""
        tar_data = self._create_tar_with_file("empty.txt", b"")

        mock_container = MagicMock()
        mock_container.id = "container123"
        mock_container.get_archive.return_value = (iter([tar_data]), {"size": len(tar_data)})

        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        file_copier._docker_client = mock_client

        result = await file_copier.read_file_from_container(
            episode_id="ep-123",
            file_path="/tmp/empty.txt",
            container_name="test-container",
        )

        assert result.success is True
        assert result.content == ""
        assert result.bytes_read == 0
