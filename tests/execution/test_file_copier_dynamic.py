"""Tests for SandboxFileCopier.copy_file_to_episode_dynamic method.

This module tests the dynamic file upload functionality that copies files from
the server's base directory into running Docker containers via REST API.
"""

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import NotFound

from saber.server.execution.sandbox.file_copier import (
    FileUploadResult,
    SandboxFileCopier,
)


class TestCopyFileToEpisodeDynamic:
    """Test cases for copy_file_to_episode_dynamic method."""

    @pytest.fixture
    def temp_base_dir(self, tmp_path: Path) -> Path:
        """Create a temporary base directory with test files."""
        base_dir = tmp_path / "server"
        base_dir.mkdir()

        # Create data/ subdirectory with files
        data_dir = base_dir / "data"
        data_dir.mkdir()
        (data_dir / "test_file.txt").write_text("Hello, World!")
        (data_dir / "script.sh").write_text("#!/bin/bash\necho 'test'")

        # Create a directory with files
        test_dir = data_dir / "test_directory"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("Content 1")
        (test_dir / "file2.txt").write_text("Content 2")
        nested_dir = test_dir / "nested"
        nested_dir.mkdir()
        (nested_dir / "file3.txt").write_text("Content 3")

        return base_dir

    @pytest.fixture
    def file_copier(self, temp_base_dir: Path) -> SandboxFileCopier:
        """Create a SandboxFileCopier instance."""
        return SandboxFileCopier(base_dir=temp_base_dir)

    @pytest.fixture
    def mock_docker_client(self) -> tuple[MagicMock, MagicMock]:
        """Create a mock Docker client."""
        client = MagicMock()
        container = MagicMock()
        container.id = "abc123def456"
        container.put_archive = MagicMock(return_value=True)
        # Mock exec_run for directory creation check
        container.exec_run = MagicMock(return_value=(0, b""))
        client.containers.get.return_value = container
        return client, container

    @pytest.mark.asyncio
    async def test_copy_single_file_success(
        self,
        file_copier: SandboxFileCopier,
        temp_base_dir: Path,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test successful copying of a single file."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        result = await file_copier.copy_file_to_episode_dynamic(
            episode_id="episode_123",
            source_path="data/test_file.txt",
            destination_path="/root/test_file.txt",
        )

        assert result.success is True
        assert result.bytes_copied == len("Hello, World!")
        assert result.is_directory is False
        assert result.source_path == "data/test_file.txt"
        assert result.destination_path == "/root/test_file.txt"
        mock_container.put_archive.assert_called_once()

    @pytest.mark.asyncio
    async def test_copy_directory_success(
        self,
        file_copier: SandboxFileCopier,
        temp_base_dir: Path,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test successful copying of a directory."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        result = await file_copier.copy_file_to_episode_dynamic(
            episode_id="episode_456",
            source_path="data/test_directory",
            destination_path="/app/test_directory",
        )

        assert result.success is True
        assert result.is_directory is True
        # Should contain bytes for file1.txt, file2.txt, and nested/file3.txt
        assert result.bytes_copied > 0
        mock_container.put_archive.assert_called_once()

    @pytest.mark.asyncio
    async def test_copy_with_custom_container_name(
        self,
        file_copier: SandboxFileCopier,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test copying with a custom container name."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        result = await file_copier.copy_file_to_episode_dynamic(
            episode_id="episode_789",
            source_path="data/test_file.txt",
            destination_path="/root/test.txt",
            container_name="custom-container",
        )

        assert result.success is True
        mock_client.containers.get.assert_called_once_with("custom-container")

    @pytest.mark.asyncio
    async def test_container_not_found(
        self,
        file_copier: SandboxFileCopier,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test handling when container is not found."""
        mock_client, _ = mock_docker_client
        mock_client.containers.get.side_effect = NotFound("Container not found")
        file_copier._docker_client = mock_client

        with pytest.raises(NotFound, match="Container 'default-episode_notfound' not found"):
            await file_copier.copy_file_to_episode_dynamic(
                episode_id="episode_notfound",
                source_path="data/test_file.txt",
                destination_path="/root/test.txt",
            )

    @pytest.mark.asyncio
    async def test_source_file_not_found(
        self,
        file_copier: SandboxFileCopier,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test handling when source file doesn't exist."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        with pytest.raises(FileNotFoundError, match="Source file not found"):
            await file_copier.copy_file_to_episode_dynamic(
                episode_id="episode_nosrc",
                source_path="data/nonexistent.txt",
                destination_path="/root/test.txt",
            )

    @pytest.mark.asyncio
    async def test_reject_absolute_source_path(
        self,
        file_copier: SandboxFileCopier,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test that absolute source paths are rejected."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        with pytest.raises(ValueError, match="must be relative"):
            await file_copier.copy_file_to_episode_dynamic(
                episode_id="episode_abs",
                source_path="/etc/passwd",
                destination_path="/root/test.txt",
            )

    @pytest.mark.asyncio
    async def test_reject_source_traversal(
        self,
        file_copier: SandboxFileCopier,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test that source path traversal is rejected."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        with pytest.raises(ValueError, match="must be relative to the base directory"):
            await file_copier.copy_file_to_episode_dynamic(
                episode_id="episode_traversal",
                source_path="../../../etc/passwd",
                destination_path="/root/test.txt",
            )

    @pytest.mark.asyncio
    async def test_reject_relative_destination(
        self,
        file_copier: SandboxFileCopier,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test that relative destination paths are rejected."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        with pytest.raises(ValueError, match="path must be absolute"):
            await file_copier.copy_file_to_episode_dynamic(
                episode_id="episode_reldest",
                source_path="data/test_file.txt",
                destination_path="relative/path.txt",
            )

    @pytest.mark.asyncio
    async def test_reject_destination_traversal(
        self,
        file_copier: SandboxFileCopier,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test that destination path traversal is rejected."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        with pytest.raises(ValueError, match="path traversal detected"):
            await file_copier.copy_file_to_episode_dynamic(
                episode_id="episode_desttraversal",
                source_path="data/test_file.txt",
                destination_path="/root/../etc/passwd",
            )

    @pytest.mark.asyncio
    async def test_reject_root_destination(
        self,
        file_copier: SandboxFileCopier,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test that root (/) destination is rejected."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        with pytest.raises(ValueError, match="cannot target container root"):
            await file_copier.copy_file_to_episode_dynamic(
                episode_id="episode_root",
                source_path="data/test_file.txt",
                destination_path="/",
            )

    @pytest.mark.asyncio
    async def test_reject_oversized_file(
        self,
        file_copier: SandboxFileCopier,
        temp_base_dir: Path,
        mock_docker_client: tuple[MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that files exceeding size limit are rejected."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        # Patch the size limit to something small
        monkeypatch.setattr(
            "saber.server.execution.sandbox.file_copier.MAX_SINGLE_FILE_BYTES",
            5,
        )

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            await file_copier.copy_file_to_episode_dynamic(
                episode_id="episode_large",
                source_path="data/test_file.txt",  # "Hello, World!" is 13 bytes
                destination_path="/root/test.txt",
            )

    @pytest.mark.asyncio
    async def test_reject_oversized_directory(
        self,
        file_copier: SandboxFileCopier,
        temp_base_dir: Path,
        mock_docker_client: tuple[MagicMock, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that directories exceeding size limit are rejected."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        monkeypatch.setattr(
            "saber.server.execution.sandbox.file_copier.MAX_DIRECTORY_BYTES",
            5,
        )

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            await file_copier.copy_file_to_episode_dynamic(
                episode_id="episode_largedir",
                source_path="data/test_directory",
                destination_path="/app/test_directory",
            )

    @pytest.mark.asyncio
    async def test_reject_symlink_file(
        self,
        file_copier: SandboxFileCopier,
        temp_base_dir: Path,
        mock_docker_client: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test that symlinks are rejected."""
        import os

        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        # Create a symlink
        data_dir = temp_base_dir / "data"
        link_path = data_dir / "symlink.txt"
        target = data_dir / "test_file.txt"

        try:
            os.symlink(target, link_path)
        except OSError:
            pytest.skip("Cannot create symlinks on this platform")

        with pytest.raises(ValueError, match="Symlinks are not allowed"):
            await file_copier.copy_file_to_episode_dynamic(
                episode_id="episode_symlink",
                source_path="data/symlink.txt",
                destination_path="/root/test.txt",
            )


class TestFileUploadResult:
    """Test cases for FileUploadResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful result."""
        result = FileUploadResult(
            success=True,
            source_path="data/file.txt",
            destination_path="/root/file.txt",
            bytes_copied=1024,
            is_directory=False,
            error_message=None,
        )
        assert result.success is True
        assert result.bytes_copied == 1024
        assert result.is_directory is False
        assert result.error_message is None

    def test_create_failure_result(self) -> None:
        """Test creating a failure result."""
        result = FileUploadResult(
            success=False,
            source_path="data/file.txt",
            destination_path="/root/file.txt",
            bytes_copied=0,
            is_directory=False,
            error_message="File not found",
        )
        assert result.success is False
        assert result.bytes_copied == 0
        assert result.error_message == "File not found"

    def test_result_is_frozen(self) -> None:
        """Test that the result is immutable."""
        from dataclasses import FrozenInstanceError

        result = FileUploadResult(
            success=True,
            source_path="data/file.txt",
            destination_path="/root/file.txt",
            bytes_copied=100,
            is_directory=False,
            error_message=None,
        )
        with pytest.raises(FrozenInstanceError):
            result.success = False
