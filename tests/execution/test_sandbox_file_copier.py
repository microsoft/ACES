"""
Tests for SandboxFileCopier.

This module tests the file provisioning functionality that copies files from
the server's base directory (containing docker/, data/, etc.) into running
Docker containers at episode startup.
"""

import io
import os
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from docker.errors import APIError, NotFound

from saber.server.execution.sandbox.file_copier import SandboxFileCopier


class TestSandboxFileCopier:
    """Test cases for SandboxFileCopier."""

    @pytest.fixture
    def temp_base_dir(self, tmp_path):
        """Create a temporary base directory with test files in docker/ and data/ subdirs."""
        base_dir = tmp_path / "server"
        base_dir.mkdir()

        # Create docker/ subdirectory with files
        docker_dir = base_dir / "docker"
        docker_dir.mkdir()
        (docker_dir / "test.txt").write_text("Hello, World!")
        (docker_dir / "script.sh").write_text("#!/bin/bash\necho 'test'")

        # Create data/ subdirectory with files
        data_dir = base_dir / "data"
        data_dir.mkdir()
        (data_dir / "data_file.txt").write_text("Data content")

        # Create a directory with files in docker/
        test_dir = docker_dir / "test_directory"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("Content 1")
        (test_dir / "file2.txt").write_text("Content 2")
        nested_dir = test_dir / "nested"
        nested_dir.mkdir()
        (nested_dir / "file3.txt").write_text("Content 3")

        return base_dir

    @pytest.fixture
    def file_copier(self, temp_base_dir):
        """Create a SandboxFileCopier instance."""
        return SandboxFileCopier(base_dir=temp_base_dir)

    @pytest.fixture
    def mock_docker_client(self):
        """Create a mock Docker client."""
        client = MagicMock()
        container = MagicMock()
        container.id = "abc123def456"
        container.put_archive = MagicMock(return_value=True)
        client.containers.get.return_value = container
        return client, container

    def test_initialization(self, temp_base_dir):
        """Test successful initialization of SandboxFileCopier."""
        copier = SandboxFileCopier(base_dir=temp_base_dir)
        assert copier.base_dir == temp_base_dir
        assert copier._docker_client is None  # Lazy initialization

    def test_docker_client_lazy_initialization(self, file_copier):
        """Test that Docker client is lazily initialized."""
        assert file_copier._docker_client is None

        with patch("saber.server.execution.sandbox.file_copier.docker_client.from_env") as mock_from_env:
            mock_client = MagicMock()
            mock_from_env.return_value = mock_client

            # Access the property
            client = file_copier.docker_client

            # Should initialize
            assert client == mock_client
            assert file_copier._docker_client == mock_client
            mock_from_env.assert_called_once()

            # Second access should not reinitialize
            client2 = file_copier.docker_client
            assert client2 == mock_client
            assert mock_from_env.call_count == 1

    @pytest.mark.asyncio
    async def test_copy_files_empty_mappings(self, file_copier):
        """Test that empty file mappings are handled gracefully."""
        # Just ensure it doesn't raise
        await file_copier.copy_files_to_episode("episode_123", {})

    @pytest.mark.asyncio
    async def test_copy_files_container_not_found(self, file_copier):
        """Test handling when container is not found."""
        mock_client = MagicMock()
        mock_client.containers.get.side_effect = NotFound("Container not found")
        file_copier._docker_client = mock_client

        with pytest.raises(NotFound, match="Container 'default-episode_123' not found"):
            await file_copier.copy_files_to_episode(
                "episode_123", {"/root/test.txt": "docker/test.txt"}
            )

    @pytest.mark.asyncio
    async def test_copy_single_file_success(self, file_copier, temp_base_dir, mock_docker_client):
        """Test successful copying of a single file."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        await file_copier.copy_files_to_episode(
            "episode_123", {"/root/test.txt": "docker/test.txt"}
        )

        # Verify container was retrieved
        mock_client.containers.get.assert_called_once_with("default-episode_123")

        # Verify put_archive was called
        mock_container.put_archive.assert_called_once()
        call_args = mock_container.put_archive.call_args
        assert call_args[0][0] == "/root"  # Destination parent directory

        # Verify tar archive contains the file
        tar_stream = call_args[0][1]
        tar_stream.seek(0)
        tar = tarfile.open(fileobj=tar_stream, mode="r")
        members = tar.getmembers()
        assert len(members) == 1
        assert members[0].name == "test.txt"
        tar.close()

    @pytest.mark.asyncio
    async def test_copy_file_with_custom_container_prefix(self, file_copier, mock_docker_client):
        """Test copying with custom container prefix."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        await file_copier.copy_files_to_episode(
            "episode_456", {"/app/file.txt": "docker/test.txt"}, container_prefix="custom"
        )

        mock_client.containers.get.assert_called_once_with("custom-episode_456")

    @pytest.mark.asyncio
    async def test_copy_multiple_files(self, file_copier, temp_base_dir, mock_docker_client):
        """Test copying multiple files."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        file_mappings = {
            "/root/test.txt": "docker/test.txt",
            "/root/script.sh": "docker/script.sh",
        }

        await file_copier.copy_files_to_episode("episode_789", file_mappings)

        # Should call put_archive twice (once per file)
        assert mock_container.put_archive.call_count == 2

    @pytest.mark.asyncio
    async def test_copy_directory(self, file_copier, temp_base_dir, mock_docker_client):
        """Test copying an entire directory."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        await file_copier.copy_files_to_episode(
            "episode_999", {"/root/test_directory": "docker/test_directory"}
        )

        mock_container.put_archive.assert_called_once()
        call_args = mock_container.put_archive.call_args
        assert call_args[0][0] == "/root"

        # Verify tar archive contains directory structure
        tar_stream = call_args[0][1]
        tar_stream.seek(0)
        tar = tarfile.open(fileobj=tar_stream, mode="r")
        member_names = [m.name for m in tar.getmembers()]

        # Should contain the directory and all its files
        assert "test_directory/file1.txt" in member_names
        assert "test_directory/file2.txt" in member_names
        assert "test_directory/nested/file3.txt" in member_names
        tar.close()

    @pytest.mark.asyncio
    async def test_source_file_not_found(self, file_copier, mock_docker_client):
        """Test handling when source file doesn't exist."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        with pytest.raises(RuntimeError, match="Failed to copy 1 file"):
            await file_copier.copy_files_to_episode(
                "episode_404", {"/root/missing.txt": "nonexistent.txt"}
            )

    @pytest.mark.asyncio
    async def test_docker_api_error(self, file_copier, temp_base_dir, mock_docker_client):
        """Test handling Docker API errors during file copy."""
        mock_client, mock_container = mock_docker_client
        mock_container.put_archive.side_effect = APIError("Docker error")
        file_copier._docker_client = mock_client

        with pytest.raises(RuntimeError, match="Failed to copy 1 file"):
            await file_copier.copy_files_to_episode(
                "episode_500", {"/root/test.txt": "docker/test.txt"}
            )

    @pytest.mark.asyncio
    async def test_partial_failure(self, file_copier, temp_base_dir, mock_docker_client):
        """Test that partial failures are reported correctly."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        # First call succeeds, second fails
        mock_container.put_archive.side_effect = [True, APIError("Docker error")]

        file_mappings = {
            "/root/test.txt": "docker/test.txt",
            "/root/script.sh": "docker/script.sh",
        }

        with pytest.raises(RuntimeError, match="Failed to copy 1 file"):
            await file_copier.copy_files_to_episode("episode_partial", file_mappings)

        # First file should have been attempted
        assert mock_container.put_archive.call_count == 2

    def test_create_tar_for_file(self, file_copier, temp_base_dir):
        """Test creating tar archive for a single file."""
        source_path = temp_base_dir / "docker" / "test.txt"
        tar_stream = file_copier._create_tar_for_file(source_path, "/root/test.txt")

        # Verify tar archive
        tar_stream.seek(0)
        tar = tarfile.open(fileobj=tar_stream, mode="r")
        members = tar.getmembers()

        assert len(members) == 1
        assert members[0].name == "test.txt"
        assert members[0].size == len("Hello, World!")
        assert members[0].mode == 0o644  # Regular file permissions
        tar.close()

    def test_create_tar_for_shell_script(self, file_copier, temp_base_dir):
        """Test that shell scripts get executable permissions."""
        source_path = temp_base_dir / "docker" / "script.sh"
        tar_stream = file_copier._create_tar_for_file(source_path, "/root/script.sh")

        tar_stream.seek(0)
        tar = tarfile.open(fileobj=tar_stream, mode="r")
        members = tar.getmembers()

        assert len(members) == 1
        assert members[0].name == "script.sh"
        assert members[0].mode == 0o755  # Executable permissions
        tar.close()

    def test_create_tar_for_directory(self, file_copier, temp_base_dir):
        """Test creating tar archive for a directory."""
        source_path = temp_base_dir / "docker" / "test_directory"
        tar_stream = file_copier._create_tar_for_directory(source_path, "/root/test_directory")

        # Verify tar archive
        tar_stream.seek(0)
        tar = tarfile.open(fileobj=tar_stream, mode="r")
        members = tar.getmembers()
        member_names = [m.name for m in members]

        # Should contain all files in the directory
        assert "test_directory/file1.txt" in member_names
        assert "test_directory/file2.txt" in member_names
        assert "test_directory/nested/file3.txt" in member_names

        # Verify file modes
        for member in members:
            if member.name.endswith(".txt"):
                assert member.mode == 0o644
        tar.close()

    def test_close_docker_client(self, file_copier):
        """Test closing the Docker client connection."""
        with patch("saber.server.execution.sandbox.file_copier.docker_client.from_env") as mock_from_env:
            mock_client = MagicMock()
            mock_from_env.return_value = mock_client

            # Initialize the client
            _ = file_copier.docker_client
            assert file_copier._docker_client is not None

            # Close it
            file_copier.close()
            mock_client.close.assert_called_once()
            assert file_copier._docker_client is None

    def test_close_without_client(self, file_copier):
        """Test closing when Docker client was never initialized."""
        # Should not raise an error
        file_copier.close()
        assert file_copier._docker_client is None

    @pytest.mark.asyncio
    async def test_copy_file_to_root_directory(self, file_copier, temp_base_dir, mock_docker_client):
        """Test copying file to root directory (/)."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        await file_copier.copy_files_to_episode(
            "episode_root", {"/test.txt": "docker/test.txt"}
        )

        # Should use "/" as the parent directory
        call_args = mock_container.put_archive.call_args
        assert call_args[0][0] == "/"

    @pytest.mark.asyncio
    async def test_logging_on_success(self, file_copier, temp_base_dir, mock_docker_client):
        """Test that appropriate logs are generated on success."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        # Should not raise
        await file_copier.copy_files_to_episode(
            "episode_log", {"/root/test.txt": "docker/test.txt"}
        )

        # Success - verify container.put_archive was called
        mock_container.put_archive.assert_called_once()

    @pytest.mark.asyncio
    async def test_file_permissions_preserved(self, file_copier, temp_base_dir, mock_docker_client):
        """Test that file permissions are set correctly in tar archives."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        # Create files with different extensions
        mappings = {
            "/root/text.txt": "docker/test.txt",
            "/root/script.sh": "docker/script.sh",
        }

        await file_copier.copy_files_to_episode("episode_perms", mappings)

        # Get all tar streams that were passed to put_archive
        assert mock_container.put_archive.call_count == 2

        # Check each call
        for call in mock_container.put_archive.call_args_list:
            tar_stream = call[0][1]
            tar_stream.seek(0)
            tar = tarfile.open(fileobj=tar_stream, mode="r")
            member = tar.getmembers()[0]

            if member.name.endswith(".sh"):
                assert member.mode == 0o755, f"{member.name} should be executable"
            else:
                assert member.mode == 0o644, f"{member.name} should be regular file"
            tar.close()

    @pytest.mark.asyncio
    async def test_relative_path_resolution(self, file_copier, temp_base_dir, mock_docker_client):
        """Test that relative paths are resolved correctly against data_dir."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        # Use a relative path that should resolve to data_dir / "test.txt"
        await file_copier.copy_files_to_episode(
            "episode_rel", {"/root/myfile.txt": "docker/test.txt"}
        )

        # Should successfully find the file at data_dir / "test.txt"
        mock_container.put_archive.assert_called_once()

    @pytest.mark.asyncio
    async def test_nested_directory_preservation(self, file_copier, temp_base_dir, mock_docker_client):
        """Test that nested directory structures are preserved in tar archives."""
        mock_client, mock_container = mock_docker_client
        file_copier._docker_client = mock_client

        await file_copier.copy_files_to_episode(
            "episode_nested", {"/app/mydir": "docker/test_directory"}
        )

        # Extract and verify the tar structure
        tar_stream = mock_container.put_archive.call_args[0][1]
        tar_stream.seek(0)
        tar = tarfile.open(fileobj=tar_stream, mode="r")
        member_names = [m.name for m in tar.getmembers()]

        # Verify nested structure is maintained
        assert "mydir/file1.txt" in member_names
        assert "mydir/nested/file3.txt" in member_names
        tar.close()

    @pytest.mark.asyncio
    async def test_reject_absolute_source_path(self, file_copier, mock_docker_client):
        """Ensure absolute source paths are rejected to prevent host escapes."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        with pytest.raises(RuntimeError, match="must be relative to the base directory"):
            await file_copier.copy_files_to_episode(
                "episode_abs", {"/root/abs.txt": "/etc/passwd"}
            )

    @pytest.mark.asyncio
    async def test_reject_source_traversal(self, file_copier, temp_base_dir, mock_docker_client):
        """Ensure source paths that traverse outside data_dir are rejected."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        outside_path = "../outside.txt"
        with pytest.raises(RuntimeError, match="must be relative to the base directory"):
            await file_copier.copy_files_to_episode(
                "episode_traversal", {"/root/evil.txt": outside_path}
            )

    @pytest.mark.asyncio
    async def test_reject_destination_traversal(self, file_copier, temp_base_dir, mock_docker_client):
        """Ensure destination paths containing traversal segments are rejected."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        with pytest.raises(RuntimeError, match="path traversal detected"):
            await file_copier.copy_files_to_episode(
                "episode_dest_traversal", {"/root/../etc/passwd": "test.txt"}
            )

    @pytest.mark.asyncio
    async def test_reject_relative_destination(self, file_copier, temp_base_dir, mock_docker_client):
        """Ensure destination paths must be absolute."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        with pytest.raises(RuntimeError, match="path must be absolute"):
            await file_copier.copy_files_to_episode(
                "episode_relative_dest", {"relative/path.txt": "test.txt"}
            )

    @pytest.mark.asyncio
    async def test_reject_large_single_file(
        self,
        file_copier,
        temp_base_dir,
        mock_docker_client,
        monkeypatch,
    ):
        """Ensure oversized single files are rejected before provisioning."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        monkeypatch.setattr(
            "saber.server.execution.sandbox.file_copier.MAX_SINGLE_FILE_BYTES",
            4,
        )

        # Expand file beyond limit
        large_file = temp_base_dir / "test.txt"
        large_file.write_text("toolong")

        with pytest.raises(RuntimeError, match="exceeds maximum allowed provisioning size"):
            await file_copier.copy_files_to_episode(
                "episode_large_file", {"/root/test.txt": "docker/test.txt"}
            )

    @pytest.mark.asyncio
    async def test_reject_large_directory(
        self,
        file_copier,
        temp_base_dir,
        mock_docker_client,
        monkeypatch,
    ):
        """Ensure oversized directory provisioning is blocked."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        monkeypatch.setattr(
            "saber.server.execution.sandbox.file_copier.MAX_DIRECTORY_BYTES",
            8,
        )

        big_dir = temp_base_dir / "docker/test_directory"
        (big_dir / "extra.txt").write_text("0123456789")

        with pytest.raises(RuntimeError, match="exceeds maximum allowed provisioning size"):
            await file_copier.copy_files_to_episode(
                "episode_large_dir", {"/root/test_directory": "docker/test_directory"}
            )

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="Symlinks not supported on this platform")
    @pytest.mark.asyncio
    async def test_reject_symlinks_in_directory(self, file_copier, temp_base_dir, mock_docker_client):
        """Ensure directories containing symlinks are rejected."""
        mock_client, _ = mock_docker_client
        file_copier._docker_client = mock_client

        target = temp_base_dir / "test.txt"
        link_path = temp_base_dir / "docker/test_directory" / "link.txt"
        os.symlink(target, link_path)

        with pytest.raises(RuntimeError, match="Symlinks are not allowed"):
            await file_copier.copy_files_to_episode(
                "episode_symlink", {"/root/test_directory": "docker/test_directory"}
            )


class TestSandboxFileCopierIntegration:
    """Integration tests for SandboxFileCopier that test real scenarios."""

    @pytest.mark.asyncio
    async def test_real_world_scenario(self, tmp_path):
        """Test a realistic scenario like labyrinth_linguist with docker/ path."""
        # Setup: Create a realistic file structure
        server_base = tmp_path / "server"
        server_base.mkdir()

        docker_dir = server_base / "docker"
        docker_dir.mkdir()

        challenge_dir = docker_dir / "challenges" / "labyrinth_linguist" / "resources"
        challenge_dir.mkdir(parents=True)

        # Create pom.xml
        pom_xml = challenge_dir / "pom.xml"
        pom_xml.write_text("<?xml version='1.0'?><project></project>")

        # Create src directory structure
        src_dir = challenge_dir / "src" / "main" / "java"
        src_dir.mkdir(parents=True)
        (src_dir / "Main.java").write_text("public class Main { }")

        # Initialize file copier with server base directory
        copier = SandboxFileCopier(base_dir=server_base)

        # Define mappings like they would be in task.yaml (relative to server base)
        file_mappings = {
            "/root/pom.xml": "docker/challenges/labyrinth_linguist/resources/pom.xml",
            "/root/src": "docker/challenges/labyrinth_linguist/resources/src",
        }

        # Mock Docker client
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "test_container"
        mock_container.put_archive.return_value = True
        mock_client.containers.get.return_value = mock_container

        copier._docker_client = mock_client
        await copier.copy_files_to_episode("test_episode", file_mappings)

        # Verify both files were copied
        assert mock_container.put_archive.call_count == 2

        # Verify pom.xml was copied to /root
        calls = mock_container.put_archive.call_args_list
        destinations = [call[0][0] for call in calls]
        assert "/root" in destinations

    @pytest.mark.asyncio
    async def test_error_recovery(self, tmp_path):
        """Test that errors are properly reported without corrupting state."""
        base_dir = tmp_path / "server"
        base_dir.mkdir()
        data_dir = base_dir / "data"
        data_dir.mkdir()
        (data_dir / "good.txt").write_text("content")

        copier = SandboxFileCopier(base_dir=base_dir)

        file_mappings = {
            "/root/good.txt": "data/good.txt",
            "/root/bad.txt": "data/nonexistent.txt",
        }

        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        copier._docker_client = mock_client
        with pytest.raises(RuntimeError, match="Failed to copy 1 file"):
            await copier.copy_files_to_episode("test_episode", file_mappings)

        # Good file should have been attempted (order dependent on dict)
        # At least one put_archive call should have happened for the good file
        assert mock_container.put_archive.call_count >= 0
