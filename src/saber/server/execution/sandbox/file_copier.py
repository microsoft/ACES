"""Sandbox file provisioning for copying files into Docker containers.

This module provides functionality to copy files from the server's base directory
into running Docker containers at episode startup time. Files can be from any
subdirectory (docker/, data/, etc.) as specified in task configuration.

Logging category: ``LogCategory.EXECUTION``.
"""

import asyncio
import io
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Dict, Optional

from docker.errors import APIError, NotFound

from docker import client as docker_client
from saber.logging_config import LogCategory, get_saber_logger

if TYPE_CHECKING:
    from docker.client import DockerClient
    from docker.models.containers import Container

logger = get_saber_logger(LogCategory.EXECUTION, __name__)

# Hard limits to prevent resource exhaustion during provisioning.
MAX_SINGLE_FILE_BYTES = 50 * 1024 * 1024  # 50 MiB per file
MAX_DIRECTORY_BYTES = 200 * 1024 * 1024  # 200 MiB per directory copy

# File permission defaults applied to provisioned content.
EXECUTABLE_SUFFIXES = {".sh", ""}
EXECUTABLE_MODE = 0o755
REGULAR_FILE_MODE = 0o644


class SandboxFileCopier:
    """Copies files from server base directory into running containers."""

    def __init__(self, base_dir: Path):
        """
        Initialize the SandboxFileCopier.

        Args:
            base_dir: Server base directory (parent of config/, docker/, data/, logs/).
                     Task files specify paths relative to this directory.
        """
        self.base_dir = Path(base_dir).resolve()
        self._docker_client: Optional["DockerClient"] = None

        logger.info(
            "SandboxFileCopier initialized",
            extra={
                "event": "file_copier_initialized",
                "base_dir": str(base_dir),
            },
        )

    @property
    def docker_client(self) -> "DockerClient":
        """Get or create Docker client (lazy initialization)."""
        if self._docker_client is None:
            self._docker_client = docker_client.from_env()
            logger.debug("Docker client initialized", extra={"event": "docker_client_initialized"})
        return self._docker_client

    async def copy_files_to_episode(
        self,
        episode_id: str,
        file_mappings: Dict[str, str],
        container_prefix: str = "default",
        container_name: Optional[str] = None,
    ) -> None:
        """
        Copy files into a running episode container.

        Args:
            episode_id: The episode ID (used to construct container name if container_name not provided)
            file_mappings: Dictionary mapping destination paths in container to source paths
                          relative to data_dir. Example:
                          {"/root/pom.xml": "sandbox_files/challenge/pom.xml"}
            container_prefix: Container name prefix (default: "default") - used if container_name not provided
            container_name: Full container name (optional) - if provided, overrides container_prefix

        Raises:
            NotFound: If container or source file not found
            APIError: If Docker API operation fails
            ValueError: If file_mappings contains invalid paths
        """
        if not file_mappings:
            logger.debug(
                "No files to copy",
                extra={
                    "event": "file_copy_skip_empty",
                    "episode_id": episode_id,
                },
            )
            return

        # Use provided container_name or construct from prefix
        if container_name is None:
            container_name = f"{container_prefix}-{episode_id}"

        logger.info(
            "File copy operation starting",
            extra={
                "event": "file_copy_start",
                "episode_id": episode_id,
                "container_name": container_name,
                "file_count": len(file_mappings),
                "mappings": file_mappings,
            },
        )

        try:
            # Get the container without blocking the event loop
            container = await asyncio.to_thread(self.docker_client.containers.get, container_name)
            logger.debug(
                "Container found for file copy",
                extra={
                    "event": "container_found",
                    "container_name": container_name,
                    "container_id": container.id[:12],
                },
            )

        except NotFound as e:
            logger.error(
                "Container not found for file copy",
                extra={
                    "event": "file_copy_container_not_found",
                    "episode_id": episode_id,
                    "container_name": container_name,
                    "error": str(e),
                },
            )
            raise NotFound(f"Container '{container_name}' not found. Cannot copy files.") from e

        # Copy each file/directory
        success_count = 0
        failed_files = []

        for dest_path, source_rel_path in file_mappings.items():
            try:
                await asyncio.to_thread(
                    self._copy_single_file_or_directory,
                    container,
                    dest_path,
                    source_rel_path,
                    episode_id,
                )
                success_count += 1
            except Exception as e:
                logger.error(
                    "Failed to copy file",
                    extra={
                        "event": "file_copy_failed",
                        "episode_id": episode_id,
                        "dest_path": dest_path,
                        "source_path": source_rel_path,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                failed_files.append((dest_path, str(e)))

        # Log final status
        if failed_files:
            logger.warning(
                "File copy completed with errors",
                extra={
                    "event": "file_copy_partial_failure",
                    "episode_id": episode_id,
                    "success_count": success_count,
                    "failed_count": len(failed_files),
                    "failed_files": failed_files,
                },
            )
            # Re-raise the first error for visibility
            raise RuntimeError(f"Failed to copy {len(failed_files)} file(s) to container: {failed_files[0][1]}")
        else:
            logger.info(
                "File copy operation completed successfully",
                extra={
                    "event": "file_copy_success",
                    "episode_id": episode_id,
                    "file_count": success_count,
                },
            )

    def _copy_single_file_or_directory(
        self,
        container: "Container",
        dest_path: str,
        source_rel_path: str,
        episode_id: str,
    ) -> None:
        """
        Copy a single file or directory into the container.

        Args:
            container: Docker container object
            dest_path: Destination path in container (absolute path)
            source_rel_path: Source path relative to data_dir
            episode_id: Episode ID for logging

        Raises:
            FileNotFoundError: If source file/directory not found
            APIError: If Docker API operation fails
        """
        dest_container_path = self._validate_and_normalize_container_path(dest_path)
        source_path = self._resolve_source_path(source_rel_path)

        logger.debug(
            "Copying file to container",
            extra={
                "event": "file_copy_single_start",
                "episode_id": episode_id,
                "source_path": str(source_path),
                "dest_path": dest_container_path.as_posix(),
                "is_directory": source_path.is_dir(),
            },
        )

        # Determine the parent directory where the tar should be extracted
        # Docker's put_archive extracts to the specified path, so we need the parent
        dest_parent = dest_container_path.parent.as_posix() or "/"

        # Ensure parent directory exists in container
        self._ensure_directory_exists(container, dest_parent, episode_id)

        # Create tar archive in memory
        if source_path.is_dir():
            tar_stream = self._create_tar_for_directory(source_path, dest_container_path)
        else:
            tar_stream = self._create_tar_for_file(source_path, dest_container_path)

        try:
            container.put_archive(dest_parent, tar_stream)
            logger.debug(
                "File copied successfully",
                extra={
                    "event": "file_copy_single_success",
                    "episode_id": episode_id,
                    "dest_path": dest_path,
                },
            )
        except APIError as e:
            logger.error(
                "Docker API error during file copy",
                extra={
                    "event": "file_copy_docker_error",
                    "episode_id": episode_id,
                    "dest_path": dest_path,
                    "error": str(e),
                },
            )
            raise

    def _ensure_directory_exists(
        self,
        container: "Container",
        directory_path: str,
        episode_id: str,
    ) -> None:
        """
        Ensure a directory exists in the container, creating it if necessary.

        Args:
            container: Docker container object
            directory_path: Directory path in container (absolute path)
            episode_id: Episode ID for logging
        """
        try:
            # Try to stat the directory
            exit_code, _ = container.exec_run(
                f"test -d {directory_path}",
                demux=False,
            )

            if exit_code == 0:
                # Directory exists
                logger.debug(
                    "Parent directory exists",
                    extra={
                        "event": "parent_dir_exists",
                        "episode_id": episode_id,
                        "directory": directory_path,
                    },
                )
                return

            # Directory doesn't exist, create it
            logger.info(
                "Creating parent directory in container",
                extra={
                    "event": "parent_dir_create",
                    "episode_id": episode_id,
                    "directory": directory_path,
                },
            )

            exit_code, output = container.exec_run(
                f"mkdir -p {directory_path}",
                demux=False,
            )

            if exit_code != 0:
                error_msg = output.decode() if output else "unknown error"
                raise RuntimeError(f"Failed to create directory {directory_path}: {error_msg}")

            logger.debug(
                "Parent directory created successfully",
                extra={
                    "event": "parent_dir_created",
                    "episode_id": episode_id,
                    "directory": directory_path,
                },
            )

        except Exception as e:
            logger.warning(
                "Error checking/creating parent directory",
                extra={
                    "event": "parent_dir_check_error",
                    "episode_id": episode_id,
                    "directory": directory_path,
                    "error": str(e),
                },
            )
            # Don't fail - let the put_archive fail if the directory really doesn't work

    def _create_tar_for_file(self, source_path: Path, dest_path: PurePosixPath | str) -> io.BytesIO:
        """
        Create an in-memory tar archive for a single file.

        Args:
            source_path: Path to source file
            dest_path: Destination path in container

        Returns:
            BytesIO object containing tar archive
        """
        if not isinstance(dest_path, PurePosixPath):
            dest_path = PurePosixPath(dest_path)

        file_size = source_path.stat().st_size
        self._enforce_file_size_limit(file_size, source_path)

        tar_stream = io.BytesIO()
        tar = tarfile.open(fileobj=tar_stream, mode="w")

        # Read file content
        with open(source_path, "rb") as f:
            data = f.read()

        # Create tar entry with proper metadata
        tarinfo = tarfile.TarInfo(name=dest_path.name)
        tarinfo.size = len(data)

        # Set appropriate permissions based on file extension
        if source_path.suffix in EXECUTABLE_SUFFIXES:
            tarinfo.mode = EXECUTABLE_MODE
        else:
            tarinfo.mode = REGULAR_FILE_MODE

        tarinfo.mtime = int(source_path.stat().st_mtime)

        # Add to archive
        tar.addfile(tarinfo, io.BytesIO(data))
        tar.close()

        tar_stream.seek(0)
        return tar_stream

    def _create_tar_for_directory(self, source_path: Path, dest_path: PurePosixPath | str) -> io.BytesIO:
        """
        Create an in-memory tar archive for a directory and its contents.

        Args:
            source_path: Path to source directory
            dest_path: Destination path in container

        Returns:
            BytesIO object containing tar archive
        """
        if not isinstance(dest_path, PurePosixPath):
            dest_path = PurePosixPath(dest_path)

        tar_stream = io.BytesIO()
        tar = tarfile.open(fileobj=tar_stream, mode="w")

        dest_basename = dest_path.name

        if not dest_basename:
            raise ValueError("Destination path must include a directory name")

        # Walk through directory and add all files
        total_bytes = 0
        for root, dirs, files in os.walk(source_path, followlinks=False):
            current_dir = Path(root)

            if current_dir.is_symlink():
                raise ValueError(f"Symlinks are not allowed in sandbox provisioning: {current_dir}")

            for dir_name in list(dirs):
                dir_path = current_dir / dir_name
                if dir_path.is_symlink():
                    raise ValueError(f"Symlinks are not allowed in sandbox provisioning: {dir_path}")

            for file_name in files:
                file_path = current_dir / file_name

                if file_path.is_symlink():
                    raise ValueError(f"Symlinks are not allowed in sandbox provisioning: {file_path}")

                rel_path = file_path.relative_to(source_path)
                arcname_path = PurePosixPath(dest_basename) / PurePosixPath(rel_path.as_posix())

                file_size = file_path.stat().st_size
                self._enforce_file_size_limit(file_size, file_path)
                total_bytes += file_size
                self._enforce_directory_size_limit(total_bytes, source_path)

                # Read file content
                with open(file_path, "rb") as f:
                    data = f.read()

                # Create tar entry
                tarinfo = tarfile.TarInfo(name=str(arcname_path))
                tarinfo.size = len(data)

                # Set appropriate permissions
                if file_path.suffix in EXECUTABLE_SUFFIXES:
                    tarinfo.mode = EXECUTABLE_MODE
                else:
                    tarinfo.mode = REGULAR_FILE_MODE

                tarinfo.mtime = int(file_path.stat().st_mtime)

                tar.addfile(tarinfo, io.BytesIO(data))

        tar.close()
        tar_stream.seek(0)
        return tar_stream

    def close(self) -> None:
        """Close the Docker client connection."""
        if self._docker_client is not None:
            self._docker_client.close()
            self._docker_client = None
            logger.debug("Docker client closed", extra={"event": "docker_client_closed"})

    def _resolve_source_path(self, source_rel_path: str) -> Path:
        """Validate and resolve a source path against the base directory."""

        if not isinstance(source_rel_path, str) or not source_rel_path.strip():
            raise ValueError("Source path must be a non-empty string relative to the base directory")

        source_rel = Path(source_rel_path)
        if source_rel.is_absolute():
            raise ValueError(f"Source path '{source_rel_path}' must be relative to the base directory")

        candidate = (self.base_dir / source_rel).resolve()

        if not self._is_within_base_dir(candidate):
            raise ValueError(f"Source path '{source_rel_path}' must be relative to the base directory")

        if not candidate.exists():
            raise FileNotFoundError(f"Source file not found: {candidate} (resolved from {source_rel_path})")

        return candidate

    def _validate_and_normalize_container_path(self, dest_path: str) -> PurePosixPath:
        """Ensure destination path is absolute and free of traversal characters."""

        if not isinstance(dest_path, str) or not dest_path.strip():
            raise ValueError("Destination path must be a non-empty absolute container path")

        container_path = PurePosixPath(dest_path)

        if not container_path.is_absolute():
            raise ValueError(f"Invalid container destination path '{dest_path}': path must be absolute")

        if container_path == PurePosixPath("/"):
            raise ValueError("Invalid container destination path '/': cannot target container root")

        if any(part in {"..", ""} for part in container_path.parts if part not in {"/"}):
            raise ValueError(f"Invalid container destination path '{dest_path}': path traversal detected")

        return container_path

    def _is_within_base_dir(self, candidate: Path) -> bool:
        """Return True if candidate resides within the configured base directory."""

        try:
            candidate.relative_to(self.base_dir)
            return True
        except ValueError:
            return False

    def _enforce_file_size_limit(self, file_size: int, path: Path) -> None:
        """Ensure a single file does not exceed the configured size limit."""

        if file_size > MAX_SINGLE_FILE_BYTES:
            raise ValueError(
                f"File '{path}' exceeds maximum allowed provisioning size of {MAX_SINGLE_FILE_BYTES} bytes"
            )

    def _enforce_directory_size_limit(self, total_bytes: int, source_path: Path) -> None:
        """Ensure aggregate directory size stays within limits."""

        if total_bytes > MAX_DIRECTORY_BYTES:
            raise ValueError(
                f"Directory '{source_path}' exceeds maximum allowed provisioning size of {MAX_DIRECTORY_BYTES} bytes"
            )
