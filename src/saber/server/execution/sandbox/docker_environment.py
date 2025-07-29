"""
Docker execution environment for secure command execution.

This module provides Docker-based execution environments that isolate
command execution in secure containers.
"""

import io
import logging
import os
import tarfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from docker.errors import APIError, ImageNotFound

import docker

from ..exceptions import ContainerCommunicationError, ContainerCreationError, SandboxExecutionError

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of command execution in container."""

    exit_code: int
    stdout: str
    stderr: str
    execution_time: float


class DockerExecutionEnvironment:
    """
    Docker-based execution environment for secure command execution.

    Provides isolated container environment for executing commands with
    security restrictions and resource limits.
    """

    def __init__(self, session_id: str, sandbox_config: Dict[str, Any]) -> None:
        """
        Initialize Docker execution environment.

        Args:
            session_id: Unique session identifier
            sandbox_config: Sandbox configuration dictionary

        Raises:
            ContainerCreationError: If Docker client cannot be initialized
        """
        self.session_id = session_id
        self.sandbox_config = sandbox_config
        self.container: Optional[Any] = None
        self._container_id: Optional[str] = None

        try:
            self.docker_client = docker.from_env()  # type: ignore
        except Exception as e:
            raise ContainerCreationError(f"Failed to initialize Docker client: {e}")

        # Extract configuration
        self.image = sandbox_config.get("image", "saber/base-sandbox:latest")
        self.working_dir = sandbox_config.get("working_dir", "/workspace")
        self.network_mode = sandbox_config.get("network_mode", "none")
        self.read_only_root = sandbox_config.get("read_only_root", True)
        self.user = sandbox_config.get("user", "tooluser:tooluser")
        self.resource_limits = sandbox_config.get("resource_limits", {})
        self.tmpfs = sandbox_config.get("tmpfs", [])
        self.volumes = sandbox_config.get("volumes", [])

        logger.info(f"Docker environment initialized for session {session_id}")

    def start(self) -> None:
        """
        Create and start the container.

        Raises:
            ContainerCreationError: If container cannot be created or started
        """
        try:
            # Prepare container configuration
            container_config = {
                "image": self.image,
                "name": f"saber-session-{self.session_id}",
                "working_dir": self.working_dir,
                "network_mode": self.network_mode,
                "read_only": self.read_only_root,
                "user": self.user,
                "detach": True,
                "stdin_open": True,
                "tty": True,
                "remove": False,  # We'll remove manually for cleanup control
            }

            # Add resource limits
            if self.resource_limits:
                mem_limit = self.resource_limits.get("memory")
                cpu_quota = self.resource_limits.get("cpu")
                if mem_limit:
                    container_config["mem_limit"] = mem_limit
                if cpu_quota:
                    # Convert CPU limit to quota (e.g., "0.5" -> 50000)
                    container_config["cpu_quota"] = int(float(cpu_quota) * 100000)
                    container_config["cpu_period"] = 100000

            # Add tmpfs mounts
            if self.tmpfs:
                tmpfs_dict = {}
                for tmpfs_mount in self.tmpfs:
                    if ":" in tmpfs_mount:
                        path, options = tmpfs_mount.split(":", 1)
                        tmpfs_dict[path] = options
                    else:
                        tmpfs_dict[tmpfs_mount] = ""
                container_config["tmpfs"] = tmpfs_dict

            # Add volumes
            if self.volumes:
                for volume in self.volumes:
                    if volume.get("type") == "tmpfs":
                        target = volume.get("target")
                        options = volume.get("options", "")
                        if target:
                            if "tmpfs" not in container_config:
                                container_config["tmpfs"] = {}
                            container_config["tmpfs"][target] = options

            # Create and start container
            logger.info(f"Creating container for session {self.session_id}")
            self.container = self.docker_client.containers.run(**container_config)
            if self.container is None:
                raise ContainerCreationError("Failed to create container")
            self._container_id = self.container.id

            # Wait for container to be ready
            self.container.reload()
            if self.container.status != "running":
                raise ContainerCreationError(f"Container failed to start: {self.container.status}")

            logger.info(f"Container {self._container_id[:12]} started for session {self.session_id}")

        except ImageNotFound as e:
            raise ContainerCreationError(f"Docker image not found: {self.image}. {e}")
        except APIError as e:
            raise ContainerCreationError(f"Docker API error creating container: {e}")
        except Exception as e:
            raise ContainerCreationError(f"Unexpected error creating container: {e}")

    def stop(self) -> None:
        """
        Stop and remove the container.

        Raises:
            ContainerCommunicationError: If container cannot be stopped or removed
        """
        if not self.container:
            logger.warning(f"No container to stop for session {self.session_id}")
            return

        try:
            container_id = self._container_id[:12] if self._container_id else "unknown"
            logger.info(f"Stopping container {container_id} for session {self.session_id}")

            # Stop the container
            self.container.stop(timeout=10)

            # Remove the container
            self.container.remove()

            logger.info(f"Container {container_id} stopped and removed for session {self.session_id}")

        except APIError as e:
            logger.error(f"Docker API error stopping container: {e}")
            raise ContainerCommunicationError(f"Docker API error stopping container: {e}")
        except Exception as e:
            # Log but don't raise - cleanup should be best effort
            logger.error(f"Error stopping container for session {self.session_id}: {e}")
        finally:
            self.container = None
            self._container_id = None

    async def execute_command(self, command: List[str], working_dir: str = "/workspace") -> CommandResult:
        """
        Execute command in the container.

        Args:
            command: Command and arguments to execute
            working_dir: Working directory for command execution

        Returns:
            CommandResult with execution results

        Raises:
            SandboxExecutionError: If container is not running or command execution fails
            ContainerCommunicationError: If communication with container fails
        """
        if not self.container:
            raise SandboxExecutionError("Container not started")

        try:
            # Refresh container status
            self.container.reload()
            if self.container.status != "running":
                raise SandboxExecutionError(f"Container not running: {self.container.status}")

            start_time = time.time()

            # Execute command
            logger.debug(
                f"Executing command in container\
                    {self._container_id[:12] if self._container_id else 'unknown'}: {' '.join(command)}"
            )

            exec_result = self.container.exec_run(
                cmd=command,
                workdir=working_dir,
                user=self.user,
                environment={},  # Use container's default environment
                stdout=True,
                stderr=True,
                stdin=False,
                tty=False,
                privileged=False,
                demux=True,  # Separate stdout and stderr
            )

            execution_time = time.time() - start_time

            # Handle output
            if exec_result.output:
                if isinstance(exec_result.output, tuple):
                    # Demuxed output (stdout, stderr)
                    stdout, stderr = exec_result.output
                    stdout = stdout.decode("utf-8", errors="replace") if stdout else ""
                    stderr = stderr.decode("utf-8", errors="replace") if stderr else ""
                else:
                    # Combined output
                    stdout = exec_result.output.decode("utf-8", errors="replace")
                    stderr = ""
            else:
                stdout = ""
                stderr = ""

            result = CommandResult(
                exit_code=exec_result.exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
            )

            logger.debug(f"Command completed in {execution_time:.2f}s with exit code {exec_result.exit_code}")
            return result

        except APIError as e:
            logger.error(f"Docker API error executing command: {e}")
            raise ContainerCommunicationError(f"Docker API error executing command: {e}")
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise SandboxExecutionError(f"Command execution failed: {e}")

    def copy_to_container(self, host_path: str, container_path: str) -> None:
        """
        Copy file from host to container.

        Args:
            host_path: Path on host system
            container_path: Path in container

        Raises:
            ContainerCommunicationError: If copy operation fails
        """
        if not self.container:
            raise SandboxExecutionError("Container not started")

        try:
            # Create tar archive of the file
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(host_path, arcname=os.path.basename(container_path))

            tar_stream.seek(0)

            # Copy to container
            self.container.put_archive(path=os.path.dirname(container_path), data=tar_stream.getvalue())

            logger.debug(
                f"Copied {host_path} to {container_path} in container\
                    {self._container_id[:12] if self._container_id else 'unknown'}"
            )

        except APIError as e:
            logger.error(f"Docker API error copying to container: {e}")
            raise ContainerCommunicationError(f"Docker API error copying to container: {e}")
        except Exception as e:
            logger.error(f"Failed to copy file to container: {e}")
            raise ContainerCommunicationError(f"Failed to copy file to container: {e}")

    def copy_from_container(self, container_path: str, host_path: str) -> None:
        """
        Copy file from container to host.

        Args:
            container_path: Path in container
            host_path: Path on host system

        Raises:
            ContainerCommunicationError: If copy operation fails
        """
        if not self.container:
            raise SandboxExecutionError("Container not started")

        try:

            # Get archive from container
            bits, _ = self.container.get_archive(container_path)

            # Extract file from archive
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)

            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                tar.extractall(path=os.path.dirname(host_path))

            logger.debug(
                f"Copied {container_path} from container\
                    {self._container_id[:12] if self._container_id else 'unknown'} to {host_path}"
            )

        except APIError as e:
            logger.error(f"Docker API error copying from container: {e}")
            raise ContainerCommunicationError(f"Docker API error copying from container: {e}")
        except Exception as e:
            logger.error(f"Failed to copy file from container: {e}")
            raise ContainerCommunicationError(f"Failed to copy file from container: {e}")

    def get_container_id(self) -> str:
        """
        Get container ID.

        Returns:
            Container ID or empty string if not started

        """
        return self._container_id or ""

    def is_healthy(self) -> bool:
        """
        Check if container is healthy and running.

        Returns:
            True if container is running, False otherwise
        """
        if not self.container:
            return False

        try:
            self.container.reload()
            return bool(self.container.status == "running")
        except Exception as e:
            logger.warning(f"Error checking container health: {e}")
            return False
