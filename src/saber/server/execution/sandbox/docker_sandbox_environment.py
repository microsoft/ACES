"""
Docker sandbox environment for multi-container orchestration.

This module provides Docker Compose-based execution environments that support
both single container and multi-container scenarios for complex security tasks.
"""

import io
import logging
import os
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml
from docker.models.containers import Container

import docker

from ..exceptions import ContainerCreationError, SandboxExecutionError
from .environment_spec import EnvironmentSpec

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of command execution in container."""

    exit_code: int
    stdout: str
    stderr: str
    execution_time: float


class DockerSandboxEnvironment:
    """
    Docker Compose-based sandbox environment for multi-container orchestration.

    Supports both single container execution and complex multi-service environments
    using Docker Compose for coordinated container management.
    """

    def __init__(self, session_id: str, environment_spec: EnvironmentSpec) -> None:
        """
        Initialize Docker sandbox environment.

        Args:
            session_id: Unique session identifier
            environment_spec: Environment specification for container orchestration

        Raises:
            ContainerCreationError: If Docker client cannot be initialized
        """
        self.session_id = session_id
        self.environment_spec = environment_spec
        self.active_services: Dict[str, Container] = {}
        self.compose_project_name = f"saber-session-{session_id}"
        self.compose_file_path: Optional[str] = None

        try:
            self.docker_client = docker.from_env()  # type: ignore
        except Exception as e:
            raise ContainerCreationError(f"Failed to initialize Docker client: {e}")

        # Validate environment specification
        self.environment_spec.validate()

        logger.info(f"Docker sandbox environment initialized for session {session_id}")

    def start(self) -> None:
        """
        Create and start the multi-container environment using Docker Compose.

        Raises:
            ContainerCreationError: If environment cannot be created or started
        """
        try:
            # Generate Docker Compose configuration
            compose_config = self.environment_spec.to_compose_dict()

            # Set project name
            compose_config["name"] = self.compose_project_name

            # Write compose file to temporary location
            self.compose_file_path = self._write_compose_file(compose_config)

            # Start services using docker-compose
            self._start_compose_services()

            # Track service containers
            self._track_service_containers()

            # Wait for services to be healthy
            self._wait_for_services_healthy()

            logger.info(f"Docker sandbox environment started for session {self.session_id}")

        except Exception as e:
            self._cleanup_compose_file()
            raise ContainerCreationError(f"Failed to start sandbox environment: {e}")

    def execute_command(self, command: List[str], timeout: int = 300) -> CommandResult:
        """
        Execute command in the execution container.

        Args:
            command: Command to execute as list of strings
            timeout: Execution timeout in seconds

        Returns:
            CommandResult with execution details

        Raises:
            SandboxExecutionError: If command execution fails
        """
        if not self.active_services:
            raise SandboxExecutionError("Environment not started")

        execution_container = self.get_execution_container()
        if not execution_container:
            raise SandboxExecutionError("Execution container not available")

        try:
            start_time = time.time()

            # Execute command in container
            # Note: exec_run doesn't support timeout parameter directly
            result = execution_container.exec_run(
                command,
                tty=False,
                stdout=True,
                stderr=True,
                stream=False,
                demux=True,
            )

            execution_time = time.time() - start_time  # Process output
            stdout = ""
            stderr = ""

            if result.output:
                if isinstance(result.output, tuple):
                    stdout_bytes, stderr_bytes = result.output
                    stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
                    stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""
                else:
                    stdout = result.output.decode("utf-8") if result.output else ""

            return CommandResult(
                exit_code=result.exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
            )

        except Exception as e:
            raise SandboxExecutionError(f"Command execution failed: {e}")

    def copy_to_container(self, src_path: str, dst_path: str) -> None:
        """
        Copy file from host to execution container.

        Args:
            src_path: Source file path on host
            dst_path: Destination path in container

        Raises:
            SandboxExecutionError: If copy operation fails
        """
        execution_container = self.get_execution_container()
        if not execution_container:
            raise SandboxExecutionError("Execution container not available")

        try:
            with open(src_path, "rb") as f:
                file_data = f.read()

            tar_buffer = io.BytesIO()
            tar = tarfile.open(fileobj=tar_buffer, mode="w")

            # Add file to tar
            tarinfo = tarfile.TarInfo(name=os.path.basename(dst_path))
            tarinfo.size = len(file_data)
            tar.addfile(tarinfo, io.BytesIO(file_data))
            tar.close()

            tar_buffer.seek(0)

            # Extract to container
            container_dir = os.path.dirname(dst_path)
            if not container_dir:
                container_dir = "/"

            execution_container.put_archive(container_dir, tar_buffer)

        except Exception as e:
            raise SandboxExecutionError(f"Failed to copy file to container: {e}")

    def copy_from_container(self, src_path: str, dst_path: str) -> None:
        """
        Copy file from execution container to host.

        Args:
            src_path: Source file path in container
            dst_path: Destination path on host

        Raises:
            SandboxExecutionError: If copy operation fails
        """
        execution_container = self.get_execution_container()
        if not execution_container:
            raise SandboxExecutionError("Execution container not available")

        try:
            logger.debug(f"Getting archive for {src_path}")
            # Get tar archive from container
            archive_data, _ = execution_container.get_archive(src_path)

            # Extract file from tar
            import io
            import tarfile

            # Combine archive chunks
            archive_bytes = b"".join(archive_data)
            logger.debug(f"Archive size: {len(archive_bytes)} bytes")
            tar_buffer = io.BytesIO(archive_bytes)

            tar = tarfile.open(fileobj=tar_buffer, mode="r")

            # Extract the first file (should be our file)
            members = tar.getmembers()
            logger.debug(f"Found {len(members)} members in tar")
            if not members:
                raise SandboxExecutionError(f"No files found in archive for {src_path}")

            # Get the first file member
            file_member = members[0]
            logger.debug(f"First member name: {file_member.name}, size: {file_member.size}")

            # Extract file content
            file_obj = tar.extractfile(file_member)
            if file_obj:
                logger.debug(f"Writing to {dst_path}")
                with open(dst_path, "wb") as f:
                    data = file_obj.read()
                    logger.debug(f"Writing {len(data)} bytes")
                    f.write(data)
                logger.debug("File written successfully")
            else:
                raise SandboxExecutionError(f"Could not extract file {src_path}")

            tar.close()

        except Exception as e:
            logger.debug(f"Exception in copy_from_container: {e}")
            raise SandboxExecutionError(f"Failed to copy file from container: {e}")

    def get_execution_container(self) -> Optional[Container]:
        """Get the execution container for command execution."""
        execution_service = self.environment_spec.get_execution_service()
        return self.active_services.get(execution_service)

    def get_service_container(self, service_name: str) -> Optional[Container]:
        """Get container for a specific service."""
        return self.active_services.get(service_name)

    def is_service_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy and running."""
        container = self.active_services.get(service_name)
        if not container:
            return False

        try:
            container.reload()

            # Check if container is running
            if container.status != "running":
                return False

            # Check health status if available
            health = container.attrs.get("State", {}).get("Health", {})
            if health:
                status = health.get("Status")
                return str(status) == "healthy" if status is not None else True

            # If no health check, consider running containers healthy
            return True

        except Exception as e:
            logger.warning(f"Error checking health for service {service_name}: {e}")
            return False

    def get_service_logs(self, service_name: str, tail: int = 100) -> str:
        """Get logs from a specific service."""
        container = self.active_services.get(service_name)
        if not container:
            return f"Service {service_name} not found"

        try:
            logs_bytes = container.logs(tail=tail, timestamps=True)
            logs = logs_bytes.decode("utf-8") if isinstance(logs_bytes, bytes) else str(logs_bytes)
            return logs
        except Exception as e:
            return f"Error getting logs for {service_name}: {e}"

    def list_active_services(self) -> List[str]:
        """Get list of active service names."""
        return list(self.active_services.keys())

    def get_service_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all services."""
        info = {}
        for service_name, container in self.active_services.items():
            try:
                container.reload()
                info[service_name] = {
                    "id": container.id,
                    "status": container.status,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "ports": container.ports,
                    "healthy": self.is_service_healthy(service_name),
                }
            except Exception as e:
                info[service_name] = {"error": str(e)}

        return info

    def stop(self) -> None:
        """Stop and clean up the sandbox environment."""
        try:
            # Stop Docker Compose services
            self._stop_compose_services()

            # Clear tracked containers
            self.active_services.clear()

            # Clean up compose file
            self._cleanup_compose_file()

            logger.info(f"Docker sandbox environment stopped for session {self.session_id}")

        except Exception as e:
            logger.error(f"Error stopping sandbox environment: {e}")
            raise SandboxExecutionError(f"Failed to stop environment: {e}")

    def _write_compose_file(self, compose_config: Dict[str, Any]) -> str:
        """Write Docker Compose configuration to temporary file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", prefix=f"saber-compose-{self.session_id}-", delete=False
        )

        try:
            yaml.dump(compose_config, temp_file, default_flow_style=False)
            temp_file.flush()
            return temp_file.name
        finally:
            temp_file.close()

    def _start_compose_services(self) -> None:
        """Start services using docker-compose."""
        if not self.compose_file_path:
            raise ContainerCreationError("Compose file not available")

        cmd = ["docker", "compose", "-f", self.compose_file_path, "-p", self.compose_project_name, "up", "-d"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            logger.debug(f"Docker Compose up output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            raise ContainerCreationError(f"Docker Compose failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise ContainerCreationError("Docker Compose startup timed out")

    def _stop_compose_services(self) -> None:
        """Stop services using docker-compose."""
        if not self.compose_file_path:
            return

        cmd = [
            "docker",
            "compose",
            "-f",
            self.compose_file_path,
            "-p",
            self.compose_project_name,
            "down",
            "--remove-orphans",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            logger.debug(f"Docker Compose down output: {result.stdout}")

        except subprocess.TimeoutExpired:
            logger.warning("Docker Compose shutdown timed out")

    def _track_service_containers(self) -> None:
        """Find and track containers for all services."""
        try:
            # Get containers by project label
            containers = self.docker_client.containers.list(
                filters={"label": f"com.docker.compose.project={self.compose_project_name}"}
            )

            for container in containers:
                service_name = container.labels.get("com.docker.compose.service")
                if service_name:
                    self.active_services[service_name] = container
                    logger.debug(f"Tracked container for service: {service_name}")

        except Exception as e:
            raise ContainerCreationError(f"Failed to track service containers: {e}")

    def _wait_for_services_healthy(self, timeout: int = 120) -> None:
        """Wait for all services to become healthy."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_healthy = True

            for service_name in self.environment_spec.get_all_services():
                if not self.is_service_healthy(service_name):
                    all_healthy = False
                    break

            if all_healthy:
                logger.info("All services are healthy")
                return

            time.sleep(2)

        # Log service status for debugging
        unhealthy_services = []
        for service_name in self.environment_spec.get_all_services():
            if not self.is_service_healthy(service_name):
                unhealthy_services.append(service_name)

        if unhealthy_services:
            logger.warning(f"Services not healthy after {timeout}s: {unhealthy_services}")

    def _cleanup_compose_file(self) -> None:
        """Clean up temporary compose file."""
        if self.compose_file_path and os.path.exists(self.compose_file_path):
            try:
                os.unlink(self.compose_file_path)
                self.compose_file_path = None
            except Exception as e:
                logger.warning(f"Failed to cleanup compose file: {e}")
