"""
Docker sandbox environment for multi-container orchestration.

This module provides Docker Compose-based execution environments that support
both single container and multi-container scenarios for complex security tasks.
"""

import asyncio
import io
import os
import subprocess
import tarfile
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import docker
import yaml

if TYPE_CHECKING:
    from docker.models.containers import Container
else:
    try:
        from docker.models.containers import Container
    except ImportError:
        Container = Any

from ....logging_config import (
    get_docker_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
    log_timeout,
)
from ...base import CommandResult
from ..exceptions import ContainerCreationError, SandboxExecutionError
from ..logging import ContainerLoggingManager
from .environment_spec import EnvironmentSpec

logger = get_docker_logger(__name__)


class DockerSandboxEnvironment:
    """
    Docker Compose-based sandbox environment for multi-container orchestration.

    Supports both single container execution and complex multi-service environments
    using Docker Compose for coordinated container management.
    """

    def __init__(
        self,
        session_id: str,
        environment_spec: EnvironmentSpec,
        container_logging_config: Optional[Dict[str, Any]] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        """
        Initialize Docker sandbox environment.

        Args:
            session_id: Unique session identifier
            environment_spec: Environment specification for container orchestration
            container_logging_config: Configuration for container logging
            episode_id: Optional episode identifier for unique container naming

        Raises:
            ContainerCreationError: If Docker client cannot be initialized
        """
        self.session_id = session_id
        self.episode_id = episode_id
        self.environment_spec = environment_spec
        self.active_services: Dict[str, Container] = {}
        self.compose_project_name = f"saber-session-{session_id}"
        self.compose_file_path: Optional[str] = None

        # Initialize container logging manager
        if container_logging_config:
            self.container_logger = ContainerLoggingManager(container_logging_config)
        else:
            # Use default config with session-specific identifier
            default_config = {"domain": "sandbox", "session_id": session_id, "logs_directory": "/app/logs"}
            self.container_logger = ContainerLoggingManager(default_config)

        try:
            self.docker_client = docker.from_env()  # type: ignore
        except Exception as e:
            raise ContainerCreationError(f"Failed to initialize Docker client: {e}")

        # Validate environment specification
        self.environment_spec.validate()

        logger.info("Docker sandbox environment initialized", session_id)

    def start(self) -> None:
        """
        Create and start the multi-container environment using Docker Compose.

        Raises:
            ContainerCreationError: If environment cannot be created or started
        """
        try:
            # Generate Docker Compose configuration
            compose_config = self.environment_spec.to_compose_dict(
                session_id=self.session_id, episode_id=self.episode_id
            )

            # Set project name
            compose_config["name"] = self.compose_project_name

            # Log the docker-compose configuration for debugging
            self.container_logger.log_compose_config(
                compose_config=compose_config,
                config_type="sandbox",
                identifier=self.session_id,
                additional_metadata={
                    "environment_spec_type": type(self.environment_spec).__name__,
                    "services_count": len(compose_config.get("services", {})),
                    "networks_count": len(compose_config.get("networks", {})),
                    "project_name": self.compose_project_name,
                },
            )

            # Log container lifecycle event
            self.container_logger.log_container_lifecycle_event(
                event_type="start_attempt",
                container_info={
                    "session_id": self.session_id,
                    "project_name": self.compose_project_name,
                    "config_type": "sandbox",
                    "services": list(compose_config.get("services", {}).keys()),
                },
            )

            # Write compose file to temporary location
            self.compose_file_path = self._write_compose_file(compose_config)

            # Start services using docker-compose
            self._start_compose_services()

            # Track service containers
            self._track_service_containers()

            # Wait for services to be healthy
            self._wait_for_services_healthy()

            # Log successful start and collect initial logs
            self.container_logger.log_container_lifecycle_event(
                event_type="start_success",
                container_info={
                    "session_id": self.session_id,
                    "project_name": self.compose_project_name,
                    "config_type": "sandbox",
                },
            )

            # Collect initial container logs
            self.container_logger.log_all_project_containers(
                project_name=self.compose_project_name, config_type="sandbox"
            )

            logger.info("Docker sandbox environment started", self.session_id)

        except Exception as e:
            # Log the failure
            self.container_logger.log_container_lifecycle_event(
                event_type="start_failure",
                container_info={
                    "session_id": self.session_id,
                    "project_name": self.compose_project_name,
                    "config_type": "sandbox",
                },
                additional_data={"error": str(e)},
            )

            # Try to collect any available logs even on failure
            self.container_logger.log_all_project_containers(
                project_name=self.compose_project_name, config_type="sandbox"
            )

            self._cleanup_compose_file()
            raise ContainerCreationError(f"Failed to start sandbox environment: {e}")

    async def execute_command(self, command: List[str], timeout: int = 300) -> CommandResult:
        """
        Execute command in the execution container asynchronously.

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

            # Execute command in container asynchronously with timeout
            # This prevents blocking the event loop and allows proper timeout handling
            try:
                log_operation_start(
                    logger, "Command execution", self.session_id, timeout=timeout, command=" ".join(command)
                )
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,  # Use default thread pool
                        lambda: execution_container.exec_run(
                            command,
                            tty=False,
                            stdout=True,
                            stderr=True,
                            stream=False,
                            demux=True,
                        ),
                    ),
                    timeout=timeout,
                )
                execution_time = time.time() - start_time
                log_operation_success(
                    logger, "Command execution", self.session_id, execution_time=f"{execution_time:.2f}s"
                )
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                log_timeout(
                    logger,
                    "Command execution",
                    timeout,
                    self.session_id,
                    execution_time=f"{execution_time:.2f}s",
                    command=" ".join(command),
                )
                # Return a CommandResult instead of raising an exception
                timeout_message = (
                    f"Command execution timed out after {timeout} seconds. Commands are limited to "
                    f"{timeout} second execution time. The command was automatically terminated to "
                    f"prevent blocking. Consider breaking down long-running operations into smaller "
                    f"steps or using shorter commands."
                )
                return CommandResult(
                    exit_code=124,  # Standard timeout exit code
                    stdout="",
                    stderr=timeout_message,
                    execution_time=execution_time,
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
            logger.debug(f"Service {service_name} not found in active_services: {list(self.active_services.keys())}")
            return False

        try:
            container.reload()

            # Log container status details
            logger.debug(f"Service {service_name} container status: {container.status}")
            logger.debug(f"Service {service_name} container ID: {container.short_id}")

            # Check if container is running
            if container.status != "running":
                logger.debug(f"Service {service_name} is not running (status: {container.status})")
                return False

            # Check health status if available
            health = container.attrs.get("State", {}).get("Health", {})
            if health:
                status = health.get("Status")
                logger.debug(f"Service {service_name} has health check - status: {status}")

                # If there's a health check, it must be healthy
                is_healthy = str(status) == "healthy" if status is not None else True
                logger.debug(f"Service {service_name} health check result: {is_healthy}")
                return is_healthy
            else:
                logger.debug(f"Service {service_name} has no health check defined - considering healthy since running")

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
        from ....logging_config import get_cleanup_logger

        cleanup_logger = get_cleanup_logger(__name__)

        cleanup_logger.info("Docker environment stop initiated", self.session_id)
        try:
            # Log stop attempt
            self.container_logger.log_container_lifecycle_event(
                event_type="stop_attempt",
                container_info={
                    "session_id": self.session_id,
                    "project_name": self.compose_project_name,
                    "config_type": "sandbox",
                },
            )

            # Collect final logs before stopping containers
            cleanup_logger.info("Collecting final container logs", self.session_id)
            self.container_logger.log_all_project_containers(
                project_name=self.compose_project_name, config_type="sandbox"
            )

            cleanup_logger.info("Stopping Docker Compose services", self.session_id)
            # Stop Docker Compose services
            self._stop_compose_services()

            cleanup_logger.info("Clearing container tracking", self.session_id)
            # Clear tracked containers
            self.active_services.clear()

            cleanup_logger.info("Cleaning up temporary files", self.session_id)
            # Clean up compose file
            self._cleanup_compose_file()

            # Log successful stop
            self.container_logger.log_container_lifecycle_event(
                event_type="stop_success",
                container_info={
                    "session_id": self.session_id,
                    "project_name": self.compose_project_name,
                    "config_type": "sandbox",
                },
            )

            logger.info("Docker sandbox environment stopped", self.session_id)

        except Exception as e:
            # Log the failure
            self.container_logger.log_container_lifecycle_event(
                event_type="stop_failure",
                container_info={
                    "session_id": self.session_id,
                    "project_name": self.compose_project_name,
                    "config_type": "sandbox",
                },
                additional_data={"error": str(e)},
            )

            log_operation_failure(cleanup_logger, "Docker environment stop", str(e), self.session_id)
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

        # Check if modern docker compose is available
        try:
            check_cmd = ["docker", "compose", "version"]
            subprocess.run(check_cmd, capture_output=True, text=True, check=True, timeout=10)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            raise ContainerCreationError(
                "Docker Compose plugin not available. Please ensure Docker Compose v2+ is installed. "
                "Modern Docker installations should include the 'docker compose' command "
                "(not legacy 'docker-compose'). Install with: apt-get install docker-compose-plugin"
            )

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
        from ....logging_config import get_cleanup_logger

        cleanup_logger = get_cleanup_logger(__name__)

        if not self.compose_file_path:
            cleanup_logger.warning("No compose file available for cleanup", self.session_id)
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

        log_operation_start(cleanup_logger, "Docker Compose down", self.session_id, command=" ".join(cmd))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            log_operation_success(cleanup_logger, "Docker Compose down", self.session_id, return_code=result.returncode)
            logger.debug(f"Docker Compose down output: {result.stdout}")
            if result.stderr:
                cleanup_logger.warning(f"Docker Compose stderr: {result.stderr}", self.session_id)

        except subprocess.TimeoutExpired:
            log_timeout(cleanup_logger, "Docker Compose down", 60, self.session_id)
        except Exception as e:
            log_operation_failure(cleanup_logger, "Docker Compose down", str(e), self.session_id)

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
