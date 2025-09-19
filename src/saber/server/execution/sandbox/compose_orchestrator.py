"""
Simple Docker Compose orchestrator for environment management.

Provides basic start/stop operations for Docker Compose files with episode isolation support.
Includes command execution capabilities for designated execution services.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..logging import ContainerLoggingManager
from .compose_health_checker import ComposeHealthChecker
from .environment_config import ComposeEnvironmentConfig

try:
    import docker
except ImportError:
    docker = None  # type: ignore

DOCKER_AVAILABLE = docker is not None

from ...base import CommandResult

logger = logging.getLogger(__name__)


class ComposeOrchestrator:
    """
    Simple Docker Compose orchestrator with command execution capabilities.

    Provides basic start/stop operations for any Docker Compose file and
    command execution via designated execution services marked with labels.
    Handles episode isolation when needed.
    """

    def __init__(self, logging_config: Optional[Dict[str, Any]] = None):
        """Initialize ComposeOrchestrator with optional container logging."""
        self.execution_service_name: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.compose_file_path: Optional[Path] = None
        self.compose_data: Optional[dict] = None
        self._docker_client = None
        self.config_type: str = "sandbox"  # Store config type for stop operations

        # Initialize container logging if config provided
        self.container_logger = None
        if logging_config:
            self.container_logger = ContainerLoggingManager(logging_config)

        # Initialize health checker for episode service verification
        self.health_checker = ComposeHealthChecker()

    @property
    def docker_client(self) -> Any:
        """Get or create Docker client."""
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker package not available. Install with: pip install docker")

        if self._docker_client is None:
            if docker is None:
                raise RuntimeError("Docker package not available. Install with: pip install docker")
            self._docker_client = docker.from_env()  # type: ignore
        return self._docker_client

    def start_environment(
        self, compose_file_path: str, config: ComposeEnvironmentConfig
    ) -> subprocess.CompletedProcess:
        """
        Start the environment using the provided compose file and configuration.

        Args:
            compose_file_path: Path to the compose file
            config: ComposeEnvironmentConfig with all required settings

        Returns:
            CompletedProcess result from docker compose up
        """
        # Set the project name for this orchestrator instance
        self.project_name = config.get_project_name()
        if not self.project_name:
            raise RuntimeError("Project name is required but not provided by config")

        self.config_type = config.config_type  # Store for stop operations
        self.episode_id = config.episode_id  # Store episode ID for container resolution
        self.compose_file_path = Path(compose_file_path)  # Store compose file path for execution

        # Convert config to environment variables
        env_vars = config.to_env_dict()

        # Apply network auto-injection for saber-episode-network and resolve environment variables
        processed_compose_path = self._inject_episode_network(compose_file_path, config, env_vars)

        # Add environment variables to the current environment
        env = os.environ.copy()
        env.update(env_vars)

        # Log lifecycle event - starting environment
        if self.container_logger:
            self.container_logger.log_container_lifecycle_event(
                event_type="start_attempt",
                container_info={
                    "project_name": self.project_name,
                    "config_type": config.config_type,
                    "compose_file": str(compose_file_path),
                    "episode_id": config.episode_id,
                },
            )

        # Log the resolved compose configuration (use processed path to include network injection)
        self._log_resolved_compose_config(processed_compose_path, env_vars)

        # Validate execution service exists before running Docker command (fail-fast)
        temp_execution_service = self._identify_execution_service(Path(processed_compose_path))

        # Run docker compose up with the processed compose file and provided environment variables
        command = ["docker", "compose", "-f", processed_compose_path, "-p", self.project_name, "up", "-d"]

        logger.info(f"Starting compose environment with project name: {self.project_name}")
        logger.debug(f"Environment variables: {env_vars}")

        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)

            # Log successful start and collect initial container logs
            if self.container_logger:
                self.container_logger.log_container_lifecycle_event(
                    event_type="start_success",
                    container_info={
                        "project_name": self.project_name,
                        "config_type": config.config_type,
                        "compose_file": str(compose_file_path),
                        "episode_id": config.episode_id,
                    },
                )

                # Collect initial container logs
                self.container_logger.log_all_project_containers(
                    project_name=self.project_name, config_type=config.config_type
                )

            # Store the execution service name that was validated earlier
            self.execution_service_name = temp_execution_service
            logger.info(f"Identified execution service: {self.execution_service_name}")

            # MANDATORY HEALTH CHECK: Wait for all services to become healthy
            # FAIL-FAST: Episode creation will fail if any service is not healthy
            logger.info(f"🔍 Starting mandatory health checks for all services in project: {self.project_name}")
            logger.info(f"🔍 Health check will use processed compose file: {processed_compose_path}")
            logger.info(f"🔍 Original compose file was: {compose_file_path}")

            # Use the processed compose file (with variables resolved) for health checks
            # NO TRY-CATCH: Health check failures will propagate up and fail episode creation
            self.health_checker.wait_for_all_services_healthy(
                compose_file_path=processed_compose_path,  # Use processed file, not original
                project_name=self.project_name,
                timeout_seconds=180,  # 3 minutes for health checks - reasonable for complex environments
                check_interval=2.0,  # Check every 2 seconds
            )
            logger.info(f"✅ All services are healthy for project: {self.project_name}")

            logger.info(f"Successfully started environment for project: {self.project_name}")
            return result
        except subprocess.CalledProcessError as e:
            # Log the failure
            if self.container_logger:
                self.container_logger.log_container_lifecycle_event(
                    event_type="start_failure",
                    container_info={
                        "project_name": self.project_name,
                        "config_type": config.config_type,
                        "compose_file": str(compose_file_path),
                        "episode_id": config.episode_id,
                    },
                    additional_data={"error": str(e)},
                )

            logger.error(f"Failed to start environment: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"Failed to start environment: {e}")
        except subprocess.TimeoutExpired as e:
            # Log timeout failure
            if self.container_logger:
                self.container_logger.log_container_lifecycle_event(
                    event_type="start_timeout",
                    container_info={
                        "project_name": self.project_name,
                        "config_type": config.config_type,
                        "compose_file": str(compose_file_path),
                        "episode_id": config.episode_id,
                    },
                    additional_data={"timeout": e.timeout},
                )

            logger.error(f"Environment start timed out: {e}")
            raise RuntimeError(f"Environment start timed out after {e.timeout} seconds")

    def _log_resolved_compose_config(self, compose_file_path: str, env_vars: Dict[str, str]) -> None:
        """
        Log the resolved compose configuration with variable substitution.

        Args:
            compose_file_path: Path to the compose file
            env_vars: Environment variables for substitution
        """
        try:
            import re

            import yaml

            # Read the compose file
            with open(compose_file_path, "r") as f:
                compose_content = f.read()

            # Resolve variables in the content
            def replace_var(match: Any) -> str:
                var_expr = match.group(1)
                if ":-" in var_expr:
                    var_name, default = var_expr.split(":-", 1)
                    return str(env_vars.get(var_name, default))
                else:
                    return str(env_vars.get(var_expr, match.group(0)))  # Return original if not found

            # Pattern to match ${VAR} or ${VAR:-default}
            pattern = r"\$\{([^}]+)\}"
            resolved_content = re.sub(pattern, replace_var, compose_content)

            # Parse and log the resolved YAML
            resolved_config = yaml.safe_load(resolved_content)

            logger.info("Resolved compose configuration:")
            logger.info(f"Project: {self.project_name}")

            # Save resolved compose file to disk
            self._save_resolved_compose_config(compose_file_path, resolved_content, resolved_config)

            # Log network configuration specifically
            if "networks" in resolved_config:
                logger.info("Networks:")
                for network_name, network_config in resolved_config["networks"].items():
                    logger.info(f"  {network_name}: {network_config}")

            # Log services and their network connections
            if "services" in resolved_config:
                logger.info("Service network connections:")
                for service_name, service_config in resolved_config["services"].items():
                    if "networks" in service_config:
                        networks = service_config["networks"]
                        if isinstance(networks, dict):
                            network_names = list(networks.keys())
                        elif isinstance(networks, list):
                            network_names = networks
                        else:
                            network_names = [str(networks)]
                        logger.info(f"  {service_name}: {network_names}")

        except Exception as e:
            logger.error(f"Could not log resolved compose config: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

    def _save_resolved_compose_config(
        self, compose_file_path: str, resolved_content: str, resolved_config: Dict[str, Any]
    ) -> None:
        """
        Save the resolved compose configuration to disk for debugging and audit purposes.

        Args:
            compose_file_path: Path to the original compose file
            resolved_content: Resolved YAML content as string
            resolved_config: Parsed resolved configuration
        """
        try:
            if not self.container_logger:
                return

            from datetime import datetime
            from pathlib import Path

            # Get logs directory from container logger
            logs_dir = Path(self.container_logger.logs_directory)
            compose_configs_dir = logs_dir / "compose-configs" / self.config_type
            compose_configs_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp and filename
            timestamp = datetime.now().isoformat()
            original_filename = Path(compose_file_path).stem

            # Create filename: timestamp_configtype_projectname_filename.yml
            safe_project_name = (
                self.project_name.replace("-", "_").replace(".", "_") if self.project_name else "unknown"
            )
            filename = f"{timestamp}_{self.config_type}_{safe_project_name}_{original_filename}.yml"

            output_file = compose_configs_dir / filename

            # Write resolved content to file
            with open(output_file, "w") as f:
                f.write("# Resolved compose configuration\n")
                f.write(f"# Original file: {compose_file_path}\n")
                f.write(f"# Project name: {self.project_name}\n")
                f.write(f"# Config type: {self.config_type}\n")
                f.write(f"# Resolved at: {timestamp}\n")
                f.write("# Variables resolved and substituted\n\n")
                f.write(resolved_content)

            logger.info(f"Saved resolved compose config to: {output_file}")

        except Exception as e:
            logger.warning(f"Could not save resolved compose config to disk: {e}")

    def _parse_compose_file(self, compose_file_path: Path) -> Dict[str, Any]:
        """
        Parse compose file and return the data structure.

        Args:
            compose_file_path: Path to Docker Compose file

        Returns:
            Parsed compose file data

        Raises:
            RuntimeError: If compose file cannot be parsed
        """
        try:
            with open(compose_file_path, "r") as f:
                compose_data = yaml.safe_load(f)

            # Handle empty or null YAML files
            if compose_data is None:
                raise RuntimeError(f"Compose file {compose_file_path} is empty or invalid")

            if not isinstance(compose_data, dict):
                raise RuntimeError(f"Compose file {compose_file_path} does not contain a valid dictionary")

            return compose_data
        except Exception as e:
            raise RuntimeError(f"Failed to parse compose file {compose_file_path}: {e}")

    def _substitute_env_vars(self, value: str) -> str:
        """
        Substitute environment variables in a string using episode_id from orchestrator.

        Supports: ${VAR}, ${VAR:-default}, ${VAR-default}
        FAIL-FAST: Does not fall back to defaults when episode_id is missing.

        Args:
            value: String that may contain environment variable references

        Returns:
            String with environment variables substituted

        Raises:
            RuntimeError: If EPISODE_ID is required but not available
        """
        import re

        def replace_var(match: Any) -> str:
            var_expr = match.group(1)

            # Handle ${VAR:-default} and ${VAR-default} patterns
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                if var_name == "EPISODE_ID":
                    if not self.episode_id:
                        raise RuntimeError("EPISODE_ID required for container naming but not set in orchestrator")
                    return self.episode_id
                return os.environ.get(var_name) or default
            elif "-" in var_expr:
                var_name, default = var_expr.split("-", 1)
                if var_name == "EPISODE_ID":
                    if not self.episode_id:
                        raise RuntimeError("EPISODE_ID required for container naming but not set in orchestrator")
                    return self.episode_id
                return os.environ.get(var_name, default)
            else:
                # Simple ${VAR} pattern
                if var_expr == "EPISODE_ID":
                    if not self.episode_id:
                        raise RuntimeError("EPISODE_ID required for container naming but not set in orchestrator")
                    return self.episode_id
                return os.environ.get(var_expr, "")

        # Match ${...} patterns
        return re.sub(r"\$\{([^}]+)\}", replace_var, value)

    def _get_actual_container_name(self, service_name: str) -> str:
        """
        Get the actual container name for a service by parsing the compose file.

        Args:
            service_name: Name of the service in compose file

        Returns:
            Actual container name that Docker will create

        Raises:
            RuntimeError: If compose file not available, service not found, or episode_id required but missing
        """
        # Ensure compose data is loaded
        if not self.compose_data:
            if not self.compose_file_path:
                raise RuntimeError("No compose file path available. Call start_environment first.")
            self.compose_data = self._parse_compose_file(self.compose_file_path)

        services = self.compose_data.get("services", {})
        service_config = services.get(service_name)

        if not service_config:
            raise RuntimeError(f"Service '{service_name}' not found in compose file")

        # Check if service has custom container_name
        container_name = service_config.get("container_name")
        if container_name:
            # Substitute environment variables (especially EPISODE_ID) - will fail if missing
            return self._substitute_env_vars(container_name)
        else:
            # Docker Compose default naming requires episode_id for episode-specific containers
            if not self.episode_id:
                raise RuntimeError(
                    f"Episode ID required for container name resolution but not set. "
                    f"Service '{service_name}' uses default Docker Compose naming which requires episode context."
                )
            project_name = f"saber-episode-{self.episode_id}"
            return f"{project_name}-{service_name}-1"

    def _identify_execution_service(self, compose_file_path: Path) -> str:
        """
        Parse compose file to identify execution service by label.

        Looks for services with 'saber.execution.service=true' label.

        Args:
            compose_file_path: Path to Docker Compose file

        Returns:
            Name of the execution service

        Raises:
            RuntimeError: If no execution service found or multiple services marked
        """
        try:
            with open(compose_file_path, "r") as f:
                compose_data = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to parse compose file {compose_file_path}: {e}")

        # Handle empty or null YAML files
        if compose_data is None:
            raise RuntimeError(f"Compose file {compose_file_path} is empty or invalid")

        if not isinstance(compose_data, dict):
            raise RuntimeError(f"Compose file {compose_file_path} does not contain a valid dictionary")

        services = compose_data.get("services", {})
        if not services:
            raise RuntimeError(f"No services found in compose file: {compose_file_path}")

        execution_services = []

        for service_name, service_config in services.items():
            labels = service_config.get("labels", [])

            # Handle both list and dict label formats
            is_execution_service = False
            if isinstance(labels, list):
                # List format: ["saber.execution.service=true", "other.label=value"]
                if "saber.execution.service=true" in labels:
                    is_execution_service = True
            elif isinstance(labels, dict):
                # Dict format: {"saber.execution.service": "true", "other.label": "value"}
                if labels.get("saber.execution.service") == "true":
                    is_execution_service = True

            if is_execution_service:
                execution_services.append(str(service_name))

        # Fail-fast validation
        if not execution_services:
            raise RuntimeError(
                f"No execution service found in {compose_file_path}. "
                f"Add 'saber.execution.service=true' label to the service that should execute commands."
            )
        if len(execution_services) > 1:
            raise RuntimeError(
                f"Multiple execution services found in {compose_file_path}: {execution_services}. "
                f"Only one service can be marked with 'saber.execution.service=true'."
            )

        return execution_services[0]

    async def execute_command(
        self, command: List[str], timeout: int = 30, working_dir: Optional[str] = None
    ) -> CommandResult:
        """
        Execute command in designated execution service container.

        Args:
            command: Command to execute as list of strings
            timeout: Command timeout in seconds
            working_dir: Working directory for command (optional)

        Returns:
            CommandResult with execution details

        Raises:
            RuntimeError: If execution service not available or command fails
        """
        # Fail-fast validations
        if not self.execution_service_name:
            raise RuntimeError("No execution service identified. Environment must be started first.")

        if not self.episode_id:
            raise RuntimeError(
                "Episode ID required for container resolution but not set in orchestrator. "
                "This indicates an improper orchestrator initialization."
            )

        # Get execution container using episode-aware resolution
        container = self.get_execution_container()
        if not container:
            expected_name = self._get_actual_container_name(self.execution_service_name)
            raise RuntimeError(
                f"Execution container '{expected_name}' not found or not running. "
                f"Episode: {self.episode_id}, Service: {self.execution_service_name}. "
                f"Ensure environment is started and container is healthy."
            )

        logger.debug(f"Executing command in container {container.name}: {' '.join(command)}")

        start_time = time.time()

        try:
            # Execute command via docker exec
            exec_result = container.exec_run(
                cmd=command,
                workdir=working_dir,
                detach=False,
                stdout=True,
                stderr=True,
                stream=False,
                demux=True,  # Separate stdout and stderr
                tty=False,
                privileged=False,
                user=None,  # Use container's default user
                environment=None,
                socket=False,
            )

            execution_time = time.time() - start_time

            # Handle demuxed output (stdout, stderr are separate)
            stdout_bytes, stderr_bytes = exec_result.output
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            # Create CommandResult
            result = CommandResult(
                stdout=stdout, stderr=stderr, exit_code=exec_result.exit_code, execution_time=execution_time
            )

            logger.debug(f"Command completed in {execution_time:.2f}s with exit code {exec_result.exit_code}")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Command execution failed: {str(e)}"
            logger.error(f"{error_msg} (after {execution_time:.2f}s)")

            # Return error result instead of raising
            return CommandResult(stdout="", stderr=error_msg, exit_code=1, execution_time=execution_time)

    def get_execution_container(self) -> Any:
        """
        Get the container designated for command execution using episode-aware resolution.

        Returns:
            Docker container object if found, None otherwise
        """
        if not self.execution_service_name:
            logger.error("No execution service identified. Environment must be started first.")
            return None

        try:
            # Get the actual container name using compose file + episode_id
            actual_container_name = self._get_actual_container_name(self.execution_service_name)
            logger.info(f"Looking for execution container: {actual_container_name}")

            # Try to get the container
            try:
                container = self.docker_client.containers.get(actual_container_name)

                # Verify container is running
                container.reload()  # Refresh container state
                if container.status != "running":
                    logger.error(
                        f"Execution container '{actual_container_name}' exists but is not running "
                        f"(status: {container.status})"
                    )
                    return None

                logger.info(f"Found running execution container: {actual_container_name}")
                return container

            except Exception as e:
                # Handle docker.errors.NotFound and other exceptions
                if docker and hasattr(docker, "errors") and isinstance(e, docker.errors.NotFound):
                    # Log available containers for debugging
                    all_containers = self.docker_client.containers.list(all=True)
                    logger.error(f"Execution container '{actual_container_name}' not found.")
                    logger.error(f"Available containers ({len(all_containers)}):")
                    for c in all_containers:
                        logger.error(f"  - {c.name} ({c.image.tags[0] if c.image.tags else 'no-tag'}) - {c.status}")
                else:
                    logger.error(f"Error finding execution container: {e}")

                return None

        except Exception as e:
            logger.error(f"Error resolving execution container: {e}")
            return None

    def stop_environment(
        self, compose_file_path: Path, episode_id: Optional[str] = None, project_name: Optional[str] = None
    ) -> None:
        """
        Stop environment from compose file.

        Args:
            compose_file_path: Path to the Docker Compose file
            episode_id: Optional episode ID that was used when starting
            project_name: Optional explicit project name (overrides episode-based naming)

        Raises:
            FileNotFoundError: If compose file doesn't exist
            RuntimeError: If docker compose command fails
        """
        if not compose_file_path.exists():
            raise FileNotFoundError(f"Compose file not found: {compose_file_path}")

        # Setup environment variables
        env = os.environ.copy()

        # Build docker compose command
        cmd = ["docker", "compose", "-f", str(compose_file_path)]

        # Determine project name (explicit project_name takes precedence over episode-based naming)
        if project_name:
            cmd.extend(["-p", project_name])
            env["COMPOSE_PROJECT_NAME"] = project_name
        elif episode_id:
            episode_project_name = f"saber-episode-{episode_id}"
            cmd.extend(["-p", episode_project_name])
            env["EPISODE_ID"] = episode_id
            env["COMPOSE_PROJECT_NAME"] = episode_project_name

        cmd.extend(["down", "--volumes", "--remove-orphans"])

        # Determine display name for logging
        if project_name:
            display_name = f" (project: {project_name})"
            active_project_name = project_name
        elif episode_id:
            display_name = f" (episode: {episode_id})"
            active_project_name = f"saber-episode-{episode_id}"
        else:
            display_name = ""
            active_project_name = None

        # Log lifecycle event - stopping environment
        if self.container_logger and active_project_name:
            self.container_logger.log_container_lifecycle_event(
                event_type="stop_attempt",
                container_info={
                    "project_name": active_project_name,
                    "config_type": self.config_type,
                    "compose_file": str(compose_file_path),
                    "episode_id": episode_id,
                },
            )

            # Collect final logs before stopping containers
            self.container_logger.log_all_project_containers(
                project_name=active_project_name, config_type=self.config_type
            )

        logger.info(f"Stopping environment: {compose_file_path}{display_name}")

        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True, timeout=60)

            logger.info(f"Environment stopped successfully: {compose_file_path}")
            if result.stdout:
                logger.debug(f"Docker compose output: {result.stdout}")

            # Log successful stop
            if self.container_logger and active_project_name:
                self.container_logger.log_container_lifecycle_event(
                    event_type="stop_success",
                    container_info={
                        "project_name": active_project_name,
                        "config_type": self.config_type,
                        "compose_file": str(compose_file_path),
                        "episode_id": episode_id,
                    },
                )

            # Clear execution state after successful stop
            self.execution_service_name = None
            self.episode_id = None
            self.compose_file_path = None
            self.compose_data = None

        except subprocess.CalledProcessError as e:
            # Log the failure
            if self.container_logger and active_project_name:
                self.container_logger.log_container_lifecycle_event(
                    event_type="stop_failure",
                    container_info={
                        "project_name": active_project_name,
                        "config_type": self.config_type,
                        "compose_file": str(compose_file_path),
                        "episode_id": episode_id,
                    },
                    additional_data={"error": str(e)},
                )

            error_msg = f"Failed to stop environment {compose_file_path}: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout stopping environment {compose_file_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def cleanup_episode(self, episode_id: str) -> None:
        """
        Clean up all containers for a specific episode.

        Args:
            episode_id: Episode ID to clean up

        Raises:
            RuntimeError: If cleanup command fails
        """
        project_name = f"saber-episode-{episode_id}"

        logger.info(f"Cleaning up episode: {episode_id}")

        try:
            # Stop and remove all containers for this episode project
            cmd = ["docker", "compose", "-p", project_name, "down", "--volumes", "--remove-orphans"]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)

            logger.info(f"Episode cleanup completed: {episode_id}")
            if result.stdout:
                logger.debug(f"Docker compose cleanup output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to cleanup episode {episode_id}: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout cleaning up episode {episode_id}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def validate_compose_file(self, compose_file_path: Path) -> None:
        """
        Validate compose file syntax.

        Args:
            compose_file_path: Path to the Docker Compose file

        Raises:
            FileNotFoundError: If compose file doesn't exist
            RuntimeError: If compose file is invalid
        """
        if not compose_file_path.exists():
            raise FileNotFoundError(f"Compose file not found: {compose_file_path}")

        try:
            cmd = ["docker", "compose", "-f", str(compose_file_path), "config"]

            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)

            logger.debug(f"Compose file validation passed: {compose_file_path}")

        except subprocess.CalledProcessError as e:
            error_msg = f"Invalid compose file {compose_file_path}: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout validating compose file {compose_file_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _inject_episode_network(
        self, compose_file_path: str, config: ComposeEnvironmentConfig, env_vars: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a processed compose file with episode network injection and environment variable resolution.

        Reads the original compose file, injects the appropriate episode network configuration,
        resolves environment variables, and writes to a temporary file. Returns the path to the
        temporary processed file.

        Args:
            compose_file_path: Path to original compose file
            config: Environment configuration containing target_episode_id
            env_vars: Environment variables for variable substitution

        Returns:
            Path to processed compose file with network injection and variable resolution

        Raises:
            ValueError: If compose file uses non-standard isolation networks
        """
        import re
        import tempfile

        try:
            # Parse the original compose file
            with open(compose_file_path, "r") as f:
                compose_content = f.read()

            # First resolve environment variables if provided
            if env_vars:

                def replace_var(match: Any) -> str:
                    var_expr = match.group(1)
                    if ":-" in var_expr:
                        var_name, default = var_expr.split(":-", 1)
                        return str(env_vars.get(var_name, default))
                    else:
                        return str(env_vars.get(var_expr, match.group(0)))  # Return original if not found

                # Pattern to match ${VAR} or ${VAR:-default}
                pattern = r"\$\{([^}]+)\}"
                compose_content = re.sub(pattern, replace_var, compose_content)
                logger.debug("Resolved environment variables in compose file")

            # Parse the resolved content
            compose_data = yaml.safe_load(compose_content)

            if compose_data is None:
                compose_data = {}

            # Validate network usage - fail fast if non-standard networks detected
            if "services" in compose_data:
                for service_name, service_config in compose_data["services"].items():
                    if "networks" in service_config:
                        networks = service_config["networks"]
                        if isinstance(networks, list):
                            network_list = networks
                        elif isinstance(networks, dict):
                            network_list = list(networks.keys())
                        else:
                            continue

                        # Check for problematic isolation network names
                        for network in network_list:
                            if (
                                any(keyword in network.lower() for keyword in ["isolated", "episode", "sandbox"])
                                and network != "saber-episode-network"
                            ):
                                error_msg = (
                                    f"SABER Network Naming Error: Service '{service_name}' uses non-standard "
                                    f"isolation network '{network}'. Please use 'saber-episode-network' for "
                                    f"episode isolation instead. This ensures proper network attachment functionality."
                                )
                                logger.error(error_msg)
                                raise ValueError(error_msg)

            # Ensure networks section exists
            if "networks" not in compose_data:
                compose_data["networks"] = {}

            # Auto-inject saber-episode-network definition
            if config.target_episode_id:
                # ATTACHED MODE: Reference existing target episode's network
                compose_data["networks"]["saber-episode-network"] = {
                    "external": True,
                    "name": f"saber-episode-{config.target_episode_id}",
                }
                logger.info(f"Injected external network reference to episode {config.target_episode_id}")
            else:
                # NORMAL MODE: Create new isolated network for this episode
                compose_data["networks"]["saber-episode-network"] = {
                    "name": f"saber-episode-{config.episode_id}",
                    "internal": True,
                    "driver": "bridge",
                    "labels": ["saber.network.type=isolated", f"saber.episode.id={config.episode_id}"],
                }
                logger.info(f"Injected new isolated network for episode {config.episode_id}")

            # Create temporary file for the modified compose content with resolved variables
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp_file:
                yaml.dump(compose_data, temp_file, default_flow_style=False)
                temp_path = temp_file.name

            logger.debug(f"Created processed compose file with resolved variables: {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"Failed to inject episode network: {e}")
            # Re-raise validation errors to fail fast
            if isinstance(e, ValueError):
                raise
            # Fall back to original file on other errors
            return compose_file_path

    def _cleanup_failed_environment(self, project_name: str) -> None:
        """
        Cleanup a failed environment by stopping and removing containers.

        Used when health checks fail during environment startup to ensure
        no orphaned containers are left running.

        Args:
            project_name: Docker Compose project name to clean up
        """
        logger.warning(f"🧹 Cleaning up failed environment: {project_name}")
        try:
            # Stop and remove containers for this project
            cleanup_command = ["docker", "compose", "-p", project_name, "down", "--volumes", "--remove-orphans"]

            result = subprocess.run(cleanup_command, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info(f"✅ Successfully cleaned up failed environment: {project_name}")
            else:
                logger.warning(
                    f"⚠️ Partial cleanup for {project_name} - some resources may remain. " f"stderr: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Cleanup timeout for project {project_name}")
        except Exception as e:
            logger.error(f"❌ Error during cleanup for project {project_name}: {e}")
