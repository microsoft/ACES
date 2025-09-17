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

        # Log the resolved compose configuration
        self._log_resolved_compose_config(compose_file_path, env_vars)

        # Run docker compose up with the provided environment variables
        command = ["docker", "compose", "-f", compose_file_path, "-p", self.project_name, "up", "-d"]

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

            # Identify the execution service after successful startup
            self.execution_service_name = self._identify_execution_service(Path(compose_file_path))
            logger.info(f"Identified execution service: {self.execution_service_name}")

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

            # Save resolved compose file to disk
            if self.container_logger:
                self._save_resolved_compose_file(compose_file_path, resolved_content, env_vars)

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

    def _save_resolved_compose_file(
        self, original_compose_path: str, resolved_content: str, env_vars: Dict[str, str]
    ) -> None:
        """
        Save the resolved compose configuration to the compose-configs directory.

        Args:
            original_compose_path: Path to the original compose file
            resolved_content: Resolved compose content with variables substituted
            env_vars: Environment variables used for substitution
        """
        try:
            if not self.container_logger:
                return

            from datetime import datetime
            from pathlib import Path

            # Get logs directory from container logger config
            logs_dir = Path(self.container_logger.logs_directory)
            compose_configs_dir = logs_dir / "compose-configs"
            compose_configs_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp and filename
            timestamp = datetime.now().isoformat()
            original_filename = Path(original_compose_path).stem
            config_type = self.config_type
            project_name = self.project_name or "unknown"

            # Create filename: timestamp_configtype_projectname_originalname.yml
            resolved_filename = f"{timestamp}_{config_type}_{project_name}_{original_filename}.yml"
            resolved_file_path = compose_configs_dir / resolved_filename

            # Write resolved compose content
            with open(resolved_file_path, "w") as f:
                f.write(resolved_content)

            # Also write environment variables used
            env_filename = f"{timestamp}_{config_type}_{project_name}_{original_filename}.env"
            env_file_path = compose_configs_dir / env_filename

            with open(env_file_path, "w") as f:
                f.write("# Environment variables used for compose resolution\n")
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")

            logger.info(f"Saved resolved compose config: {resolved_file_path}")
            logger.info(f"Saved environment variables: {env_file_path}")

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
