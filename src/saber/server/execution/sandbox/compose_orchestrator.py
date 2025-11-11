"""Simple Docker Compose orchestrator for environment management.

Logging Category: DOCKER

Provides basic start/stop operations for Docker Compose files with episode isolation support.
Includes command execution capabilities for designated execution services.
"""

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from saber.logging_config import LogCategory, get_saber_logger

from ..logging import ContainerLoggingManager
from .compose_health_checker import ComposeHealthChecker
from .environment_config import ComposeEnvironmentConfig

try:
    import docker
except ImportError:
    docker = None  # type: ignore

DOCKER_AVAILABLE = docker is not None

from ...base import CommandResult

logger = get_saber_logger(LogCategory.DOCKER, __name__)


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

        logger.info(
            "Compose environment start requested",
            extra={
                "event": "compose_environment_start_requested",
                "project_name": self.project_name,
                "config_type": config.config_type,
                "compose_file": str(compose_file_path),
                "episode_id": config.episode_id,
            },
        )
        logger.debug(
            "Compose environment variables prepared",
            extra={
                "event": "compose_environment_variables_prepared",
                "project_name": self.project_name,
                "env_vars": env_vars,
            },
        )

        # Log the resolved compose configuration (use processed path to include network injection)
        self._log_resolved_compose_config(processed_compose_path, env_vars)

        # Validate execution service exists before running Docker command (fail-fast)
        temp_execution_service = self._identify_execution_service(Path(processed_compose_path))

        # Run docker compose up with the processed compose file and provided environment variables
        command = ["docker", "compose", "-f", processed_compose_path, "-p", self.project_name, "up", "-d"]

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
            logger.debug(
                "Execution service identified",
                extra={
                    "event": "compose_execution_service_identified",
                    "project_name": self.project_name,
                    "execution_service": self.execution_service_name,
                },
            )

            # MANDATORY HEALTH CHECK: Wait for all services to become healthy
            logger.debug(
                "Compose environment health checks starting",
                extra={
                    "event": "compose_environment_health_checks_starting",
                    "project_name": self.project_name,
                    "processed_compose_file": processed_compose_path,
                    "original_compose_file": str(compose_file_path),
                },
            )

            # Use the processed compose file (with variables resolved) for health checks
            # NO TRY-CATCH: Health check failures will propagate up and fail episode creation
            self.health_checker.wait_for_all_services_healthy(
                compose_file_path=processed_compose_path,  # Use processed file, not original
                project_name=self.project_name,
                timeout_seconds=180,  # 3 minutes for health checks - reasonable for complex environments
                check_interval=2.0,  # Check every 2 seconds
            )
            logger.info(
                "Compose environment healthy",
                extra={
                    "event": "compose_environment_health_checks_completed",
                    "project_name": self.project_name,
                },
            )

            logger.info(
                "Compose environment started",
                extra={
                    "event": "compose_environment_started",
                    "project_name": self.project_name,
                    "config_type": config.config_type,
                    "compose_file": str(compose_file_path),
                    "episode_id": config.episode_id,
                },
            )
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

            logger.error(
                "Compose environment start failed",
                extra={
                    "event": "compose_environment_start_failed",
                    "project_name": self.project_name,
                    "config_type": config.config_type,
                    "compose_file": str(compose_file_path),
                    "episode_id": config.episode_id,
                    "return_code": getattr(e, "returncode", None),
                    "stdout": e.stdout,
                    "stderr": e.stderr,
                },
            )
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

            logger.error(
                "Compose environment start timed out",
                extra={
                    "event": "compose_environment_start_timeout",
                    "project_name": self.project_name,
                    "config_type": config.config_type,
                    "compose_file": str(compose_file_path),
                    "episode_id": config.episode_id,
                    "timeout_seconds": e.timeout,
                },
            )
            raise RuntimeError(f"Environment start timed out after {e.timeout} seconds")

    def start_environment_async(
        self, compose_file_path: str, config: ComposeEnvironmentConfig
    ) -> subprocess.CompletedProcess:
        """
        Start environment without waiting for health checks.

        Returns immediately after docker compose up succeeds.
        Health checks must be performed separately via wait_for_healthy().

        Note: This is deliberately synchronous; callers invoke it via
        `await asyncio.to_thread(orchestrator.start_environment_async, ...)`
        so the event loop stays responsive.

        Args:
            compose_file_path: Path to compose file
            config: Environment configuration

        Returns:
            CompletedProcess from docker compose up

        Raises:
            RuntimeError: If docker compose up fails
        """
        # Set project name and config
        self.project_name = config.get_project_name()
        if not self.project_name:
            raise RuntimeError("Project name is required but not provided by config")

        self.config_type = config.config_type
        self.episode_id = config.episode_id
        self.compose_file_path = Path(compose_file_path)

        # Prepare environment variables
        env_vars = config.to_env_dict()
        processed_compose_path = self._inject_episode_network(compose_file_path, config, env_vars)

        # Store processed path for health checks
        self.processed_compose_path = processed_compose_path

        env = os.environ.copy()
        env.update(env_vars)

        # Log start attempt
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

        logger.info(
            "Async compose environment start requested",
            extra={
                "event": "compose_environment_async_start",
                "project_name": self.project_name,
                "episode_id": config.episode_id,
            },
        )

        # Log resolved compose config
        self._log_resolved_compose_config(processed_compose_path, env_vars)

        # Validate execution service exists
        temp_execution_service = self._identify_execution_service(Path(processed_compose_path))

        # Run docker compose up
        command = ["docker", "compose", "-f", processed_compose_path, "-p", self.project_name, "up", "-d"]

        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)

            # Log successful start
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

            self.execution_service_name = temp_execution_service

            logger.info(
                "Compose environment started (async - health checks pending)",
                extra={
                    "event": "compose_environment_async_started",
                    "project_name": self.project_name,
                    "episode_id": config.episode_id,
                },
            )

            return result

        except subprocess.CalledProcessError as e:
            # Parse stderr to extract actual Docker error
            stderr_text = e.stderr if e.stderr else ""

            # Detect specific error conditions
            is_subnet_exhausted = "all predefined address pools have been fully subnetted" in stderr_text
            is_network_error = "failed to create network" in stderr_text
            is_port_conflict = "port is already allocated" in stderr_text or "address already in use" in stderr_text

            # Build enhanced error message with diagnostic hints
            error_hints = []
            if is_subnet_exhausted:
                error_hints.append(
                    "Docker subnet pool exhausted - clean up unused networks with 'docker network prune'"
                )
            if is_network_error:
                error_hints.append("Network creation failed - check for orphaned networks")
            if is_port_conflict:
                error_hints.append("Port conflict detected - another container may be using the same port")

            if self.container_logger:
                self.container_logger.log_container_lifecycle_event(
                    event_type="start_failure",
                    container_info={
                        "project_name": self.project_name,
                        "config_type": config.config_type,
                        "compose_file": str(compose_file_path),
                        "episode_id": config.episode_id,
                    },
                    additional_data={
                        "error": str(e),
                        "stderr": stderr_text,
                        "is_subnet_exhausted": is_subnet_exhausted,
                        "is_network_error": is_network_error,
                        "is_port_conflict": is_port_conflict,
                        "error_hints": error_hints,
                    },
                )

            logger.error(
                "Compose environment async start failed",
                extra={
                    "event": "compose_environment_async_start_failed",
                    "project_name": self.project_name,
                    "episode_id": config.episode_id,
                    "stderr": stderr_text,
                    "exit_code": e.returncode,
                    "is_subnet_exhausted": is_subnet_exhausted,
                    "is_network_error": is_network_error,
                    "is_port_conflict": is_port_conflict,
                    "error_hints": error_hints,
                },
            )

            # Include diagnostic hint in exception message
            error_msg = f"Failed to start environment: {e}"
            if error_hints:
                error_msg += f" | Hints: {'; '.join(error_hints)}"
            raise RuntimeError(error_msg)

    def wait_for_healthy(self, compose_file_path: str, timeout_seconds: int = 180, check_interval: float = 2.0) -> None:
        """
        Wait for all services in this environment to become healthy.

        Must be called after start_environment_async().

        Note: Invoke via `await asyncio.to_thread(...)` to avoid blocking the event loop.

        Args:
            compose_file_path: Path to compose file (processed with env vars resolved)
            timeout_seconds: Maximum time to wait (default 180s)
            check_interval: Seconds between health checks (default 2s)

        Raises:
            ComposeHealthCheckError: If services don't become healthy in time
        """
        logger.debug(
            "Waiting for compose environment health checks",
            extra={
                "event": "compose_environment_health_wait_start",
                "project_name": self.project_name,
                "timeout_seconds": timeout_seconds,
            },
        )

        # Ensure project_name is set before waiting for health
        if self.project_name is None:
            raise RuntimeError("Project name must be set before waiting for services to become healthy")

        self.health_checker.wait_for_all_services_healthy(
            compose_file_path=compose_file_path,
            project_name=self.project_name,
            timeout_seconds=timeout_seconds,
            check_interval=check_interval,
        )

        logger.info(
            "Compose environment healthy",
            extra={
                "event": "compose_environment_health_ready",
                "project_name": self.project_name,
            },
        )

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
            resolved_config = yaml.safe_load(resolved_content) or {}

            logger.debug(
                "Compose configuration resolved",
                extra={
                    "event": "compose_configuration_resolved",
                    "project_name": self.project_name,
                    "processed_compose_file": compose_file_path,
                },
            )

            # Save resolved compose file to disk
            self._save_resolved_compose_config(compose_file_path, resolved_content, resolved_config)

            # Log network configuration specifically
            if "networks" in resolved_config:
                logger.debug(
                    "Compose networks discovered",
                    extra={
                        "event": "compose_networks_discovered",
                        "project_name": self.project_name,
                        "networks": resolved_config["networks"],
                    },
                )

            # Log services and their network connections
            if "services" in resolved_config:
                service_networks: Dict[str, List[str]] = {}
                for service_name, service_config in resolved_config["services"].items():
                    if "networks" in service_config:
                        networks = service_config["networks"]
                        if isinstance(networks, dict):
                            network_names = list(networks.keys())
                        elif isinstance(networks, list):
                            network_names = [str(entry) for entry in networks]
                        else:
                            network_names = [str(networks)]
                        service_networks[service_name] = network_names
                if service_networks:
                    logger.debug(
                        "Compose service network connections discovered",
                        extra={
                            "event": "compose_service_networks_discovered",
                            "project_name": self.project_name,
                            "service_networks": service_networks,
                        },
                    )

        except Exception as e:
            logger.error(
                "Compose configuration logging failed",
                extra={
                    "event": "compose_configuration_logging_failed",
                    "processed_compose_file": compose_file_path,
                    "error": str(e),
                },
            )

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

            logger.debug(
                "Compose configuration persisted",
                extra={
                    "event": "compose_configuration_persisted",
                    "project_name": self.project_name,
                    "output_file": str(output_file),
                    "config_type": self.config_type,
                },
            )

        except Exception as e:
            logger.warning(
                "Compose configuration persistence skipped",
                extra={
                    "event": "compose_configuration_persist_failed",
                    "project_name": self.project_name,
                    "compose_file": compose_file_path,
                    "error": str(e),
                },
            )

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

        # Get execution container using episode-aware resolution (offload to thread pool)
        container = await self.get_execution_container_async()
        if not container:
            expected_name = self._get_actual_container_name(self.execution_service_name)
            raise RuntimeError(
                f"Execution container '{expected_name}' not found or not running. "
                f"Episode: {self.episode_id}, Service: {self.execution_service_name}. "
                f"Ensure environment is started and container is healthy."
            )

        logger.debug(
            "Container command execution started",
            extra={
                "event": "container_command_execution_started",
                "container_name": container.name,
                "command": command,
                "working_dir": working_dir,
                "timeout_seconds": timeout,
            },
        )

        start_time = time.time()

        try:
            # Execute command via docker exec in thread pool with timeout enforcement
            # This allows concurrent tool execution across multiple episodes
            # Wrap with asyncio.wait_for to enforce timeout (Docker SDK doesn't support timeouts natively)
            exec_result = await asyncio.wait_for(
                asyncio.to_thread(
                    container.exec_run,
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
                ),
                timeout=timeout,  # Enforce timeout from execution_config
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

            logger.debug(
                "Container command execution completed",
                extra={
                    "event": "container_command_execution_completed",
                    "container_name": container.name,
                    "command": command,
                    "exit_code": exec_result.exit_code,
                    "execution_time_seconds": round(execution_time, 2),
                    "timeout_seconds": timeout,
                },
            )

            return result

        except asyncio.TimeoutError:
            # Command exceeded timeout - kill and return timeout error
            execution_time = time.time() - start_time
            logger.error(
                "Container command execution timed out",
                extra={
                    "event": "container_command_execution_timeout",
                    "container_name": container.name,
                    "command": command,
                    "timeout_seconds": timeout,
                    "execution_time_seconds": round(execution_time, 2),
                },
            )

            # Return timeout error result with exit code 124 (standard timeout exit code)
            return CommandResult(
                stdout="",
                stderr=f"Command execution timed out after {timeout} seconds",
                exit_code=124,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Container command execution failed",
                extra={
                    "event": "container_command_execution_failed",
                    "container_name": container.name,
                    "command": command,
                    "error": str(e),
                    "execution_time_seconds": round(execution_time, 2),
                },
            )

            # Return error result instead of raising
            return CommandResult(
                stdout="",
                stderr=f"Command execution failed: {str(e)}",
                exit_code=1,
                execution_time=execution_time,
            )

    async def get_execution_container_async(self) -> Any:
        """
        Get the container designated for command execution using episode-aware resolution.

        This async version offloads blocking Docker calls to a thread pool.

        Returns:
            Docker container object if found, None otherwise
        """
        if not self.execution_service_name:
            logger.error(
                "Execution service missing",
                extra={
                    "event": "execution_service_missing",
                },
            )
            return None

        try:
            # Get the actual container name using compose file + episode_id
            actual_container_name = self._get_actual_container_name(self.execution_service_name)
            logger.info(
                "Execution container lookup started",
                extra={
                    "event": "execution_container_lookup_started",
                    "container_name": actual_container_name,
                    "execution_service": self.execution_service_name,
                    "episode_id": self.episode_id,
                },
            )

            # Try to get the container (offload to thread pool to avoid blocking)
            try:
                container = await asyncio.to_thread(self.docker_client.containers.get, actual_container_name)

                # Verify container is running (offload to thread pool)
                await asyncio.to_thread(container.reload)  # Refresh container state
                if container.status != "running":
                    logger.error(
                        "Execution container not running",
                        extra={
                            "event": "execution_container_not_running",
                            "container_name": actual_container_name,
                            "status": container.status,
                            "execution_service": self.execution_service_name,
                            "episode_id": self.episode_id,
                        },
                    )
                    return None

                logger.info(
                    "Execution container ready",
                    extra={
                        "event": "execution_container_ready",
                        "container_name": actual_container_name,
                        "execution_service": self.execution_service_name,
                        "episode_id": self.episode_id,
                    },
                )
                return container

            except Exception as e:
                # Handle docker.errors.NotFound and other exceptions
                if docker and hasattr(docker, "errors") and isinstance(e, docker.errors.NotFound):
                    # Log available containers for debugging (offload to thread pool)
                    all_containers = await asyncio.to_thread(self.docker_client.containers.list, all=True)
                    available_containers = []
                    for container_item in all_containers:
                        image_obj = getattr(container_item, "image", None)
                        image_tags = getattr(image_obj, "tags", []) if image_obj else []
                        available_containers.append(
                            {
                                "name": container_item.name,
                                "status": container_item.status,
                                "image": image_tags[0] if image_tags else None,
                            }
                        )
                    logger.error(
                        "Execution container not found",
                        extra={
                            "event": "execution_container_not_found",
                            "container_name": actual_container_name,
                            "execution_service": self.execution_service_name,
                            "episode_id": self.episode_id,
                            "available_containers": available_containers,
                        },
                    )
                else:
                    logger.error(
                        "Execution container lookup failed",
                        extra={
                            "event": "execution_container_lookup_failed",
                            "container_name": actual_container_name,
                            "execution_service": self.execution_service_name,
                            "episode_id": self.episode_id,
                            "error": str(e),
                        },
                    )

                return None

        except Exception as e:
            logger.error(
                "Execution container resolution failed",
                extra={
                    "event": "execution_container_resolution_failed",
                    "execution_service": self.execution_service_name,
                    "episode_id": self.episode_id,
                    "error": str(e),
                },
            )
            return None

    def get_execution_container(self) -> Any:
        """
        Get the container designated for command execution (synchronous wrapper).

        Deprecated: Use get_execution_container_async() instead for better concurrency.
        This method exists for backward compatibility but may block the event loop.

        Returns:
            Docker container object if found, None otherwise
        """
        # Synchronous fallback - may block event loop
        if not self.execution_service_name:
            logger.error(
                "Execution service missing",
                extra={"event": "execution_service_missing"},
            )
            return None

        try:
            actual_container_name = self._get_actual_container_name(self.execution_service_name)
            container = self.docker_client.containers.get(actual_container_name)
            container.reload()
            if container.status != "running":
                return None
            return container
        except Exception:
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

        logger.info(
            "Compose environment stop requested",
            extra={
                "event": "compose_environment_stop_requested",
                "compose_file": str(compose_file_path),
                "project_name": active_project_name,
                "episode_id": episode_id,
                "display_name": display_name.strip() or None,
            },
        )

        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True, timeout=60)

            logger.info(
                "Compose environment stopped",
                extra={
                    "event": "compose_environment_stopped",
                    "compose_file": str(compose_file_path),
                    "project_name": active_project_name,
                    "episode_id": episode_id,
                },
            )
            if result.stdout:
                logger.debug(
                    "Compose stop command output",
                    extra={
                        "event": "compose_environment_stop_output",
                        "compose_file": str(compose_file_path),
                        "project_name": active_project_name,
                        "stdout": result.stdout,
                    },
                )

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
            logger.error(
                "Compose environment stop failed",
                extra={
                    "event": "compose_environment_stop_failed",
                    "compose_file": str(compose_file_path),
                    "project_name": active_project_name,
                    "episode_id": episode_id,
                    "return_code": getattr(e, "returncode", None),
                    "stderr": e.stderr,
                },
            )
            raise RuntimeError(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout stopping environment {compose_file_path}"
            logger.error(
                "Compose environment stop timed out",
                extra={
                    "event": "compose_environment_stop_timeout",
                    "compose_file": str(compose_file_path),
                    "project_name": active_project_name,
                    "episode_id": episode_id,
                },
            )
            raise RuntimeError(error_msg)

    def cleanup_episode(self, episode_id: str) -> bool:
        """
        Clean up all containers and networks for a specific episode.

        Args:
            episode_id: Episode ID to clean up

        Returns:
            bool: True if cleanup succeeded, False otherwise
        """
        project_name = f"saber-episode-{episode_id}"
        network_name = f"saber-episode-{episode_id}"

        logger.info(
            "Episode cleanup requested",
            extra={
                "event": "episode_cleanup_requested",
                "episode_id": episode_id,
                "project_name": project_name,
                "network_name": network_name,
            },
        )

        cleanup_success = True

        try:
            # First, try docker compose down (works for fully created environments)
            cmd = ["docker", "compose", "-p", project_name, "down", "--volumes", "--remove-orphans"]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)

            logger.info(
                "Episode compose cleanup completed",
                extra={
                    "event": "episode_compose_cleanup_completed",
                    "episode_id": episode_id,
                    "project_name": project_name,
                },
            )
            if result.stdout:
                logger.debug(
                    "Episode cleanup output",
                    extra={
                        "event": "episode_cleanup_output",
                        "episode_id": episode_id,
                        "stdout": result.stdout,
                    },
                )

        except subprocess.CalledProcessError as e:
            # Compose down failed - log but continue to network cleanup
            logger.warning(
                "Episode compose cleanup failed (may not have been fully created)",
                extra={
                    "event": "episode_compose_cleanup_failed",
                    "episode_id": episode_id,
                    "project_name": project_name,
                    "error": str(e),
                    "stderr": e.stderr,
                },
            )
            cleanup_success = False

        # Additionally, explicitly remove the episode network if it exists
        # This handles cases where 'docker compose up' failed partway through
        try:
            # Check if network exists first
            check_cmd = ["docker", "network", "inspect", network_name]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)

            if check_result.returncode == 0:
                # Network exists, remove it
                remove_cmd = ["docker", "network", "rm", network_name]
                subprocess.run(remove_cmd, capture_output=True, text=True, check=True, timeout=10)

                logger.info(
                    "Episode network removed",
                    extra={
                        "event": "episode_network_removed",
                        "episode_id": episode_id,
                        "network_name": network_name,
                    },
                )
                cleanup_success = True
            else:
                # Network doesn't exist, nothing to clean up
                logger.debug(
                    "Episode network does not exist (already cleaned up or never created)",
                    extra={
                        "event": "episode_network_not_found",
                        "episode_id": episode_id,
                        "network_name": network_name,
                    },
                )

        except subprocess.CalledProcessError as e:
            # Network removal failed
            logger.error(
                "Failed to remove episode network",
                extra={
                    "event": "episode_network_removal_failed",
                    "episode_id": episode_id,
                    "network_name": network_name,
                    "error": str(e),
                    "stderr": e.stderr,
                },
            )
            cleanup_success = False

        if cleanup_success:
            logger.info(
                "Episode cleanup completed successfully",
                extra={
                    "event": "episode_cleanup_completed",
                    "episode_id": episode_id,
                    "project_name": project_name,
                },
            )

        return cleanup_success

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

            logger.debug(
                "Compose file validation passed",
                extra={
                    "event": "compose_file_validation_passed",
                    "compose_file": str(compose_file_path),
                },
            )

        except subprocess.CalledProcessError as e:
            error_msg = f"Invalid compose file {compose_file_path}: {e.stderr}"
            logger.error(
                "Compose file validation failed",
                extra={
                    "event": "compose_file_validation_failed",
                    "compose_file": str(compose_file_path),
                    "stderr": e.stderr,
                    "return_code": getattr(e, "returncode", None),
                },
            )
            raise RuntimeError(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout validating compose file {compose_file_path}"
            logger.error(
                "Compose file validation timed out",
                extra={
                    "event": "compose_file_validation_timeout",
                    "compose_file": str(compose_file_path),
                },
            )
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
                logger.debug(
                    "Compose file environment variables resolved",
                    extra={
                        "event": "compose_file_env_variables_resolved",
                        "compose_file": compose_file_path,
                        "resolved_variables": list(env_vars.keys()),
                    },
                )

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
                                logger.error(
                                    "Compose network validation failed",
                                    extra={
                                        "event": "compose_network_validation_failed",
                                        "compose_file": compose_file_path,
                                        "service_name": service_name,
                                        "network": network,
                                    },
                                )
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
                logger.debug(
                    "Compose network injection configured",
                    extra={
                        "event": "compose_network_injection_configured",
                        "mode": "attached",
                        "target_episode_id": config.target_episode_id,
                        "compose_file": compose_file_path,
                    },
                )
            else:
                # NORMAL MODE: Create new isolated network for this episode
                compose_data["networks"]["saber-episode-network"] = {
                    "name": f"saber-episode-{config.episode_id}",
                    "internal": True,
                    "driver": "bridge",
                    "labels": ["saber.network.type=isolated", f"saber.episode.id={config.episode_id}"],
                }
                logger.debug(
                    "Compose network injection configured",
                    extra={
                        "event": "compose_network_injection_configured",
                        "mode": "isolated",
                        "episode_id": config.episode_id,
                        "compose_file": compose_file_path,
                    },
                )

            # Create temporary file for the modified compose content with resolved variables
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp_file:
                yaml.dump(compose_data, temp_file, default_flow_style=False)
                temp_path = temp_file.name

            logger.debug(
                "Processed compose file created",
                extra={
                    "event": "processed_compose_file_created",
                    "compose_file": compose_file_path,
                    "processed_file": temp_path,
                },
            )
            return temp_path

        except Exception as e:
            logger.error(
                "Compose network injection failed",
                extra={
                    "event": "compose_network_injection_failed",
                    "compose_file": compose_file_path,
                    "episode_id": config.episode_id,
                    "target_episode_id": config.target_episode_id,
                    "error": str(e),
                },
            )
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
        logger.warning(
            "Failed environment cleanup started",
            extra={
                "event": "failed_environment_cleanup_started",
                "project_name": project_name,
            },
        )
        try:
            # Stop and remove containers for this project
            cleanup_command = ["docker", "compose", "-p", project_name, "down", "--volumes", "--remove-orphans"]

            result = subprocess.run(cleanup_command, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info(
                    "Failed environment cleanup completed",
                    extra={
                        "event": "failed_environment_cleanup_completed",
                        "project_name": project_name,
                    },
                )
            else:
                logger.warning(
                    "Failed environment cleanup partial",
                    extra={
                        "event": "failed_environment_cleanup_partial",
                        "project_name": project_name,
                        "stderr": result.stderr,
                        "return_code": result.returncode,
                    },
                )
        except subprocess.TimeoutExpired:
            logger.error(
                "Failed environment cleanup timed out",
                extra={
                    "event": "failed_environment_cleanup_timeout",
                    "project_name": project_name,
                },
            )
        except Exception as e:
            logger.error(
                "Failed environment cleanup error",
                extra={
                    "event": "failed_environment_cleanup_error",
                    "project_name": project_name,
                    "error": str(e),
                },
            )
