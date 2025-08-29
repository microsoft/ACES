"""
Permanent environment manager for Docker execution environments.

This module manages Docker permanent environments that persist across all sessions,
providing lifecycle management for long-running services.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ....logging_config import get_execution_logger
from ..exceptions import SandboxExecutionError
from ..logging import ContainerLoggingManager
from .environment_spec import PermanentEnvironmentSpec

logger = get_execution_logger(__name__)


class PermanentEnvironmentManager:
    """
    Manager for permanent Docker environments that persist across sessions.

    Handles creation, tracking, and cleanup of Docker permanent environments
    that provide persistent services for all benchmark sessions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PermanentEnvironmentManager.

        Args:
            config: Configuration dictionary for permanent environment settings

        Raises:
            SandboxExecutionError: If permanent environment configuration is invalid
        """
        self.config = config
        self.permanent_spec: Optional[PermanentEnvironmentSpec] = None
        self.compose_project_name = "saber-permanent"
        self._is_running = False

        # Initialize container logging manager
        self.container_logger = ContainerLoggingManager(config)

        logger.info("PermanentEnvironmentManager initializing...")

    def start_permanent_environment(self, environment_spec: PermanentEnvironmentSpec) -> None:
        """
        Start the permanent environment with all persistent services.

        Args:
            environment_spec: Specification for permanent services and networks

        Raises:
            SandboxExecutionError: If permanent environment cannot be started
        """
        if self._is_running:
            logger.warning("Permanent environment is already running")
            return

        try:
            self.permanent_spec = environment_spec

            # Generate Docker Compose configuration
            compose_config = environment_spec.to_compose_dict()

            # Log the docker-compose configuration for debugging
            self.container_logger.log_compose_config(
                compose_config=compose_config,
                config_type="permanent",
                identifier=self.compose_project_name,
                additional_metadata={
                    "environment_spec_type": type(environment_spec).__name__,
                    "services_count": len(compose_config.get("services", {})),
                    "networks_count": len(compose_config.get("networks", {})),
                },
            )

            # Log container lifecycle event
            self.container_logger.log_container_lifecycle_event(
                event_type="start_attempt",
                container_info={
                    "project_name": self.compose_project_name,
                    "config_type": "permanent",
                    "services": list(compose_config.get("services", {}).keys()),
                },
            )

            # Write compose file to temporary location
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as compose_file:
                yaml.dump(compose_config, compose_file, default_flow_style=False)
                compose_file_path = compose_file.name

            # Start services using docker-compose
            cmd = ["docker-compose", "-f", compose_file_path, "-p", self.compose_project_name, "up", "-d"]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Clean up temporary file
            Path(compose_file_path).unlink()

            self._is_running = True

            # Log successful start and collect initial logs
            self.container_logger.log_container_lifecycle_event(
                event_type="start_success",
                container_info={"project_name": self.compose_project_name, "config_type": "permanent"},
                additional_data={"stdout": result.stdout},
            )

            # Collect initial container logs
            self.container_logger.log_all_project_containers(
                project_name=self.compose_project_name, config_type="permanent"
            )
            logger.info(f"Permanent environment started successfully with project name: {self.compose_project_name}")

        except subprocess.CalledProcessError as e:
            # Log the failure with container logs for debugging
            self.container_logger.log_container_lifecycle_event(
                event_type="start_failure",
                container_info={"project_name": self.compose_project_name, "config_type": "permanent"},
                additional_data={"error": str(e), "stderr": e.stderr, "command": e.cmd},
            )

            # Try to collect any available logs even on failure
            self.container_logger.log_all_project_containers(
                project_name=self.compose_project_name, config_type="permanent"
            )

            logger.error(f"Failed to start permanent environment: {e.stderr}")
            raise SandboxExecutionError(f"Failed to start permanent environment: {e.stderr}")
        except Exception as e:
            # Log unexpected failure
            self.container_logger.log_container_lifecycle_event(
                event_type="start_error",
                container_info={"project_name": self.compose_project_name, "config_type": "permanent"},
                additional_data={"error": str(e)},
            )

            logger.error(f"Unexpected error starting permanent environment: {e}")
            raise SandboxExecutionError(f"Failed to start permanent environment: {e}")

    def stop_permanent_environment(self) -> None:
        """
        Stop the permanent environment and all persistent services.

        Raises:
            SandboxExecutionError: If permanent environment cannot be stopped
        """
        if not self._is_running:
            logger.warning("Permanent environment is not running")
            return

        try:
            # Collect final logs before stopping
            self.container_logger.log_container_lifecycle_event(
                event_type="stop_attempt",
                container_info={"project_name": self.compose_project_name, "config_type": "permanent"},
            )

            # Collect logs before stopping containers
            self.container_logger.log_all_project_containers(
                project_name=self.compose_project_name, config_type="permanent"
            )

            # Stop services using docker-compose
            cmd = ["docker-compose", "-p", self.compose_project_name, "down", "-v"]  # Remove volumes as well

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            self._is_running = False
            self.permanent_spec = None

            # Log successful stop
            self.container_logger.log_container_lifecycle_event(
                event_type="stop_success",
                container_info={"project_name": self.compose_project_name, "config_type": "permanent"},
                additional_data={"stdout": result.stdout},
            )

            logger.info("Permanent environment stopped successfully")

        except subprocess.CalledProcessError as e:
            # Log the failure
            self.container_logger.log_container_lifecycle_event(
                event_type="stop_failure",
                container_info={"project_name": self.compose_project_name, "config_type": "permanent"},
                additional_data={"error": str(e), "stderr": e.stderr, "command": e.cmd},
            )

            logger.error(f"Failed to stop permanent environment: {e.stderr}")
            raise SandboxExecutionError(f"Failed to stop permanent environment: {e.stderr}")
        except Exception as e:
            # Log unexpected failure
            self.container_logger.log_container_lifecycle_event(
                event_type="stop_error",
                container_info={"project_name": self.compose_project_name, "config_type": "permanent"},
                additional_data={"error": str(e)},
            )

            logger.error(f"Unexpected error stopping permanent environment: {e}")
            raise SandboxExecutionError(f"Failed to stop permanent environment: {e}")

    def get_service_endpoints(self, service_name: str) -> Dict[str, Any]:
        """
        Get connection endpoints for a permanent service.

        Args:
            service_name: Name of the permanent service

        Returns:
            Dictionary containing service connection information

        Raises:
            SandboxExecutionError: If service not found or not running
        """
        if not self._is_running or not self.permanent_spec:
            raise SandboxExecutionError("Permanent environment is not running")

        service_spec = self.permanent_spec.get_service(service_name)
        if not service_spec:
            raise SandboxExecutionError(f"Permanent service '{service_name}' not found")

        try:
            # Get container information using docker-compose
            cmd = ["docker-compose", "-p", self.compose_project_name, "ps", "-q", service_name]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()

            if not container_id:
                raise SandboxExecutionError(f"Permanent service '{service_name}' container not found")

            # Get container details
            cmd = ["docker", "inspect", container_id]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            import json

            container_info = json.loads(result.stdout)[0]

            # Extract network information
            networks = container_info.get("NetworkSettings", {}).get("Networks", {})
            endpoints = {}

            for network_name, network_info in networks.items():
                endpoints[network_name] = {
                    "ip_address": network_info.get("IPAddress"),
                    "ports": service_spec.ports,
                }

            return {
                "service_name": service_name,
                "container_id": container_id,
                "endpoints": endpoints,
                "status": "running" if container_info.get("State", {}).get("Running") else "stopped",
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get service endpoints for '{service_name}': {e.stderr}")
            raise SandboxExecutionError(f"Failed to get service endpoints: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error getting service endpoints for '{service_name}': {e}")
            raise SandboxExecutionError(f"Failed to get service endpoints: {e}")

    def is_running(self) -> bool:
        """
        Check if the permanent environment is running.

        Returns:
            True if permanent environment is running, False otherwise
        """
        return self._is_running

    def get_permanent_services(self) -> List[str]:
        """
        Get list of permanent service names.

        Returns:
            List of permanent service names
        """
        if not self.permanent_spec:
            return []

        return list(self.permanent_spec.services.keys())

    def get_permanent_networks(self) -> List[str]:
        """
        Get list of permanent network names.

        Returns:
            List of permanent network names
        """
        if not self.permanent_spec:
            return []

        return list(self.permanent_spec.networks.keys())
