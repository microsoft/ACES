"""
Permanent environment manager for Docker execution environments.

This module manages Docker permanent environments that persist across all sessions,
providing lifecycle management for long-running services and networks.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ....logging_config import get_execution_logger
from ..exceptions import SandboxExecutionError
from ..logging import ContainerLoggingManager
from .environment_spec import PermanentEnvironmentSpec, PermanentNetworkSpec

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

        # Configuration and metadata storage
        self.logs_dir = Path(config.get("config_dir", "/app/logs"))
        self.compose_configs_dir = self.logs_dir / "compose-configs" / "permanent-environments"
        self.metadata_dir = self.compose_configs_dir / "metadata"
        self.compose_configs_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Initialize container logging manager
        self.container_logger = ContainerLoggingManager(config)

        logger.info("PermanentEnvironmentManager initializing...")
        logger.info(f"Configuration storage: {self.compose_configs_dir}")
        logger.info(f"Metadata storage: {self.metadata_dir}")

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
            cmd = ["docker", "compose", "-f", compose_file_path, "-p", self.compose_project_name, "up", "-d"]

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
            cmd = ["docker", "compose", "-p", self.compose_project_name, "down", "-v"]  # Remove volumes as well

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
            cmd = ["docker", "compose", "-p", self.compose_project_name, "ps", "-q", service_name]

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

    def get_permanent_network_spec(self, network_name: str) -> Optional[PermanentNetworkSpec]:
        """
        Get permanent network specification by name.

        Args:
            network_name: Name of the permanent network

        Returns:
            PermanentNetworkSpec if found, None otherwise
        """
        if not self.permanent_spec:
            return None

        return self.permanent_spec.networks.get(network_name)

    def _compute_configuration_hash(self, environment_spec: PermanentEnvironmentSpec) -> str:
        """
        Compute hash of the permanent environment configuration.

        Args:
            environment_spec: The permanent environment specification

        Returns:
            SHA-256 hash of the configuration
        """
        compose_config = environment_spec.to_compose_dict()
        # Create deterministic JSON string for hashing
        config_str = json.dumps(compose_config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _get_stored_configuration_hash(self) -> Optional[str]:
        """
        Get the stored configuration hash from metadata.

        Returns:
            Stored configuration hash or None if not found
        """
        hash_file = self.metadata_dir / "configuration_hash.txt"
        if hash_file.exists():
            return hash_file.read_text().strip()
        return None

    def _store_configuration_hash(self, config_hash: str) -> None:
        """
        Store the configuration hash to metadata.

        Args:
            config_hash: Configuration hash to store
        """
        hash_file = self.metadata_dir / "configuration_hash.txt"
        hash_file.write_text(config_hash)

    def is_configuration_current(self, environment_spec: PermanentEnvironmentSpec) -> bool:
        """
        Check if running permanent environments match current config.

        Args:
            environment_spec: Current environment specification

        Returns:
            True if configuration is current, False if recreation needed
        """
        if not self._is_running:
            return False

        current_hash = self._compute_configuration_hash(environment_spec)
        stored_hash = self._get_stored_configuration_hash()

        logger.info(
            f"Configuration check: current={current_hash[:12]}..., "
            f"stored={stored_hash[:12] if stored_hash else 'None'}..."
        )
        return current_hash == stored_hash

    def ensure_permanent_environments_current(self, environment_spec: PermanentEnvironmentSpec) -> None:
        """
        Ensure permanent environments match current configuration.

        Args:
            environment_spec: Environment specification to ensure
        """
        logger.info("Ensuring permanent environments are current...")

        if not self.is_configuration_current(environment_spec):
            logger.info("Configuration changed or first startup - recreating permanent environments")
            self.recreate_environment(environment_spec)
        else:
            logger.info("Permanent environment configuration is current")

    def recreate_environment(self, environment_spec: PermanentEnvironmentSpec) -> None:
        """
        Recreate permanent environment with new configuration.

        Args:
            environment_spec: New environment specification
        """
        logger.info("Recreating permanent environment...")

        # Stop existing environment if running
        if self._is_running:
            logger.info("Stopping existing permanent environment")
            self.stop_permanent_environment()

        # Ensure required networks exist
        self.ensure_shared_networks(environment_spec)

        # Start with new configuration
        self.start_permanent_environment(environment_spec)

        # Store new configuration hash
        config_hash = self._compute_configuration_hash(environment_spec)
        self._store_configuration_hash(config_hash)

        logger.info("Permanent environment recreation complete")

    def ensure_shared_networks(self, environment_spec: PermanentEnvironmentSpec) -> None:
        """
        Create required external networks if they don't exist.

        Args:
            environment_spec: Environment specification containing network definitions
        """
        logger.info("Ensuring shared networks exist...")

        for network_name, network_spec in environment_spec.networks.items():
            # Skip external networks - they should already exist or be managed externally
            if network_spec.external:
                effective_name = network_spec.external_name or network_name
                logger.info(f"Skipping external network (should exist): {effective_name}")

                # Optionally verify external network exists
                try:
                    check_cmd = ["docker", "network", "inspect", effective_name]
                    result = subprocess.run(check_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.warning(f"External network '{effective_name}' does not exist - creating it")
                        # Create the external network if it doesn't exist
                        self._create_network(effective_name, network_spec)
                    else:
                        logger.info(f"External network '{effective_name}' exists")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Could not verify external network '{effective_name}': {e.stderr}")
                continue

            try:
                # Check if network already exists
                check_cmd = ["docker", "network", "inspect", network_name]
                result = subprocess.run(check_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    # Network doesn't exist, create it
                    logger.info(f"Creating network: {network_name}")
                    self._create_network(network_name, network_spec)
                else:
                    logger.info(f"Network already exists: {network_name}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to ensure network {network_name}: {e.stderr}")
                raise SandboxExecutionError(f"Failed to ensure network {network_name}: {e.stderr}")

    def _create_network(self, network_name: str, network_spec: PermanentNetworkSpec) -> None:
        """
        Create a Docker network with the given specification.

        Args:
            network_name: Name of the network to create
            network_spec: Network specification
        """
        create_cmd = ["docker", "network", "create"]

        # Add driver
        create_cmd.extend(["--driver", network_spec.driver])

        # Add internal flag if specified
        if network_spec.internal:
            create_cmd.append("--internal")

        # Add IPAM configuration if specified
        if network_spec.ipam_config:
            for subnet_config in network_spec.ipam_config.get("config", []):
                if "subnet" in subnet_config:
                    create_cmd.extend(["--subnet", subnet_config["subnet"]])

        create_cmd.append(network_name)

        subprocess.run(create_cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully created network: {network_name}")

    def cleanup_managed_networks(self) -> None:
        """
        Remove networks created by this manager.

        Note: Only removes networks that are not in use by other containers.
        Does not remove external networks.
        """
        if not self.permanent_spec:
            return

        logger.info("Cleaning up managed networks...")

        for network_name, network_spec in self.permanent_spec.networks.items():
            # Skip external networks - we don't manage their lifecycle
            if network_spec.external:
                logger.info(f"Skipping cleanup of external network: {network_spec.external_name or network_name}")
                continue

            try:
                # Check if network exists and is not in use
                inspect_cmd = ["docker", "network", "inspect", network_name]
                result = subprocess.run(inspect_cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    # Network exists, check if it's in use
                    network_info = json.loads(result.stdout)[0]
                    containers = network_info.get("Containers", {})

                    if not containers:
                        # Network is not in use, safe to remove
                        logger.info(f"Removing unused network: {network_name}")
                        remove_cmd = ["docker", "network", "rm", network_name]
                        subprocess.run(remove_cmd, capture_output=True, text=True, check=True)
                        logger.info(f"Successfully removed network: {network_name}")
                    else:
                        logger.info(f"Network {network_name} is still in use, skipping removal")

            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to cleanup network {network_name}: {e.stderr}")
            except Exception as e:
                logger.warning(f"Unexpected error cleaning up network {network_name}: {e}")

    def cleanup_on_server_shutdown(self) -> None:
        """
        Cleanup permanent environments on server shutdown.

        This method provides a clean shutdown hook for server lifecycle.
        """
        logger.info("Cleaning up permanent environments on server shutdown...")

        if self._is_running:
            self.stop_permanent_environment()

        # Optionally cleanup networks (commented out to preserve shared networks)
        # self.cleanup_managed_networks()

        logger.info("Permanent environment cleanup complete")
