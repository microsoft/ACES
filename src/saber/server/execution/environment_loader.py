"""
Environment loader for resolving environment templates and configurations.

This module handles loading environment templates from YAML files and resolving
them to complete EnvironmentSpec instances.
"""

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..execution.exceptions import InvalidEnvironmentSpecException
from ..execution.sandbox.environment_spec import (
    EnvironmentSpec,
    HealthCheck,
    NetworkSpec,
    PermanentEnvironmentSpec,
    PermanentNetworkSpec,
    PermanentServiceSpec,
    ServiceSpec,
)

logger = logging.getLogger(__name__)


class EnvironmentLoader:
    """
    Loads and resolves environment configurations from YAML templates.

    Supports template references, granular configuration, and hybrid approaches.
    """

    def __init__(self, environments_file_path: str):
        """
        Initialize EnvironmentLoader with configuration file.

        Args:
            environments_file_path: Path to environments.yaml configuration file
        """
        self.environments_file_path = Path(environments_file_path)
        self._templates_cache: Optional[Dict[str, Any]] = None

        logger.info(f"EnvironmentLoader initialized with config: {self.environments_file_path}")

    def resolve_environment(self, task_env_config: Any) -> EnvironmentSpec:
        """
        Resolve environment configuration to EnvironmentSpec.

        Args:
            task_env_config: Environment configuration from task YAML

        Returns:
            EnvironmentSpec instance

        Raises:
            InvalidEnvironmentSpecException: If configuration is invalid
        """
        if isinstance(task_env_config, str):
            # Template reference: "dvwa_pentest"
            return self.load_template(task_env_config)
        elif isinstance(task_env_config, dict):
            if "base_template" in task_env_config:
                # Hybrid: template + additions
                return self._merge_template_with_additions(task_env_config)
            else:
                # Granular: full config
                return self.build_from_granular(task_env_config)
        else:
            raise InvalidEnvironmentSpecException(f"Invalid environment configuration type: {type(task_env_config)}")

    def load_template(self, template_name: str) -> EnvironmentSpec:
        """
        Load environment template by name.

        Args:
            template_name: Name of the environment template

        Returns:
            EnvironmentSpec instance

        Raises:
            InvalidEnvironmentSpecException: If template not found or invalid
        """
        templates = self._load_templates()

        if template_name not in templates.get("environments", {}):
            raise InvalidEnvironmentSpecException(f"Environment template '{template_name}' not found")

        template_config = templates["environments"][template_name]
        return self._build_from_template_config(template_config, templates)

    def build_from_granular(self, config: Dict[str, Any]) -> EnvironmentSpec:
        """
        Build EnvironmentSpec from granular configuration.

        Args:
            config: Granular environment configuration

        Returns:
            EnvironmentSpec instance
        """
        templates = self._load_templates()

        # Get network configuration (support both single network and list of networks)
        network_config = config.get("network")
        if not network_config:
            raise InvalidEnvironmentSpecException("Network must be specified in granular config")

        network_specs = []
        if isinstance(network_config, str):
            # Single network
            network_spec = self._resolve_network(network_config, templates)
            network_specs.append(network_spec)
        elif isinstance(network_config, list):
            # Multiple networks
            for network_name in network_config:
                network_spec = self._resolve_network(network_name, templates)
                network_specs.append(network_spec)
        else:
            raise InvalidEnvironmentSpecException("Network must be a string or list of strings")

        # Get execution service
        execution_service = config.get("execution")
        if not execution_service:
            raise InvalidEnvironmentSpecException("Execution service must be specified in granular config")

        execution_config = self._resolve_container_config(execution_service, templates)

        # Get target services
        target_services = []
        for service_config in config.get("services", []):
            service_spec = self._resolve_service_config(service_config, templates)
            target_services.append(service_spec)

        # Resource limits
        resource_limits = config.get("resource_limits", {})

        return EnvironmentSpec(
            networks=network_specs,
            execution_service=execution_service,
            execution_config=execution_config,
            target_services=target_services,
            resource_limits=resource_limits,
        )

    def load_permanent_environment(self, template_name: str) -> PermanentEnvironmentSpec:
        """
        Load permanent environment specification by template name.

        Args:
            template_name: Name of the permanent environment template

        Returns:
            PermanentEnvironmentSpec instance

        Raises:
            InvalidEnvironmentSpecException: If template not found or invalid
        """
        templates = self._load_templates()

        if template_name not in templates.get("environments", {}):
            raise InvalidEnvironmentSpecException(f"Permanent environment template '{template_name}' not found")

        template_config = templates["environments"][template_name]

        # Validate this is a permanent environment
        if not template_config.get("permanent", False):
            raise InvalidEnvironmentSpecException(f"Environment '{template_name}' is not marked as permanent")

        return self._build_permanent_environment_spec(template_config, templates)

    def _build_permanent_environment_spec(
        self, config: Dict[str, Any], templates: Dict[str, Any]
    ) -> PermanentEnvironmentSpec:
        """
        Build PermanentEnvironmentSpec from configuration.

        Args:
            config: Permanent environment configuration
            templates: All available templates

        Returns:
            PermanentEnvironmentSpec instance
        """
        services = {}
        networks = {}

        # Get network configuration (support both singular and plural)
        network_name = config.get("network")
        network_names = config.get("networks", [])

        if network_name:
            network_spec = self._resolve_permanent_network(network_name, templates)
            networks[network_name] = network_spec

        for network_name in network_names:
            if network_name not in networks:  # Avoid duplicates
                network_spec = self._resolve_permanent_network(network_name, templates)
                networks[network_name] = network_spec

        # Get services
        for service_config in config.get("services", []):
            service_spec = self._resolve_permanent_service_config(service_config, templates)
            services[service_spec.name] = service_spec

        return PermanentEnvironmentSpec(
            services=services,
            networks=networks,
        )

    def _resolve_permanent_network(self, network_name: str, templates: Dict[str, Any]) -> PermanentNetworkSpec:
        """Resolve permanent network configuration by name."""
        network_templates = templates.get("networks", {})
        if network_name not in network_templates:
            raise InvalidEnvironmentSpecException(f"Network '{network_name}' not found in templates")

        network_config = network_templates[network_name]

        return PermanentNetworkSpec(
            name=network_name,
            driver=network_config.get("driver", "bridge"),
            internal=network_config.get("internal", False),
            ipam_config=network_config.get("ipam", {}),
        )

    def _resolve_permanent_service_config(
        self, service_config: Dict[str, Any], templates: Dict[str, Any]
    ) -> PermanentServiceSpec:
        """Resolve permanent service configuration."""
        service_name = service_config.get("name")
        container_name = service_config.get("container")

        if not service_name or not container_name:
            raise InvalidEnvironmentSpecException("Service config must have 'name' and 'container' fields")

        # Get container configuration
        container_config = self._resolve_container_config(container_name, templates)

        # Build health check if present (support both 'healthcheck' and 'health_check')
        health_check = None
        hc_config = container_config.get("healthcheck") or container_config.get("health_check")
        if hc_config:
            health_check = HealthCheck(
                test=hc_config.get("test", []),
                interval=hc_config.get("interval", "30s"),
                timeout=hc_config.get("timeout", "10s"),
                retries=hc_config.get("retries", 3),
                start_period=hc_config.get("start_period", "30s"),
            )

        # Get the image and ensure it's not None
        image = container_config.get("image")
        if image is None:
            raise InvalidEnvironmentSpecException(f"Container '{container_name}' must specify an image")

        return PermanentServiceSpec(
            name=service_name,
            image=image,
            ports=container_config.get("ports", []),
            environment=container_config.get("environment", []),
            volumes=container_config.get("volumes", []),
            health_check=health_check,
            resource_limits=container_config.get("resource_limits", {}),
        )

    def _merge_template_with_additions(self, config: Dict[str, Any]) -> EnvironmentSpec:
        """
        Merge base template with additional configuration.

        Args:
            config: Hybrid configuration with base_template and additions

        Returns:
            EnvironmentSpec instance
        """
        base_template_name = config["base_template"]
        base_spec = self.load_template(base_template_name)

        # Add additional services
        additional_services = config.get("additional_services", [])
        templates = self._load_templates()

        for service_config in additional_services:
            service_spec = self._resolve_service_config(service_config, templates)
            base_spec.target_services.append(service_spec)

        # Apply network overrides (apply to the primary network)
        network_overrides = config.get("network_overrides", {})
        if network_overrides:
            primary_network = base_spec.get_primary_network()
            for key, value in network_overrides.items():
                setattr(primary_network, key, value)

        # Apply resource limit overrides
        resource_overrides = config.get("resource_limits", {})
        if resource_overrides:
            base_spec.resource_limits.update(resource_overrides)

        return base_spec

    def _build_from_template_config(
        self, template_config: Dict[str, Any], templates: Dict[str, Any]
    ) -> EnvironmentSpec:
        """Build EnvironmentSpec from template configuration."""
        # Get network configuration (support both single network and list of networks)
        network_config = template_config.get("network")
        if not network_config:
            raise InvalidEnvironmentSpecException("Template must specify network")

        network_specs = []
        if isinstance(network_config, str):
            # Single network
            network_spec = self._resolve_network(network_config, templates)
            network_specs.append(network_spec)
        elif isinstance(network_config, list):
            # Multiple networks
            for network_name in network_config:
                network_spec = self._resolve_network(network_name, templates)
                network_specs.append(network_spec)
        else:
            raise InvalidEnvironmentSpecException("Network must be a string or list of strings")

        # Get execution service
        execution_service = template_config.get("execution")
        if not execution_service:
            raise InvalidEnvironmentSpecException("Template must specify execution service")

        execution_config = self._resolve_container_config(execution_service, templates)

        # Get target services
        target_services = []
        for service_config in template_config.get("services", []):
            service_spec = self._resolve_service_config(service_config, templates)
            target_services.append(service_spec)

        # Resource limits
        resource_limits = template_config.get("resource_limits", {})

        return EnvironmentSpec(
            networks=network_specs,
            execution_service=execution_service,
            execution_config=execution_config,
            target_services=target_services,
            resource_limits=resource_limits,
        )

    def _resolve_network(self, network_name: str, templates: Dict[str, Any]) -> NetworkSpec:
        """Resolve network configuration by name with dynamic subnet allocation."""
        networks = templates.get("networks", {})
        if network_name not in networks:
            raise InvalidEnvironmentSpecException(f"Network '{network_name}' not found in configuration")

        network_config = networks[network_name].copy()

        # Always allocate a unique subnet for session isolation
        unique_subnet = self._allocate_unique_subnet()

        # Ensure IPAM configuration exists
        if "ipam" not in network_config:
            network_config["ipam"] = {"config": [{}]}
        elif "config" not in network_config["ipam"]:
            network_config["ipam"]["config"] = [{}]
        elif not network_config["ipam"]["config"]:
            network_config["ipam"]["config"] = [{}]

        # Set the unique subnet in the first IPAM config
        network_config["ipam"]["config"][0]["subnet"] = unique_subnet
        logger.info(f"Allocated dynamic subnet {unique_subnet} for network {network_name}")

        return NetworkSpec(
            name=network_name,
            driver=network_config.get("driver", "bridge"),
            internal=network_config.get("internal", False),
            ipam_config=network_config.get("ipam", {}),
            options=network_config.get("options", {}),
        )

    def _allocate_unique_subnet(self) -> str:
        """
        Allocate a unique subnet for the session network.
        Uses sequential allocation in the 172.20.0.0/16 range.

        Returns:
            Unique subnet string (e.g., "172.20.1.0/24")
        """
        try:
            # Get list of existing Docker networks and their subnets
            existing_subnets = self._get_existing_subnets()

            # Find next available subnet in 172.20.x.0/24 range
            base_network = "172.20"
            for subnet_id in range(1, 255):  # 172.20.1.0/24 to 172.20.254.0/24
                candidate_subnet = f"{base_network}.{subnet_id}.0/24"
                if candidate_subnet not in existing_subnets:
                    return candidate_subnet

            # Fallback to 172.21.x.0/24 range if 172.20 is exhausted
            base_network = "172.21"
            for subnet_id in range(1, 255):
                candidate_subnet = f"{base_network}.{subnet_id}.0/24"
                if candidate_subnet not in existing_subnets:
                    return candidate_subnet

            # If both ranges exhausted, use hash-based fallback
            logger.warning("Standard subnet ranges exhausted, using hash-based allocation")
            return self._generate_hash_based_subnet()

        except Exception as e:
            logger.warning(f"Failed to allocate unique subnet, using fallback: {e}")
            return self._generate_hash_based_subnet()

    def _get_existing_subnets(self) -> set:
        """Get set of existing Docker network subnets."""
        try:
            # Query Docker networks for their IPAM configurations
            result = subprocess.run(
                ["docker", "network", "ls", "--format", "{{.Name}}"], capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                logger.warning("Failed to query Docker networks")
                return set()

            existing_subnets = set()
            network_names = result.stdout.strip().split("\n")

            for network_name in network_names:
                if not network_name or network_name in ["bridge", "host", "none"]:
                    continue

                # Inspect each network for subnet information
                inspect_result = subprocess.run(
                    [
                        "docker",
                        "network",
                        "inspect",
                        network_name,
                        "--format",
                        "{{range .IPAM.Config}}{{.Subnet}}{{end}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if inspect_result.returncode == 0 and inspect_result.stdout.strip():
                    subnet = inspect_result.stdout.strip()
                    if subnet and "/" in subnet:  # Valid CIDR notation
                        existing_subnets.add(subnet)

            logger.debug(f"Found existing subnets: {existing_subnets}")
            return existing_subnets

        except Exception as e:
            logger.warning(f"Error querying existing subnets: {e}")
            return set()

    def _generate_hash_based_subnet(self) -> str:
        """Generate subnet based on process ID and timestamp as fallback."""
        import os
        import time

        # Create unique identifier from PID and timestamp
        unique_id = f"{os.getpid()}-{int(time.time() * 1000)}"
        hash_obj = hashlib.md5(unique_id.encode())
        hash_int = int(hash_obj.hexdigest()[:4], 16)

        # Map to 172.22-31.x.x range for hash-based allocation
        subnet_second = 22 + (hash_int >> 8) % 10  # 172.22-31.x.x
        subnet_third = hash_int & 0xFF  # 172.x.0-255.x

        return f"172.{subnet_second}.{subnet_third}.0/24"

    def _resolve_container_config(self, container_name: str, templates: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve container configuration by name."""
        containers = templates.get("containers", {})
        if container_name not in containers:
            raise InvalidEnvironmentSpecException(f"Container '{container_name}' not found in configuration")

        container_config = containers[container_name].copy()

        # Convert to Docker Compose format
        compose_config = {}

        if "image" in container_config:
            compose_config["image"] = container_config["image"]
        if "command" in container_config:
            compose_config["command"] = container_config["command"]
        if "working_dir" in container_config:
            compose_config["working_dir"] = container_config["working_dir"]
        if "user" in container_config:
            compose_config["user"] = container_config["user"]
        if "environment" in container_config:
            compose_config["environment"] = container_config["environment"]
        if "volumes" in container_config:
            compose_config["volumes"] = container_config["volumes"]
        if "ports" in container_config:
            compose_config["ports"] = container_config["ports"]
        if "healthcheck" in container_config:
            compose_config["healthcheck"] = container_config["healthcheck"]
        if "health_check" in container_config:
            compose_config["health_check"] = container_config["health_check"]

        # Resource limits
        resource_limits = container_config.get("resource_limits", {})
        if resource_limits:
            if "memory" in resource_limits:
                compose_config["mem_limit"] = resource_limits["memory"]
            if "cpu" in resource_limits:
                compose_config["cpus"] = str(resource_limits["cpu"])

        return compose_config

    def _resolve_service_config(self, service_config: Dict[str, Any], templates: Dict[str, Any]) -> ServiceSpec:
        """Resolve service configuration to ServiceSpec."""
        service_name = service_config.get("name")
        container_name = service_config.get("container")

        if not service_name:
            raise InvalidEnvironmentSpecException("Service name must be specified")
        if not container_name:
            raise InvalidEnvironmentSpecException("Service container must be specified")

        # Get base container configuration
        containers = templates.get("containers", {})
        if container_name not in containers:
            raise InvalidEnvironmentSpecException(f"Container '{container_name}' not found in configuration")

        container_config = containers[container_name]

        # Create health check if specified (support both 'healthcheck' and 'health_check')
        health_check = None
        hc_config = container_config.get("healthcheck") or container_config.get("health_check")
        if hc_config:
            health_check = HealthCheck(
                test=hc_config["test"],
                interval=hc_config.get("interval", "30s"),
                timeout=hc_config.get("timeout", "10s"),
                retries=hc_config.get("retries", 3),
                start_period=hc_config.get("start_period", "30s"),
            )

        return ServiceSpec(
            name=service_name,
            container=container_name,
            image=container_config.get("image"),
            ports=container_config.get("ports", []),
            environment=container_config.get("environment", []),
            volumes=container_config.get("volumes", []),
            depends_on=container_config.get("depends_on", []),
            health_check=health_check,
            working_dir=container_config.get("working_dir"),
            user=container_config.get("user"),
            resource_limits=container_config.get("resource_limits", {}),
        )

    def _load_templates(self) -> Dict[str, Any]:
        """Load environment templates from YAML file with caching."""
        if self._templates_cache is not None:
            return self._templates_cache

        try:
            if not self.environments_file_path.exists():
                raise InvalidEnvironmentSpecException(f"Environments file not found: {self.environments_file_path}")

            with open(self.environments_file_path, "r", encoding="utf-8") as file:
                templates = yaml.safe_load(file)

            if not isinstance(templates, dict):
                raise InvalidEnvironmentSpecException("Environments file must contain a dictionary")

            self._templates_cache = templates
            logger.info(f"Loaded environment templates from {self.environments_file_path}")

            return templates

        except yaml.YAMLError as e:
            raise InvalidEnvironmentSpecException(f"YAML parsing error in environments file: {e}")
        except Exception as e:
            raise InvalidEnvironmentSpecException(f"Error loading environments file: {e}")
