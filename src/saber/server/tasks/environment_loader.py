"""
Environment loader for resolving environment templates and configurations.

This module handles loading environment templates from YAML files and resolving
them to complete EnvironmentSpec instances.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..execution.exceptions import InvalidEnvironmentSpecException
from ..execution.sandbox.environment_spec import EnvironmentSpec, HealthCheck, NetworkSpec, ServiceSpec

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

        # Get network configuration
        network_name = config.get("network")
        if not network_name:
            raise InvalidEnvironmentSpecException("Network must be specified in granular config")

        network_spec = self._resolve_network(network_name, templates)

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
            network=network_spec,
            execution_service=execution_service,
            execution_config=execution_config,
            target_services=target_services,
            resource_limits=resource_limits,
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

        # Apply network overrides
        network_overrides = config.get("network_overrides", {})
        if network_overrides:
            for key, value in network_overrides.items():
                setattr(base_spec.network, key, value)

        # Apply resource limit overrides
        resource_overrides = config.get("resource_limits", {})
        if resource_overrides:
            base_spec.resource_limits.update(resource_overrides)

        return base_spec

    def _build_from_template_config(
        self, template_config: Dict[str, Any], templates: Dict[str, Any]
    ) -> EnvironmentSpec:
        """Build EnvironmentSpec from template configuration."""
        # Get network
        network_name = template_config.get("network")
        if not network_name:
            raise InvalidEnvironmentSpecException("Template must specify network")

        network_spec = self._resolve_network(network_name, templates)

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
            network=network_spec,
            execution_service=execution_service,
            execution_config=execution_config,
            target_services=target_services,
            resource_limits=resource_limits,
        )

    def _resolve_network(self, network_name: str, templates: Dict[str, Any]) -> NetworkSpec:
        """Resolve network configuration by name."""
        networks = templates.get("networks", {})
        if network_name not in networks:
            raise InvalidEnvironmentSpecException(f"Network '{network_name}' not found in configuration")

        network_config = networks[network_name]
        return NetworkSpec(
            name=network_name,
            driver=network_config.get("driver", "bridge"),
            internal=network_config.get("internal", False),
            ipam_config=network_config.get("ipam", {}),
            options=network_config.get("options", {}),
        )

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

        # Create health check if specified
        health_check = None
        if "health_check" in container_config:
            hc_config = container_config["health_check"]
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
