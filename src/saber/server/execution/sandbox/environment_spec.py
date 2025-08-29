"""
Environment specification classes for multi-container orchestration.

This module defines specifications for Docker Compose-based execution environments,
supporting both template references and granular container configuration.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..exceptions import InvalidEnvironmentSpecException

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Health check configuration for a container service."""

    test: List[str]
    interval: str = "30s"
    timeout: str = "10s"
    retries: int = 3
    start_period: str = "30s"

    def to_compose_health_check(self) -> Dict[str, Any]:
        """Convert to Docker Compose health check format."""
        return {
            "test": self.test,
            "interval": self.interval,
            "timeout": self.timeout,
            "retries": self.retries,
            "start_period": self.start_period,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test": self.test,
            "interval": self.interval,
            "timeout": self.timeout,
            "retries": self.retries,
            "start_period": self.start_period,
        }


@dataclass
class ServiceSpec:
    """Specification for an individual container service."""

    name: str
    container: str  # Reference to container definition
    image: Optional[str] = None
    ports: List[str] = field(default_factory=list)
    environment: List[str] = field(default_factory=list)
    volumes: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    health_check: Optional[HealthCheck] = None
    working_dir: Optional[str] = None
    user: Optional[str] = None
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    def to_compose_service(self) -> Dict[str, Any]:
        """Convert to Docker Compose service definition."""
        service_config: Dict[str, Any] = {}

        if self.image:
            service_config["image"] = self.image

        if self.ports:
            service_config["ports"] = self.ports

        if self.environment:
            service_config["environment"] = self.environment

        if self.volumes:
            service_config["volumes"] = self.volumes

        if self.depends_on:
            service_config["depends_on"] = self.depends_on

        if self.health_check:
            service_config["healthcheck"] = self.health_check.to_compose_health_check()

        if self.working_dir:
            service_config["working_dir"] = self.working_dir

        if self.user:
            service_config["user"] = self.user

        # Add resource limits
        if self.resource_limits:
            if "memory" in self.resource_limits:
                service_config["mem_limit"] = self.resource_limits["memory"]
            if "cpu" in self.resource_limits:
                service_config["cpus"] = str(self.resource_limits["cpu"])

        return service_config

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "container": self.container,
            "image": self.image,
            "ports": self.ports,
            "environment": self.environment,
            "volumes": self.volumes,
            "depends_on": self.depends_on,
            "working_dir": self.working_dir,
            "user": self.user,
            "resource_limits": self.resource_limits,
        }
        if self.health_check:
            result["health_check"] = self.health_check.to_dict()
        return result


@dataclass
class NetworkSpec:
    """Network configuration specification."""

    name: str
    driver: str = "bridge"
    internal: bool = False
    external: bool = False
    external_name: Optional[str] = None
    ipam_config: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, str] = field(default_factory=dict)

    def to_compose_network(self) -> Dict[str, Any]:
        """Convert to Docker Compose network definition."""
        if self.external:
            # For external networks, only specify external property and name
            external_config: Dict[str, Any] = {"external": True}
            if self.external_name:
                external_config["name"] = self.external_name
            return external_config

        standard_config: Dict[str, Any] = {
            "driver": self.driver,
            "internal": self.internal,
        }

        if self.ipam_config:
            standard_config["ipam"] = self.ipam_config

        if self.options:
            standard_config["driver_opts"] = self.options

        return standard_config

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "driver": self.driver,
            "internal": self.internal,
            "external": self.external,
            "external_name": self.external_name,
            "ipam_config": self.ipam_config,
            "options": self.options,
        }


@dataclass
class PermanentServiceSpec:
    """Specification for a permanent service that persists across episodes."""

    name: str
    image: str
    ports: List[str] = field(default_factory=list)
    environment: List[str] = field(default_factory=list)
    volumes: List[str] = field(default_factory=list)
    health_check: Optional[HealthCheck] = None
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    container_name: Optional[str] = None  # Add container_name support
    command: Optional[str] = None  # Add command support

    def to_compose_service(self) -> Dict[str, Any]:
        """Convert to Docker Compose service definition."""
        service_config: Dict[str, Any] = {
            "image": self.image,
            "restart": "unless-stopped",  # Permanent services should restart
        }

        if self.container_name:
            service_config["container_name"] = self.container_name

        if self.command:
            service_config["command"] = self.command

        if self.ports:
            service_config["ports"] = self.ports

        if self.environment:
            service_config["environment"] = self.environment

        if self.volumes:
            service_config["volumes"] = self.volumes

        if self.health_check:
            service_config["healthcheck"] = self.health_check.to_compose_health_check()

        # Add resource limits
        if self.resource_limits:
            if "memory" in self.resource_limits:
                service_config["mem_limit"] = self.resource_limits["memory"]
            if "cpu" in self.resource_limits:
                service_config["cpus"] = str(self.resource_limits["cpu"])

        return service_config

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "image": self.image,
            "ports": self.ports,
            "environment": self.environment,
            "volumes": self.volumes,
            "resource_limits": self.resource_limits,
        }
        if self.container_name:
            result["container_name"] = self.container_name
        if self.command:
            result["command"] = self.command
        if self.health_check:
            result["healthcheck"] = self.health_check.to_dict()
        return result


@dataclass
class PermanentNetworkSpec:
    """Network specification for permanent networks."""

    name: str
    driver: str = "bridge"
    internal: bool = False
    external: bool = False
    external_name: Optional[str] = None
    ipam_config: Dict[str, Any] = field(default_factory=dict)

    def to_compose_network(self) -> Dict[str, Any]:
        """Convert to Docker Compose network definition."""
        if self.external:
            # For external networks, only specify external property and name
            external_config: Dict[str, Any] = {"external": True}
            if self.external_name:
                external_config["name"] = self.external_name
            return external_config
        else:
            return {
                "driver": self.driver,
                "internal": self.internal,
                "ipam": self.ipam_config,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "driver": self.driver,
            "internal": self.internal,
            "external": self.external,
            "external_name": self.external_name,
            "ipam": self.ipam_config,
        }


@dataclass
class PermanentEnvironmentSpec:
    """Specification for permanent environment that persists across episodes."""

    services: Dict[str, PermanentServiceSpec] = field(default_factory=dict)
    networks: Dict[str, PermanentNetworkSpec] = field(default_factory=dict)

    def get_service(self, name: str) -> Optional[PermanentServiceSpec]:
        """Get permanent service by name."""
        return self.services.get(name)

    def get_network(self, name: str) -> Optional[PermanentNetworkSpec]:
        """Get permanent network by name."""
        return self.networks.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "services": {name: service.to_dict() for name, service in self.services.items()},
            "networks": {name: network.to_dict() for name, network in self.networks.items()},
        }

    def to_compose_dict(self) -> Dict[str, Any]:
        """Generate Docker Compose configuration for permanent services."""
        compose_config: Dict[str, Any] = {
            "version": "3.8",
            "services": {},
            "networks": {},
            "volumes": {},
        }

        # Add services
        for service_name, service_spec in self.services.items():
            service_config = service_spec.to_compose_service()

            # Connect service to all available networks
            if self.networks:
                service_config["networks"] = list(self.networks.keys())

            compose_config["services"][service_name] = service_config

        # Add networks
        for network_name, network_spec in self.networks.items():
            compose_config["networks"][network_name] = network_spec.to_compose_network()

        # Extract volume definitions from services
        volumes = set()
        for service_spec in self.services.values():
            for volume in service_spec.volumes:
                if ":" in volume:
                    volume_name = volume.split(":")[0]
                    if not volume_name.startswith("/"):  # Named volume, not host bind mount
                        volumes.add(volume_name)

        # Add volume definitions
        for volume in volumes:
            compose_config["volumes"][volume] = {}

        return compose_config


@dataclass
class SandboxEnvironmentSpec:
    """Main sandbox environment specification for multi-container orchestration."""

    networks: List[NetworkSpec]  # List of networks to connect to
    execution_service: str  # Name of service used for command execution
    execution_config: Dict[str, Any]  # Configuration for execution container
    target_services: List[ServiceSpec] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    def get_primary_network(self) -> NetworkSpec:
        """Get the primary (first) network."""
        if not self.networks:
            raise InvalidEnvironmentSpecException("At least one network must be specified")
        return self.networks[0]

    def get_network_names(self) -> List[str]:
        """Get names of all networks."""
        return [network.name for network in self.networks]

    def get_execution_service(self) -> str:
        """Get the name of the execution service."""
        return self.execution_service

    def get_all_services(self) -> List[str]:
        """Get names of all services (execution + targets)."""
        services = [self.execution_service]
        services.extend([service.name for service in self.target_services])
        return services

    def get_service_by_name(self, name: str) -> Optional[ServiceSpec]:
        """Get service specification by name."""
        for service in self.target_services:
            if service.name == name:
                return service
        return None

    def to_compose_dict(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete Docker Compose configuration.

        Args:
            session_id: Session identifier for container labeling
        """
        services = {}

        # Add execution service
        exec_config = self.execution_config.copy()
        exec_config["networks"] = self.get_network_names()
        # Add SABER labels to execution service
        if session_id:
            if "labels" not in exec_config:
                exec_config["labels"] = []
            exec_config["labels"].extend(
                [
                    f"saber.session_id={session_id}",
                    "saber.role=execution",
                ]
            )
        services[self.execution_service] = exec_config

        # Add target services
        for service_spec in self.target_services:
            service_config = service_spec.to_compose_service()
            service_config["networks"] = self.get_network_names()
            # Add SABER labels to target services
            if session_id:
                if "labels" not in service_config:
                    service_config["labels"] = []
                service_config["labels"].extend(
                    [
                        f"saber.session_id={session_id}",
                        "saber.role=target",
                    ]
                )
            services[service_spec.name] = service_config

        # Build networks configuration
        networks = {}
        for network_spec in self.networks:
            networks[network_spec.name] = network_spec.to_compose_network()

        compose_config = {
            "version": "3.8",
            "services": services,
            "networks": networks,
        }

        return compose_config

    def validate(self) -> None:
        """Validate the environment specification."""
        if not self.execution_service:
            raise InvalidEnvironmentSpecException("Execution service must be specified")

        if not self.networks:
            raise InvalidEnvironmentSpecException("At least one network must be specified")

        for network in self.networks:
            if not network.name:
                raise InvalidEnvironmentSpecException("Network name must be specified")

        # Validate service names are unique
        all_services = self.get_all_services()
        if len(all_services) != len(set(all_services)):
            raise InvalidEnvironmentSpecException("Service names must be unique")

        # Validate dependencies exist
        service_names = set(all_services)
        for service in self.target_services:
            for dep in service.depends_on:
                if dep not in service_names:
                    raise InvalidEnvironmentSpecException(
                        f"Service '{service.name}' depends on '{dep}' which is not defined"
                    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "networks": {network.name: network.to_dict() for network in self.networks},
            "execution_service": self.execution_service,
            "execution_config": self.execution_config,
            "target_services": [service.to_dict() for service in self.target_services],
        }
        if self.resource_limits:
            result["resource_limits"] = self.resource_limits
        return result


# Compatibility alias
EnvironmentSpec = SandboxEnvironmentSpec
