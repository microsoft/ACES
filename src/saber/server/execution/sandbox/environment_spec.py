"""
Environment specification classes for multi-container orchestration.

This module defines specifications for Docker Compose-based execution environments,
supporting both template references and granular container configuration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..exceptions import InvalidEnvironmentSpecException


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


@dataclass
class NetworkSpec:
    """Network configuration specification."""

    name: str
    driver: str = "bridge"
    internal: bool = False
    ipam_config: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, str] = field(default_factory=dict)

    def to_compose_network(self) -> Dict[str, Any]:
        """Convert to Docker Compose network definition."""
        network_config = {
            "driver": self.driver,
            "internal": self.internal,
        }

        if self.ipam_config:
            network_config["ipam"] = self.ipam_config

        if self.options:
            network_config["driver_opts"] = self.options

        return network_config


@dataclass
class EnvironmentSpec:
    """Main environment specification for multi-container orchestration."""

    network: NetworkSpec
    execution_service: str  # Name of service used for command execution
    execution_config: Dict[str, Any]  # Configuration for execution container
    target_services: List[ServiceSpec] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)

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
        cleanup_token: Optional[str] = None,
        saber_host_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete Docker Compose configuration.

        Args:
            session_id: Session identifier for orchestrator integration
            cleanup_token: Cleanup token for orchestrator authentication
            saber_host_url: SABER server URL for orchestrator polling

        Raises:
            ValueError: If session_id is provided but cleanup_token is missing
        """
        # Validate orchestrator requirements
        if session_id and not cleanup_token:
            raise ValueError(
                "cleanup_token is required when session_id is provided. "
                "The orchestrator service requires both session_id and cleanup_token for proper integration."
            )

        services = {}

        # Add SABER orchestrator service (if session info provided)
        if session_id and cleanup_token:
            orchestrator_config = {
                "image": "saber-orchestrator:latest",
                "container_name": f"saber-orchestrator-{session_id}",
                "environment": [
                    f"SABER_SESSION_ID={session_id}",
                    f"SABER_CLEANUP_TOKEN={cleanup_token}",
                    f"SABER_HOST_URL={saber_host_url or 'http://host.docker.internal:8000'}",
                    "SABER_POLL_INTERVAL=30",
                    f"SABER_COMPOSE_PROJECT=saber-session-{session_id}",
                ],
                "volumes": ["/var/run/docker.sock:/var/run/docker.sock", ".:/app"],
                "labels": [
                    f"saber.session_id={session_id}",
                    "saber.role=orchestrator",
                    f"saber.cleanup_token={cleanup_token}",
                ],
                "restart": "unless-stopped",
                "networks": [self.network.name],
            }
            services["saber-orchestrator"] = orchestrator_config

        # Add execution service
        exec_config = self.execution_config.copy()
        exec_config["networks"] = [self.network.name]
        # Add SABER labels to execution service
        if session_id:
            if "labels" not in exec_config:
                exec_config["labels"] = []
            exec_config["labels"].extend(
                [
                    f"saber.session_id={session_id}",
                    "saber.role=execution",
                    f"saber.cleanup_token={cleanup_token}" if cleanup_token else f"saber.session_id={session_id}",
                ]
            )
            # Add dependency on orchestrator
            exec_config["depends_on"] = ["saber-orchestrator"]
        services[self.execution_service] = exec_config

        # Add target services
        for service_spec in self.target_services:
            service_config = service_spec.to_compose_service()
            service_config["networks"] = [self.network.name]
            # Add SABER labels to target services
            if session_id:
                if "labels" not in service_config:
                    service_config["labels"] = []
                service_config["labels"].extend(
                    [
                        f"saber.session_id={session_id}",
                        "saber.role=target",
                        f"saber.cleanup_token={cleanup_token}" if cleanup_token else f"saber.session_id={session_id}",
                    ]
                )
                # Add dependency on orchestrator
                if "depends_on" not in service_config:
                    service_config["depends_on"] = []
                service_config["depends_on"].append("saber-orchestrator")
            services[service_spec.name] = service_config

        # Add SABER labels to network
        network_config = self.network.to_compose_network()
        if session_id:
            if "labels" not in network_config:
                network_config["labels"] = []
            network_config["labels"].append(f"saber.session_id={session_id}")

        compose_config = {
            "version": "3.8",
            "services": services,
            "networks": {self.network.name: network_config},
        }

        return compose_config

    def validate(self) -> None:
        """Validate the environment specification."""
        if not self.execution_service:
            raise InvalidEnvironmentSpecException("Execution service must be specified")

        if not self.network.name:
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
