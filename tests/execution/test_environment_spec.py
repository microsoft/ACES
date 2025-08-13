"""
Unit tests for EnvironmentSpec classes.

Tests the environment specification system for multi-container orchestration.
"""

import pytest
from unittest.mock import Mock, patch

from saber.server.execution.sandbox.environment_spec import (
    EnvironmentSpec,
    ServiceSpec,
    NetworkSpec,
    HealthCheck,
)
from saber.server.execution.exceptions import InvalidEnvironmentSpecException


class TestHealthCheck:
    """Test HealthCheck configuration."""

    def test_health_check_creation(self):
        """Test creating a health check configuration."""
        health_check = HealthCheck(
            test=["CMD", "curl", "-f", "http://localhost/"],
            interval="30s",
            timeout="10s",
            retries=3,
            start_period="60s"
        )

        assert health_check.test == ["CMD", "curl", "-f", "http://localhost/"]
        assert health_check.interval == "30s"
        assert health_check.timeout == "10s"
        assert health_check.retries == 3
        assert health_check.start_period == "60s"

    def test_health_check_defaults(self):
        """Test health check with default values."""
        health_check = HealthCheck(test=["CMD", "echo", "ok"])

        assert health_check.test == ["CMD", "echo", "ok"]
        assert health_check.interval == "30s"
        assert health_check.timeout == "10s"
        assert health_check.retries == 3
        assert health_check.start_period == "30s"

    def test_to_compose_health_check(self):
        """Test conversion to Docker Compose format."""
        health_check = HealthCheck(
            test=["CMD", "curl", "-f", "http://localhost/"],
            interval="15s",
            timeout="5s",
            retries=5,
            start_period="30s"
        )

        compose_config = health_check.to_compose_health_check()

        expected = {
            "test": ["CMD", "curl", "-f", "http://localhost/"],
            "interval": "15s",
            "timeout": "5s",
            "retries": 5,
            "start_period": "30s"
        }

        assert compose_config == expected


class TestServiceSpec:
    """Test ServiceSpec configuration."""

    def test_service_spec_creation(self):
        """Test creating a service specification."""
        health_check = HealthCheck(test=["CMD", "echo", "ok"])

        service = ServiceSpec(
            name="webapp",
            container="nginx_container",
            image="nginx:latest",
            ports=["80", "443"],
            environment=["ENV=production"],
            volumes=["/data:/app/data"],
            depends_on=["database"],
            health_check=health_check,
            working_dir="/app",
            user="nginx:nginx",
            resource_limits={"memory": "512m", "cpu": "0.5"}
        )

        assert service.name == "webapp"
        assert service.container == "nginx_container"
        assert service.image == "nginx:latest"
        assert service.ports == ["80", "443"]
        assert service.environment == ["ENV=production"]
        assert service.volumes == ["/data:/app/data"]
        assert service.depends_on == ["database"]
        assert service.health_check == health_check
        assert service.working_dir == "/app"
        assert service.user == "nginx:nginx"
        assert service.resource_limits == {"memory": "512m", "cpu": "0.5"}

    def test_service_spec_defaults(self):
        """Test service spec with minimal configuration."""
        service = ServiceSpec(name="webapp", container="nginx_container")

        assert service.name == "webapp"
        assert service.container == "nginx_container"
        assert service.image is None
        assert service.ports == []
        assert service.environment == []
        assert service.volumes == []
        assert service.depends_on == []
        assert service.health_check is None
        assert service.working_dir is None
        assert service.user is None
        assert service.resource_limits == {}

    def test_to_compose_service(self):
        """Test conversion to Docker Compose service format."""
        health_check = HealthCheck(test=["CMD", "echo", "ok"])

        service = ServiceSpec(
            name="webapp",
            container="nginx_container",
            image="nginx:latest",
            ports=["80"],
            environment=["ENV=test"],
            volumes=["/data:/app"],
            depends_on=["db"],
            health_check=health_check,
            working_dir="/app",
            user="nginx",
            resource_limits={"memory": "1g", "cpu": "1.0"}
        )

        compose_config = service.to_compose_service()

        assert compose_config["image"] == "nginx:latest"
        assert compose_config["ports"] == ["80"]
        assert compose_config["environment"] == ["ENV=test"]
        assert compose_config["volumes"] == ["/data:/app"]
        assert compose_config["depends_on"] == ["db"]
        assert compose_config["working_dir"] == "/app"
        assert compose_config["user"] == "nginx"
        assert compose_config["mem_limit"] == "1g"
        assert compose_config["cpus"] == "1.0"
        assert "healthcheck" in compose_config

    def test_to_compose_service_minimal(self):
        """Test compose conversion with minimal configuration."""
        service = ServiceSpec(name="webapp", container="nginx_container")

        compose_config = service.to_compose_service()

        # Should be empty dict for minimal config
        assert compose_config == {}


class TestNetworkSpec:
    """Test NetworkSpec configuration."""

    def test_network_spec_creation(self):
        """Test creating a network specification."""
        network = NetworkSpec(
            name="test_network",
            driver="bridge",
            internal=True,
            ipam_config={"config": [{"subnet": "172.20.0.0/16"}]},
            options={"com.docker.network.bridge.name": "test-br"}
        )

        assert network.name == "test_network"
        assert network.driver == "bridge"
        assert network.internal is True
        assert network.ipam_config == {"config": [{"subnet": "172.20.0.0/16"}]}
        assert network.options == {"com.docker.network.bridge.name": "test-br"}

    def test_network_spec_defaults(self):
        """Test network spec with default values."""
        network = NetworkSpec(name="test_network")

        assert network.name == "test_network"
        assert network.driver == "bridge"
        assert network.internal is False
        assert network.ipam_config == {}
        assert network.options == {}

    def test_to_compose_network(self):
        """Test conversion to Docker Compose network format."""
        network = NetworkSpec(
            name="test_network",
            driver="overlay",
            internal=True,
            ipam_config={"config": [{"subnet": "172.20.0.0/16"}]},
            options={"encrypted": "true"}
        )

        compose_config = network.to_compose_network()

        expected = {
            "driver": "overlay",
            "internal": True,
            "ipam": {"config": [{"subnet": "172.20.0.0/16"}]},
            "driver_opts": {"encrypted": "true"}
        }

        assert compose_config == expected

    def test_to_compose_network_minimal(self):
        """Test compose conversion with minimal configuration."""
        network = NetworkSpec(name="test_network")

        compose_config = network.to_compose_network()

        expected = {
            "driver": "bridge",
            "internal": False
        }

        assert compose_config == expected


class TestEnvironmentSpec:
    """Test EnvironmentSpec configuration."""

    def test_environment_spec_creation(self):
        """Test creating an environment specification."""
        network = NetworkSpec(name="test_network")
        service = ServiceSpec(name="webapp", container="nginx_container")

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"},
            target_services=[service],
            resource_limits={"total_memory": "2g"}
        )

        assert env_spec.network == network
        assert env_spec.execution_service == "execution"
        assert env_spec.execution_config == {"image": "ubuntu:latest"}
        assert env_spec.target_services == [service]
        assert env_spec.resource_limits == {"total_memory": "2g"}

    def test_environment_spec_defaults(self):
        """Test environment spec with minimal configuration."""
        network = NetworkSpec(name="test_network")

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"}
        )

        assert env_spec.target_services == []
        assert env_spec.resource_limits == {}

    def test_get_execution_service(self):
        """Test getting execution service name."""
        network = NetworkSpec(name="test_network")
        env_spec = EnvironmentSpec(
            network=network,
            execution_service="my_executor",
            execution_config={"image": "ubuntu:latest"}
        )

        assert env_spec.get_execution_service() == "my_executor"

    def test_get_all_services(self):
        """Test getting all service names."""
        network = NetworkSpec(name="test_network")
        service1 = ServiceSpec(name="webapp", container="nginx")
        service2 = ServiceSpec(name="database", container="mysql")

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"},
            target_services=[service1, service2]
        )

        all_services = env_spec.get_all_services()
        assert all_services == ["execution", "webapp", "database"]

    def test_get_service_by_name(self):
        """Test getting service by name."""
        network = NetworkSpec(name="test_network")
        service1 = ServiceSpec(name="webapp", container="nginx")
        service2 = ServiceSpec(name="database", container="mysql")

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"},
            target_services=[service1, service2]
        )

        found_service = env_spec.get_service_by_name("webapp")
        assert found_service == service1

        not_found = env_spec.get_service_by_name("nonexistent")
        assert not_found is None

    def test_to_compose_dict(self):
        """Test generating Docker Compose configuration."""
        network = NetworkSpec(name="test_network")
        service = ServiceSpec(
            name="webapp",
            container="nginx",
            image="nginx:latest",
            ports=["80"]
        )

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="execution",
            execution_config={
                "image": "ubuntu:latest",
                "working_dir": "/workspace"
            },
            target_services=[service]
        )

        compose_dict = env_spec.to_compose_dict()

        # Check structure
        assert "version" in compose_dict
        assert "services" in compose_dict
        assert "networks" in compose_dict

        # Check services
        services = compose_dict["services"]
        assert "execution" in services
        assert "webapp" in services

        # Check execution service
        exec_service = services["execution"]
        assert exec_service["image"] == "ubuntu:latest"
        assert exec_service["working_dir"] == "/workspace"
        assert exec_service["networks"] == ["test_network"]

        # Check webapp service
        webapp_service = services["webapp"]
        assert webapp_service["image"] == "nginx:latest"
        assert webapp_service["ports"] == ["80"]
        assert webapp_service["networks"] == ["test_network"]

        # Check networks
        networks = compose_dict["networks"]
        assert "test_network" in networks

    def test_validate_success(self):
        """Test successful validation."""
        network = NetworkSpec(name="test_network")
        service = ServiceSpec(name="webapp", container="nginx")

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"},
            target_services=[service]
        )

        # Should not raise
        env_spec.validate()

    def test_validate_missing_execution_service(self):
        """Test validation with missing execution service."""
        network = NetworkSpec(name="test_network")

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="",
            execution_config={"image": "ubuntu:latest"}
        )

        with pytest.raises(InvalidEnvironmentSpecException, match="Execution service must be specified"):
            env_spec.validate()

    def test_validate_missing_network_name(self):
        """Test validation with missing network name."""
        network = NetworkSpec(name="")

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"}
        )

        with pytest.raises(InvalidEnvironmentSpecException, match="Network name must be specified"):
            env_spec.validate()

    def test_validate_duplicate_service_names(self):
        """Test validation with duplicate service names."""
        network = NetworkSpec(name="test_network")
        service1 = ServiceSpec(name="webapp", container="nginx1")
        service2 = ServiceSpec(name="webapp", container="nginx2")  # Duplicate name

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="webapp",  # Also duplicate
            execution_config={"image": "ubuntu:latest"},
            target_services=[service1, service2]
        )

        with pytest.raises(InvalidEnvironmentSpecException, match="Service names must be unique"):
            env_spec.validate()

    def test_validate_missing_dependency(self):
        """Test validation with missing dependency."""
        network = NetworkSpec(name="test_network")
        service = ServiceSpec(
            name="webapp",
            container="nginx",
            depends_on=["nonexistent_service"]
        )

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"},
            target_services=[service]
        )

        with pytest.raises(InvalidEnvironmentSpecException, match="depends on 'nonexistent_service' which is not defined"):
            env_spec.validate()

    def test_validate_valid_dependency(self):
        """Test validation with valid dependencies."""
        network = NetworkSpec(name="test_network")
        db_service = ServiceSpec(name="database", container="mysql")
        webapp_service = ServiceSpec(
            name="webapp",
            container="nginx",
            depends_on=["database"]
        )

        env_spec = EnvironmentSpec(
            network=network,
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"},
            target_services=[db_service, webapp_service]
        )

        # Should not raise
        env_spec.validate()
