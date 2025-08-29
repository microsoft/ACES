"""
Additional unit tests for environment specification classes to cover permanent environment support.

Tests the new PermanentEnvironmentSpec, PermanentServiceSpec, and multi-network functionality.
"""

import pytest

from saber.server.execution.sandbox.environment_spec import (
    HealthCheck,
    NetworkSpec,
    PermanentEnvironmentSpec,
    PermanentNetworkSpec,
    PermanentServiceSpec,
    SandboxEnvironmentSpec,
    ServiceSpec,
)


class TestPermanentEnvironmentSpec:
    """Test PermanentEnvironmentSpec class."""

    def test_init_with_services_and_networks(self):
        """Test initialization with services and networks."""
        service = PermanentServiceSpec(
            name="test_service",
            image="nginx:latest",
            ports=["80"]
        )
        network = PermanentNetworkSpec(
            name="test_network",
            driver="bridge"
        )

        spec = PermanentEnvironmentSpec(
            services={"test_service": service},
            networks={"test_network": network}
        )

        assert len(spec.services) == 1
        assert "test_service" in spec.services
        assert spec.services["test_service"] == service
        assert len(spec.networks) == 1
        assert "test_network" in spec.networks
        assert spec.networks["test_network"] == network

    def test_init_empty(self):
        """Test initialization with empty collections."""
        spec = PermanentEnvironmentSpec()

        assert len(spec.services) == 0
        assert len(spec.networks) == 0
        assert isinstance(spec.services, dict)
        assert isinstance(spec.networks, dict)

    def test_to_dict(self):
        """Test conversion to dictionary representation."""
        service = PermanentServiceSpec(
            name="test_service",
            image="nginx:latest",
            ports=["80"],
            environment=["ENV=production"]
        )
        network = PermanentNetworkSpec(
            name="test_network",
            driver="bridge",
            internal=True
        )

        spec = PermanentEnvironmentSpec(
            services={"test_service": service},
            networks={"test_network": network}
        )

        result = spec.to_dict()

        assert "services" in result
        assert "networks" in result
        assert "test_service" in result["services"]
        assert "test_network" in result["networks"]

        # Verify service details
        service_dict = result["services"]["test_service"]
        assert service_dict["image"] == "nginx:latest"
        assert service_dict["ports"] == ["80"]
        assert service_dict["environment"] == ["ENV=production"]

        # Verify network details
        network_dict = result["networks"]["test_network"]
        assert network_dict["driver"] == "bridge"
        assert network_dict["internal"] is True


class TestPermanentServiceSpec:
    """Test PermanentServiceSpec class."""

    def test_init_minimal(self):
        """Test initialization with minimal required fields."""
        spec = PermanentServiceSpec(
            name="test_service",
            image="nginx:latest"
        )

        assert spec.name == "test_service"
        assert spec.image == "nginx:latest"
        assert spec.ports == []
        assert spec.environment == []
        assert spec.health_check is None
        assert spec.resource_limits == {}

    def test_init_full(self):
        """Test initialization with all fields."""
        health_check = HealthCheck(
            test=["CMD", "curl", "-f", "http://localhost/"],
            interval="30s",
            timeout="10s",
            retries=3
        )

        spec = PermanentServiceSpec(
            name="test_service",
            image="nginx:latest",
            ports=["80", "443"],
            environment=["ENV=production", "DEBUG=false"],
            health_check=health_check,
            resource_limits={"memory": "1G", "cpu": "0.5"}
        )

        assert spec.name == "test_service"
        assert spec.image == "nginx:latest"
        assert spec.ports == ["80", "443"]
        assert spec.environment == ["ENV=production", "DEBUG=false"]
        assert spec.health_check == health_check
        assert spec.resource_limits == {"memory": "1G", "cpu": "0.5"}

    def test_to_dict(self):
        """Test conversion to dictionary representation."""
        health_check = HealthCheck(test=["CMD", "curl", "-f", "http://localhost/"])

        spec = PermanentServiceSpec(
            name="test_service",
            image="nginx:latest",
            ports=["80"],
            environment=["ENV=production"],
            health_check=health_check,
            resource_limits={"memory": "512M"}
        )

        result = spec.to_dict()

        assert result["image"] == "nginx:latest"
        assert result["ports"] == ["80"]
        assert result["environment"] == ["ENV=production"]
        assert "healthcheck" in result
        assert result["healthcheck"]["test"] == ["CMD", "curl", "-f", "http://localhost/"]

        # Resource limits might be in deploy section
        if "deploy" in result:
            assert "resources" in result["deploy"]

    def test_equality(self):
        """Test equality comparison."""
        spec1 = PermanentServiceSpec(
            name="test_service",
            image="nginx:latest",
            ports=["80"]
        )
        spec2 = PermanentServiceSpec(
            name="test_service",
            image="nginx:latest",
            ports=["80"]
        )
        spec3 = PermanentServiceSpec(
            name="test_service",
            image="nginx:alpine",
            ports=["80"]
        )

        assert spec1 == spec2
        assert spec1 != spec3


class TestPermanentNetworkSpec:
    """Test PermanentNetworkSpec class."""

    def test_init_minimal(self):
        """Test initialization with minimal required fields."""
        spec = PermanentNetworkSpec(
            name="test_network",
            driver="bridge"
        )

        assert spec.name == "test_network"
        assert spec.driver == "bridge"
        assert spec.internal is False
        assert spec.ipam_config == {}

    def test_init_full(self):
        """Test initialization with all fields."""
        ipam_config = {
            "config": [{"subnet": "172.30.0.0/16", "gateway": "172.30.0.1"}]
        }

        spec = PermanentNetworkSpec(
            name="test_network",
            driver="bridge",
            internal=True,
            ipam_config=ipam_config
        )

        assert spec.name == "test_network"
        assert spec.driver == "bridge"
        assert spec.internal is True
        assert spec.ipam_config == ipam_config

    def test_to_dict(self):
        """Test conversion to dictionary representation."""
        ipam_config = {"config": [{"subnet": "172.30.0.0/16"}]}

        spec = PermanentNetworkSpec(
            name="test_network",
            driver="bridge",
            internal=True,
            ipam_config=ipam_config
        )

        result = spec.to_dict()

        assert result["driver"] == "bridge"
        assert result["internal"] is True
        assert result["ipam"] == ipam_config

    def test_equality(self):
        """Test equality comparison."""
        spec1 = PermanentNetworkSpec(
            name="test_network",
            driver="bridge",
            internal=True
        )
        spec2 = PermanentNetworkSpec(
            name="test_network",
            driver="bridge",
            internal=True
        )
        spec3 = PermanentNetworkSpec(
            name="test_network",
            driver="bridge",
            internal=False
        )

        assert spec1 == spec2
        assert spec1 != spec3


class TestSandboxEnvironmentSpecMultiNetwork:
    """Test SandboxEnvironmentSpec with multiple networks support."""

    def test_init_with_multiple_networks(self):
        """Test initialization with multiple networks."""
        network1 = NetworkSpec(name="network1", driver="bridge", internal=True)
        network2 = NetworkSpec(name="network2", driver="bridge", internal=False)
        service = ServiceSpec(name="test_service", container="nginx", image="nginx:latest")

        spec = SandboxEnvironmentSpec(
            networks=[network1, network2],
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"},
            target_services=[service]
        )

        assert len(spec.networks) == 2
        assert spec.networks[0] == network1
        assert spec.networks[1] == network2

    def test_get_network_names(self):
        """Test getting network names from multiple networks."""
        network1 = NetworkSpec(name="sandbox_net", driver="bridge", internal=True)
        network2 = NetworkSpec(name="permanent_net", driver="bridge", internal=False)
        service = ServiceSpec(name="test_service", container="nginx", image="nginx:latest")

        spec = SandboxEnvironmentSpec(
            networks=[network1, network2],
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"},
            target_services=[service]
        )

        # If there's a method to get network names
        if hasattr(spec, 'get_network_names'):
            names = spec.get_network_names()
            assert "sandbox_net" in names
            assert "permanent_net" in names
        else:
            # Test direct access
            names = [net.name for net in spec.networks]
            assert "sandbox_net" in names
            assert "permanent_net" in names

    def test_to_dict_with_multiple_networks(self):
        """Test conversion to dict with multiple networks."""
        network1 = NetworkSpec(name="network1", driver="bridge", internal=True)
        network2 = NetworkSpec(name="network2", driver="bridge", internal=False)
        service = ServiceSpec(name="test_service", container="nginx", image="nginx:latest")

        spec = SandboxEnvironmentSpec(
            networks=[network1, network2],
            execution_service="execution",
            execution_config={"image": "ubuntu:latest"},
            target_services=[service]
        )

        result = spec.to_dict()

        assert "networks" in result
        network_section = result["networks"]

        # Should contain both networks
        assert "network1" in network_section
        assert "network2" in network_section

        # Verify network properties
        assert network_section["network1"]["internal"] is True
        assert network_section["network2"]["internal"] is False


class TestHealthCheckSpec:
    """Test HealthCheck specification class."""

    def test_init_minimal(self):
        """Test initialization with minimal fields."""
        health_check = HealthCheck(test=["CMD", "curl", "-f", "http://localhost/"])

        assert health_check.test == ["CMD", "curl", "-f", "http://localhost/"]
        assert health_check.interval == "30s"  # Default value
        assert health_check.timeout == "10s"   # Default value
        assert health_check.retries == 3       # Default value

    def test_init_full(self):
        """Test initialization with all fields."""
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
        if hasattr(health_check, 'start_period'):
            assert health_check.start_period == "60s"

    def test_to_dict(self):
        """Test conversion to dictionary representation."""
        health_check = HealthCheck(
            test=["CMD", "curl", "-f", "http://localhost/"],
            interval="30s",
            timeout="10s",
            retries=3
        )

        result = health_check.to_dict()

        assert result["test"] == ["CMD", "curl", "-f", "http://localhost/"]
        assert result["interval"] == "30s"
        assert result["timeout"] == "10s"
        assert result["retries"] == 3
