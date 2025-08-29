"""
Unit tests for network specifications and external network handling.

Tests the core network specification functionality for permanent environments.
"""

import pytest

from saber.server.execution.sandbox.environment_spec import (
    PermanentEnvironmentSpec,
    PermanentNetworkSpec,
    PermanentServiceSpec,
)


class TestNetworkSpecifications:
    """Test network specification functionality."""

    def test_permanent_network_spec_external(self):
        """Test external network specification."""
        network_spec = PermanentNetworkSpec(
            name="shared_network",
            driver="bridge",
            external=True,
            external_name="excytin-shared-network",
            internal=False,
            ipam_config={}
        )

        assert network_spec.name == "shared_network"
        assert network_spec.external is True
        assert network_spec.external_name == "excytin-shared-network"
        assert network_spec.internal is False

    def test_permanent_network_spec_internal(self):
        """Test internal network specification."""
        network_spec = PermanentNetworkSpec(
            name="internal_network",
            driver="bridge",
            external=False,
            internal=True,
            ipam_config={"config": [{"subnet": "172.20.0.0/16"}]}
        )

        assert network_spec.name == "internal_network"
        assert network_spec.external is False
        assert network_spec.internal is True
        assert network_spec.ipam_config == {"config": [{"subnet": "172.20.0.0/16"}]}

    def test_permanent_network_spec_to_compose_external(self):
        """Test external network compose configuration."""
        network_spec = PermanentNetworkSpec(
            name="shared_network",
            driver="bridge",
            external=True,
            external_name="excytin-shared-network"
        )

        compose_config = network_spec.to_compose_network()

        expected = {
            "external": True,
            "name": "excytin-shared-network"
        }
        assert compose_config == expected

    def test_permanent_network_spec_to_compose_internal(self):
        """Test internal network compose configuration."""
        network_spec = PermanentNetworkSpec(
            name="internal_network",
            driver="bridge",
            internal=True,
            ipam_config={"config": [{"subnet": "172.20.0.0/16"}]}
        )

        compose_config = network_spec.to_compose_network()

        expected = {
            "driver": "bridge",
            "internal": True,
            "ipam": {"config": [{"subnet": "172.20.0.0/16"}]}
        }
        assert compose_config == expected

    def test_permanent_environment_spec_to_compose_dict(self):
        """Test permanent environment compose dictionary generation."""
        # Create service
        service_spec = PermanentServiceSpec(
            name="test_service",
            image="nginx:latest",
            ports=["80:80"]
        )

        # Create external network
        external_network = PermanentNetworkSpec(
            name="shared_network",
            external=True,
            external_name="excytin-shared-network"
        )

        # Create internal network
        internal_network = PermanentNetworkSpec(
            name="internal_network",
            driver="bridge",
            internal=True
        )

        # Create environment spec
        env_spec = PermanentEnvironmentSpec(
            services={"test_service": service_spec},
            networks={
                "shared_network": external_network,
                "internal_network": internal_network
            }
        )

        compose_config = env_spec.to_compose_dict()

        # Verify structure
        assert "version" in compose_config
        assert "services" in compose_config
        assert "networks" in compose_config
        assert "volumes" in compose_config

        # Verify services
        assert "test_service" in compose_config["services"]
        assert compose_config["services"]["test_service"]["image"] == "nginx:latest"

        # Verify networks
        assert "shared_network" in compose_config["networks"]
        assert compose_config["networks"]["shared_network"]["external"] is True
        assert compose_config["networks"]["shared_network"]["name"] == "excytin-shared-network"

        assert "internal_network" in compose_config["networks"]
        assert compose_config["networks"]["internal_network"]["internal"] is True

    def test_external_network_name_resolution(self):
        """Test that external network names are correctly resolved."""
        network_spec = PermanentNetworkSpec(
            name="local_name",
            external=True,
            external_name="actual-docker-network-name"
        )

        # When external=True, the compose config should use the external_name
        compose_config = network_spec.to_compose_network()
        assert compose_config["name"] == "actual-docker-network-name"
        assert compose_config["external"] is True

    def test_network_spec_dict_representation(self):
        """Test network spec dictionary representation."""
        network_spec = PermanentNetworkSpec(
            name="test_network",
            driver="bridge",
            external=True,
            external_name="external-name",
            internal=False,
            ipam_config={"config": [{"subnet": "172.20.0.0/16"}]}
        )

        dict_repr = network_spec.to_dict()

        expected = {
            "name": "test_network",
            "driver": "bridge",
            "external": True,
            "external_name": "external-name",
            "internal": False,
            "ipam": {"config": [{"subnet": "172.20.0.0/16"}]}
        }
        assert dict_repr == expected

    def test_service_spec_with_container_name(self):
        """Test service spec with container name for DNS resolution."""
        service_spec = PermanentServiceSpec(
            name="incident-5-db",
            image="saber-excytin-incident-5:latest",
            ports=["3306:3306"],
            environment=["MYSQL_ROOT_PASSWORD=admin"],
            container_name="saber-excytin-incident-5"
        )

        compose_config = service_spec.to_compose_service()

        expected = {
            "image": "saber-excytin-incident-5:latest",
            "restart": "unless-stopped",
            "container_name": "saber-excytin-incident-5",
            "ports": ["3306:3306"],
            "environment": ["MYSQL_ROOT_PASSWORD=admin"]
        }
        assert compose_config == expected

    def test_permanent_environment_spec_with_container_names(self):
        """Test permanent environment with container names in compose dict."""
        # Create service with container name
        service_spec = PermanentServiceSpec(
            name="incident-5-db",
            image="saber-excytin-incident-5:latest",
            ports=["3306:3306"],
            environment=["MYSQL_ROOT_PASSWORD=admin"],
            container_name="saber-excytin-incident-5"
        )

        # Create network
        network_spec = PermanentNetworkSpec(
            name="shared_network",
            external=True,
            external_name="excytin-shared-network"
        )

        # Create environment spec
        env_spec = PermanentEnvironmentSpec(
            services={"incident-5-db": service_spec},
            networks={"shared_network": network_spec}
        )

        compose_config = env_spec.to_compose_dict()

        # Verify container_name is preserved in the compose configuration
        assert "services" in compose_config
        assert "incident-5-db" in compose_config["services"]

        service_config = compose_config["services"]["incident-5-db"]
        assert service_config["container_name"] == "saber-excytin-incident-5"
        assert service_config["image"] == "saber-excytin-incident-5:latest"
