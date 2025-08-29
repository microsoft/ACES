"""
Tests for permanent environment connectivity feature in environment loader.

This module tests the ability for sandbox environments to automatically connect
to permanent environment networks when permanent_environment_connectivity is enabled.
"""

import pytest
from unittest.mock import MagicMock, patch

from saber.server.execution.environment_loader import EnvironmentLoader
from saber.server.execution.sandbox.environment_spec import PermanentNetworkSpec
from saber.server.execution.exceptions import InvalidEnvironmentSpecException


class TestPermanentEnvironmentConnectivity:
    """Test permanent environment connectivity feature."""

    @pytest.fixture
    def mock_permanent_manager(self):
        """Create a mock permanent environment manager."""
        manager = MagicMock()
        manager.get_permanent_networks.return_value = ["excytin-shared-network"]

        # Create a mock permanent network spec with external name
        mock_network_spec = PermanentNetworkSpec(
            name="excytin-shared-network",
            external=True,
            external_name="excytin-shared-network"
        )
        manager.get_permanent_network_spec.return_value = mock_network_spec

        return manager

    @pytest.fixture
    def environments_config(self, tmp_path):
        """Create a temporary environments.yaml config file."""
        config_content = """
containers:
  saber-python-sandbox:
    image: saber-python-sandbox:latest
    container_name: saber-python-sandbox
    working_dir: /workspace
    environment:
      PYTHONPATH: /workspace
      PYTHONUNBUFFERED: "1"

environments:
  # Environment with permanent connectivity enabled
  python_sandbox_with_permanent:
    network: "excytin-shared-network"
    permanent_environment_connectivity: true
    execution: "saber-python-sandbox"
    resource_limits:
      total_memory: "512m"
      total_cpu: "0.5"
      execution_timeout: 900

  # Environment with permanent connectivity disabled
  python_sandbox_without_permanent:
    network: "excytin-shared-network"
    permanent_environment_connectivity: false
    execution: "saber-python-sandbox"
    resource_limits:
      total_memory: "512m"
      total_cpu: "0.5"
      execution_timeout: 900

  # Environment without permanent connectivity option (defaults to false)
  python_sandbox_default:
    network: "excytin-shared-network"
    execution: "saber-python-sandbox"
    resource_limits:
      total_memory: "512m"
      total_cpu: "0.5"
      execution_timeout: 900

networks:
  excytin-shared-network:
    driver: bridge
    internal: false
"""

        config_file = tmp_path / "environments.yaml"
        config_file.write_text(config_content)
        return str(config_file)

    def test_permanent_connectivity_enabled(self, environments_config, mock_permanent_manager):
        """Test sandbox uses permanent network when permanent_environment_connectivity is true."""
        # Create environment loader with permanent manager
        loader = EnvironmentLoader(environments_config, mock_permanent_manager)

        # Load environment with permanent connectivity enabled
        env_spec = loader.load_template("python_sandbox_with_permanent")

        # Verify the network is configured as external
        assert len(env_spec.networks) == 1
        network_spec = env_spec.networks[0]
        assert network_spec.name == "excytin-shared-network"
        assert network_spec.external is True
        assert network_spec.external_name == "excytin-shared-network"

        # Verify permanent manager methods were called
        mock_permanent_manager.get_permanent_networks.assert_called_once()
        mock_permanent_manager.get_permanent_network_spec.assert_called_once_with("excytin-shared-network")

    def test_permanent_connectivity_disabled(self, environments_config, mock_permanent_manager):
        """Test sandbox creates new network when permanent_environment_connectivity is false."""
        # Create environment loader with permanent manager
        loader = EnvironmentLoader(environments_config, mock_permanent_manager)

        # Mock the subnet allocation to avoid Docker calls
        with patch.object(loader, '_allocate_unique_subnet', return_value="172.20.1.0/24"):
            env_spec = loader.load_template("python_sandbox_without_permanent")

        # Verify the network is configured as internal (not external)
        assert len(env_spec.networks) == 1
        network_spec = env_spec.networks[0]
        assert network_spec.name == "excytin-shared-network"
        assert network_spec.external is False
        assert network_spec.external_name is None
        assert "172.20.1.0/24" in str(network_spec.ipam_config)

    def test_permanent_connectivity_default_false(self, environments_config, mock_permanent_manager):
        """Test sandbox creates new network when permanent_environment_connectivity is not specified."""
        # Create environment loader with permanent manager
        loader = EnvironmentLoader(environments_config, mock_permanent_manager)

        # Mock the subnet allocation to avoid Docker calls
        with patch.object(loader, '_allocate_unique_subnet', return_value="172.20.2.0/24"):
            env_spec = loader.load_template("python_sandbox_default")

        # Verify the network is configured as internal (not external)
        assert len(env_spec.networks) == 1
        network_spec = env_spec.networks[0]
        assert network_spec.name == "excytin-shared-network"
        assert network_spec.external is False
        assert network_spec.external_name is None

    def test_permanent_connectivity_without_manager(self, environments_config):
        """Test sandbox works normally when no permanent manager is provided."""
        # Create environment loader without permanent manager
        loader = EnvironmentLoader(environments_config, None)

        # Mock the subnet allocation to avoid Docker calls
        with patch.object(loader, '_allocate_unique_subnet', return_value="172.20.3.0/24"):
            env_spec = loader.load_template("python_sandbox_with_permanent")

        # Verify the network is configured as internal (permanent connectivity is ignored)
        assert len(env_spec.networks) == 1
        network_spec = env_spec.networks[0]
        assert network_spec.name == "excytin-shared-network"
        assert network_spec.external is False
        assert network_spec.external_name is None

    def test_permanent_network_not_found_fallback(self, environments_config):
        """Test fallback to regular network creation when permanent network is not found."""
        # Create mock manager that doesn't have the requested network
        mock_manager = MagicMock()
        mock_manager.get_permanent_networks.return_value = ["other-network"]

        loader = EnvironmentLoader(environments_config, mock_manager)

        # Mock the subnet allocation to avoid Docker calls
        with patch.object(loader, '_allocate_unique_subnet', return_value="172.20.4.0/24"):
            env_spec = loader.load_template("python_sandbox_with_permanent")

        # Verify it falls back to regular network creation
        assert len(env_spec.networks) == 1
        network_spec = env_spec.networks[0]
        assert network_spec.name == "excytin-shared-network"
        assert network_spec.external is False
        assert network_spec.external_name is None

    def test_granular_config_permanent_connectivity(self, environments_config, mock_permanent_manager):
        """Test permanent connectivity works with granular configuration."""
        loader = EnvironmentLoader(environments_config, mock_permanent_manager)

        granular_config = {
            "network": "excytin-shared-network",
            "permanent_environment_connectivity": True,
            "execution": "saber-python-sandbox",
            "services": [],
            "resource_limits": {
                "total_memory": "512m",
                "total_cpu": "0.5"
            }
        }

        env_spec = loader.build_from_granular(granular_config)

        # Verify the network is configured as external
        assert len(env_spec.networks) == 1
        network_spec = env_spec.networks[0]
        assert network_spec.name == "excytin-shared-network"
        assert network_spec.external is True
        assert network_spec.external_name == "excytin-shared-network"

    def test_multiple_networks_permanent_connectivity(self, environments_config, mock_permanent_manager):
        """Test permanent connectivity works with multiple networks."""
        # Update mock to return multiple networks
        mock_permanent_manager.get_permanent_networks.return_value = ["excytin-shared-network", "another-network"]

        # Add network specs for both networks
        def mock_get_network_spec(network_name):
            if network_name == "excytin-shared-network":
                return PermanentNetworkSpec(
                    name="excytin-shared-network",
                    external=True,
                    external_name="excytin-shared-network"
                )
            elif network_name == "another-network":
                return PermanentNetworkSpec(
                    name="another-network",
                    external=True,
                    external_name="another-shared-network"
                )
            return None

        mock_permanent_manager.get_permanent_network_spec.side_effect = mock_get_network_spec

        loader = EnvironmentLoader(environments_config, mock_permanent_manager)

        granular_config = {
            "network": ["excytin-shared-network", "another-network"],
            "permanent_environment_connectivity": True,
            "execution": "saber-python-sandbox",
            "services": [],
            "resource_limits": {"total_memory": "512m"}
        }

        env_spec = loader.build_from_granular(granular_config)

        # Verify both networks are configured as external
        assert len(env_spec.networks) == 2

        network_names = [net.name for net in env_spec.networks]
        assert "excytin-shared-network" in network_names
        assert "another-network" in network_names

        for network_spec in env_spec.networks:
            assert network_spec.external is True
            if network_spec.name == "excytin-shared-network":
                assert network_spec.external_name == "excytin-shared-network"
            elif network_spec.name == "another-network":
                assert network_spec.external_name == "another-shared-network"
