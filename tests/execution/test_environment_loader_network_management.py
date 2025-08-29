"""
Unit tests for environment loader network management.

Tests the environment loader's ability to handle external networks
and extract network specifications for permanent environments.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from saber.server.execution.environment_loader import EnvironmentLoader
from saber.server.execution.exceptions import InvalidEnvironmentSpecException
from saber.server.execution.sandbox.environment_spec import (
    PermanentEnvironmentSpec,
    PermanentNetworkSpec,
)


class TestEnvironmentLoaderNetworkManagement:
    """Test environment loader network handling functionality."""

    @pytest.fixture
    def temp_environments_file(self):
        """Create a temporary environments.yaml file."""
        environments_config = {
            "containers": {
                "test-db": {
                    "image": "mysql:8.0",
                    "container_name": "test-db",
                    "ports": ["3306:3306"],
                    "environment": ["MYSQL_ROOT_PASSWORD=admin"],
                },
                "test-app": {
                    "image": "nginx:latest",
                    "container_name": "test-app",
                    "ports": ["80:80"],
                }
            },
            "networks": {
                "shared_network": {
                    "driver": "bridge",
                    "external": True,
                    "name": "excytin-shared-network"
                },
                "internal_network": {
                    "driver": "bridge",
                    "internal": True,
                    "ipam": {
                        "config": [{"subnet": "172.20.0.0/16"}]
                    }
                },
                "simple_network": {
                    "driver": "bridge"
                }
            },
            "environments": {
                "permanent_env": {
                    "permanent": True,
                    "network": "shared_network",
                    "services": [
                        {
                            "name": "database",
                            "container": "test-db"
                        }
                    ],
                    "resource_limits": {
                        "total_memory": "1g",
                        "total_cpu": "0.5",
                        "execution_timeout": None
                    }
                },
                "multi_network_env": {
                    "permanent": True,
                    "networks": ["shared_network", "internal_network"],
                    "services": [
                        {
                            "name": "app",
                            "container": "test-app"
                        }
                    ]
                },
                "sandbox_env": {
                    "network": "shared_network",
                    "execution": "test-app",
                    "resource_limits": {
                        "total_memory": "512m",
                        "total_cpu": "0.5",
                        "execution_timeout": 900
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(environments_config, f)
            yield f.name

        Path(f.name).unlink()

    @pytest.fixture
    def environment_loader(self, temp_environments_file):
        """Create an EnvironmentLoader instance."""
        return EnvironmentLoader(temp_environments_file)

    def test_load_permanent_environment_with_external_network(self, environment_loader):
        """Test loading permanent environment with external network."""
        permanent_spec = environment_loader.load_permanent_environment("permanent_env")

        assert permanent_spec is not None
        assert isinstance(permanent_spec, PermanentEnvironmentSpec)

        # Check that services were loaded
        assert "database" in permanent_spec.services

        # Check that external network was properly configured
        assert "shared_network" in permanent_spec.networks
        shared_network = permanent_spec.networks["shared_network"]

        assert shared_network.external is True
        assert shared_network.external_name == "excytin-shared-network"
        assert shared_network.driver == "bridge"

    def test_load_permanent_environment_with_multiple_networks(self, environment_loader):
        """Test loading permanent environment with multiple networks."""
        # Modify the environment to use multi_network_env as permanent
        with open(environment_loader.environments_file_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set multi_network_env as the only permanent environment
        config['environments']['permanent_env']['permanent'] = False
        config['environments']['multi_network_env']['permanent'] = True

        with open(environment_loader.environments_file_path, 'w') as f:
            yaml.dump(config, f)

        permanent_spec = environment_loader.load_permanent_environment("multi_network_env")

        assert permanent_spec is not None
        assert len(permanent_spec.networks) == 2

        # Check external network
        assert "shared_network" in permanent_spec.networks
        shared_network = permanent_spec.networks["shared_network"]
        assert shared_network.external is True
        assert shared_network.external_name == "excytin-shared-network"

        # Check internal network
        assert "internal_network" in permanent_spec.networks
        internal_network = permanent_spec.networks["internal_network"]
        assert internal_network.external is False
        assert internal_network.internal is True
        assert internal_network.ipam_config == {"config": [{"subnet": "172.20.0.0/16"}]}

    def test_load_permanent_environment_no_permanent_configured(self, environment_loader):
        """Test loading when no permanent environment is configured."""
        # Modify config to have no permanent environments
        with open(environment_loader.environments_file_path, 'r') as f:
            config = yaml.safe_load(f)

        config['environments']['permanent_env']['permanent'] = False

        with open(environment_loader.environments_file_path, 'w') as f:
            yaml.dump(config, f)

        # Should raise exception when trying to load non-permanent environment
        with pytest.raises(InvalidEnvironmentSpecException):
            environment_loader.load_permanent_environment("permanent_env")

    def test_resolve_permanent_network_external(self, environment_loader):
        """Test resolving external network specification."""
        templates = environment_loader._load_templates()

        network_spec = environment_loader._resolve_permanent_network("shared_network", templates)

        assert isinstance(network_spec, PermanentNetworkSpec)
        assert network_spec.name == "shared_network"
        assert network_spec.external is True
        assert network_spec.external_name == "excytin-shared-network"
        assert network_spec.driver == "bridge"

    def test_resolve_permanent_network_internal(self, environment_loader):
        """Test resolving internal network specification."""
        templates = environment_loader._load_templates()

        network_spec = environment_loader._resolve_permanent_network("internal_network", templates)

        assert isinstance(network_spec, PermanentNetworkSpec)
        assert network_spec.name == "internal_network"
        assert network_spec.external is False
        assert network_spec.internal is True
        assert network_spec.ipam_config == {"config": [{"subnet": "172.20.0.0/16"}]}

    def test_resolve_permanent_network_simple(self, environment_loader):
        """Test resolving simple network specification."""
        templates = environment_loader._load_templates()

        network_spec = environment_loader._resolve_permanent_network("simple_network", templates)

        assert isinstance(network_spec, PermanentNetworkSpec)
        assert network_spec.name == "simple_network"
        assert network_spec.external is False
        assert network_spec.internal is False
        assert network_spec.driver == "bridge"

    def test_permanent_environment_service_configuration(self, environment_loader):
        """Test that permanent environment services are properly configured."""
        permanent_spec = environment_loader.load_permanent_environment("permanent_env")

        assert permanent_spec is not None
        assert "database" in permanent_spec.services

        db_service = permanent_spec.services["database"]
        assert db_service.name == "database"
        assert db_service.image == "mysql:8.0"
        assert "3306:3306" in db_service.ports
        assert "MYSQL_ROOT_PASSWORD=admin" in db_service.environment

    def test_external_network_name_resolution(self, environment_loader):
        """Test that external network names are properly resolved."""
        permanent_spec = environment_loader.load_permanent_environment("permanent_env")

        # Get the compose configuration
        compose_config = permanent_spec.to_compose_dict()

        # Check that external networks reference the external name
        networks_config = compose_config.get("networks", {})
        shared_network_config = networks_config.get("shared_network", {})

        # External networks should have the external name and external flag
        if shared_network_config.get("external"):
            assert shared_network_config.get("name") == "excytin-shared-network"

    def test_invalid_permanent_environment_configuration(self, temp_environments_file):
        """Test handling of invalid permanent environment configuration."""
        # Create config with invalid network reference
        invalid_config = {
            "containers": {},
            "networks": {},
            "environments": {
                "invalid_env": {
                    "permanent": True,
                    "network": "nonexistent_network",
                    "services": []
                }
            }
        }

        with open(temp_environments_file, 'w') as f:
            yaml.dump(invalid_config, f)

        loader = EnvironmentLoader(temp_environments_file)

        with pytest.raises(InvalidEnvironmentSpecException):
            loader.load_permanent_environment("invalid_env")

    def test_permanent_environment_with_no_networks(self, temp_environments_file):
        """Test permanent environment with no network configuration."""
        config_no_networks = {
            "containers": {
                "test-app": {
                    "image": "nginx:latest",
                    "container_name": "test-app"
                }
            },
            "networks": {},
            "environments": {
                "permanent_env": {
                    "permanent": True,
                    "services": [
                        {
                            "name": "app",
                            "container": "test-app"
                        }
                    ]
                }
            }
        }

        with open(temp_environments_file, 'w') as f:
            yaml.dump(config_no_networks, f)

        loader = EnvironmentLoader(temp_environments_file)
        permanent_spec = loader.load_permanent_environment("permanent_env")

        assert permanent_spec is not None
        assert len(permanent_spec.networks) == 0
        assert len(permanent_spec.services) == 1
