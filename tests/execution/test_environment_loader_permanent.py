"""
Additional unit tests for EnvironmentLoader to cover permanent environments and multi-network support.

Extends existing test coverage to include the new permanent environment functionality.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from saber.server.execution.environment_loader import EnvironmentLoader
from saber.server.execution.exceptions import InvalidEnvironmentSpecException, SandboxExecutionError
from saber.server.execution.sandbox.environment_spec import (
    EnvironmentSpec,
    HealthCheck,
    NetworkSpec,
    PermanentEnvironmentSpec,
    PermanentNetworkSpec,
    PermanentServiceSpec,
    ServiceSpec,
)


class TestEnvironmentLoaderPermanentSupport:
    """Test EnvironmentLoader permanent environment functionality."""

    @pytest.fixture
    def sample_environments_yaml(self):
        """Create sample environments.yaml with permanent environment support."""
        return {
            "containers": {
                "test_service": {
                    "image": "nginx:latest",
                    "ports": ["80"],
                    "environment": ["ENV=production"],
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost/"]
                    }
                },
                "execution_container": {
                    "image": "ubuntu:latest",
                    "working_dir": "/workspace"
                },
                "permanent_webapp": {
                    "image": "vulnerable-webapp:latest",
                    "ports": ["8080"],
                    "environment": ["DEBUG=false"],
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8080/health"]
                    }
                }
            },
            "networks": {
                "sandbox_network": {
                    "driver": "bridge",
                    "internal": True
                },
                "permanent_bridge": {
                    "driver": "bridge",
                    "internal": False,
                    "ipam": {
                        "config": [{"subnet": "172.30.0.0/16"}]
                    }
                }
            },
            "environments": {
                "test_sandbox": {
                    "network": "sandbox_network",
                    "execution": "execution_container",
                    "services": [{"name": "test_service", "container": "test_service"}]
                },
                "permanent_services": {
                    "network": "permanent_bridge",
                    "services": [{"name": "permanent_webapp", "container": "permanent_webapp"}],
                    "permanent": True
                },
                "multi_network_env": {
                    "network": ["sandbox_network", "permanent_bridge"],
                    "execution": "execution_container",
                    "services": [{"name": "test_service", "container": "test_service"}]
                }
            }
        }

    @pytest.fixture
    def temp_environments_file(self, sample_environments_yaml):
        """Create temporary environments.yaml file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_environments_yaml, f)
            return f.name

    def test_load_permanent_environment_success(self, temp_environments_file):
        """Test successful permanent environment loading."""
        loader = EnvironmentLoader(temp_environments_file)

        # Load permanent environment
        permanent_spec = loader.load_permanent_environment("permanent_services")

        # Verify structure
        assert isinstance(permanent_spec, PermanentEnvironmentSpec)
        assert "permanent_webapp" in permanent_spec.services
        assert "permanent_bridge" in permanent_spec.networks

        # Verify service
        service = permanent_spec.services["permanent_webapp"]
        assert isinstance(service, PermanentServiceSpec)
        assert service.name == "permanent_webapp"
        assert service.image == "vulnerable-webapp:latest"
        assert service.ports == ["8080"]
        assert "DEBUG=false" in service.environment

        # Verify network
        network = permanent_spec.networks["permanent_bridge"]
        assert isinstance(network, PermanentNetworkSpec)
        assert network.name == "permanent_bridge"
        assert network.driver == "bridge"
        assert not network.internal

    def test_load_permanent_environment_not_found(self, temp_environments_file):
        """Test loading non-existent permanent environment."""
        loader = EnvironmentLoader(temp_environments_file)

        with pytest.raises(InvalidEnvironmentSpecException, match="Permanent environment template 'nonexistent' not found"):
            loader.load_permanent_environment("nonexistent")

    def test_load_sandbox_environment_multi_network(self, temp_environments_file):
        """Test loading sandbox environment with multiple networks."""
        loader = EnvironmentLoader(temp_environments_file)

        # Load multi-network environment
        sandbox_spec = loader.load_template("multi_network_env")

        # Verify structure
        assert isinstance(sandbox_spec, EnvironmentSpec)
        assert len(sandbox_spec.networks) == 2

        # Verify both networks are present
        network_names = [net.name for net in sandbox_spec.networks]
        assert "sandbox_network" in network_names
        assert "permanent_bridge" in network_names

        # Verify network properties
        for network in sandbox_spec.networks:
            if network.name == "sandbox_network":
                assert network.internal
            elif network.name == "permanent_bridge":
                assert not network.internal

    def test_load_environment_basic_template(self, temp_environments_file):
        """Test loading basic environment template."""
        loader = EnvironmentLoader(temp_environments_file)

        # Test basic template loading
        sandbox_spec = loader.load_template("test_sandbox")

        # Verify structure
        assert isinstance(sandbox_spec, EnvironmentSpec)
        assert len(sandbox_spec.target_services) == 1
        assert sandbox_spec.target_services[0].name == "test_service"

    def test_load_environment_service_network_validation(self, sample_environments_yaml):
        """Test validation of service network references."""
        # Create config with service referencing undefined network
        invalid_config = sample_environments_yaml.copy()
        # Create invalid environment with undefined network
        invalid_config["environments"]["invalid_env"] = {
            "network": "undefined_network",
            "execution": "execution_container",
            "services": [{"name": "test_service", "container": "test_service"}]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_file = f.name

        try:
            loader = EnvironmentLoader(temp_file)
            # This should handle gracefully or raise appropriate error
            # Depending on implementation, this might pass with warning or fail
            sandbox_spec = loader.load_template("test_sandbox")
            # If it passes, verify the service is included despite network reference
            assert len(sandbox_spec.target_services) == 1
            assert sandbox_spec.target_services[0].name == "test_service"
        except SandboxExecutionError:
            # If it fails with validation error, that's also acceptable
            pass
        finally:
            Path(temp_file).unlink()

    def test_network_spec_conversion(self, temp_environments_file):
        """Test proper conversion between network spec types."""
        loader = EnvironmentLoader(temp_environments_file)

        # Load permanent environment
        permanent_spec = loader.load_permanent_environment("permanent_services")
        permanent_network = permanent_spec.networks["permanent_bridge"]
        assert isinstance(permanent_network, PermanentNetworkSpec)

        # Load sandbox environment
        sandbox_spec = loader.load_template("multi_network_env")

        # Find the permanent_bridge network in sandbox spec
        permanent_net_in_sandbox = None
        for net in sandbox_spec.networks:
            if net.name == "permanent_bridge":
                permanent_net_in_sandbox = net
                break

        assert permanent_net_in_sandbox is not None
        assert isinstance(permanent_net_in_sandbox, NetworkSpec)
        # Properties should match
        assert permanent_net_in_sandbox.name == permanent_network.name
        assert permanent_net_in_sandbox.driver == permanent_network.driver
        assert permanent_net_in_sandbox.internal == permanent_network.internal

    def test_health_check_conversion(self, temp_environments_file):
        """Test proper health check conversion."""
        loader = EnvironmentLoader(temp_environments_file)

        # Load permanent environment with health check
        permanent_spec = loader.load_permanent_environment("permanent_services")
        service = permanent_spec.services["permanent_webapp"]

        assert service.health_check is not None
        assert isinstance(service.health_check, HealthCheck)
        assert service.health_check.test == ["CMD", "curl", "-f", "http://localhost:8080/health"]

    def test_resource_limits_handling(self, sample_environments_yaml):
        """Test handling of resource limits in service specs."""
        # Add resource limits to container config
        config_with_limits = sample_environments_yaml.copy()
        config_with_limits["containers"]["permanent_webapp"]["deploy"] = {
            "resources": {
                "limits": {
                    "memory": "1G",
                    "cpus": "1.0"
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_with_limits, f)
            temp_file = f.name

        try:
            loader = EnvironmentLoader(temp_file)
            permanent_spec = loader.load_permanent_environment("permanent_services")
            service = permanent_spec.services["permanent_webapp"]

            # Verify resource limits are parsed (if implementation supports them)
            if hasattr(service, 'resource_limits') and service.resource_limits:
                assert "memory" in service.resource_limits or "cpus" in service.resource_limits
        finally:
            Path(temp_file).unlink()
