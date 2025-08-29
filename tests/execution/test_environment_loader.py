"""
Unit tests for EnvironmentLoader.

Tests the environment template loading and resolution system.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml

from saber.server.execution.environment_loader import EnvironmentLoader
from saber.server.execution.exceptions import InvalidEnvironmentSpecException
from saber.server.execution.sandbox.environment_spec import EnvironmentSpec


class TestEnvironmentLoader:
    """Test EnvironmentLoader functionality."""

    @pytest.fixture
    def sample_environments_data(self):
        """Sample environment configuration data."""
        return {
            "containers": {
                "execution_container": {
                    "image": "ubuntu:latest",
                    "working_dir": "/workspace",
                    "user": "user:user",
                    "resource_limits": {"memory": "512m", "cpu": "0.5"},
                },
                "webapp_container": {
                    "image": "nginx:latest",
                    "ports": ["80"],
                    "environment": ["ENV=production"],
                    "health_check": {
                        "test": ["CMD", "curl", "-f", "http://localhost/"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3,
                    },
                    "depends_on": ["database_container"],
                },
                "database_container": {
                    "image": "mysql:5.7",
                    "environment": ["MYSQL_ROOT_PASSWORD=root"],
                    "volumes": ["/data:/var/lib/mysql"],
                },
            },
            "networks": {
                "test_network": {
                    "driver": "bridge",
                    "internal": True,
                    "ipam": {"config": [{"subnet": "172.20.0.0/16"}]},
                },
                "external_network": {"driver": "bridge", "internal": False},
            },
            "environments": {
                "simple_env": {
                    "network": "test_network",
                    "execution": "execution_container",
                    "services": [{"name": "webapp", "container": "webapp_container"}],
                    "resource_limits": {"total_memory": "1g"},
                },
                "complex_env": {
                    "network": "test_network",
                    "execution": "execution_container",
                    "services": [
                        {"name": "webapp", "container": "webapp_container"},
                        {"name": "database", "container": "database_container"},
                    ],
                    "resource_limits": {"total_memory": "2g"},
                },
            },
        }

    @pytest.fixture
    def temp_environments_file(self, sample_environments_data):
        """Create temporary environments.yaml file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_environments_data, f)
            return f.name

    def test_init(self, temp_environments_file):
        """Test EnvironmentLoader initialization."""
        loader = EnvironmentLoader(temp_environments_file)
        assert loader.environments_file_path == Path(temp_environments_file)
        assert loader._templates_cache is None

    def test_load_templates_success(self, temp_environments_file):
        """Test successful template loading."""
        loader = EnvironmentLoader(temp_environments_file)
        templates = loader._load_templates()

        assert isinstance(templates, dict)
        assert "containers" in templates
        assert "networks" in templates
        assert "environments" in templates
        assert loader._templates_cache is not None

    def test_load_templates_caching(self, temp_environments_file):
        """Test template caching."""
        loader = EnvironmentLoader(temp_environments_file)

        # First load
        templates1 = loader._load_templates()

        # Second load should use cache
        templates2 = loader._load_templates()

        assert templates1 is templates2  # Same object reference

    def test_load_templates_file_not_found(self):
        """Test loading with non-existent file."""
        loader = EnvironmentLoader("nonexistent.yaml")

        with pytest.raises(InvalidEnvironmentSpecException, match="Environments file not found"):
            loader._load_templates()

    def test_load_templates_invalid_yaml(self):
        """Test loading with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = f.name

        loader = EnvironmentLoader(temp_file)

        with pytest.raises(InvalidEnvironmentSpecException, match="YAML parsing error"):
            loader._load_templates()

    def test_load_templates_not_dict(self):
        """Test loading when file doesn't contain a dictionary."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["not", "a", "dict"], f)
            temp_file = f.name

        loader = EnvironmentLoader(temp_file)

        with pytest.raises(InvalidEnvironmentSpecException, match="must contain a dictionary"):
            loader._load_templates()

    def test_load_template_success(self, temp_environments_file):
        """Test loading a specific template."""
        loader = EnvironmentLoader(temp_environments_file)
        env_spec = loader.load_template("simple_env")

        assert isinstance(env_spec, EnvironmentSpec)
        assert env_spec.execution_service == "execution_container"
        assert env_spec.get_primary_network().name == "test_network"
        assert len(env_spec.target_services) == 1
        assert env_spec.target_services[0].name == "webapp"

    def test_load_template_not_found(self, temp_environments_file):
        """Test loading non-existent template."""
        loader = EnvironmentLoader(temp_environments_file)

        with pytest.raises(InvalidEnvironmentSpecException, match="Environment template 'nonexistent' not found"):
            loader.load_template("nonexistent")

    def test_resolve_environment_string_reference(self, temp_environments_file):
        """Test resolving environment with string reference."""
        loader = EnvironmentLoader(temp_environments_file)
        env_spec = loader.resolve_environment("simple_env")

        assert isinstance(env_spec, EnvironmentSpec)
        assert env_spec.execution_service == "execution_container"

    def test_resolve_environment_granular_config(self, temp_environments_file):
        """Test resolving environment with granular configuration."""
        loader = EnvironmentLoader(temp_environments_file)

        config = {
            "network": "test_network",
            "execution": "execution_container",
            "services": [{"name": "webapp", "container": "webapp_container"}],
        }

        env_spec = loader.resolve_environment(config)

        assert isinstance(env_spec, EnvironmentSpec)
        assert env_spec.execution_service == "execution_container"
        assert env_spec.get_primary_network().name == "test_network"
        assert len(env_spec.target_services) == 1

    def test_resolve_environment_hybrid_config(self, temp_environments_file):
        """Test resolving environment with hybrid configuration."""
        loader = EnvironmentLoader(temp_environments_file)

        config = {
            "base_template": "simple_env",
            "additional_services": [{"name": "database", "container": "database_container"}],
            "resource_limits": {"total_memory": "3g"},
        }

        env_spec = loader.resolve_environment(config)

        assert isinstance(env_spec, EnvironmentSpec)
        assert env_spec.execution_service == "execution_container"
        assert len(env_spec.target_services) == 2  # original + additional
        assert env_spec.resource_limits["total_memory"] == "3g"

    def test_resolve_environment_invalid_type(self, temp_environments_file):
        """Test resolving environment with invalid configuration type."""
        loader = EnvironmentLoader(temp_environments_file)

        with pytest.raises(InvalidEnvironmentSpecException, match="Invalid environment configuration type"):
            loader.resolve_environment(123)  # Invalid type

    def test_build_from_granular_missing_network(self, temp_environments_file):
        """Test granular build with missing network."""
        loader = EnvironmentLoader(temp_environments_file)

        config = {"execution": "execution_container", "services": []}

        with pytest.raises(InvalidEnvironmentSpecException, match="Network must be specified"):
            loader.build_from_granular(config)

    def test_build_from_granular_missing_execution(self, temp_environments_file):
        """Test granular build with missing execution service."""
        loader = EnvironmentLoader(temp_environments_file)

        config = {"network": "test_network", "services": []}

        with pytest.raises(InvalidEnvironmentSpecException, match="Execution service must be specified"):
            loader.build_from_granular(config)

    def test_resolve_network_success(self, temp_environments_file):
        """Test resolving network configuration."""
        loader = EnvironmentLoader(temp_environments_file)
        templates = loader._load_templates()

        network_spec = loader._resolve_network("test_network", templates)

        assert network_spec.name == "test_network"
        assert network_spec.driver == "bridge"
        assert network_spec.internal is True

    def test_resolve_network_not_found(self, temp_environments_file):
        """Test resolving non-existent network."""
        loader = EnvironmentLoader(temp_environments_file)
        templates = loader._load_templates()

        with pytest.raises(InvalidEnvironmentSpecException, match="Network 'nonexistent' not found"):
            loader._resolve_network("nonexistent", templates)

    def test_resolve_container_config_success(self, temp_environments_file):
        """Test resolving container configuration."""
        loader = EnvironmentLoader(temp_environments_file)
        templates = loader._load_templates()

        config = loader._resolve_container_config("execution_container", templates)

        assert config["image"] == "ubuntu:latest"
        assert config["working_dir"] == "/workspace"
        assert config["user"] == "user:user"
        assert config["mem_limit"] == "512m"
        assert config["cpus"] == "0.5"

    def test_resolve_container_config_not_found(self, temp_environments_file):
        """Test resolving non-existent container."""
        loader = EnvironmentLoader(temp_environments_file)
        templates = loader._load_templates()

        with pytest.raises(InvalidEnvironmentSpecException, match="Container 'nonexistent' not found"):
            loader._resolve_container_config("nonexistent", templates)

    def test_resolve_service_config_success(self, temp_environments_file):
        """Test resolving service configuration."""
        loader = EnvironmentLoader(temp_environments_file)
        templates = loader._load_templates()

        service_config = {"name": "webapp", "container": "webapp_container"}

        service_spec = loader._resolve_service_config(service_config, templates)

        assert service_spec.name == "webapp"
        assert service_spec.container == "webapp_container"
        assert service_spec.image == "nginx:latest"
        assert service_spec.ports == ["80"]
        assert service_spec.environment == ["ENV=production"]
        assert service_spec.depends_on == ["database_container"]
        assert service_spec.health_check is not None

    def test_resolve_service_config_missing_name(self, temp_environments_file):
        """Test resolving service config without name."""
        loader = EnvironmentLoader(temp_environments_file)
        templates = loader._load_templates()

        service_config = {"container": "webapp_container"}

        with pytest.raises(InvalidEnvironmentSpecException, match="Service name must be specified"):
            loader._resolve_service_config(service_config, templates)

    def test_resolve_service_config_missing_container(self, temp_environments_file):
        """Test resolving service config without container."""
        loader = EnvironmentLoader(temp_environments_file)
        templates = loader._load_templates()

        service_config = {"name": "webapp"}

        with pytest.raises(InvalidEnvironmentSpecException, match="Service container must be specified"):
            loader._resolve_service_config(service_config, templates)

    def test_resolve_service_config_container_not_found(self, temp_environments_file):
        """Test resolving service config with non-existent container."""
        loader = EnvironmentLoader(temp_environments_file)
        templates = loader._load_templates()

        service_config = {"name": "webapp", "container": "nonexistent_container"}

        with pytest.raises(InvalidEnvironmentSpecException, match="Container 'nonexistent_container' not found"):
            loader._resolve_service_config(service_config, templates)

    def test_merge_template_with_additions(self, temp_environments_file):
        """Test merging template with additional services."""
        loader = EnvironmentLoader(temp_environments_file)

        config = {
            "base_template": "simple_env",
            "additional_services": [{"name": "database", "container": "database_container"}],
            "network_overrides": {"internal": False},
            "resource_limits": {"total_cpu": "2.0"},
        }

        env_spec = loader._merge_template_with_additions(config)

        # Check that additional service was added
        assert len(env_spec.target_services) == 2
        service_names = [s.name for s in env_spec.target_services]
        assert "webapp" in service_names
        assert "database" in service_names

        # Check network override was applied
        assert env_spec.get_primary_network().internal is False

        # Check resource limits were merged
        assert "total_memory" in env_spec.resource_limits
        assert "total_cpu" in env_spec.resource_limits
