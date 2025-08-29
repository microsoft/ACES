"""
Unit tests for permanent environment lifecycle management.

Tests the new configuration change detection, network management, and server lifecycle integration.
"""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.sandbox.environment_spec import (
    PermanentEnvironmentSpec,
    PermanentNetworkSpec,
    PermanentServiceSpec,
)
from saber.server.execution.sandbox.permanent_environment_manager import PermanentEnvironmentManager


class TestPermanentEnvironmentLifecycle:
    """Test permanent environment lifecycle and configuration management."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def base_config(self, temp_config_dir):
        """Create base configuration for tests."""
        return {
            "domain": "test_domain",
            "config_dir": str(temp_config_dir),
            "logs_directory": "/app/logs",
            "enable_logging": True,
        }

    @pytest.fixture
    def sample_network_spec(self):
        """Create a sample network specification."""
        return PermanentNetworkSpec(
            name="test_network",
            driver="bridge",
            internal=False,
            external=False,
            external_name=None,
            ipam_config={"config": [{"subnet": "172.30.0.0/16"}]}
        )

    @pytest.fixture
    def external_network_spec(self):
        """Create an external network specification."""
        return PermanentNetworkSpec(
            name="shared_network",
            driver="bridge",
            internal=False,
            external=True,
            external_name="excytin-shared-network",
            ipam_config={}
        )

    @pytest.fixture
    def sample_service_spec(self):
        """Create a sample service specification."""
        return PermanentServiceSpec(
            name="test_service",
            image="nginx:latest",
            ports=["80:80"],
            environment=["ENV=production"],
            resource_limits={"memory": "512m", "cpu": "0.5"}
        )

    @pytest.fixture
    def permanent_env_spec(self, sample_service_spec, sample_network_spec):
        """Create a permanent environment specification."""
        return PermanentEnvironmentSpec(
            services={"test_service": sample_service_spec},
            networks={"test_network": sample_network_spec}
        )

    @pytest.fixture
    def manager(self, base_config):
        """Create a PermanentEnvironmentManager instance."""
        with patch('saber.server.execution.sandbox.permanent_environment_manager.ContainerLoggingManager'):
            return PermanentEnvironmentManager(base_config)

    def test_configuration_hash_computation(self, manager, permanent_env_spec):
        """Test configuration hash computation is deterministic."""
        hash1 = manager._compute_configuration_hash(permanent_env_spec)
        hash2 = manager._compute_configuration_hash(permanent_env_spec)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

        # Test that different configs produce different hashes
        different_spec = PermanentEnvironmentSpec(
            services={"different_service": permanent_env_spec.services["test_service"]},
            networks=permanent_env_spec.networks
        )
        hash3 = manager._compute_configuration_hash(different_spec)
        assert hash1 != hash3

    def test_configuration_storage_and_retrieval(self, manager, permanent_env_spec):
        """Test storing and retrieving configuration hashes."""
        # Initially no stored hash
        assert manager._get_stored_configuration_hash() is None

        # Store a hash
        test_hash = "abcd1234"
        manager._store_configuration_hash(test_hash)

        # Retrieve the hash
        retrieved_hash = manager._get_stored_configuration_hash()
        assert retrieved_hash == test_hash

        # Verify file was created
        hash_file = manager.metadata_dir / 'configuration_hash.txt'
        assert hash_file.exists()
        assert hash_file.read_text().strip() == test_hash

    def test_is_configuration_current_first_run(self, manager, permanent_env_spec):
        """Test configuration check on first run (no stored hash)."""
        manager._is_running = True

        # No stored hash means configuration is not current
        assert not manager.is_configuration_current(permanent_env_spec)

    def test_is_configuration_current_same_config(self, manager, permanent_env_spec):
        """Test configuration check with same configuration."""
        manager._is_running = True

        # Store current configuration hash
        current_hash = manager._compute_configuration_hash(permanent_env_spec)
        manager._store_configuration_hash(current_hash)

        # Configuration should be current
        assert manager.is_configuration_current(permanent_env_spec)

    def test_is_configuration_current_different_config(self, manager, permanent_env_spec):
        """Test configuration check with different configuration."""
        manager._is_running = True

        # Store a different hash
        manager._store_configuration_hash("different_hash")

        # Configuration should not be current
        assert not manager.is_configuration_current(permanent_env_spec)

    def test_is_configuration_current_not_running(self, manager, permanent_env_spec):
        """Test configuration check when environment is not running."""
        manager._is_running = False

        # Should return False when not running, regardless of stored hash
        assert not manager.is_configuration_current(permanent_env_spec)

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_ensure_shared_networks_create_new(self, mock_subprocess, manager, permanent_env_spec):
        """Test creating new networks when they don't exist."""
        # Mock network inspect to return failure (network doesn't exist)
        mock_subprocess.side_effect = [
            Mock(returncode=1),  # Network inspect fails
            Mock(returncode=0)   # Network create succeeds
        ]

        manager.ensure_shared_networks(permanent_env_spec)

        # Verify network inspect was called
        assert mock_subprocess.call_args_list[0][0][0] == ["docker", "network", "inspect", "test_network"]

        # Verify network create was called
        create_call = mock_subprocess.call_args_list[1][0][0]
        assert "docker" in create_call
        assert "network" in create_call
        assert "create" in create_call
        assert "test_network" in create_call

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_ensure_shared_networks_external_network(self, mock_subprocess, manager, external_network_spec):
        """Test handling external networks that should not be created."""
        spec = PermanentEnvironmentSpec(
            services={},
            networks={"shared_network": external_network_spec}
        )

        # Mock network inspect to return success (external network exists)
        mock_subprocess.return_value = Mock(returncode=0)

        manager.ensure_shared_networks(spec)

        # Should only call inspect, not create (external networks are not created)
        assert len(mock_subprocess.call_args_list) == 1
        assert mock_subprocess.call_args_list[0][0][0] == ["docker", "network", "inspect", "excytin-shared-network"]

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_ensure_shared_networks_already_exists(self, mock_subprocess, manager, permanent_env_spec):
        """Test handling networks that already exist."""
        # Mock network inspect to return success (network exists)
        mock_subprocess.return_value = Mock(returncode=0)

        manager.ensure_shared_networks(permanent_env_spec)

        # Should only call inspect, not create
        assert len(mock_subprocess.call_args_list) == 1
        assert mock_subprocess.call_args_list[0][0][0] == ["docker", "network", "inspect", "test_network"]

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_cleanup_managed_networks_not_in_use(self, mock_subprocess, manager, permanent_env_spec):
        """Test cleaning up networks that are not in use."""
        manager.permanent_spec = permanent_env_spec

        # Mock network inspect to return empty containers
        network_info = {"Containers": {}}
        mock_subprocess.side_effect = [
            Mock(returncode=0, stdout=json.dumps([network_info])),  # Network inspect
            Mock(returncode=0)  # Network remove
        ]

        manager.cleanup_managed_networks()

        # Verify network remove was called
        assert len(mock_subprocess.call_args_list) == 2
        remove_call = mock_subprocess.call_args_list[1][0][0]
        assert "docker" in remove_call
        assert "network" in remove_call
        assert "rm" in remove_call
        assert "test_network" in remove_call

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_cleanup_managed_networks_in_use(self, mock_subprocess, manager, permanent_env_spec):
        """Test skipping cleanup of networks that are in use."""
        manager.permanent_spec = permanent_env_spec

        # Mock network inspect to return containers in use
        network_info = {"Containers": {"container1": {}}}
        mock_subprocess.return_value = Mock(returncode=0, stdout=json.dumps([network_info]))

        manager.cleanup_managed_networks()

        # Should only call inspect, not remove
        assert len(mock_subprocess.call_args_list) == 1

    @patch.object(PermanentEnvironmentManager, 'stop_permanent_environment')
    @patch.object(PermanentEnvironmentManager, 'ensure_shared_networks')
    @patch.object(PermanentEnvironmentManager, 'start_permanent_environment')
    def test_recreate_environment(self, mock_start, mock_ensure_networks, mock_stop, manager, permanent_env_spec):
        """Test complete environment recreation."""
        manager._is_running = True

        manager.recreate_environment(permanent_env_spec)

        # Verify sequence of operations
        mock_stop.assert_called_once()
        mock_ensure_networks.assert_called_once_with(permanent_env_spec)
        mock_start.assert_called_once_with(permanent_env_spec)

        # Verify configuration hash was stored
        stored_hash = manager._get_stored_configuration_hash()
        expected_hash = manager._compute_configuration_hash(permanent_env_spec)
        assert stored_hash == expected_hash

    @patch.object(PermanentEnvironmentManager, 'recreate_environment')
    def test_ensure_permanent_environments_current_needs_recreation(self, mock_recreate, manager, permanent_env_spec):
        """Test ensuring current configuration when recreation is needed."""
        manager._is_running = True
        # No stored hash, so recreation needed

        manager.ensure_permanent_environments_current(permanent_env_spec)

        mock_recreate.assert_called_once_with(permanent_env_spec)

    @patch.object(PermanentEnvironmentManager, 'recreate_environment')
    def test_ensure_permanent_environments_current_already_current(self, mock_recreate, manager, permanent_env_spec):
        """Test ensuring current configuration when no recreation is needed."""
        manager._is_running = True

        # Store current configuration hash
        current_hash = manager._compute_configuration_hash(permanent_env_spec)
        manager._store_configuration_hash(current_hash)

        manager.ensure_permanent_environments_current(permanent_env_spec)

        # Should not recreate if configuration is current
        mock_recreate.assert_not_called()

    @patch.object(PermanentEnvironmentManager, 'stop_permanent_environment')
    def test_cleanup_on_server_shutdown(self, mock_stop, manager):
        """Test cleanup during server shutdown."""
        manager._is_running = True

        manager.cleanup_on_server_shutdown()

        mock_stop.assert_called_once()
