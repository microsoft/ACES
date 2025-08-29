"""
Integration test for the complete permanent environment management feature.

This test validates the full implementation of configuration change detection,
network management, and server lifecycle integration.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

from saber.server.execution.sandbox.environment_spec import (
    PermanentEnvironmentSpec,
    PermanentNetworkSpec,
    PermanentServiceSpec,
)
from saber.server.execution.sandbox.permanent_environment_manager import PermanentEnvironmentManager


class TestPermanentEnvironmentManagementIntegration:
    """Integration tests for complete permanent environment management."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def config(self, temp_config_dir):
        """Configuration for PermanentEnvironmentManager."""
        return {
            "domain": "test_domain",
            "config_dir": str(temp_config_dir),
            "logs_directory": "/app/logs",
            "enable_logging": True,
        }

    @pytest.fixture
    def permanent_env_spec_v1(self):
        """Version 1 of permanent environment specification."""
        service = PermanentServiceSpec(
            name="database",
            image="mysql:8.0",
            ports=["3306:3306"],
            environment=["MYSQL_ROOT_PASSWORD=admin"]
        )

        external_network = PermanentNetworkSpec(
            name="shared_network",
            external=True,
            external_name="excytin-shared-network"
        )

        return PermanentEnvironmentSpec(
            services={"database": service},
            networks={"shared_network": external_network}
        )

    @pytest.fixture
    def permanent_env_spec_v2(self):
        """Version 2 of permanent environment specification (different config)."""
        service = PermanentServiceSpec(
            name="database",
            image="mysql:8.0",
            ports=["3306:3306"],
            environment=["MYSQL_ROOT_PASSWORD=newpassword"]  # Changed password
        )

        external_network = PermanentNetworkSpec(
            name="shared_network",
            external=True,
            external_name="excytin-shared-network"
        )

        internal_network = PermanentNetworkSpec(
            name="internal_network",
            driver="bridge",
            internal=True
        )

        return PermanentEnvironmentSpec(
            services={"database": service},
            networks={
                "shared_network": external_network,
                "internal_network": internal_network  # Added new network
            }
        )

    @pytest.fixture
    def manager(self, config):
        """Create a PermanentEnvironmentManager instance."""
        with patch('saber.server.execution.sandbox.permanent_environment_manager.ContainerLoggingManager'):
            return PermanentEnvironmentManager(config)

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_full_lifecycle_configuration_change_detection(
        self,
        mock_subprocess,
        manager,
        permanent_env_spec_v1,
        permanent_env_spec_v2
    ):
        """Test complete lifecycle with configuration change detection."""

        # Mock successful subprocess calls for network operations and container management
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # === FIRST STARTUP ===
        # No previous configuration, should start environment
        manager.ensure_permanent_environments_current(permanent_env_spec_v1)

        # Verify configuration hash was stored
        stored_hash_v1 = manager._get_stored_configuration_hash()
        assert stored_hash_v1 is not None

        # Verify network creation was attempted for external network (should check if exists)
        network_inspect_calls = [
            call for call in mock_subprocess.call_args_list
            if call[0][0][:3] == ["docker", "network", "inspect"]
        ]
        assert len(network_inspect_calls) >= 1

        # === SECOND STARTUP (SAME CONFIG) ===
        mock_subprocess.reset_mock()
        manager._is_running = True  # Simulate running state

        # Same configuration, should not recreate
        manager.ensure_permanent_environments_current(permanent_env_spec_v1)

        # Should not have called docker-compose commands for recreation
        compose_calls = [
            call for call in mock_subprocess.call_args_list
            if "compose" in call[0][0]
        ]
        assert len(compose_calls) == 0  # No recreation needed

        # === THIRD STARTUP (CHANGED CONFIG) ===
        mock_subprocess.reset_mock()

        # Different configuration, should recreate
        manager.ensure_permanent_environments_current(permanent_env_spec_v2)

        # Verify new configuration hash was stored
        stored_hash_v2 = manager._get_stored_configuration_hash()
        assert stored_hash_v2 != stored_hash_v1

        # Should have called network operations for new internal network
        network_inspect_calls = [
            call for call in mock_subprocess.call_args_list
            if call[0][0][:3] == ["docker", "network", "inspect"]
        ]
        assert len(network_inspect_calls) >= 2  # Check for both networks

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_external_network_handling(self, mock_subprocess, manager, permanent_env_spec_v1):
        """Test that external networks are handled correctly."""

        # Mock network inspect to return success (external network exists)
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        manager.ensure_shared_networks(permanent_env_spec_v1)

        # Verify external network was checked but not created
        network_calls = [
            call for call in mock_subprocess.call_args_list
            if "network" in call[0][0]
        ]

        # Should only inspect, not create external networks
        inspect_calls = [call for call in network_calls if "inspect" in call[0][0]]
        create_calls = [call for call in network_calls if "create" in call[0][0]]

        assert len(inspect_calls) >= 1
        # External networks should not be created automatically
        external_create_calls = [
            call for call in create_calls
            if "excytin-shared-network" in call[0][0]
        ]
        assert len(external_create_calls) == 0

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_internal_network_creation(self, mock_subprocess, manager, permanent_env_spec_v2):
        """Test that internal networks are created when needed."""

        # Mock network inspect to fail (network doesn't exist), then creation succeeds
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # External network inspect (exists - no creation)
            Mock(returncode=1),  # Internal network inspect (doesn't exist)
            Mock(returncode=0),  # Internal network create (succeeds)
        ]

        manager.ensure_shared_networks(permanent_env_spec_v2)

        # Verify internal network creation was attempted
        create_calls = [
            call for call in mock_subprocess.call_args_list
            if call[0][0][:3] == ["docker", "network", "create"]
        ]
        assert len(create_calls) >= 1

        # Find the create call for internal network
        internal_create_calls = [
            call for call in create_calls
            if "internal_network" in call[0][0]
        ]
        assert len(internal_create_calls) == 1

    def test_configuration_hash_deterministic(self, manager, permanent_env_spec_v1):
        """Test that configuration hashes are deterministic and detect changes."""

        # Same spec should produce same hash
        hash1 = manager._compute_configuration_hash(permanent_env_spec_v1)
        hash2 = manager._compute_configuration_hash(permanent_env_spec_v1)
        assert hash1 == hash2

        # Different spec should produce different hash
        different_spec = PermanentEnvironmentSpec(
            services={},  # Empty services
            networks=permanent_env_spec_v1.networks
        )
        hash3 = manager._compute_configuration_hash(different_spec)
        assert hash1 != hash3

    def test_metadata_persistence(self, manager, permanent_env_spec_v1):
        """Test that metadata is properly persisted across manager instances."""

        # Store configuration hash
        expected_hash = manager._compute_configuration_hash(permanent_env_spec_v1)
        manager._store_configuration_hash(expected_hash)

        # Create new manager instance with same config directory
        with patch('saber.server.execution.sandbox.permanent_environment_manager.ContainerLoggingManager'):
            new_manager = PermanentEnvironmentManager(manager.config)

        # Should be able to retrieve the stored hash
        retrieved_hash = new_manager._get_stored_configuration_hash()
        assert retrieved_hash == expected_hash

        # Should recognize configuration as current
        new_manager._is_running = True
        assert new_manager.is_configuration_current(permanent_env_spec_v1)

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_cleanup_on_shutdown(self, mock_subprocess, manager):
        """Test cleanup behavior on server shutdown."""

        # Setup running state
        manager._is_running = True

        # Mock successful cleanup
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        manager.cleanup_on_server_shutdown()

        # Verify stop was called
        stop_calls = [
            call for call in mock_subprocess.call_args_list
            if call[0][0][:2] == ["docker", "compose"] and "down" in call[0][0]
        ]
        assert len(stop_calls) >= 1
