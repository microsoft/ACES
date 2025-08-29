"""
Unit tests for PermanentEnvironmentManager.

Tests the permanent environment lifecycle management and service coordination.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.sandbox.environment_spec import (
    HealthCheck,
    PermanentEnvironmentSpec,
    PermanentNetworkSpec,
    PermanentServiceSpec,
)
from saber.server.execution.sandbox.permanent_environment_manager import PermanentEnvironmentManager


class TestPermanentEnvironmentManager:
    """Test PermanentEnvironmentManager functionality."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config storage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    @pytest.fixture
    def sample_permanent_spec(self):
        """Create a sample permanent environment specification."""
        # Create network spec
        network_spec = PermanentNetworkSpec(
            name="permanent_bridge",
            driver="bridge",
            internal=False,
            ipam_config={"config": [{"subnet": "172.30.0.0/16"}]}
        )

        # Create service spec
        service_spec = PermanentServiceSpec(
            name="test_service",
            image="nginx:latest",
            ports=["80"],
            environment=["ENV=production"],
            health_check=HealthCheck(test=["CMD", "curl", "-f", "http://localhost/"]),
            resource_limits={"memory": "512m", "cpu": "0.5"}
        )

        return PermanentEnvironmentSpec(
            services={"test_service": service_spec},
            networks={"permanent_bridge": network_spec}
        )

    def test_init(self, temp_config_dir):
        """Test PermanentEnvironmentManager initialization."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)

        assert manager.config == config
        assert manager.permanent_spec is None
        assert manager.compose_project_name == "saber-permanent"
        assert not manager._is_running

    @patch('saber.server.execution.sandbox.permanent_environment_manager.Path.unlink')
    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    @patch('saber.server.execution.sandbox.permanent_environment_manager.tempfile.NamedTemporaryFile')
    def test_start_permanent_environment_success(self, mock_temp_file, mock_subprocess, mock_unlink, sample_permanent_spec, temp_config_dir):
        """Test successful permanent environment startup."""
        # Setup mocks
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)

        mock_file = Mock()
        mock_file.name = "/tmp/test_compose.yml"
        mock_temp_file.return_value.__enter__.return_value = mock_file

        mock_subprocess.return_value = Mock(stdout="Services started", stderr="")

        # Start permanent environment
        manager.start_permanent_environment(sample_permanent_spec)

        # Verify state
        assert manager._is_running
        assert manager.permanent_spec == sample_permanent_spec

        # Verify docker compose was called (new format, not docker-compose)
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "docker" in call_args
        assert "compose" in call_args
        assert "-p" in call_args
        assert "saber-permanent" in call_args
        assert "up" in call_args
        assert "-d" in call_args

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_start_permanent_environment_failure(self, mock_subprocess, sample_permanent_spec, temp_config_dir):
        """Test permanent environment startup failure."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)

        # Mock subprocess failure
        from subprocess import CalledProcessError
        mock_subprocess.side_effect = CalledProcessError(1, "docker-compose", stderr="Error message")

        # Start should raise exception
        with pytest.raises(SandboxExecutionError, match="Failed to start permanent environment"):
            manager.start_permanent_environment(sample_permanent_spec)

        # Verify state
        assert not manager._is_running
        # Note: permanent_spec is set before the error occurs, so it won't be None

    def test_start_permanent_environment_already_running(self, sample_permanent_spec, temp_config_dir):
        """Test starting permanent environment when already running."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)
        manager._is_running = True

        # Should not raise error, just log warning
        manager.start_permanent_environment(sample_permanent_spec)

        # State should not change
        assert manager._is_running
        assert manager.permanent_spec is None

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_stop_permanent_environment_success(self, mock_subprocess, temp_config_dir):
        """Test successful permanent environment stop."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)
        manager._is_running = True
        manager.permanent_spec = Mock()

        mock_subprocess.return_value = Mock(stdout="Services stopped", stderr="")

        # Stop permanent environment
        manager.stop_permanent_environment()

        # Verify state
        assert not manager._is_running
        assert manager.permanent_spec is None

        # Verify docker compose was called (new format, not docker-compose)
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "docker" in call_args
        assert "compose" in call_args
        assert "-p" in call_args
        assert "saber-permanent" in call_args
        assert "down" in call_args
        assert "-v" in call_args

    def test_stop_permanent_environment_not_running(self, temp_config_dir):
        """Test stopping permanent environment when not running."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)

        # Should not raise error, just log warning
        manager.stop_permanent_environment()

        # State should remain unchanged
        assert not manager._is_running
        assert manager.permanent_spec is None

    @patch('saber.server.execution.sandbox.permanent_environment_manager.subprocess.run')
    def test_get_service_endpoints_success(self, mock_subprocess, sample_permanent_spec, temp_config_dir):
        """Test successful service endpoints retrieval."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)
        manager._is_running = True
        manager.permanent_spec = sample_permanent_spec

        # Mock container information
        container_info = {
            "Id": "container123",
            "State": {"Running": True},
            "NetworkSettings": {
                "Networks": {
                    "permanent_bridge": {
                        "IPAddress": "172.30.0.2"
                    }
                }
            }
        }

        # First call returns container ID, second returns container info
        import json
        mock_subprocess.side_effect = [
            Mock(stdout="container123\n", stderr=""),
            Mock(stdout=json.dumps([container_info]), stderr="")
        ]

        # Get service endpoints
        endpoints = manager.get_service_endpoints("test_service")

        # Verify result
        assert endpoints["service_name"] == "test_service"
        assert endpoints["container_id"] == "container123"
        assert endpoints["status"] == "running"
        assert "permanent_bridge" in endpoints["endpoints"]
        assert endpoints["endpoints"]["permanent_bridge"]["ip_address"] == "172.30.0.2"

    def test_get_service_endpoints_not_running(self, sample_permanent_spec, temp_config_dir):
        """Test getting service endpoints when environment not running."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)

        with pytest.raises(SandboxExecutionError, match="Permanent environment is not running"):
            manager.get_service_endpoints("test_service")

    def test_get_service_endpoints_service_not_found(self, sample_permanent_spec, temp_config_dir):
        """Test getting endpoints for non-existent service."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)
        manager._is_running = True
        manager.permanent_spec = sample_permanent_spec

        with pytest.raises(SandboxExecutionError, match="Permanent service 'nonexistent' not found"):
            manager.get_service_endpoints("nonexistent")

    def test_is_running(self, temp_config_dir):
        """Test is_running status check."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)

        assert not manager.is_running()

        manager._is_running = True
        assert manager.is_running()

    def test_get_permanent_services(self, sample_permanent_spec, temp_config_dir):
        """Test getting list of permanent services."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)

        # No spec loaded
        assert manager.get_permanent_services() == []

        # With spec loaded
        manager.permanent_spec = sample_permanent_spec
        services = manager.get_permanent_services()
        assert services == ["test_service"]

    def test_get_permanent_networks(self, sample_permanent_spec, temp_config_dir):
        """Test getting list of permanent networks."""
        config = {"domain": "test_domain", "config_dir": temp_config_dir}
        manager = PermanentEnvironmentManager(config)

        # No spec loaded
        assert manager.get_permanent_networks() == []

        # With spec loaded
        manager.permanent_spec = sample_permanent_spec
        networks = manager.get_permanent_networks()
        assert networks == ["permanent_bridge"]
