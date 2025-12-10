"""
Unit tests for PermanentEnvironmentManager.

Tests the file-based permanent environment lifecycle management.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call, AsyncMock
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

        # Verify basic configuration
        assert manager.domain == "test_domain"
        assert manager.compose_project_name == "test_domain_permanent_environment"

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

        # Test
        manager.start_permanent_environment_from_file(temp_compose_file)

        # Verify
        assert manager._is_running is True
        assert manager._compose_file_path == temp_compose_file

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

        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.start_permanent_environment_from_file(nonexistent_file)

        assert "Failed to start permanent environment" in str(excinfo.value)

    def test_start_permanent_environment_orchestrator_failure(self, manager, temp_compose_file):
        """Test start handles orchestrator failure."""
        # Setup mock to fail
        manager.orchestrator.start_environment = Mock(side_effect=RuntimeError("Docker error"))

        # Test
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.start_permanent_environment_from_file(temp_compose_file)

        assert "Failed to start permanent environment" in str(excinfo.value)
        assert "Docker error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_stop_permanent_environment_success(self, manager, temp_compose_file):
        """Test successful stop of permanent environment."""
        # Setup - mark as running and setup mocks
        manager._is_running = True
        manager._compose_file_path = temp_compose_file
        manager.orchestrator.stop_environment = AsyncMock()

        # Test
        await manager.stop_permanent_environment()

        # Verify
        assert manager._is_running is False
        assert manager._compose_file_path is None
        manager.orchestrator.stop_environment.assert_called_once_with(temp_compose_file, project_name=manager.compose_project_name)

    @pytest.mark.asyncio
    async def test_stop_permanent_environment_not_running(self, manager):
        """Test stop when environment is not running."""
        # Setup - ensure not running
        manager._is_running = False
        manager.orchestrator.stop_environment = AsyncMock()

        # Test
        await manager.stop_permanent_environment()

        # Verify - should not call orchestrator since not running
        manager.orchestrator.stop_environment.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_permanent_environment_no_file_path(self, manager):
        """Test stop fails when no compose file path is stored."""
        # Setup - mark as running but no file path
        manager._is_running = True
        manager._compose_file_path = None

        # Test
        with pytest.raises(SandboxExecutionError) as excinfo:
            await manager.stop_permanent_environment()

        assert "No compose file path stored" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_stop_permanent_environment_orchestrator_failure(self, manager, temp_compose_file):
        """Test stop handles orchestrator failure."""
        # Setup - mark as running and make orchestrator fail
        manager._is_running = True
        manager._compose_file_path = temp_compose_file
        manager.orchestrator.stop_environment = AsyncMock(side_effect=RuntimeError("Docker error"))

        # Test
        with pytest.raises(SandboxExecutionError) as excinfo:
            await manager.stop_permanent_environment()

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
