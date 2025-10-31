"""
Test suite for SandboxEnvironmentManager with static compose files.

This module tests the updated SandboxEnvironmentManager that uses ComposeOrchestrator
with static compose files instead of dynamic generation.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


class TestSandboxEnvironmentManager:
    """Test cases for SandboxEnvironmentManager."""

    @pytest.fixture
    def sandbox_config(self):
        """Standard sandbox configuration for testing."""
        return {
            "domain": "test_domain"
        }

    @pytest.fixture
    def temp_compose_file(self):
        """Create a temporary compose file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.compose.yml', delete=False) as f:
            f.write("""
version: '3.8'
services:
  test-service:
    image: alpine:latest
    command: sleep 3600
    networks:
      - test-network
networks:
  test-network:
    driver: bridge
""")
            compose_path = Path(f.name)

        yield compose_path

        # Cleanup
        if compose_path.exists():
            compose_path.unlink()

    @pytest.fixture
    def mock_environments_dir(self, temp_compose_file):
        """Mock the environments directory structure."""
        # Create temporary directory structure
        temp_dir = Path(tempfile.mkdtemp())
        environments_dir = temp_dir / "domains" / "test_domain" / "server" / "config" / "environments" / "sandbox"
        environments_dir.mkdir(parents=True, exist_ok=True)

        # Copy the temp compose file to the expected location
        test_compose_file = environments_dir / "test_sandbox.compose.yml"
        test_compose_file.write_text(temp_compose_file.read_text())

        # Change to the temp directory so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        yield environments_dir

        # Cleanup
        os.chdir(original_cwd)
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, sandbox_config, mock_environments_dir):
        """Create SandboxEnvironmentManager instance for testing."""
        manager = SandboxEnvironmentManager(sandbox_config)
        manager._is_ready = True  # Mark as ready for testing
        return manager

    def test_init_success(self, sandbox_config):
        """Test successful initialization."""
        manager = SandboxEnvironmentManager(sandbox_config)

        assert manager.domain == "test_domain"
        assert str(manager.environments_base_path) == "domains/test_domain/server/config/environments/sandbox"
        assert len(manager.active_orchestrators) == 0
        assert len(manager.episode_compose_files) == 0

    def test_init_default_domain(self):
        """Test initialization with default domain when not specified."""
        config = {}

        manager = SandboxEnvironmentManager(config)

        assert manager.domain == "excytin_demo"

    def test_is_ready_status(self, manager):
        """Test is_ready status tracking."""
        assert manager.is_ready() is True

        # Simulate not ready
        manager._is_ready = False
        assert manager.is_ready() is False

    def test_wait_for_ready(self, manager):
        """Test waiting for manager to be ready."""
        # Should return immediately when ready
        result = manager.wait_for_ready(timeout=1)
        assert result is True

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_create_episode_environment_success(self, mock_orchestrator_class, manager):
        """Test successful episode environment creation."""
        # Setup mock
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator

        # Test
        result = manager.create_episode_environment("test-episode-123", "test_sandbox")

        # Verify
        assert result is True
        assert "test-episode-123" in manager.active_orchestrators
        assert "test-episode-123" in manager.episode_compose_files
        mock_orchestrator.start_environment.assert_called_once()

    def test_create_episode_environment_not_ready(self, manager):
        """Test episode creation fails when manager not ready."""
        manager._is_ready = False

        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.create_episode_environment("test-episode", "test_sandbox")

        assert "not ready yet" in str(excinfo.value)

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_create_episode_environment_already_exists(self, mock_orchestrator_class, manager):
        """Test episode creation fails when episode already exists."""
        # Create first environment
        manager.create_episode_environment("test-episode", "test_sandbox")

        # Try to create again
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.create_episode_environment("test-episode", "test_sandbox")

        assert "already exists" in str(excinfo.value)

    def test_create_episode_environment_missing_compose_file(self, manager):
        """Test episode creation fails when compose file doesn't exist."""
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.create_episode_environment("test-episode", "nonexistent_sandbox")

        assert "Compose file not found" in str(excinfo.value)

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_create_episode_environment_orchestrator_failure(self, mock_orchestrator_class, manager):
        """Test episode creation handles orchestrator failure."""
        # Setup mock to fail
        mock_orchestrator = Mock()
        mock_orchestrator.start_environment.side_effect = RuntimeError("Docker error")
        mock_orchestrator_class.return_value = mock_orchestrator

        # Test
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.create_episode_environment("test-episode", "test_sandbox")

        assert "Failed to create sandbox environment" in str(excinfo.value)

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_get_episode_environment_exists(self, mock_orchestrator_class, manager):
        """Test getting orchestrator for existing episode."""
        # Create environment
        manager.create_episode_environment("test-episode", "test_sandbox")

        # Get environment
        orchestrator = manager.get_episode_environment("test-episode")

        assert orchestrator is not None

    def test_get_episode_environment_not_exists(self, manager):
        """Test getting orchestrator for non-existing episode."""
        orchestrator = manager.get_episode_environment("nonexistent-episode")

        assert orchestrator is None

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_stop_episode_environment_success(self, mock_orchestrator_class, manager):
        """Test successful episode environment stop."""
        # Setup - create environment first
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        manager.create_episode_environment("test-episode", "test_sandbox")

        # Test
        result = manager.stop_episode_environment("test-episode")

        # Verify
        assert result is True
        assert "test-episode" not in manager.active_orchestrators
        assert "test-episode" not in manager.episode_compose_files
        mock_orchestrator.stop_environment.assert_called_once()

    def test_stop_episode_environment_not_exists(self, manager):
        """Test stopping non-existent episode environment."""
        result = manager.stop_episode_environment("nonexistent-episode")

        assert result is False

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_stop_episode_environment_orchestrator_failure(self, mock_orchestrator_class, manager):
        """Test stop episode handles orchestrator failure."""
        # Setup - create environment and make stop fail
        mock_orchestrator = Mock()
        mock_orchestrator.stop_environment.side_effect = RuntimeError("Docker error")
        mock_orchestrator_class.return_value = mock_orchestrator
        manager.create_episode_environment("test-episode", "test_sandbox")

        # Test
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.stop_episode_environment("test-episode")

        assert "Failed to stop sandbox environment" in str(excinfo.value)

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_cleanup_all_episodes_success(self, mock_orchestrator_class, manager):
        """Test cleanup of all environments."""
        # Setup - create multiple environments
        mock_orchestrator1 = Mock()
        mock_orchestrator2 = Mock()
        mock_orchestrator_class.side_effect = [mock_orchestrator1, mock_orchestrator2]

        manager.create_episode_environment("episode-1", "test_sandbox")
        manager.create_episode_environment("episode-2", "test_sandbox")

        # Test
        manager.cleanup_all_episodes()

        # Verify all environments cleaned up
        assert len(manager.active_orchestrators) == 0
        assert len(manager.episode_compose_files) == 0

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_cleanup_all_episodes_some_failures(self, mock_orchestrator_class, manager):
        """Test cleanup handles some environment failures."""
        # Setup - create environments with one failing
        mock_orchestrator1 = Mock()
        mock_orchestrator2 = Mock()
        mock_orchestrator1.stop_environment.side_effect = RuntimeError("Docker error")
        mock_orchestrator_class.side_effect = [mock_orchestrator1, mock_orchestrator2]

        manager.create_episode_environment("episode-1", "test_sandbox")
        manager.create_episode_environment("episode-2", "test_sandbox")

        # Test - should not raise exception even with failures
        manager.cleanup_all_episodes()

        # Failed episode should still be tracked (since cleanup failed)
        # Successful episode should be cleaned up
        assert len(manager.active_orchestrators) == 1  # episode-1 failed, still tracked
        assert "episode-1" in manager.active_orchestrators
        assert "episode-2" not in manager.active_orchestrators

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_is_episode_active(self, mock_orchestrator_class, manager):
        """Test checking if episode is active."""
        # Initially not active
        assert manager.is_episode_active("test-episode") is False

        # Create episode
        manager.create_episode_environment("test-episode", "test_sandbox")

        # Now should be active
        assert manager.is_episode_active("test-episode") is True

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_get_active_episodes(self, mock_orchestrator_class, manager):
        """Test getting list of active episodes."""
        # Initially empty
        assert manager.get_active_episodes() == []

        # Create episodes
        manager.create_episode_environment("episode-1", "test_sandbox")
        manager.create_episode_environment("episode-2", "test_sandbox")

        # Should return both episodes
        active_episodes = manager.get_active_episodes()
        assert len(active_episodes) == 2
        assert "episode-1" in active_episodes
        assert "episode-2" in active_episodes

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_get_episode_status(self, mock_orchestrator_class, manager):
        """Test getting episode status information."""
        # Non-existent episode
        status = manager.get_episode_status("nonexistent")
        assert status is None

        # Create episode
        manager.create_episode_environment("test-episode", "test_sandbox")

        # Get status
        status = manager.get_episode_status("test-episode")
        assert status is not None
        assert status["episode_id"] == "test-episode"
        assert status["status"] == "active"
        assert status["domain"] == "test_domain"
        assert "compose_file" in status

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_multiple_episode_isolation(self, mock_orchestrator_class, manager):
        """Test that multiple episodes are properly isolated."""
        # Create multiple orchestrators
        mock_orchestrator1 = Mock()
        mock_orchestrator2 = Mock()
        mock_orchestrator3 = Mock()
        mock_orchestrator_class.side_effect = [mock_orchestrator1, mock_orchestrator2, mock_orchestrator3]

        # Create multiple episodes
        manager.create_episode_environment("episode-a", "test_sandbox")
        manager.create_episode_environment("episode-b", "test_sandbox")
        manager.create_episode_environment("episode-c", "test_sandbox")

        # Verify each episode has its own orchestrator
        assert len(manager.active_orchestrators) == 3
        assert manager.get_episode_environment("episode-a") is mock_orchestrator1
        assert manager.get_episode_environment("episode-b") is mock_orchestrator2
        assert manager.get_episode_environment("episode-c") is mock_orchestrator3

        # Verify each orchestrator was called with episode-specific parameters
        mock_orchestrator1.start_environment.assert_called_once()
        mock_orchestrator2.start_environment.assert_called_once()
        mock_orchestrator3.start_environment.assert_called_once()

    def test_get_compose_file_path_valid(self, manager):
        """Test getting compose file path for valid environment."""
        # The mock_environments_dir fixture creates test_sandbox.compose.yml, so this should succeed
        compose_path = manager._get_compose_file_path("test_sandbox")

        assert compose_path.name == "test_sandbox.compose.yml"
        assert "test_sandbox.compose.yml" in str(compose_path)

    def test_get_compose_file_path_missing_environments_dir(self):
        """Test error when environments directory doesn't exist."""
        config = {"domain": "nonexistent_domain"}
        manager = SandboxEnvironmentManager(config)
        manager._is_ready = True

        with pytest.raises(SandboxExecutionError) as excinfo:
            manager._get_compose_file_path("test_sandbox")

        assert "Sandbox environments directory not found" in str(excinfo.value)


class TestSandboxEnvironmentManagerAsync:
    """Test cases for SandboxEnvironmentManager async episode creation."""

    @pytest.fixture
    def sandbox_config(self):
        """Standard sandbox configuration for testing."""
        return {
            "domain": "test_domain"
        }

    @pytest.fixture
    def manager(self, mock_environments_dir, sandbox_config):
        """Create a SandboxEnvironmentManager instance."""
        with patch.dict(os.environ, {}, clear=False):
            manager = SandboxEnvironmentManager(sandbox_config)
            manager._is_ready = True
            manager.environments_path = mock_environments_dir
            manager.environments_base_path = mock_environments_dir
            yield manager

    @pytest.fixture
    def mock_environments_dir(self):
        """Create a temporary environments directory with a test compose file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_dir = Path(temp_dir)

            # Create test compose file
            compose_file = env_dir / "test_sandbox.compose.yml"
            compose_file.write_text("""
version: '3.8'
services:
  test-service:
    image: alpine:latest
    command: sleep 3600
    labels:
      - "saber.execution.service=true"
      - "saber.service.type=sandbox"
    networks:
      - test-network
networks:
  test-network:
    driver: bridge
""")
            yield env_dir

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_create_episode_environment_async_success(self, mock_orchestrator_class, manager):
        """Test successful async episode environment creation."""
        # Setup mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.processed_compose_path = "/tmp/processed_test.yml"
        mock_orchestrator_class.return_value = mock_orchestrator

        # Create episode async
        orchestrator, processed_path = manager.create_episode_environment_async(
            "test-episode-async-001",
            "test_sandbox"
        )

        # Verify orchestrator was created
        mock_orchestrator_class.assert_called_once()

        # CRITICAL: Verify start_environment_async was called (not start_environment)
        mock_orchestrator.start_environment_async.assert_called_once()
        mock_orchestrator.start_environment.assert_not_called()

        # Verify episode is tracked
        assert "test-episode-async-001" in manager.active_orchestrators
        assert manager.active_orchestrators["test-episode-async-001"] is mock_orchestrator

        # Verify compose file is tracked
        assert "test-episode-async-001" in manager.episode_compose_files

        # Verify returned values
        assert orchestrator is mock_orchestrator
        assert processed_path is not None

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_wait_for_episode_healthy_success(self, mock_orchestrator_class, manager):
        """Test successful episode health check after async creation."""
        # Setup mock orchestrator with processed_compose_path attribute
        mock_orchestrator = Mock()
        mock_orchestrator.processed_compose_path = "/tmp/processed_test.yml"
        mock_orchestrator.wait_for_healthy = Mock()  # Mock the health check method
        mock_orchestrator_class.return_value = mock_orchestrator

        # Create episode async first
        manager.create_episode_environment_async("test-episode-health-002", "test_sandbox")

        # Now wait for healthy
        manager.wait_for_episode_healthy("test-episode-health-002", timeout_seconds=180)

        # CRITICAL: Verify wait_for_healthy was called with the processed compose path
        mock_orchestrator.wait_for_healthy.assert_called_once()
        call_args = mock_orchestrator.wait_for_healthy.call_args

        # Verify the compose_file_path argument is the processed path from orchestrator
        assert call_args[1]['compose_file_path'] == mock_orchestrator.processed_compose_path
        assert call_args[1]['timeout_seconds'] == 180

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_wait_for_episode_healthy_no_orchestrator(self, mock_orchestrator_class, manager):
        """Test wait_for_healthy fails if episode not found."""
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.wait_for_episode_healthy("nonexistent-episode", timeout_seconds=180)

        assert "No active environment found" in str(excinfo.value)
        assert "nonexistent-episode" in str(excinfo.value)

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_wait_for_episode_healthy_no_processed_path(self, mock_orchestrator_class, manager):
        """Test wait_for_healthy fails if orchestrator has no processed path."""
        # Setup mock orchestrator WITHOUT processed_compose_path attribute
        mock_orchestrator = Mock(spec=['start_environment_async'])  # No processed_compose_path
        mock_orchestrator_class.return_value = mock_orchestrator

        # Create episode async
        manager.create_episode_environment_async("test-episode-no-path-003", "test_sandbox")

        # Try to wait for healthy - should fail
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.wait_for_episode_healthy("test-episode-no-path-003", timeout_seconds=180)

        assert "No processed compose path found" in str(excinfo.value)

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_async_full_workflow(self, mock_orchestrator_class, manager):
        """Test the complete async workflow: create_async -> wait_healthy."""
        # Setup mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.processed_compose_path = "/tmp/processed_workflow.yml"
        mock_orchestrator.wait_for_healthy = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator

        episode_id = "test-episode-workflow-004"

        # Step 1: Create episode async
        orchestrator, processed_path = manager.create_episode_environment_async(
            episode_id,
            "test_sandbox"
        )

        # Verify async start was called, not sync start
        mock_orchestrator.start_environment_async.assert_called_once()
        mock_orchestrator.start_environment.assert_not_called()
        mock_orchestrator.wait_for_healthy.assert_not_called()  # Not called yet

        # Step 2: Wait for healthy
        manager.wait_for_episode_healthy(episode_id, timeout_seconds=180)

        # Verify health check was called with processed path
        mock_orchestrator.wait_for_healthy.assert_called_once()
        health_call = mock_orchestrator.wait_for_healthy.call_args
        assert health_call[1]['compose_file_path'] == mock_orchestrator.processed_compose_path

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_create_episode_environment_async_duplicate(self, mock_orchestrator_class, manager):
        """Test that creating duplicate async episode fails."""
        mock_orchestrator = Mock()
        mock_orchestrator.processed_compose_path = "/tmp/processed.yml"
        mock_orchestrator_class.return_value = mock_orchestrator

        # Create first episode
        manager.create_episode_environment_async("duplicate-episode", "test_sandbox")

        # Try to create same episode again - should fail
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.create_episode_environment_async("duplicate-episode", "test_sandbox")

        assert "already exists" in str(excinfo.value)

    @patch('saber.server.execution.sandbox.sandbox_environment_manager.ComposeOrchestrator')
    def test_wait_for_episode_healthy_propagates_errors(self, mock_orchestrator_class, manager):
        """Test that health check errors are properly propagated."""
        # Setup mock orchestrator that fails health check
        mock_orchestrator = Mock()
        mock_orchestrator.processed_compose_path = "/tmp/processed.yml"
        mock_orchestrator.wait_for_healthy.side_effect = Exception("Health check timeout")
        mock_orchestrator_class.return_value = mock_orchestrator

        # Create episode async
        manager.create_episode_environment_async("test-episode-fail-005", "test_sandbox")

        # Try to wait for healthy - should propagate error
        with pytest.raises(SandboxExecutionError) as excinfo:
            manager.wait_for_episode_healthy("test-episode-fail-005", timeout_seconds=180)

        assert "failed health checks" in str(excinfo.value)
        assert "Health check timeout" in str(excinfo.value)
