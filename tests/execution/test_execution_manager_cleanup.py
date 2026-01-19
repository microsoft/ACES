"""
Test suite for ExecutionManager cleanup functionality after CleanupManager removal.

This module tests the new direct component cleanup approach where ExecutionManager
calls SandboxEnvironmentManager and PermanentEnvironmentManager directly.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, PropertyMock, AsyncMock

from saber.server.execution.execution_manager import ExecutionManager


class TestExecutionManagerCleanup:
    """Test cases for ExecutionManager cleanup functionality."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "environments": {
                "excytin_demo": {
                    "domain": "excytin_demo",
                    "sandbox": {
                        "type": "docker_compose",
                        "config": {"timeout": 600}
                    }
                }
            },
            "executors": {
                "available": ["bash", "python"],
                "default": "bash"
            }
        }

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Mock sandbox environment manager."""
        mock = MagicMock()
        mock.is_ready.return_value = True
        # Make async methods return coroutines
        mock.stop_episode_environment = AsyncMock(return_value=True)
        mock.cleanup_all_episodes = AsyncMock(return_value=None)
        mock.get_active_episodes.return_value = []
        return mock

    @pytest.fixture
    def mock_permanent_manager(self):
        """Mock permanent environment manager."""
        mock = MagicMock()
        mock.is_running.return_value = True
        mock.stop_permanent_environment = AsyncMock(return_value=None)
        return mock

    @pytest.fixture
    def execution_manager(self, sample_config, mock_sandbox_manager):
        """Create ExecutionManager with mocked dependencies."""
        with patch('saber.server.execution.execution_manager.SandboxEnvironmentManager') as mock_sem, \
             patch('saber.server.execution.execution_manager.ExecutorFactory') as mock_ef:

            mock_sem.return_value = mock_sandbox_manager
            mock_ef.return_value = MagicMock()

            # ExecutionManager constructor expects config_dir, not config dict
            manager = ExecutionManager("/tmp/test_config")

            # Manually set the sandbox manager since it's not initialized by default
            manager._sandbox_environment_manager = mock_sandbox_manager

            return manager

    @pytest.mark.asyncio
    async def test_cleanup_episode_success(self, execution_manager, mock_sandbox_manager):
        """Test successful episode cleanup through direct sandbox manager call."""
        episode_id = "test-episode-123"
        context = {"test": "context"}

        # Mock successful cleanup
        mock_sandbox_manager.stop_episode_environment = AsyncMock(return_value=True)

        # Call cleanup (now async)
        result = await execution_manager.cleanup_episode(episode_id, context)

        # Verify direct sandbox manager call
        mock_sandbox_manager.stop_episode_environment.assert_called_once_with(episode_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_cleanup_episode_failure(self, execution_manager, mock_sandbox_manager):
        """Test episode cleanup failure handling."""
        episode_id = "test-episode-123"

        # Mock cleanup failure
        mock_sandbox_manager.stop_episode_environment = AsyncMock(return_value=False)

        # Call cleanup (now async)
        result = await execution_manager.cleanup_episode(episode_id)

        # Verify result reflects failure
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_episode_exception(self, execution_manager, mock_sandbox_manager):
        """Test episode cleanup exception handling."""
        episode_id = "test-episode-123"

        # Mock cleanup exception
        mock_sandbox_manager.stop_episode_environment = AsyncMock(side_effect=Exception("Cleanup failed"))

        # Call cleanup (now async)
        result = await execution_manager.cleanup_episode(episode_id)

        # Verify result reflects failure
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_all_containers_success(self, execution_manager, mock_sandbox_manager, mock_permanent_manager):
        """Test successful cleanup of all containers."""
        # Setup permanent manager
        execution_manager._permanent_environment_manager = mock_permanent_manager

        # Mock successful cleanup
        mock_sandbox_manager.get_active_episodes.return_value = ["ep1", "ep2"]

        # Call cleanup (now async)
        result = await execution_manager.cleanup_all_containers({"test": "context"})

        # Verify both sandbox and permanent cleanup were called
        mock_sandbox_manager.cleanup_all_episodes.assert_called_once()
        mock_permanent_manager.stop_permanent_environment.assert_called_once()

        # Verify result structure - now a CleanupResult typed object
        assert hasattr(result, "ephemeral_episodes_cleaned")
        assert hasattr(result, "permanent_environment_stopped")
        assert hasattr(result, "total_cleanup_success")

    @pytest.mark.asyncio
    async def test_cleanup_all_containers_no_permanent_manager(self, execution_manager, mock_sandbox_manager):
        """Test cleanup when no permanent manager is configured."""
        # Ensure no permanent manager
        execution_manager._permanent_environment_manager = None

        # Call cleanup (now async)
        result = await execution_manager.cleanup_all_containers()

        # Verify only sandbox cleanup was called
        mock_sandbox_manager.cleanup_all_episodes.assert_called_once()

        # Verify result indicates permanent environment was "stopped" (no-op)
        assert result.permanent_environment_stopped is True

    @pytest.mark.asyncio
    async def test_stop_permanent_environment_success(self, execution_manager, mock_permanent_manager):
        """Test successful permanent environment stop."""
        execution_manager._permanent_environment_manager = mock_permanent_manager

        # Call stop (now async)
        await execution_manager.stop_permanent_environment()

        # Verify direct manager call
        mock_permanent_manager.stop_permanent_environment.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_permanent_environment_no_manager(self, execution_manager):
        """Test permanent environment stop when no manager configured."""
        execution_manager._permanent_environment_manager = None

        # Should not raise exception (now async)
        await execution_manager.stop_permanent_environment()

    @pytest.mark.asyncio
    async def test_stop_permanent_environment_failure(self, execution_manager, mock_permanent_manager):
        """Test permanent environment stop failure."""
        execution_manager._permanent_environment_manager = mock_permanent_manager
        mock_permanent_manager.stop_permanent_environment = AsyncMock(side_effect=Exception("Stop failed"))

        # Should raise RuntimeError (now async)
        with pytest.raises(RuntimeError, match="Failed to stop permanent environment"):
            await execution_manager.stop_permanent_environment()

    def test_is_permanent_environment_running(self, execution_manager, mock_permanent_manager):
        """Test permanent environment running status check."""
        execution_manager._permanent_environment_manager = mock_permanent_manager
        mock_permanent_manager.is_running.return_value = True

        result = execution_manager.is_permanent_environment_running()

        assert result is True
        mock_permanent_manager.is_running.assert_called_once()

    def test_is_permanent_environment_running_no_manager(self, execution_manager):
        """Test permanent environment status when no manager configured."""
        execution_manager._permanent_environment_manager = None

        result = execution_manager.is_permanent_environment_running()

        assert result is False

    def test_cleanup_session_success(self, execution_manager, mock_sandbox_manager):
        """Test successful session cleanup."""
        session_id = "test-session-123"
        context = {"test": "context"}

        # Call cleanup
        result = execution_manager.cleanup_session(session_id, context)

        # Should return True for success
        assert result is True

    def test_cleanup_session_failure(self, execution_manager, mock_sandbox_manager):
        """Test session cleanup failure handling."""
        session_id = "test-session-123"

        # Mock exception in cleanup
        mock_sandbox_manager.cleanup_session = MagicMock(side_effect=Exception("Cleanup failed"))

        # Call cleanup
        result = execution_manager.cleanup_session(session_id)

        # Should return False for failure
        assert result is False
