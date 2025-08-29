"""
Unit tests for session manager lifecycle integration.

Tests the integration between SessionManager and PermanentEnvironmentManager
for server startup and shutdown hooks.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from saber.server.execution.sandbox.environment_spec import (
    PermanentEnvironmentSpec,
    PermanentNetworkSpec,
    PermanentServiceSpec,
)
from saber.server.session_manager import SessionManager


class TestSessionManagerLifecycleIntegration:
    """Test SessionManager integration with permanent environment lifecycle."""

    @pytest.fixture
    def mock_permanent_env_manager(self):
        """Create a mock PermanentEnvironmentManager."""
        manager = Mock()
        manager.is_running.return_value = False
        manager.ensure_permanent_environments_current = Mock()
        manager.stop_permanent_environment = Mock()
        manager.cleanup_on_server_shutdown = Mock()
        return manager

    @pytest.fixture
    def session_manager_config(self):
        """Configuration for SessionManager."""
        return {
            "domain_name": "test_domain",
            "config_dir": "/test/config",
            "host": "0.0.0.0",
            "port": 8000,
            "mcp_host": "0.0.0.0",
            "mcp_port": 3001,
        }

    @pytest.fixture
    def sample_permanent_spec(self):
        """Create a sample permanent environment specification."""
        service_spec = PermanentServiceSpec(
            name="test_service",
            image="nginx:latest",
            ports=["80:80"],
        )
        network_spec = PermanentNetworkSpec(
            name="test_network",
            driver="bridge"
        )
        return PermanentEnvironmentSpec(
            services={"test_service": service_spec},
            networks={"test_network": network_spec}
        )

    @patch('saber.server.session_manager.PolicyManager')
    @patch('saber.server.session_manager.EvaluationManager')
    @patch('saber.server.session_manager.ContainerCleanupManager')
    @patch('saber.server.session_manager.ExecutionManager')
    @patch('saber.server.session_manager.EpisodeManager')
    @patch('saber.server.session_manager.BenchmarkManager')
    @patch('saber.server.session_manager.SessionMCPAPI')
    @patch('saber.server.session_manager.SessionRestAPI')
    @patch('saber.server.session_manager.PermanentEnvironmentManager')
    def test_session_manager_initialization_with_permanent_env(
        self,
        mock_perm_env_manager_class,
        mock_rest_api,
        mock_mcp_api,
        mock_benchmark_manager,
        mock_episode_manager,
        mock_execution_manager,
        mock_cleanup_manager,
        mock_evaluation_manager,
        mock_policy_manager,
        session_manager_config,
        mock_permanent_env_manager
    ):
        """Test SessionManager initialization with permanent environment manager."""
        mock_perm_env_manager_class.return_value = mock_permanent_env_manager

        session_manager = SessionManager(**session_manager_config)

        # Verify PermanentEnvironmentManager was initialized with correct config
        expected_config = {
            "domain": "test_domain",
            "config_dir": "/test/config",
            "enable_logging": True,
        }
        mock_perm_env_manager_class.assert_called_once_with(expected_config)
        assert session_manager.permanent_environment_manager == mock_permanent_env_manager

    @pytest.mark.asyncio
    @patch('saber.server.session_manager.PolicyManager')
    @patch('saber.server.session_manager.EvaluationManager')
    @patch('saber.server.session_manager.ContainerCleanupManager')
    @patch('saber.server.session_manager.ExecutionManager')
    @patch('saber.server.session_manager.EpisodeManager')
    @patch('saber.server.session_manager.BenchmarkManager')
    @patch('saber.server.session_manager.SessionMCPAPI')
    @patch('saber.server.session_manager.SessionRestAPI')
    @patch('saber.server.session_manager.PermanentEnvironmentManager')
    async def test_start_server_with_permanent_environment(
        self,
        mock_perm_env_manager_class,
        mock_rest_api,
        mock_mcp_api,
        mock_benchmark_manager,
        mock_episode_manager,
        mock_execution_manager,
        mock_cleanup_manager,
        mock_evaluation_manager,
        mock_policy_manager,
        session_manager_config,
        mock_permanent_env_manager,
        sample_permanent_spec
    ):
        """Test server startup with permanent environment initialization."""
        # Setup mocks
        mock_perm_env_manager_class.return_value = mock_permanent_env_manager

        # Mock execution manager's environment loader
        mock_env_loader = Mock()
        mock_env_loader.load_permanent_environment.return_value = sample_permanent_spec
        mock_execution_manager.return_value._environment_loader = mock_env_loader

        # Mock benchmark manager config loader
        mock_config_loader = Mock()
        mock_config_loader.get_permanent_environment.return_value = "test_permanent_env"
        mock_benchmark_manager.return_value.config_loader = mock_config_loader

        # Mock REST and MCP API start methods
        mock_rest_api.return_value.start_server = AsyncMock()
        mock_mcp_api.return_value.start_mcp_server = AsyncMock()

        session_manager = SessionManager(**session_manager_config)

        # Call _start_permanent_environment directly to test the logic
        await session_manager._start_permanent_environment()

        # Verify permanent environment was started with correct spec
        mock_permanent_env_manager.ensure_permanent_environments_current.assert_called_once_with(sample_permanent_spec)

    @pytest.mark.asyncio
    @patch('saber.server.session_manager.PolicyManager')
    @patch('saber.server.session_manager.EvaluationManager')
    @patch('saber.server.session_manager.ContainerCleanupManager')
    @patch('saber.server.session_manager.ExecutionManager')
    @patch('saber.server.session_manager.EpisodeManager')
    @patch('saber.server.session_manager.BenchmarkManager')
    @patch('saber.server.session_manager.SessionMCPAPI')
    @patch('saber.server.session_manager.SessionRestAPI')
    @patch('saber.server.session_manager.PermanentEnvironmentManager')
    async def test_shutdown_with_permanent_environment_cleanup(
        self,
        mock_perm_env_manager_class,
        mock_rest_api,
        mock_mcp_api,
        mock_benchmark_manager,
        mock_episode_manager,
        mock_execution_manager,
        mock_cleanup_manager,
        mock_evaluation_manager,
        mock_policy_manager,
        session_manager_config,
        mock_permanent_env_manager
    ):
        """Test server shutdown with permanent environment cleanup."""
        # Setup mocks
        mock_perm_env_manager_class.return_value = mock_permanent_env_manager
        mock_permanent_env_manager.is_running.return_value = True

        mock_mcp_api.return_value.shutdown_mcp_server = AsyncMock()

        session_manager = SessionManager(**session_manager_config)
        session_manager.cleanup_task = None  # No cleanup task to cancel

        await session_manager.shutdown()

        # Verify permanent environment cleanup was called
        mock_permanent_env_manager.cleanup_on_server_shutdown.assert_called_once()

    @pytest.mark.asyncio
    @patch('saber.server.session_manager.PolicyManager')
    @patch('saber.server.session_manager.EvaluationManager')
    @patch('saber.server.session_manager.ContainerCleanupManager')
    @patch('saber.server.session_manager.ExecutionManager')
    @patch('saber.server.session_manager.EpisodeManager')
    @patch('saber.server.session_manager.BenchmarkManager')
    @patch('saber.server.session_manager.SessionMCPAPI')
    @patch('saber.server.session_manager.SessionRestAPI')
    @patch('saber.server.session_manager.PermanentEnvironmentManager')
    async def test_start_permanent_environment_no_permanent_config(
        self,
        mock_perm_env_manager_class,
        mock_rest_api,
        mock_mcp_api,
        mock_benchmark_manager,
        mock_episode_manager,
        mock_execution_manager,
        mock_cleanup_manager,
        mock_evaluation_manager,
        mock_policy_manager,
        session_manager_config,
        mock_permanent_env_manager
    ):
        """Test startup when no permanent environment is configured."""
        # Setup mocks
        mock_perm_env_manager_class.return_value = mock_permanent_env_manager

        # Mock benchmark manager config loader to return None (no permanent env)
        mock_config_loader = Mock()
        mock_config_loader.get_permanent_environment.return_value = None
        mock_benchmark_manager.return_value.config_loader = mock_config_loader

        session_manager = SessionManager(**session_manager_config)

        await session_manager._start_permanent_environment()

        # Should not call ensure_permanent_environments_current if no permanent config
        mock_permanent_env_manager.ensure_permanent_environments_current.assert_not_called()

    @pytest.mark.asyncio
    @patch('saber.server.session_manager.PolicyManager')
    @patch('saber.server.session_manager.EvaluationManager')
    @patch('saber.server.session_manager.ContainerCleanupManager')
    @patch('saber.server.session_manager.ExecutionManager')
    @patch('saber.server.session_manager.EpisodeManager')
    @patch('saber.server.session_manager.BenchmarkManager')
    @patch('saber.server.session_manager.SessionMCPAPI')
    @patch('saber.server.session_manager.SessionRestAPI')
    @patch('saber.server.session_manager.PermanentEnvironmentManager')
    async def test_shutdown_with_no_permanent_environment_running(
        self,
        mock_perm_env_manager_class,
        mock_rest_api,
        mock_mcp_api,
        mock_benchmark_manager,
        mock_episode_manager,
        mock_execution_manager,
        mock_cleanup_manager,
        mock_evaluation_manager,
        mock_policy_manager,
        session_manager_config,
        mock_permanent_env_manager
    ):
        """Test server shutdown when no permanent environment is running."""
        # Setup mocks
        mock_perm_env_manager_class.return_value = mock_permanent_env_manager
        mock_permanent_env_manager.is_running.return_value = False

        mock_mcp_api.return_value.shutdown_mcp_server = AsyncMock()

        session_manager = SessionManager(**session_manager_config)
        session_manager.cleanup_task = None  # No cleanup task to cancel

        await session_manager.shutdown()

        # Verify cleanup was NOT called when not running (matching the actual logic)
        mock_permanent_env_manager.cleanup_on_server_shutdown.assert_not_called()

    @pytest.mark.asyncio
    @patch('saber.server.session_manager.PolicyManager')
    @patch('saber.server.session_manager.EvaluationManager')
    @patch('saber.server.session_manager.ContainerCleanupManager')
    @patch('saber.server.session_manager.ExecutionManager')
    @patch('saber.server.session_manager.EpisodeManager')
    @patch('saber.server.session_manager.BenchmarkManager')
    @patch('saber.server.session_manager.SessionMCPAPI')
    @patch('saber.server.session_manager.SessionRestAPI')
    @patch('saber.server.session_manager.PermanentEnvironmentManager')
    async def test_start_permanent_environment_exception_handling(
        self,
        mock_perm_env_manager_class,
        mock_rest_api,
        mock_mcp_api,
        mock_benchmark_manager,
        mock_episode_manager,
        mock_execution_manager,
        mock_cleanup_manager,
        mock_evaluation_manager,
        mock_policy_manager,
        session_manager_config,
        mock_permanent_env_manager,
        sample_permanent_spec
    ):
        """Test exception handling during permanent environment startup."""
        # Setup mocks
        mock_perm_env_manager_class.return_value = mock_permanent_env_manager

        # Mock execution manager's environment loader
        mock_env_loader = Mock()
        mock_env_loader.load_permanent_environment.return_value = sample_permanent_spec
        mock_execution_manager.return_value._environment_loader = mock_env_loader

        # Mock benchmark manager config loader
        mock_config_loader = Mock()
        mock_config_loader.get_permanent_environment.return_value = "test_permanent_env"
        mock_benchmark_manager.return_value.config_loader = mock_config_loader

        # Make ensure_permanent_environments_current raise an exception
        mock_permanent_env_manager.ensure_permanent_environments_current.side_effect = Exception("Network error")

        session_manager = SessionManager(**session_manager_config)

        # Should raise exception since it propagates up
        with pytest.raises(Exception, match="Network error"):
            await session_manager._start_permanent_environment()

        # Verify the call was attempted
        mock_permanent_env_manager.ensure_permanent_environments_current.assert_called_once_with(sample_permanent_spec)
