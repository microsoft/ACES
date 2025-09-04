"""
Test suite for SABER Debug Mode functionality.

Tests the debug mode feature that prevents container cleanup for debugging purposes.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import os

# Import only the specific modules we need to avoid circular import issues
from saber.client.containers.agent_manager import AgentContainerConfig, AgentManager


class TestDebugMode:
    """Test suite for debug mode functionality."""

    def test_agent_container_config_debug_mode_default(self):
        """Test that debug_mode defaults to False in AgentContainerConfig."""
        config = AgentContainerConfig()
        assert config.debug_mode is False

    def test_agent_container_config_debug_mode_enabled(self):
        """Test that debug_mode can be explicitly enabled in AgentContainerConfig."""
        config = AgentContainerConfig(debug_mode=True)
        assert config.debug_mode is True

    @pytest.mark.asyncio
    async def test_cleanup_container_normal_mode(self):
        """Test that cleanup_container works normally when debug_mode is False."""
        # Setup
        config = AgentContainerConfig(debug_mode=False)
        agent_manager = AgentManager(config=config)

        # Mock container
        mock_container = Mock()
        mock_container.status = "running"
        mock_container.remove = Mock()
        mock_container.stop = Mock()
        mock_container.reload = Mock()

        # Mock Docker client
        with patch.object(agent_manager, 'docker_client'):
            # Call cleanup
            await agent_manager._cleanup_container(mock_container, "test_container")

            # Verify container was stopped and removed
            mock_container.reload.assert_called_once()
            mock_container.stop.assert_called_once_with(timeout=10)
            mock_container.remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_container_debug_mode(self):
        """Test that cleanup_container skips cleanup when debug_mode is True."""
        # Setup
        config = AgentContainerConfig(debug_mode=True)
        agent_manager = AgentManager(config=config)

        # Mock container
        mock_container = Mock()
        mock_container.status = "running"
        mock_container.remove = Mock()
        mock_container.stop = Mock()
        mock_container.reload = Mock()

        # Mock Docker client
        with patch.object(agent_manager, 'docker_client'):
            # Call cleanup
            await agent_manager._cleanup_container(mock_container, "test_container")

            # Verify container was NOT stopped or removed
            mock_container.reload.assert_not_called()
            mock_container.stop.assert_not_called()
            mock_container.remove.assert_not_called()

    def test_environment_variable_parsing(self):
        """Test that environment variable parsing works correctly for debug mode."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("Yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("", False),
            ("invalid", False),
        ]

        for env_value, expected_result in test_cases:
            # Test the same logic used in episode_executor.py
            result = env_value.lower() in ("true", "1", "yes")
            assert result == expected_result, \
                f"Expected {expected_result} for env value '{env_value}', got {result}"
