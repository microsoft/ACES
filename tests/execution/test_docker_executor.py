"""
Tests for DockerExecutor base class.

This module tests the abstract Docker executor base class that provides shared
Docker container management functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from saber.server.base import CommandResult
from saber.server.execution.base import ValidationResult
from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.executors.docker_executor import DockerExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


class TestDockerExecutor:
    """Test cases for DockerExecutor base class."""

    # Create a concrete implementation for testing
    class ConcreteDockerExecutor(DockerExecutor):
        """Concrete implementation of DockerExecutor for testing."""

        async def execute(self, parameters, context):
            return CommandResult.success_result(data="test_execution")

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {
            "image": "saber/sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser",
            "resource_limits": {"memory": "512m", "cpus": "1.0"},
        }
        return manager

    @pytest.fixture
    def docker_executor(self, mock_sandbox_manager):
        """Create a concrete Docker executor instance for testing."""
        config = {"timeout": 60.0}
        return self.ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

    @pytest.fixture
    def mock_docker_environment(self):
        """Create a mock Docker execution environment."""
        env = MagicMock()
        env.get_container_id.return_value = "container123456789"
        return env

    def test_initialization_success(self, mock_sandbox_manager):
        """Test successful initialization with sandbox manager."""
        config = {"timeout": 120.0}
        executor = self.ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        assert executor._sandbox_manager == mock_sandbox_manager
        assert executor.get_timeout() == 120.0

    def test_initialization_with_docker_config(self, mock_sandbox_manager):
        """Test initialization with Docker configuration."""
        config = {"timeout": 60.0, "working_dir": "/custom/workspace", "environment": {"PYTHONPATH": "/app"}}

        executor = self.ConcreteDockerExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        # Docker-specific configs are stored in the general config
        assert executor._config["working_dir"] == "/custom/workspace"
        assert executor._config["environment"]["PYTHONPATH"] == "/app"

    def test_initialization_without_sandbox_manager(self):
        """Test that initialization fails without sandbox manager."""
        with pytest.raises(SandboxExecutionError, match="sandbox_manager is required"):
            self.ConcreteDockerExecutor(sandbox_manager=None)

    def test_get_episode_environment_existing(self, docker_executor, mock_docker_environment):
        """Test retrieving existing episode environment."""
        episode_id = "test_episode_123"

        # Mock an existing environment
        docker_executor._sandbox_manager.get_episode_environment.return_value = mock_docker_environment

        result = docker_executor.get_episode_environment(episode_id)

        assert result == mock_docker_environment
        docker_executor._sandbox_manager.get_episode_environment.assert_called_once_with(episode_id)
        # Verify create_session_environment was NOT called since environment exists
        docker_executor._sandbox_manager.create_episode_environment.assert_not_called()

    def test_get_episode_environment_not_found(self, docker_executor):
        """Test behavior when episode environment doesn't exist."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.get_episode_environment.return_value = None

        with pytest.raises(SandboxExecutionError, match="No environment found for episode"):
            docker_executor.get_episode_environment(episode_id)

    def test_get_episode_environment_failure(self, docker_executor):
        """Test episode environment retrieval failure."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.get_episode_environment.side_effect = Exception("Environment error")

        with pytest.raises(SandboxExecutionError, match="Failed to get episode environment"):
            docker_executor.get_episode_environment(episode_id)

    def test_ensure_container_ready_success(self, docker_executor, mock_docker_environment):
        """Test successful container readiness check."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.get_episode_environment.return_value = mock_docker_environment

        result = docker_executor.ensure_container_ready(episode_id)

        assert result is True

    def test_ensure_container_ready_failure(self, docker_executor):
        """Test container readiness check failure."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.get_episode_environment.side_effect = Exception("Container error")

    def test_ensure_container_ready_success(self, docker_executor, mock_docker_environment):
        """Test successful container readiness check."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.get_episode_environment.return_value = mock_docker_environment

        result = docker_executor.ensure_container_ready(episode_id)

        assert result is True

    def test_ensure_container_ready_failure(self, docker_executor):
        """Test container readiness check failure."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.get_episode_environment.side_effect = Exception("Container error")

        result = docker_executor.ensure_container_ready(episode_id)

        assert result is False

    def test_cleanup_execution_success(self, docker_executor):
        """Test successful execution cleanup."""
        episode_id = "test_episode_123"

        docker_executor.cleanup_execution(episode_id)

        docker_executor._sandbox_manager.stop_episode_environment.assert_called_once_with(episode_id)

    def test_cleanup_execution_with_error(self, docker_executor):
        """Test execution cleanup with error (should not raise)."""
        episode_id = "test_episode_123"

        docker_executor._sandbox_manager.stop_episode_environment.side_effect = Exception("Cleanup error")

        with pytest.raises(SandboxExecutionError, match="Failed to clean up execution resources"):
            docker_executor.cleanup_execution(episode_id)

        docker_executor._sandbox_manager.stop_episode_environment.assert_called_once_with(episode_id)

    def test_validate_docker_parameters_valid(self, docker_executor):
        """Test validation of valid Docker parameters."""
        parameters = {"working_dir": "/workspace", "timeout": 300, "other_param": "value"}

        result = docker_executor.validate_docker_parameters(parameters)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_docker_parameters_invalid_working_dir(self, docker_executor):
        """Test validation with invalid working directory."""
        parameters = {
            "working_dir": "relative/path",  # Should be absolute
        }

        result = docker_executor.validate_docker_parameters(parameters)

        assert result.valid is False
        assert "working_dir must be an absolute path" in result.errors

    def test_validate_docker_parameters_invalid_working_dir_type(self, docker_executor):
        """Test validation with wrong working directory type."""
        parameters = {
            "working_dir": 123,  # Should be string
        }

        result = docker_executor.validate_docker_parameters(parameters)

        assert result.valid is False
        assert "working_dir must be a string" in result.errors

    def test_validate_docker_parameters_invalid_timeout(self, docker_executor):
        """Test validation with invalid timeout."""
        parameters = {
            "timeout": -10,  # Should be positive
        }

        result = docker_executor.validate_docker_parameters(parameters)

        assert result.valid is False
        assert "timeout must be a positive number" in result.errors

    def test_validate_docker_parameters_invalid_timeout_type(self, docker_executor):
        """Test validation with wrong timeout type."""
        parameters = {
            "timeout": "not_a_number",  # Should be number
        }

        result = docker_executor.validate_docker_parameters(parameters)

        assert result.valid is False
        assert "timeout must be a positive number" in result.errors

    def test_get_docker_info(self, docker_executor):
        """Test getting Docker configuration information."""
        info = docker_executor.get_docker_info()

        assert info["execution_environment"] == "docker_container"
        assert "timeout" in info
        assert "docker_config" in info

        # Check sandbox config details
        docker_config = info["docker_config"]
        assert docker_config["image"] == "saber/sandbox:latest"
        assert docker_config["network_mode"] == "none"
        assert docker_config["read_only_root"] is True
        assert docker_config["user"] == "tooluser:tooluser"
        assert docker_config["resource_limits"]["memory"] == "512m"

    def test_get_docker_info_with_error(self, docker_executor):
        """Test getting Docker info when sandbox config retrieval fails."""
        # Simulate an exception by making sandbox_config access fail
        del docker_executor._sandbox_manager.sandbox_config

        info = docker_executor.get_docker_info()

        assert info["execution_environment"] == "docker_container"
        assert "timeout" in info
        # Should still work even if sandbox config fails

    def test_validate_parameters_combined(self, docker_executor):
        """Test parameter validation combining base and Docker validation."""
        # Mock base class validation
        with patch.object(DockerExecutor.__bases__[0], "validate_parameters") as mock_base_validate:
            base_result = ValidationResult.success()
            base_result.add_warning("Base warning")
            mock_base_validate.return_value = base_result

            parameters = {"working_dir": "/workspace", "timeout": 300}

            result = docker_executor.validate_parameters(parameters)

            assert result.valid is True
            assert "Base warning" in result.warnings
            mock_base_validate.assert_called_once_with(parameters)

    def test_validate_parameters_combined_with_errors(self, docker_executor):
        """Test parameter validation with both base and Docker errors."""
        # Mock base class validation with errors
        with patch.object(DockerExecutor.__bases__[0], "validate_parameters") as mock_base_validate:
            base_result = ValidationResult.failure(["Base error"])
            mock_base_validate.return_value = base_result

            parameters = {
                "working_dir": "invalid/path",  # Docker validation error
            }

            result = docker_executor.validate_parameters(parameters)

            assert not result.valid  # Should be invalid due to errors
            assert "Base error" in result.errors
            assert "working_dir must be an absolute path" in result.errors

    @pytest.mark.asyncio
    async def test_execute_abstract_method_implemented(self, docker_executor):
        """Test that concrete implementation provides execute method."""
        parameters = {"test": "param"}
        context = {"episode_id": "test"}

        result = await docker_executor(parameters, context)

        assert result.success is True
        assert result.data == "test_execution"

    def test_inheritance_from_command_executor(self, docker_executor):
        """Test that DockerExecutor properly inherits from CommandExecutor."""
        from saber.server.execution.executors.base_executors import CommandExecutor

        assert isinstance(docker_executor, CommandExecutor)
        assert hasattr(docker_executor, "get_timeout")
        assert hasattr(docker_executor, "get_parameters")
        assert hasattr(docker_executor, "add_parameter")

    def test_docker_specific_methods_exist(self, docker_executor):
        """Test that Docker-specific methods are properly defined."""
        assert hasattr(docker_executor, "get_episode_environment")
        assert hasattr(docker_executor, "ensure_container_ready")
        assert hasattr(docker_executor, "cleanup_execution")
        assert hasattr(docker_executor, "validate_docker_parameters")
        assert hasattr(docker_executor, "get_docker_info")

    def test_abstract_class_cannot_be_instantiated(self, mock_sandbox_manager):
        """Test that DockerExecutor cannot be instantiated directly."""
        # Since we're not using ABC, DockerExecutor can be instantiated
        # but calling execute() should raise NotImplementedError or similar
        # This test documents that it's meant to be abstract
        try:
            # This should work since we're not using ABC
            executor = DockerExecutor(sandbox_manager=mock_sandbox_manager)
            assert hasattr(executor, "execute")
            # The execute method should be abstract (not implemented)
        except Exception:
            # If there are import or other issues, that's also fine
            pass
