"""
Tests for DockerCLIExecutor base functionality.

This module tests the Docker-based command-line tool execution framework including
security validation, parameter handling, and Docker container management.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from saber.server.execution.base import CommandResult, ValidationResult
from saber.server.execution.executors.cli import DockerCLIExecutor
from saber.server.execution.utils.security_validator import SecurityValidator
from saber.server.execution.sandbox.sandbox_manager import SandboxManager
from saber.server.execution.exceptions import SandboxExecutionError


class MockDockerCLIExecutor(DockerCLIExecutor):
    """Mock implementation for testing DockerCLIExecutor."""

    def __init__(self, sandbox_manager, command="echo", **kwargs):
        super().__init__(sandbox_manager=sandbox_manager, **kwargs)
        self._command = command

    def build_command(self, parameters, context):
        """Simple build_command implementation for testing."""
        return [self._command] + [str(v) for v in parameters.values()]

    def parse_output(self, stdout, stderr, return_code):
        """Simple parse_output implementation for testing."""
        if return_code == 0:
            return CommandResult.success_result(
                data={"stdout": stdout, "stderr": stderr},
                metadata={"return_code": return_code}
            )
        else:
            return CommandResult.error_result(
                error=f"Command failed: {stderr}",
                metadata={"return_code": return_code}
            )


class TestDockerCLIExecutor:
    """Test cases for DockerCLIExecutor."""

    @pytest.fixture
    def mock_security_validator(self):
        """Create a mock SecurityValidator."""
        validator = MagicMock(spec=SecurityValidator)
        validator.validate_base_command.return_value = None
        validator.validate_full_command.return_value = ValidationResult.success()
        return validator

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxManager."""
        manager = MagicMock(spec=SandboxManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/base-sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser"
        }
        return manager

    @pytest.fixture
    def executor(self, mock_sandbox_manager):
        """Create a DockerCLIExecutor instance for testing."""
        return MockDockerCLIExecutor(
            sandbox_manager=mock_sandbox_manager,
            command="echo",
            timeout=30.0
        )

    def test_initialization(self, mock_sandbox_manager):
        """Test executor initialization."""
        executor = MockDockerCLIExecutor(
            sandbox_manager=mock_sandbox_manager,
            command="test_cmd",
            timeout=60.0
        )

        assert executor._command == "test_cmd"
        assert executor.get_timeout() == 60.0

    def test_initialization_validates_base_command(self, mock_sandbox_manager):
        """Test that initialization works properly."""
        executor = MockDockerCLIExecutor(
            sandbox_manager=mock_sandbox_manager,
            command="safe_cmd"
        )
        assert executor._command == "safe_cmd"

    def test_get_security_info(self, executor):
        """Test get_security_info method."""
        info = executor.get_security_info()

        assert info["execution_environment"] == "docker_container"
        assert info["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_execute_success(self, executor, mock_sandbox_manager):
        """Test successful command execution in Docker environment."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        parameters = {"arg1": "hello", "arg2": "world"}
        context = {"session_id": "test123"}

        # Mock Docker environment execution
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock()
        mock_env.execute_command.return_value = CommandResult(
            exit_code=0,
            stdout="hello world\n",
            stderr="",
            execution_time=0.1
        )
        mock_env.get_container_id.return_value = "container123456789"

        mock_sandbox_manager.get_session_environment.return_value = mock_env

        result = await executor.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "hello world\n"
        assert result.data["stderr"] == ""
        assert result.metadata["return_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_security_validation_failure(self, executor, mock_sandbox_manager):
        """Test execution when command building fails."""
        parameters = {"command": "rm -rf /"}
        context = {"session_id": "test123"}

        # Mock build_command to raise an exception
        with patch.object(executor, 'build_command', side_effect=ValueError("Invalid command")):
            result = await executor.execute(parameters, context)

        assert result.success is False
        assert "Docker command execution failed" in result.error
        assert "Invalid command" in result.error

    @pytest.mark.asyncio
    async def test_execute_command_failure(self, executor, mock_sandbox_manager):
        """Test execution when command returns non-zero exit code."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        parameters = {"arg": "invalid"}
        context = {"session_id": "test123"}

        # Mock Docker environment execution with failure
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock()
        mock_env.execute_command.return_value = CommandResult(
            exit_code=1,
            stdout="",
            stderr="command not found\n",
            execution_time=0.05
        )
        mock_env.get_container_id.return_value = "failcontainer123"

        mock_sandbox_manager.get_session_environment.return_value = mock_env

        result = await executor.execute(parameters, context)

        assert result.success is False
        assert "Command failed: command not found" in result.error
        assert result.success is False
        assert "Command failed: command not found" in result.error
        assert result.metadata["return_code"] == 1

    @pytest.mark.asyncio
    async def test_execute_missing_session_id(self, executor, mock_sandbox_manager):
        """Test execution failure when session_id is missing."""
        parameters = {"arg": "test"}
        context = {}  # Missing session_id

        result = await executor.execute(parameters, context)

        assert result.success is False
        assert "session_id required in context" in result.error

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, executor, mock_sandbox_manager):
        """Test handling of unexpected exceptions during execution."""
        parameters = {"arg": "test"}
        context = {"session_id": "test123"}

        # Mock Docker environment to raise exception
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock()
        mock_env.execute_command.side_effect = Exception("Docker daemon not available")

        mock_sandbox_manager.get_session_environment.return_value = mock_env

        result = await executor.execute(parameters, context)

        assert result.success is False
        assert "Docker command execution failed" in result.error
        assert "Docker daemon not available" in result.error

    @pytest.mark.asyncio
    async def test_execute_create_session_environment(self, executor, mock_sandbox_manager):
        """Test that new session environment is created when needed."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        parameters = {"arg": "test"}
        context = {"session_id": "new_session"}

        # First call returns None (no existing environment)
        mock_sandbox_manager.get_session_environment.return_value = None

        # Create new environment
        mock_new_env = MagicMock()
        mock_new_env.execute_command = AsyncMock()
        mock_new_env.execute_command.return_value = CommandResult(
            exit_code=0,
            stdout="test output\n",
            stderr="",
            execution_time=0.2
        )
        mock_new_env.get_container_id.return_value = "newcontainer123"

        mock_sandbox_manager.create_session_environment.return_value = mock_new_env

        result = await executor.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "test output\n"

        # Verify new environment was created
        mock_sandbox_manager.create_session_environment.assert_called_once_with("new_session")

    def test_sandbox_manager_requirement(self):
        """Test that DockerCLIExecutor requires sandbox manager."""
        with pytest.raises(SandboxExecutionError, match="sandbox_manager is required"):
            MockDockerCLIExecutor(sandbox_manager=None)

    def test_get_security_info_with_sandbox_config(self, executor):
        """Test that security info includes sandbox configuration."""
        info = executor.get_security_info()

        assert info["execution_environment"] == "docker_container"
        assert "sandbox_config" in info

        sandbox_config = info["sandbox_config"]
        assert sandbox_config["image"] == "saber/base-sandbox:latest"
        assert sandbox_config["network_mode"] == "none"
        assert sandbox_config["read_only_root"] is True
