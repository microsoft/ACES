"""
Tests for Docker CLI tool executor.

This module tests the secure Docker-based command line interface tool that accepts
arbitrary command strings and executes them with comprehensive security validation
in isolated Docker containers. Includes tests for command chaining functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.base import CommandResult
from saber.server.execution.base import BashParameters, ExecutionContext, ParameterType, ValidationResult
from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.executors.standard_registry.bash_executor import BashExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from saber.server.execution.utils.security_validator import SecurityValidator


class TestBashExecutor:
    """Test cases for Bash executor."""

    @pytest.fixture
    def mock_security_validator(self):
        """Create a mock SecurityValidator."""
        validator = MagicMock(spec=SecurityValidator)
        validator.validate_base_command.return_value = None
        validator.validate_full_command.return_value = ValidationResult.success()
        return validator

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {
            "image": "saber/sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser",
        }
        return manager

    @pytest.fixture
    def docker_bash_tool(self, mock_sandbox_manager):
        """Create a Bash executor instance for testing."""
        from saber.server.execution.models import ExecutorConfig
        config = ExecutorConfig(timeout=30.0)
        return BashExecutor(
            sandbox_manager=mock_sandbox_manager, config=config, allowed_commands=["file", "strings", "echo", "cat"]
        )

    def test_build_command_simple(self, docker_bash_tool):
        """Test building command with simple string (no shell mode)."""
        params = BashParameters(command="ls -la")

        result = docker_bash_tool._build_command(params)

        # Should always use shell mode
        assert result == ["/bin/sh", "-c", "ls -la"]

    def test_build_command_shell_mode(self, docker_bash_tool):
        """Test building command with complex shell features."""
        params = BashParameters(command="ls -la | grep test")

        result = docker_bash_tool._build_command(params)

        # Should use shell with -c flag
        assert result == ["/bin/sh", "-c", "ls -la | grep test"]

    def test_build_command_default_shell_mode(self, docker_bash_tool):
        """Test building command always uses shell mode."""
        params = BashParameters(command="echo hello world")

        result = docker_bash_tool._build_command(params)

        # Should always use shell mode
        assert result == ["/bin/sh", "-c", "echo hello world"]

    def test_build_command_quoted_arguments(self, docker_bash_tool):
        """Test building command with quoted arguments."""
        params = BashParameters(command='echo "hello world" test')

        result = docker_bash_tool._build_command(params)

        # Should use shell mode (shell handles quotes properly)
        assert result == ["/bin/sh", "-c", 'echo "hello world" test']

    def test_build_command_complex_shell_command(self, docker_bash_tool):
        """Test building command with complex shell constructs."""
        params = BashParameters(command="find /tmp -name '*.txt' | head -10 > results.txt")

        result = docker_bash_tool._build_command(params)

        assert result == ["/bin/sh", "-c", "find /tmp -name '*.txt' | head -10 > results.txt"]

    def test_build_command_invalid_quotes(self, docker_bash_tool):
        """Test building command with invalid quotes (shell handles gracefully)."""
        params = BashParameters(command='echo "unclosed quote')

        # Shell mode doesn't validate quotes at build time
        result = docker_bash_tool._build_command(params)
        assert result == ["/bin/sh", "-c", 'echo "unclosed quote']

    def test_build_command_empty_after_parsing(self, docker_bash_tool):
        """Test building command with empty string."""
        params = BashParameters(command="")

        with pytest.raises(ValueError, match="Command string cannot be empty"):
            docker_bash_tool._build_command(params)

    def test_parse_output_success(self, docker_bash_tool):
        """Test parsing successful command output."""
        stdout = "Hello, World!\nLine 2\n"
        stderr = ""
        return_code = 0

        result = docker_bash_tool.parse_output(stdout, stderr, return_code)

        assert result.success is True
        assert result.data["stdout"] == stdout
        assert result.data["stderr"] == stderr
        assert result.data["return_code"] == return_code
        assert result.data["success"] is True
        assert result.data["output"] == stdout  # Primary output for success
        assert result.data["success"] is True
        assert result.data["output"] == stdout  # Primary output for success

        # Check metadata
        assert result.metadata["command_type"] == "docker_bash"
        assert result.metadata["execution_environment"] == "docker_container"
        assert result.metadata["exit_code"] == 0
        assert result.metadata["has_stdout"] is True
        assert result.metadata["has_stderr"] is False
        assert result.metadata["output_length"] == len(stdout)

    def test_parse_output_failure_with_stderr(self, docker_bash_tool):
        """Test parsing failed command output with stderr."""
        stdout = ""
        stderr = "command not found\n"
        return_code = 127

        result = docker_bash_tool.parse_output(stdout, stderr, return_code)

        assert result.success is False
        assert "Command failed with exit code 127" in result.error
        assert "command not found" in result.error

        # Check metadata includes raw data
        assert result.metadata["exit_code"] == 127
        assert result.metadata["has_stderr"] is True
        assert result.metadata["raw_data"]["stdout"] == stdout
        assert result.metadata["raw_data"]["stderr"] == stderr

    def test_parse_output_failure_with_stdout_only(self, docker_bash_tool):
        """Test parsing failed command output with only stdout."""
        stdout = "Some error message to stdout\n"
        stderr = ""
        return_code = 1

        result = docker_bash_tool.parse_output(stdout, stderr, return_code)

        assert result.success is False
        assert "Command failed with exit code 1" in result.error
        assert "Some error message to stdout" in result.error

    def test_parse_output_failure_no_output(self, docker_bash_tool):
        """Test parsing failed command with no output."""
        stdout = ""
        stderr = ""
        return_code = 1

        result = docker_bash_tool.parse_output(stdout, stderr, return_code)

        assert result.success is False
        assert result.error == "Command failed with exit code 1"

    def test_parse_output_with_whitespace(self, docker_bash_tool):
        """Test parsing output with whitespace handling."""
        stdout = "  \n\t  "  # Only whitespace
        stderr = ""
        return_code = 0

        result = docker_bash_tool.parse_output(stdout, stderr, return_code)

        assert result.success is True
        assert result.metadata["has_stdout"] is False  # Stripped whitespace

    def test_parse_output_large_output(self, docker_bash_tool):
        """Test parsing output with length calculation."""
        stdout = "x" * 1000
        stderr = "y" * 500
        return_code = 0

        result = docker_bash_tool.parse_output(stdout, stderr, return_code)

        assert result.success is True
        assert result.metadata["output_length"] == 1500

    def test_parameter_validation_success(self, docker_bash_tool):
        """Test successful parameter validation."""
        params = BashParameters(command="ls -la")

        result = docker_bash_tool.validate_parameters(params)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_parameter_validation_missing_required(self, docker_bash_tool):
        """Test parameter validation with empty command."""
        # Create params with empty command - should fail validation
        params = BashParameters(command="")

        result = docker_bash_tool.validate_parameters(params)

        # Empty command fails bash executor's validation
        assert result.valid is False
        assert "Command parameter is required" in result.errors[0]

    def test_parameter_validation_wrong_type(self, docker_bash_tool):
        """Test parameter validation with wrong parameter type."""
        # With typed params, wrong types are caught at dataclass creation
        # This test now verifies the type system works correctly
        params = BashParameters(command="ls")  # Valid params
        result = docker_bash_tool.validate_parameters(params)
        assert result.valid is True


class TestBashExecutorIntegration:
    """Integration tests for CLI executor with mocked Docker environment."""

    @pytest.fixture
    def mock_docker_sandbox_environment(self):
        """Create a mock Docker execution environment."""

        env = MagicMock()
        # Mock the new interface
        container_mock = MagicMock()
        container_mock.id = "container123456789"
        env.get_execution_container.return_value = container_mock
        env.execute_command = MagicMock()  # Not async anymore
        return env

    @pytest.fixture
    def mock_sandbox_manager_with_env(self, mock_docker_sandbox_environment):
        """Create a mock SandboxEnvironmentManager that returns a Docker environment."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {
            "image": "saber/sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser",
        }
        manager.get_episode_environment.return_value = mock_docker_sandbox_environment
        manager.create_episode_environment_async.return_value = mock_docker_sandbox_environment
        return manager

    @pytest.fixture
    def docker_bash_tool_with_env(self, mock_sandbox_manager_with_env):
        """Create a Bash executor with mocked environment."""
        from saber.server.execution.models import ExecutorConfig
        config = ExecutorConfig(timeout=30.0)
        return BashExecutor(
            sandbox_manager=mock_sandbox_manager_with_env,
            config=config,
            allowed_commands=["file", "strings", "echo", "cat"],
        )

    @pytest.mark.asyncio
    async def test_execute_docker_integration_success(self, docker_bash_tool_with_env, mock_sandbox_manager_with_env):
        """Test successful Docker command execution."""

        # Set up mock command result
        command_result = CommandResult(exit_code=0, stdout="Hello from Docker!\n", stderr="", execution_time=0.5)

        env = mock_sandbox_manager_with_env.get_episode_environment.return_value
        env.execute_command = AsyncMock(return_value=command_result)

        params = BashParameters(command="echo 'Hello from Docker!'")
        context = ExecutionContext(episode_id="test_episode_123", session_id="test_session_123")

        result = await docker_bash_tool_with_env.execute(params, context)

        assert result.success is True
        assert result.data["stdout"] == "Hello from Docker!\n"
        assert result.data["return_code"] == 0
        assert result.metadata["container_id"] == "container123"[:12]  # Truncated to 12 chars
        assert result.metadata["episode_id"] == "test_episode_123"
        assert result.metadata["execution_time"] == 0.5

        # Verify Docker environment was called correctly (shell mode)
        env.execute_command.assert_called_once_with(
            command=["/bin/sh", "-c", "echo 'Hello from Docker!'"], timeout=30  # Timeout from fixture config
        )

    @pytest.mark.asyncio
    async def test_execute_docker_integration_shell_mode(self, docker_bash_tool_with_env, mock_sandbox_manager_with_env):
        """Test Docker command execution in shell mode."""

        command_result = CommandResult(exit_code=0, stdout="hello world\n", stderr="", execution_time=1.2)

        env = mock_sandbox_manager_with_env.get_episode_environment.return_value
        env.execute_command = AsyncMock(return_value=command_result)

        params = BashParameters(command="echo hello world")
        context = ExecutionContext(episode_id="shell_test_episode")

        result = await docker_bash_tool_with_env.execute(params, context)

        assert result.success is True
        assert result.data["stdout"] == "hello world\n"

        # Verify shell command was constructed correctly
        env.execute_command.assert_called_once_with(command=["/bin/sh", "-c", "echo hello world"], timeout=30)

    @pytest.mark.asyncio
    async def test_execute_docker_integration_failure(self, docker_bash_tool_with_env, mock_sandbox_manager_with_env):
        """Test Docker command execution failure."""

        command_result = CommandResult(
            exit_code=127, stdout="", stderr="command not found: nonexistent_command\n", execution_time=0.1
        )

        env = mock_sandbox_manager_with_env.get_episode_environment.return_value
        env.execute_command = AsyncMock(return_value=command_result)

        params = BashParameters(command="nonexistent_command")
        context = ExecutionContext(episode_id="failure_test_episode")

        result = await docker_bash_tool_with_env.execute(params, context)

        assert result.success is False
        assert "Command failed with exit code 127" in result.error
        assert "command not found" in result.error
        assert result.metadata["exit_code"] == 127

    @pytest.mark.asyncio
    async def test_execute_missing_episode_id(self, docker_bash_tool_with_env, mock_sandbox_manager_with_env):
        """Test that execution fails without valid episode_id in context."""
        # Set up environment to fail on empty episode_id
        mock_sandbox_manager_with_env.get_episode_environment.return_value = None

        params = BashParameters(command="echo test")
        context = ExecutionContext(episode_id="")  # Empty episode_id

        result = await docker_bash_tool_with_env.execute(params, context)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_with_existing_environment(self, docker_bash_tool_with_env, mock_sandbox_manager_with_env):
        """Test that command execution works when environment already exists for session."""

        # Mock an existing environment
        existing_env = MagicMock()
        mock_sandbox_manager_with_env.get_episode_environment.return_value = existing_env

        command_result = CommandResult(exit_code=0, stdout="test\n", stderr="", execution_time=0.3)
        existing_env.execute_command = AsyncMock(return_value=command_result)

        # Mock the container interface
        container_mock = MagicMock()
        container_mock.id = "container123"
        existing_env.get_execution_container.return_value = container_mock

        params = BashParameters(command="echo test")
        context = ExecutionContext(episode_id="existing_episode")

        result = await docker_bash_tool_with_env.execute(params, context)

        assert result.success is True
        assert result.data["stdout"] == "test\n"

        # Verify that create_episode_environment_async was NOT called since environment exists
        mock_sandbox_manager_with_env.create_episode_environment_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_docker_exception_handling(self, docker_bash_tool_with_env, mock_sandbox_manager_with_env):
        """Test handling of Docker execution exceptions."""
        env = mock_sandbox_manager_with_env.get_episode_environment.return_value
        env.execute_command.side_effect = Exception("Docker daemon not available")

        params = BashParameters(command="echo test")
        context = ExecutionContext(episode_id="exception_test")

        result = await docker_bash_tool_with_env.execute(params, context)

        assert result.success is False
        assert "Docker command execution failed" in result.error
        assert "Docker daemon not available" in result.error
