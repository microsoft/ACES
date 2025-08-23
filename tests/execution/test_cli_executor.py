"""
Tests for Docker CLI tool executor.

This module tests the secure Docker-based command line interface tool that accepts
arbitrary command strings and executes them with comprehensive security validation
in isolated Docker containers. Includes tests for command chaining functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.base import CommandResult
from saber.server.execution.base import ParameterType, ValidationResult
from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.executors.standard_registry.cli_executor import CLIExecutor
from saber.server.execution.sandbox.sandbox_manager import SandboxManager
from saber.server.execution.utils.security_validator import SecurityValidator


class TestCLIExecutor:
    """Test cases for CLI executor."""

    @pytest.fixture
    def mock_security_validator(self):
        """Create a mock SecurityValidator."""
        validator = MagicMock(spec=SecurityValidator)
        validator.validate_base_command.return_value = None
        validator.validate_full_command.return_value = ValidationResult.success()
        validator.get_security_info.return_value = {"test": "security_info"}
        return validator

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxManager."""
        manager = MagicMock(spec=SandboxManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/base-sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser",
        }
        return manager

    @pytest.fixture
    def docker_cli_tool(self, mock_sandbox_manager):
        """Create a CLI executor instance for testing."""
        config = {
            "timeout": 30.0,
        }
        return CLIExecutor(
            sandbox_manager=mock_sandbox_manager, config=config, allowed_commands=["file", "strings", "echo", "cat"]
        )

    def test_executor_metadata(self, docker_cli_tool):
        """Test executor metadata is correctly defined."""
        metadata = docker_cli_tool._executor_metadata

        assert metadata["name"] == "execute_cli"
        assert metadata["description"] == "Execute CLI commands in secure Docker container"

    def test_build_command_simple(self, docker_cli_tool):
        """Test building command with simple string (no shell mode)."""
        parameters = {"command": "ls -la"}
        context = {}

        result = docker_cli_tool.build_command(parameters, context)

        # Should always use shell mode
        assert result == ["/bin/sh", "-c", "ls -la"]

    def test_build_command_shell_mode(self, docker_cli_tool):
        """Test building command with complex shell features."""
        parameters = {"command": "ls -la | grep test"}
        context = {}

        result = docker_cli_tool.build_command(parameters, context)

        # Should use shell with -c flag
        assert result == ["/bin/sh", "-c", "ls -la | grep test"]

    def test_build_command_default_shell_mode(self, docker_cli_tool):
        """Test building command always uses shell mode."""
        parameters = {"command": "echo hello world"}
        context = {}

        result = docker_cli_tool.build_command(parameters, context)

        # Should always use shell mode
        assert result == ["/bin/sh", "-c", "echo hello world"]

    def test_build_command_quoted_arguments(self, docker_cli_tool):
        """Test building command with quoted arguments."""
        parameters = {"command": 'echo "hello world" test'}
        context = {}

        result = docker_cli_tool.build_command(parameters, context)

        # Should use shell mode (shell handles quotes properly)
        assert result == ["/bin/sh", "-c", 'echo "hello world" test']

    def test_build_command_complex_shell_command(self, docker_cli_tool):
        """Test building command with complex shell constructs."""
        parameters = {"command": "find /tmp -name '*.txt' | head -10 > results.txt"}
        context = {}

        result = docker_cli_tool.build_command(parameters, context)

        assert result == ["/bin/sh", "-c", "find /tmp -name '*.txt' | head -10 > results.txt"]

    def test_build_command_invalid_quotes(self, docker_cli_tool):
        """Test building command with invalid quotes (shell handles gracefully)."""
        parameters = {"command": 'echo "unclosed quote'}
        context = {}

        # Shell mode doesn't validate quotes at build time
        result = docker_cli_tool.build_command(parameters, context)
        assert result == ["/bin/sh", "-c", 'echo "unclosed quote']

    def test_build_command_empty_after_parsing(self, docker_cli_tool):
        """Test building command with empty string."""
        parameters = {"command": ""}
        context = {}

        with pytest.raises(ValueError, match="Command string cannot be empty"):
            docker_cli_tool.build_command(parameters, context)

    def test_parse_output_success(self, docker_cli_tool):
        """Test parsing successful command output."""
        stdout = "Hello, World!\nLine 2\n"
        stderr = ""
        return_code = 0

        result = docker_cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is True
        assert result.data["stdout"] == stdout
        assert result.data["stderr"] == stderr
        assert result.data["return_code"] == return_code
        assert result.data["success"] is True
        assert result.data["output"] == stdout  # Primary output for success
        assert result.data["success"] is True
        assert result.data["output"] == stdout  # Primary output for success

        # Check metadata
        assert result.metadata["command_type"] == "docker_cli"
        assert result.metadata["execution_environment"] == "docker_container"
        assert result.metadata["exit_code"] == 0
        assert result.metadata["has_stdout"] is True
        assert result.metadata["has_stderr"] is False
        assert result.metadata["output_length"] == len(stdout)

    def test_parse_output_failure_with_stderr(self, docker_cli_tool):
        """Test parsing failed command output with stderr."""
        stdout = ""
        stderr = "command not found\n"
        return_code = 127

        result = docker_cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is False
        assert "Command failed with exit code 127" in result.error
        assert "command not found" in result.error

        # Check metadata includes raw data
        assert result.metadata["exit_code"] == 127
        assert result.metadata["has_stderr"] is True
        assert result.metadata["raw_data"]["stdout"] == stdout
        assert result.metadata["raw_data"]["stderr"] == stderr

    def test_parse_output_failure_with_stdout_only(self, docker_cli_tool):
        """Test parsing failed command output with only stdout."""
        stdout = "Some error message to stdout\n"
        stderr = ""
        return_code = 1

        result = docker_cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is False
        assert "Command failed with exit code 1" in result.error
        assert "Some error message to stdout" in result.error

    def test_parse_output_failure_no_output(self, docker_cli_tool):
        """Test parsing failed command with no output."""
        stdout = ""
        stderr = ""
        return_code = 1

        result = docker_cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is False
        assert result.error == "Command failed with exit code 1"

    def test_parse_output_with_whitespace(self, docker_cli_tool):
        """Test parsing output with whitespace handling."""
        stdout = "  \n\t  "  # Only whitespace
        stderr = ""
        return_code = 0

        result = docker_cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is True
        assert result.metadata["has_stdout"] is False  # Stripped whitespace

    def test_parse_output_large_output(self, docker_cli_tool):
        """Test parsing output with length calculation."""
        stdout = "x" * 1000
        stderr = "y" * 500
        return_code = 0

        result = docker_cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is True
        assert result.metadata["output_length"] == 1500

    def test_parameter_validation_success(self, docker_cli_tool):
        """Test successful parameter validation."""
        parameters = {"command": "ls -la", "shell": False}

        result = docker_cli_tool.validate_parameters(parameters)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_parameter_validation_missing_required(self, docker_cli_tool):
        """Test parameter validation with missing required parameter."""
        parameters = {"shell": True}  # Missing required 'command'

        result = docker_cli_tool.validate_parameters(parameters)

        assert result.valid is False
        assert "Required parameter 'command' is missing" in result.errors

    def test_parameter_validation_wrong_type(self, docker_cli_tool):
        """Test parameter validation with wrong parameter type."""
        parameters = {"command": 123}  # command should be string

        result = docker_cli_tool.validate_parameters(parameters)

        assert result.valid is False
        assert "Parameter 'command' must be a string" in result.errors

    def test_parameter_validation_unknown_parameter(self, docker_cli_tool):
        """Test parameter validation with unknown parameter."""
        parameters = {"command": "ls", "unknown_param": "value"}

        result = docker_cli_tool.validate_parameters(parameters)

        assert result.valid is True  # Should be valid but with warning
        assert "Unknown parameter 'unknown_param' will be ignored" in result.warnings

    def test_get_security_info(self, docker_cli_tool):
        """Test getting security information including sandbox config."""
        security_info = docker_cli_tool.get_security_info()

        assert security_info["execution_environment"] == "docker_container"
        assert security_info["timeout"] == 30.0
        assert "docker_config" in security_info

        docker_config = security_info["docker_config"]
        assert docker_config["image"] == "saber/base-sandbox:latest"
        assert docker_config["network_mode"] == "none"
        assert docker_config["read_only_root"] is True
        assert docker_config["user"] == "tooluser:tooluser"


class TestCLIExecutorIntegration:
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
        """Create a mock SandboxManager that returns a Docker environment."""
        manager = MagicMock(spec=SandboxManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/base-sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser",
        }
        manager.get_session_environment.return_value = mock_docker_sandbox_environment
        manager.create_session_environment.return_value = mock_docker_sandbox_environment
        return manager

    @pytest.fixture
    def docker_cli_tool_with_env(self, mock_sandbox_manager_with_env):
        """Create a CLI executor with mocked environment."""
        config = {"timeout": 30.0}
        return CLIExecutor(
            sandbox_manager=mock_sandbox_manager_with_env,
            config=config,
            allowed_commands=["file", "strings", "echo", "cat"],
        )

    @pytest.mark.asyncio
    async def test_execute_docker_integration_success(self, docker_cli_tool_with_env, mock_sandbox_manager_with_env):
        """Test successful Docker command execution."""

        # Set up mock command result
        command_result = CommandResult(exit_code=0, stdout="Hello from Docker!\n", stderr="", execution_time=0.5)

        env = mock_sandbox_manager_with_env.get_session_environment.return_value
        env.execute_command.return_value = command_result

        parameters = {"command": "echo 'Hello from Docker!'"}
        context = {"session_id": "test_session_123"}

        result = await docker_cli_tool_with_env(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "Hello from Docker!\n"
        assert result.data["return_code"] == 0
        assert result.metadata["container_id"] == "container123"[:12]  # Truncated to 12 chars
        assert result.metadata["session_id"] == "test_session_123"
        assert result.metadata["execution_time"] == 0.5

        # Verify Docker environment was called correctly (shell mode)
        env.execute_command.assert_called_once_with(
            command=["/bin/sh", "-c", "echo 'Hello from Docker!'"], timeout=30  # Timeout from fixture config
        )

    @pytest.mark.asyncio
    async def test_execute_docker_integration_shell_mode(self, docker_cli_tool_with_env, mock_sandbox_manager_with_env):
        """Test Docker command execution in shell mode."""

        command_result = CommandResult(exit_code=0, stdout="hello world\n", stderr="", execution_time=1.2)

        env = mock_sandbox_manager_with_env.get_session_environment.return_value
        env.execute_command.return_value = command_result

        parameters = {"command": "echo hello world", "shell": True}
        context = {"session_id": "shell_test_session"}

        result = await docker_cli_tool_with_env(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "hello world\n"

        # Verify shell command was constructed correctly
        env.execute_command.assert_called_once_with(command=["/bin/sh", "-c", "echo hello world"], timeout=30)

    @pytest.mark.asyncio
    async def test_execute_docker_integration_failure(self, docker_cli_tool_with_env, mock_sandbox_manager_with_env):
        """Test Docker command execution failure."""

        command_result = CommandResult(
            exit_code=127, stdout="", stderr="command not found: nonexistent_command\n", execution_time=0.1
        )

        env = mock_sandbox_manager_with_env.get_session_environment.return_value
        env.execute_command.return_value = command_result

        parameters = {"command": "nonexistent_command", "shell": False}
        context = {"session_id": "failure_test_session"}

        result = await docker_cli_tool_with_env(parameters, context)

        assert result.success is False
        assert "Command failed with exit code 127" in result.error
        assert "command not found" in result.error
        assert result.metadata["exit_code"] == 127

    @pytest.mark.asyncio
    async def test_execute_missing_session_id(self, docker_cli_tool_with_env):
        """Test that execution fails without session_id in context."""
        parameters = {"command": "echo test", "shell": False}
        context = {}  # Missing session_id

        result = await docker_cli_tool_with_env(parameters, context)

        assert result.success is False
        assert "session_id required in context" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_existing_environment(self, docker_cli_tool_with_env, mock_sandbox_manager_with_env):
        """Test that command execution works when environment already exists for session."""

        # Mock an existing environment
        existing_env = MagicMock()
        mock_sandbox_manager_with_env.get_session_environment.return_value = existing_env

        command_result = CommandResult(exit_code=0, stdout="test\n", stderr="", execution_time=0.3)
        existing_env.execute_command.return_value = command_result

        # Mock the container interface
        container_mock = MagicMock()
        container_mock.id = "container123"
        existing_env.get_execution_container.return_value = container_mock

        parameters = {"command": "echo test", "shell": False}
        context = {"session_id": "existing_session"}

        result = await docker_cli_tool_with_env(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "test\n"

        # Verify that create_session_environment was NOT called since environment exists
        mock_sandbox_manager_with_env.create_session_environment.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_docker_exception_handling(self, docker_cli_tool_with_env, mock_sandbox_manager_with_env):
        """Test handling of Docker execution exceptions."""
        env = mock_sandbox_manager_with_env.get_session_environment.return_value
        env.execute_command.side_effect = Exception("Docker daemon not available")

        parameters = {"command": "echo test", "shell": False}
        context = {"session_id": "exception_test"}

        result = await docker_cli_tool_with_env(parameters, context)

        assert result.success is False
        assert "Docker command execution failed" in result.error
        assert "Docker daemon not available" in result.error
