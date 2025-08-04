"""
Tests for Docker CLI tool executor.

This module tests the secure Docker-based command line interface tool that accepts
arbitrary command strings and executes them with comprehensive security validation
in isolated Docker containers.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from saber.server.execution.base import ParameterType, ValidationResult
from saber.server.execution.executors.cli import DockerCLIExecutor
from saber.server.execution.utils.security_validator import SecurityValidator
from saber.server.execution.sandbox.sandbox_manager import SandboxManager
from saber.server.execution.exceptions import SandboxExecutionError


class TestDockerCLI:
    """Test cases for Docker CLI tool executor."""

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
            "user": "tooluser:tooluser"
        }
        return manager

    @pytest.fixture
    def docker_cli_tool(self, mock_sandbox_manager):
        """Create a Docker CLI tool instance for testing."""
        return DockerCLIExecutor(sandbox_manager=mock_sandbox_manager, timeout=30.0)

    def test_initialization(self, mock_sandbox_manager):
        """Test Docker CLI tool initialization."""
        cli = DockerCLIExecutor(sandbox_manager=mock_sandbox_manager, timeout=60.0)

        assert cli.get_timeout() == 60.0

        # Check that parameters were added
        params = cli.get_parameters()
        assert "command" in params
        assert "shell" in params

        # Verify parameter definitions
        command_param = params["command"]
        assert command_param.name == "command"
        assert command_param.type == ParameterType.STRING
        assert command_param.required is True

        shell_param = params["shell"]
        assert shell_param.name == "shell"
        assert shell_param.type == ParameterType.BOOLEAN
        assert shell_param.required is False
        assert shell_param.default is False

    def test_initialization_without_sandbox_manager(self):
        """Test that initialization fails without sandbox manager."""
        with pytest.raises(SandboxExecutionError, match="sandbox_manager is required"):
            DockerCLIExecutor(sandbox_manager=None, timeout=60.0)

    def test_security_command_metadata(self, docker_cli_tool):
        """Test that security command metadata is properly set."""
        metadata = docker_cli_tool._security_command_metadata

        assert metadata["domain"] == "general"
        assert metadata["name"] == "docker_cli"
        assert metadata["description"] == "Execute validated shell commands in Docker containers"
        assert metadata["author"] == "SABER Team"
        assert metadata["security_level"] == "high"
        assert metadata["requires_validation"] is True

    def test_build_command_simple(self, docker_cli_tool):
        """Test building command with simple string (no shell mode)."""
        parameters = {"command": "ls -la", "shell": False}
        context = {}

        result = docker_cli_tool.build_command(parameters, context)

        # Should parse into individual arguments
        assert result == ["ls", "-la"]

    def test_build_command_shell_mode(self, docker_cli_tool):
        """Test building command with shell mode enabled."""
        parameters = {"command": "ls -la | grep test", "shell": True}
        context = {}

        result = docker_cli_tool.build_command(parameters, context)

        # Should use shell with -c flag
        assert result == ["/bin/sh", "-c", "ls -la | grep test"]

    def test_build_command_default_shell_mode(self, docker_cli_tool):
        """Test building command with default shell mode (False)."""
        parameters = {"command": "echo hello world"}
        context = {}

        result = docker_cli_tool.build_command(parameters, context)

        # Should parse into individual arguments (default shell=False)
        assert result == ["echo", "hello", "world"]

    def test_build_command_empty_string(self, docker_cli_tool):
        """Test building command with empty string raises ValueError."""
        parameters = {"command": "", "shell": False}
        context = {}

        with pytest.raises(ValueError, match="Command string cannot be empty"):
            docker_cli_tool.build_command(parameters, context)

    def test_build_command_whitespace_only(self, docker_cli_tool):
        """Test building command with whitespace-only string raises ValueError."""
        parameters = {"command": "   \t\n  ", "shell": False}
        context = {}

        with pytest.raises(ValueError, match="Command string cannot be empty"):
            docker_cli_tool.build_command(parameters, context)

    def test_build_command_quoted_arguments(self, docker_cli_tool):
        """Test building command with quoted arguments."""
        parameters = {"command": 'echo "hello world" test', "shell": False}
        context = {}

        result = docker_cli_tool.build_command(parameters, context)

        # Should properly parse quoted strings
        assert result == ["echo", "hello world", "test"]

    def test_build_command_complex_shell_command(self, docker_cli_tool):
        """Test building command with complex shell constructs."""
        parameters = {
            "command": "find /tmp -name '*.txt' | head -10 > results.txt",
            "shell": True
        }
        context = {}

        result = docker_cli_tool.build_command(parameters, context)

        assert result == ["/bin/sh", "-c", "find /tmp -name '*.txt' | head -10 > results.txt"]

    def test_build_command_invalid_quotes(self, docker_cli_tool):
        """Test building command with invalid quote parsing."""
        parameters = {"command": 'echo "unclosed quote', "shell": False}
        context = {}

        with pytest.raises(ValueError, match="Failed to parse command string"):
            docker_cli_tool.build_command(parameters, context)

    def test_build_command_empty_after_parsing(self, docker_cli_tool):
        """Test building command that results in empty arguments after parsing."""
        # This is a tricky case - a command that shlex can parse but results in no arguments
        with patch('shlex.split', return_value=[]):
            parameters = {"command": "some_command", "shell": False}
            context = {}

            with pytest.raises(ValueError, match="Parsed command resulted in empty argument list"):
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
        parameters = {"command": "ls", "shell": "true"}  # shell should be boolean

        result = docker_cli_tool.validate_parameters(parameters)

        assert result.valid is False
        assert "Parameter 'shell' must be a boolean" in result.errors

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
        assert "sandbox_config" in security_info

        sandbox_config = security_info["sandbox_config"]
        assert sandbox_config["image"] == "saber/base-sandbox:latest"
        assert sandbox_config["network_mode"] == "none"
        assert sandbox_config["read_only_root"] is True
        assert sandbox_config["user"] == "tooluser:tooluser"


class TestDockerCLIExecutorIntegration:
    """Integration tests for Docker CLI executor with mocked Docker environment."""

    @pytest.fixture
    def mock_docker_environment(self):
        """Create a mock Docker execution environment."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        env = MagicMock()
        env.get_container_id.return_value = "container123456789"
        env.execute_command = AsyncMock()
        return env

    @pytest.fixture
    def mock_sandbox_manager_with_env(self, mock_docker_environment):
        """Create a mock SandboxManager that returns a Docker environment."""
        manager = MagicMock(spec=SandboxManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/base-sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser"
        }
        manager.get_session_environment.return_value = mock_docker_environment
        manager.create_session_environment.return_value = mock_docker_environment
        return manager

    @pytest.fixture
    def docker_cli_tool_with_env(self, mock_sandbox_manager_with_env):
        """Create a Docker CLI tool with mocked environment."""
        return DockerCLIExecutor(sandbox_manager=mock_sandbox_manager_with_env, timeout=30.0)

    @pytest.mark.asyncio
    async def test_execute_docker_integration_success(self, docker_cli_tool_with_env, mock_sandbox_manager_with_env):
        """Test successful Docker command execution."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        # Set up mock command result
        command_result = CommandResult(
            exit_code=0,
            stdout="Hello from Docker!\n",
            stderr="",
            execution_time=0.5
        )

        env = mock_sandbox_manager_with_env.get_session_environment.return_value
        env.execute_command.return_value = command_result

        parameters = {"command": "echo 'Hello from Docker!'", "shell": False}
        context = {"session_id": "test_session_123"}

        result = await docker_cli_tool_with_env.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "Hello from Docker!\n"
        assert result.data["return_code"] == 0
        assert result.metadata["container_id"] == "container123"[:12]  # Truncated to 12 chars
        assert result.metadata["session_id"] == "test_session_123"
        assert result.metadata["execution_time"] == 0.5

        # Verify Docker environment was called correctly
        env.execute_command.assert_called_once_with(
            command=["echo", "Hello from Docker!"],
            working_dir="/workspace"
        )

    @pytest.mark.asyncio
    async def test_execute_docker_integration_shell_mode(self, docker_cli_tool_with_env, mock_sandbox_manager_with_env):
        """Test Docker command execution in shell mode."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        command_result = CommandResult(
            exit_code=0,
            stdout="file1.txt\nfile2.txt\n",
            stderr="",
            execution_time=1.2
        )

        env = mock_sandbox_manager_with_env.get_session_environment.return_value
        env.execute_command.return_value = command_result

        parameters = {"command": "ls *.txt | sort", "shell": True}
        context = {"session_id": "shell_test_session"}

        result = await docker_cli_tool_with_env.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "file1.txt\nfile2.txt\n"

        # Verify shell command was constructed correctly
        env.execute_command.assert_called_once_with(
            command=["/bin/sh", "-c", "ls *.txt | sort"],
            working_dir="/workspace"
        )

    @pytest.mark.asyncio
    async def test_execute_docker_integration_failure(self, docker_cli_tool_with_env, mock_sandbox_manager_with_env):
        """Test Docker command execution failure."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        command_result = CommandResult(
            exit_code=127,
            stdout="",
            stderr="command not found: nonexistent_command\n",
            execution_time=0.1
        )

        env = mock_sandbox_manager_with_env.get_session_environment.return_value
        env.execute_command.return_value = command_result

        parameters = {"command": "nonexistent_command", "shell": False}
        context = {"session_id": "failure_test_session"}

        result = await docker_cli_tool_with_env.execute(parameters, context)

        assert result.success is False
        assert "Command failed with exit code 127" in result.error
        assert "command not found" in result.error
        assert result.metadata["exit_code"] == 127

    @pytest.mark.asyncio
    async def test_execute_missing_session_id(self, docker_cli_tool_with_env):
        """Test that execution fails without session_id in context."""
        parameters = {"command": "echo test", "shell": False}
        context = {}  # Missing session_id

        result = await docker_cli_tool_with_env.execute(parameters, context)

        assert result.success is False
        assert "session_id required in context" in result.error

    @pytest.mark.asyncio
    async def test_execute_create_new_environment(self, docker_cli_tool_with_env, mock_sandbox_manager_with_env):
        """Test that new environment is created when none exists for session."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        # First call returns None (no existing environment), second call returns new environment
        mock_sandbox_manager_with_env.get_session_environment.return_value = None

        command_result = CommandResult(exit_code=0, stdout="test\n", stderr="", execution_time=0.3)
        new_env = mock_sandbox_manager_with_env.create_session_environment.return_value
        new_env.execute_command.return_value = command_result
        new_env.get_container_id.return_value = "newcontainer123"

        parameters = {"command": "echo test", "shell": False}
        context = {"session_id": "new_session"}

        result = await docker_cli_tool_with_env.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "test\n"

        # Verify that create_session_environment was called
        mock_sandbox_manager_with_env.create_session_environment.assert_called_once_with("new_session")

    @pytest.mark.asyncio
    async def test_execute_docker_exception_handling(self, docker_cli_tool_with_env, mock_sandbox_manager_with_env):
        """Test handling of Docker execution exceptions."""
        env = mock_sandbox_manager_with_env.get_session_environment.return_value
        env.execute_command.side_effect = Exception("Docker daemon not available")

        parameters = {"command": "echo test", "shell": False}
        context = {"session_id": "exception_test"}

        result = await docker_cli_tool_with_env.execute(parameters, context)

        assert result.success is False
        assert "Docker command execution failed" in result.error
        assert "Docker daemon not available" in result.error
