"""
Tests for CLI tool executor.

This module tests the secure command line interface tool that accepts
arbitrary command strings and executes them with comprehensive security validation.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from saber.server.tools.base import Parameter, ParameterType, ToolResult, ValidationResult
from saber.server.tools.executors.cli import CLIExecutor
from saber.server.tools.utils.security_validator import SecurityValidator


class TestCLI:
    """Test cases for CLI tool executor."""

    @pytest.fixture
    def mock_security_validator(self):
        """Create a mock SecurityValidator."""
        validator = MagicMock(spec=SecurityValidator)
        validator.validate_base_command.return_value = None
        validator.validate_full_command.return_value = ValidationResult.success()
        validator.get_security_info.return_value = {"test": "security_info"}
        return validator

    @pytest.fixture
    def cli_tool(self, mock_security_validator):
        """Create a CLI tool instance for testing."""
        return CLIExecutor(timeout=30.0)

    def test_initialization(self, mock_security_validator):
        """Test CLI tool initialization."""
        cli = CLIExecutor(timeout=60.0)

        assert cli._command == "sh"
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

    def test_security_tool_metadata(self, cli_tool):
        """Test that security tool metadata is properly set."""
        metadata = cli_tool._security_tool_metadata

        assert metadata["domain"] == "general"
        assert metadata["name"] == "cli"
        assert metadata["description"] == "Execute validated shell commands in a secure environment"
        assert metadata["author"] == "SABER Team"
        assert metadata["security_level"] == "high"
        assert metadata["requires_validation"] is True

    def test_build_command_simple(self, cli_tool):
        """Test building command with simple string (no shell mode)."""
        parameters = {"command": "ls -la", "shell": False}
        context = {}

        result = cli_tool.build_command(parameters, context)

        # Should parse into individual arguments
        assert result == ["ls", "-la"]

    def test_build_command_shell_mode(self, cli_tool):
        """Test building command with shell mode enabled."""
        parameters = {"command": "ls -la | grep test", "shell": True}
        context = {}

        result = cli_tool.build_command(parameters, context)

        # Should use shell with -c flag
        assert result == ["/bin/sh", "-c", "ls -la | grep test"]

    def test_build_command_default_shell_mode(self, cli_tool):
        """Test building command with default shell mode (False)."""
        parameters = {"command": "echo hello world"}
        context = {}

        result = cli_tool.build_command(parameters, context)

        # Should parse into individual arguments (default shell=False)
        assert result == ["echo", "hello", "world"]

    def test_build_command_empty_string(self, cli_tool):
        """Test building command with empty string raises ValueError."""
        parameters = {"command": "", "shell": False}
        context = {}

        with pytest.raises(ValueError, match="Command string cannot be empty"):
            cli_tool.build_command(parameters, context)

    def test_build_command_whitespace_only(self, cli_tool):
        """Test building command with whitespace-only string raises ValueError."""
        parameters = {"command": "   \t\n  ", "shell": False}
        context = {}

        with pytest.raises(ValueError, match="Command string cannot be empty"):
            cli_tool.build_command(parameters, context)

    def test_build_command_quoted_arguments(self, cli_tool):
        """Test building command with quoted arguments."""
        parameters = {"command": 'echo "hello world" test', "shell": False}
        context = {}

        result = cli_tool.build_command(parameters, context)

        # Should properly parse quoted strings
        assert result == ["echo", "hello world", "test"]

    def test_build_command_complex_shell_command(self, cli_tool):
        """Test building command with complex shell constructs."""
        parameters = {
            "command": "find /tmp -name '*.txt' | head -10 > results.txt",
            "shell": True
        }
        context = {}

        result = cli_tool.build_command(parameters, context)

        assert result == ["/bin/sh", "-c", "find /tmp -name '*.txt' | head -10 > results.txt"]

    def test_build_command_invalid_quotes(self, cli_tool):
        """Test building command with invalid quote parsing."""
        parameters = {"command": 'echo "unclosed quote', "shell": False}
        context = {}

        with pytest.raises(ValueError, match="Failed to parse command string"):
            cli_tool.build_command(parameters, context)

    def test_build_command_empty_after_parsing(self, cli_tool):
        """Test building command that results in empty arguments after parsing."""
        # This is a tricky case - a command that shlex can parse but results in no arguments
        with patch('shlex.split', return_value=[]):
            parameters = {"command": "some_command", "shell": False}
            context = {}

            with pytest.raises(ValueError, match="Parsed command resulted in empty argument list"):
                cli_tool.build_command(parameters, context)

    def test_parse_output_success(self, cli_tool):
        """Test parsing successful command output."""
        stdout = "Hello, World!\nLine 2\n"
        stderr = ""
        return_code = 0

        result = cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is True
        assert result.data["stdout"] == stdout
        assert result.data["stderr"] == stderr
        assert result.data["return_code"] == return_code
        assert result.data["success"] is True
        assert result.data["output"] == stdout  # Primary output for success

        # Check metadata
        assert result.metadata["command_type"] == "cli"
        assert result.metadata["exit_code"] == 0
        assert result.metadata["has_stdout"] is True
        assert result.metadata["has_stderr"] is False
        assert result.metadata["output_length"] == len(stdout)

    def test_parse_output_failure_with_stderr(self, cli_tool):
        """Test parsing failed command output with stderr."""
        stdout = ""
        stderr = "command not found\n"
        return_code = 127

        result = cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is False
        assert "Command failed with exit code 127" in result.error
        assert "command not found" in result.error

        # Check metadata includes raw data
        assert result.metadata["exit_code"] == 127
        assert result.metadata["has_stderr"] is True
        assert result.metadata["raw_data"]["stdout"] == stdout
        assert result.metadata["raw_data"]["stderr"] == stderr

    def test_parse_output_failure_with_stdout_only(self, cli_tool):
        """Test parsing failed command output with only stdout."""
        stdout = "Some error message to stdout\n"
        stderr = ""
        return_code = 1

        result = cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is False
        assert "Command failed with exit code 1" in result.error
        assert "Some error message to stdout" in result.error

    def test_parse_output_failure_no_output(self, cli_tool):
        """Test parsing failed command with no output."""
        stdout = ""
        stderr = ""
        return_code = 1

        result = cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is False
        assert result.error == "Command failed with exit code 1"

    def test_parse_output_with_whitespace(self, cli_tool):
        """Test parsing output with whitespace handling."""
        stdout = "  \n\t  "  # Only whitespace
        stderr = ""
        return_code = 0

        result = cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is True
        assert result.metadata["has_stdout"] is False  # Stripped whitespace

    def test_parse_output_large_output(self, cli_tool):
        """Test parsing output with length calculation."""
        stdout = "x" * 1000
        stderr = "y" * 500
        return_code = 0

        result = cli_tool.parse_output(stdout, stderr, return_code)

        assert result.success is True
        assert result.metadata["output_length"] == 1500

    def test_parameter_validation_success(self, cli_tool):
        """Test successful parameter validation."""
        parameters = {"command": "ls -la", "shell": False}

        result = cli_tool.validate_parameters(parameters)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_parameter_validation_missing_required(self, cli_tool):
        """Test parameter validation with missing required parameter."""
        parameters = {"shell": True}  # Missing required 'command'

        result = cli_tool.validate_parameters(parameters)

        assert result.valid is False
        assert "Required parameter 'command' is missing" in result.errors

    def test_parameter_validation_wrong_type(self, cli_tool):
        """Test parameter validation with wrong parameter type."""
        parameters = {"command": "ls", "shell": "true"}  # shell should be boolean

        result = cli_tool.validate_parameters(parameters)

        assert result.valid is False
        assert "Parameter 'shell' must be a boolean" in result.errors

    def test_parameter_validation_unknown_parameter(self, cli_tool):
        """Test parameter validation with unknown parameter."""
        parameters = {"command": "ls", "unknown_param": "value"}

        result = cli_tool.validate_parameters(parameters)

        assert result.valid is True  # Should be valid but with warning
        assert "Unknown parameter 'unknown_param' will be ignored" in result.warnings

    @pytest.mark.asyncio
    async def test_execute_integration(self, cli_tool, mock_security_validator):
        """Integration test for complete execution flow."""
        parameters = {"command": "echo test", "shell": False}
        context = {"session_id": "test123"}

        # Mock the subprocess execution
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"test\n", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await cli_tool.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "test\n"
        assert result.metadata["command"] == "echo"

    @pytest.mark.asyncio
    async def test_execute_shell_mode_integration(self, cli_tool, mock_security_validator):
        """Integration test for shell mode execution."""
        parameters = {"command": "echo hello | cat", "shell": True}
        context = {}

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"hello\n", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await cli_tool.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "hello\n"
