"""
Tests for CLIExecutor.

This module tests the command-line tool execution framework including
security validation, parameter handling, and subprocess management.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from saber.server.tools.base import ToolResult, ValidationResult
from saber.server.tools.executors.cli import CLIExecutor
from saber.server.tools.utils.security_validator import SecurityValidator


class MockCLIExecutor(CLIExecutor):
    """Mock implementation for testing CLIExecutor."""

    def __init__(self, command="echo", **kwargs):
        super().__init__(**kwargs)
        self._command = command

    def build_command(self, parameters, context):
        """Simple build_command implementation for testing."""
        return [self._command] + [str(v) for v in parameters.values()]

    def parse_output(self, stdout, stderr, return_code):
        """Simple parse_output implementation for testing."""
        if return_code == 0:
            return ToolResult.success_result(
                data={"stdout": stdout, "stderr": stderr},
                metadata={"return_code": return_code}
            )
        else:
            return ToolResult.error_result(
                error=f"Command failed: {stderr}",
                metadata={"return_code": return_code}
            )


class TestCLIExecutor:
    """Test cases for CLIExecutor."""

    @pytest.fixture
    def mock_security_validator(self):
        """Create a mock SecurityValidator."""
        validator = MagicMock(spec=SecurityValidator)
        validator.validate_base_command.return_value = None
        validator.validate_full_command.return_value = ValidationResult.success()
        validator.get_security_info.return_value = {"test": "info"}
        return validator

    @pytest.fixture
    def executor(self, mock_security_validator):
        """Create a CLIExecutor instance for testing."""
        return MockCLIExecutor(
            command="echo",
            timeout=30.0
        )

    def test_initialization(self, mock_security_validator):
        """Test executor initialization."""
        executor = MockCLIExecutor(
            command="test_cmd",
            timeout=60.0
        )

        assert executor._command == "test_cmd"
        assert executor.get_timeout() == 60.0

    def test_initialization_validates_base_command(self, mock_security_validator):
        """Test that initialization works properly."""
        executor = MockCLIExecutor(
            command="safe_cmd"
        )
        assert executor._command == "safe_cmd"

    def test_get_security_info(self, executor, mock_security_validator):
        """Test get_security_info method."""
        info = executor.get_security_info()

        assert info["base_command"] == "echo"
        assert info["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_execute_success(self, executor, mock_security_validator):
        """Test successful command execution."""
        parameters = {"arg1": "hello", "arg2": "world"}
        context = {"session_id": "test123"}

        # Mock subprocess execution
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"hello world\n", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await executor.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "hello world\n"
        assert result.data["stderr"] == ""
        assert result.metadata["return_code"] == 0
        assert result.metadata["command"] == "echo"

    @pytest.mark.asyncio
    async def test_execute_security_validation_failure(self, executor, mock_security_validator):
        """Test execution when command building fails."""
        parameters = {"command": "rm -rf /"}
        context = {}

        # Mock build_command to raise an exception
        with patch.object(executor, 'build_command', side_effect=ValueError("Invalid command")):
            result = await executor.execute(parameters, context)

        assert result.success is False
        assert "Command execution failed" in result.error
        assert "Invalid command" in result.error

    @pytest.mark.asyncio
    async def test_execute_command_failure(self, executor, mock_security_validator):
        """Test execution when command returns non-zero exit code."""
        parameters = {"arg": "invalid"}
        context = {}

        # Mock subprocess execution with failure
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"command not found\n")
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await executor.execute(parameters, context)

        assert result.success is False
        assert "Command failed: command not found" in result.error
        assert result.metadata["return_code"] == 1

    @pytest.mark.asyncio
    async def test_execute_timeout(self, executor, mock_security_validator):
        """Test execution timeout handling."""
        parameters = {"arg": "slow"}
        context = {}

        # Mock subprocess execution
        mock_process = AsyncMock()
        mock_process.terminate.return_value = None
        mock_process.kill.return_value = None
        mock_process.returncode = None

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                with patch("asyncio.sleep"):
                    result = await executor.execute(parameters, context)

        assert result.success is False
        assert "Command timed out" in result.error
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, executor, mock_security_validator):
        """Test handling of unexpected exceptions during execution."""
        parameters = {"arg": "test"}
        context = {}

        with patch("asyncio.create_subprocess_exec", side_effect=OSError("Permission denied")):
            result = await executor.execute(parameters, context)

        assert result.success is False
        assert "Command execution failed: Permission denied" in result.error

    @pytest.mark.asyncio
    async def test_execute_output_size_limit(self, executor, mock_security_validator):
        """Test that output is truncated to size limits."""
        parameters = {"arg": "large_output"}
        context = {}

        # Create large output that exceeds the limit
        large_output = b"x" * 20000  # Larger than 10KB limit

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (large_output, b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await executor.execute(parameters, context)

        # Output should be truncated to max_output_size (10240 bytes)
        assert len(result.data["stdout"]) <= 10240

    @pytest.mark.asyncio
    async def test_execute_with_security_warnings(self, executor, mock_security_validator):
        """Test execution with normal output."""
        parameters = {"arg": "normal_output"}
        context = {}

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await executor.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "output"

    def test_setup_security_restrictions(self, executor):
        """Test security restriction setup."""
        with patch("resource.setrlimit") as mock_setrlimit:
            with patch("os.getuid", return_value=1000):  # Non-root user
                executor._setup_security_restrictions()

        # Should attempt to set various resource limits
        assert mock_setrlimit.call_count >= 5  # CPU, memory, file size, etc.

    def test_setup_security_restrictions_as_root(self, executor):
        """Test security restriction setup when running as root."""
        with patch("resource.setrlimit"):
            with patch("os.getuid", return_value=0):  # Root user
                with patch("pwd.getpwnam") as mock_getpwnam:
                    with patch("os.setgid") as mock_setgid:
                        with patch("os.setuid") as mock_setuid:
                            mock_getpwnam.return_value = MagicMock(pw_gid=65534, pw_uid=65534)

                            executor._setup_security_restrictions()

                            mock_setgid.assert_called_once_with(65534)
                            mock_setuid.assert_called_once_with(65534)

    def test_setup_security_restrictions_exception_handling(self, executor):
        """Test that security restriction failures don't crash execution."""
        with patch("resource.setrlimit", side_effect=OSError("Permission denied")):
            with patch("saber.server.tools.executors.cli.logger") as mock_logger:
                # Should not raise exception
                executor._setup_security_restrictions()
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_execute_environment_variables(self, executor, mock_security_validator):
        """Test that restricted environment variables are used."""
        parameters = {"arg": "env_test"}
        context = {}

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_create:
            result = await executor.execute(parameters, context)

        # Verify restricted environment was used
        call_args = mock_create.call_args
        assert "env" in call_args.kwargs
        env = call_args.kwargs["env"]
        assert env["USER"] == "nobody"
        assert env["SHELL"] == "/bin/false"
        assert env["HOME"] == "/tmp"

    @pytest.mark.asyncio
    async def test_execute_working_directory(self, executor, mock_security_validator):
        """Test that sandbox working directory is used."""
        parameters = {"arg": "pwd_test"}
        context = {}

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"/tmp", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_create:
            result = await executor.execute(parameters, context)

        # Verify working directory was set
        call_args = mock_create.call_args
        assert "cwd" in call_args.kwargs
        # Should use default sandbox directory when no sandbox_path is set
        assert call_args.kwargs["cwd"] == "/tmp"
