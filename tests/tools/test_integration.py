"""
Integration tests for the tool execution framework.

Tests integration between CLIExecutor,
SecurityValidator, CLI executor, and ExecutionManager working together.
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path

from saber.server.tools.execution_manager import ExecutionManager
from saber.server.tools.base import ToolResult, ValidationResult
from saber.server.tools.utils.security_validator import SecurityValidator
from saber.server.tools.executors.cli import CLIExecutor


class TestToolsIntegration:
    """Integration test cases for the complete tools framework."""

    @pytest.fixture
    def test_config(self):
        """Test configuration for integration tests."""
        return {
            "execution": {
                "timeout": 30.0,
                "max_concurrent": 3
            },
            "security": {
                "allowed_commands": ["echo", "cat", "ls"],
                "max_command_length": 1000
            },
            "cli": {
                "default_shell_mode": False
            }
        }

    @pytest.fixture
    def registry(self, test_config):
        """Create ExecutionManager for integration testing."""
        return ExecutionManager(config=test_config)

    @pytest.mark.asyncio
    async def test_end_to_end_safe_command_execution(self, registry):
        """Test complete flow for safe command execution."""
        parameters = {"command": "echo hello world"}
        context = {"session_id": "integration_test_001"}

        # Mock the subprocess to avoid actual execution
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"hello world\n", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await registry.execute_command(parameters, context)

        # Verify complete success flow
        assert result.success is True
        assert result.data["stdout"] == "hello world\n"
        assert result.data["return_code"] == 0
        assert result.metadata["command"] == "echo"

    @pytest.mark.asyncio
    async def test_end_to_end_blocked_command_execution(self, registry):
        """Test complete flow for blocked command execution."""
        parameters = {"command": "sudo rm -rf /"}
        context = {"session_id": "integration_test_002"}

        result = await registry.execute_command(parameters, context)

        # Should be blocked by security validation
        assert result.success is False
        assert "Command security validation failed" in result.error

    @pytest.mark.asyncio
    async def test_end_to_end_whitelisted_command_execution(self, registry):
        """Test execution of command that's blocked but whitelisted."""
        # Note: In the real implementation, we'd need to test with a command
        # that's actually in the blocked list but also in our whitelist
        parameters = {"command": "echo test"}  # Safe command
        context = {"session_id": "integration_test_003"}

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"test\n", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await registry.execute_command(parameters, context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_concurrent_command_execution(self, registry):
        """Test concurrent execution with semaphore control."""
        parameters_list = [
            {"command": f"echo test{i}"} for i in range(5)
        ]

        # Mock subprocess for all commands
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"test\n", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Execute all commands concurrently
            tasks = [
                asyncio.create_task(registry.execute_command(params))
                for params in parameters_list
            ]

            results = await asyncio.gather(*tasks)

        # All should succeed despite concurrency limits
        assert all(result.success for result in results)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_shell_mode_integration(self, registry):
        """Test shell mode with complex commands."""
        # Use a command that would benefit from shell mode but isn't dangerous
        parameters = {"command": "echo 'hello world'", "shell": True}
        context = {"session_id": "integration_test_004"}

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"hello world\n", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            result = await registry.execute_command(parameters, context)

        assert result.success is True

        # Verify shell mode was used
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "/bin/sh"
        assert call_args[1] == "-c"
        assert call_args[2] == "echo 'hello world'"

    def test_security_validator_configuration_integration(self, registry):
        """Test that security validator is properly configured."""
        # Get security info to verify configuration
        security_info = registry.get_security_info()

        # Should include allowed commands from config
        allowed_commands = security_info.get("allowed_commands")
        assert allowed_commands is not None
        assert "echo" in allowed_commands
        assert "cat" in allowed_commands
        assert "ls" in allowed_commands

    def test_mcp_integration(self, registry):
        """Test MCP tools format integration."""
        mcp_tools = registry.to_mcp_tools()

        assert len(mcp_tools) == 1
        cli_tool = mcp_tools[0]

        # Verify MCP format compliance
        assert "name" in cli_tool
        assert "description" in cli_tool
        assert "inputSchema" in cli_tool

        schema = cli_tool["inputSchema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "command" in schema["properties"]
        assert "command" in schema["required"]

    @pytest.mark.asyncio
    async def test_parameter_validation_integration(self, registry):
        """Test parameter validation integration."""
        # Test missing required parameter
        result = await registry.execute_command({"shell": True})  # Missing command
        assert result.success is False
        assert "Parameter validation failed" in result.error

        # Test invalid parameter type
        result = await registry.execute_command({"command": "echo test", "shell": "invalid"})
        assert result.success is False
        assert "must be a boolean" in result.error

    @pytest.mark.asyncio
    async def test_timeout_integration(self, registry):
        """Test timeout handling integration."""
        parameters = {"command": "sleep 100"}  # Command that would timeout

        mock_process = AsyncMock()
        mock_process.terminate.return_value = None
        mock_process.kill.return_value = None
        mock_process.returncode = None

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                with patch("asyncio.sleep"):
                    result = await registry.execute_command(parameters)

        assert result.success is False
        assert "Command timed out" in result.error

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, registry):
        """Test error handling throughout the system."""
        parameters = {"command": "nonexistent_command_xyz"}

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"command not found")
        mock_process.returncode = 127

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await registry.execute_command(parameters)

        assert result.success is False
        assert "Command failed with exit code 127" in result.error

    def test_configuration_loading_integration(self, registry, tmp_path):
        """Test configuration loading and component updates."""
        # Create test config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
execution:
  timeout: 90.0
  max_concurrent: 8
security:
  allowed_commands:
    - "new_command"
    - "another_command"
  max_command_length: 2000
"""
        config_file.write_text(config_content)

        # Load new configuration
        registry.load_configuration(str(config_file))

        # Verify configuration was updated
        config = registry.get_configuration()
        assert config.get_execution_timeout() == 90.0
        assert config.get_max_concurrent() == 8
        assert "new_command" in config.get_allowed_commands()

    @pytest.mark.asyncio
    async def test_security_patterns_integration(self, registry):
        """Test integration of security pattern detection."""
        dangerous_commands = [
            "echo hello; rm -rf /",
            "cat file | sh",
            "echo $(whoami)",
            "ls > /etc/passwd"
        ]

        for cmd in dangerous_commands:
            parameters = {"command": cmd}
            result = await registry.execute_command(parameters)

            assert result.success is False, f"Dangerous command should be blocked: {cmd}"
            assert "Command security validation failed" in result.error

    def test_component_initialization_integration(self, test_config):
        """Test that all components are properly initialized together."""
        registry = ExecutionManager(config=test_config)

        # Verify all components exist and are correct types
        assert hasattr(registry, '_configuration')
        assert hasattr(registry, '_security_validator')
        assert hasattr(registry, '_cli_tool')
        assert hasattr(registry, '_semaphore')

        # Verify types
        assert isinstance(registry._security_validator, SecurityValidator)
        assert isinstance(registry._cli_tool, CLIExecutor)
        assert isinstance(registry._semaphore, asyncio.Semaphore)

        # Verify configuration propagation
        assert registry._cli_tool.get_timeout() == test_config["execution"]["timeout"]

    @pytest.mark.asyncio
    async def test_realistic_malware_analysis_scenario(self, registry):
        """Test realistic malware analysis commands."""
        # Simulate commands that might be used in malware analysis
        analysis_commands = [
            {"command": "echo 'Analyzing file: sample.exe'"},
            {"command": "echo 'File type: PE32 executable'"},  # Simulating file command
            {"command": "echo 'Strings found: 50'"},           # Simulating strings command
        ]

        mock_process = AsyncMock()
        mock_process.returncode = 0

        results = []

        for i, params in enumerate(analysis_commands):
            expected_output = params["command"].split("'")[1].encode() + b"\n"
            mock_process.communicate.return_value = (expected_output, b"")

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await registry.execute_command(params)
                results.append(result)

        # All analysis commands should succeed
        assert all(result.success for result in results)

        # Verify expected outputs
        assert "Analyzing file" in results[0].data["stdout"]
        assert "File type" in results[1].data["stdout"]
        assert "Strings found" in results[2].data["stdout"]
