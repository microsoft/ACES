"""
Integration tests for the Docker-based tool execution framework.

Tests integration between DockerDockerCLIExecutor, SandboxManager,
SecurityValidator, and ExecutionManager working together in Docker containers.
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from saber.server.execution.execution_manager import ExecutionManager, ExecutionConfiguration
from saber.server.execution.base import CommandResult, ValidationResult
from saber.server.execution.utils.security_validator import SecurityValidator
from saber.server.execution.executors.cli import DockerCLIExecutor
from saber.server.execution.sandbox.sandbox_manager import SandboxManager


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
            },
            "sandbox": {
                "enabled": True,
                "image": "saber/base-sandbox:latest",
                "network_mode": "none",
                "read_only_root": True,
                "user": "tooluser:tooluser"
            }
        }

    @pytest.fixture
    def registry(self, test_config):
        """Create ExecutionManager for integration testing."""
        with patch("saber.server.execution.sandbox.sandbox_manager.SandboxManager"):
            return ExecutionManager(config=test_config)

    @pytest.mark.asyncio
    async def test_end_to_end_safe_command_execution(self, registry):
        """Test complete flow for safe command execution in Docker."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        parameters = {"command": "echo hello world"}
        context = {"session_id": "integration_test_001"}

        # Mock Docker environment execution
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock()
        mock_env.execute_command.return_value = CommandResult(
            exit_code=0,
            stdout="hello world\n",
            stderr="",
            execution_time=0.1
        )
        mock_env.get_container_id.return_value = "integration_container_123"

        # Mock sandbox manager to return our environment
        with patch.object(registry._sandbox_manager, 'get_session_environment', return_value=mock_env):
            result = await registry.step(parameters, context)

        # Verify complete success flow
        assert result.success is True
        assert result.data["stdout"] == "hello world\n"
        assert result.data["return_code"] == 0
        assert result.metadata["execution_environment"] == "docker_container"

    @pytest.mark.asyncio
    async def test_end_to_end_blocked_command_execution(self, registry):
        """Test complete flow for blocked command execution."""
        parameters = {"command": "sudo rm -rf /"}
        context = {"session_id": "integration_test_002"}

        result = await registry.step(parameters, context)

        # Should be blocked by security validation
        assert result.success is False
        assert "Command security validation failed" in result.error

    @pytest.mark.asyncio
    async def test_end_to_end_whitelisted_command_execution(self, registry):
        """Test execution of allowed command in Docker container."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        parameters = {"command": "echo test"}  # Safe command from allowed list
        context = {"session_id": "integration_test_003"}

        # Mock Docker environment execution
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock()
        mock_env.execute_command.return_value = CommandResult(
            exit_code=0,
            stdout="test\n",
            stderr="",
            execution_time=0.05
        )
        mock_env.get_container_id.return_value = "whitelist_container_456"

        with patch.object(registry._sandbox_manager, 'get_session_environment', return_value=mock_env):
            result = await registry.step(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "test\n"

    @pytest.mark.asyncio
    async def test_concurrent_command_execution(self, registry):
        """Test concurrent execution with semaphore control in Docker."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        parameters_list = [
            {"command": f"echo test{i}"}
            for i in range(5)
        ]

        contexts_list = [
            {"session_id": f"concurrent_test_{i}"}
            for i in range(5)
        ]

        # Mock Docker environment for all commands
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock()
        mock_env.execute_command.return_value = CommandResult(
            exit_code=0,
            stdout="test\n",
            stderr="",
            execution_time=0.02
        )
        mock_env.get_container_id.return_value = "concurrent_container"

        with patch.object(registry._sandbox_manager, 'get_session_environment', return_value=mock_env):
            # Execute all commands concurrently
            tasks = [
                asyncio.create_task(registry.step(params, context))
                for params, context in zip(parameters_list, contexts_list)
            ]

            results = await asyncio.gather(*tasks)

        # All should succeed despite concurrency limits
        assert all(result.success for result in results)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_shell_mode_integration(self, registry):
        """Test shell mode with complex commands in Docker."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        # Use a command that would benefit from shell mode but isn't dangerous
        parameters = {"command": "echo 'hello world'", "shell": True}
        context = {"session_id": "integration_test_004"}

        # Mock Docker environment execution
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock()
        mock_env.execute_command.return_value = CommandResult(
            exit_code=0,
            stdout="hello world\n",
            stderr="",
            execution_time=0.03
        )
        mock_env.get_container_id.return_value = "shell_test_container"

        with patch.object(registry._sandbox_manager, 'get_session_environment', return_value=mock_env) as mock_get_env:
            result = await registry.step(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "hello world\n"

        # Verify Docker step was called with shell format
        mock_env.execute_command.assert_called_once_with(
            command=["/bin/sh", "-c", "echo 'hello world'"],
            working_dir="/workspace"
        )

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
        assert cli_tool["name"] == "docker_cli"
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
        result = await registry.step({"shell": True})  # Missing command
        assert result.success is False
        assert "Parameter validation failed" in result.error

        # Test invalid parameter type
        result = await registry.step({"command": "echo test", "shell": "invalid"})
        assert result.success is False
        assert "must be a boolean" in result.error

    @pytest.mark.asyncio
    async def test_session_isolation_integration(self, registry):
        """Test that different sessions are properly isolated."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        parameters1 = {"command": "echo session1"}
        parameters2 = {"command": "echo session2"}
        context1 = {"session_id": "session_isolation_1"}
        context2 = {"session_id": "session_isolation_2"}

        # Mock different environments for different sessions
        mock_env1 = MagicMock()
        mock_env1.execute_command = AsyncMock()
        mock_env1.execute_command.return_value = CommandResult(
            exit_code=0, stdout="session1\n", stderr="", execution_time=0.1
        )
        mock_env1.get_container_id.return_value = "session1_container"

        mock_env2 = MagicMock()
        mock_env2.execute_command = AsyncMock()
        mock_env2.execute_command.return_value = CommandResult(
            exit_code=0, stdout="session2\n", stderr="", execution_time=0.1
        )
        mock_env2.get_container_id.return_value = "session2_container"

        # Mock sandbox manager to return different environments per session
        def get_session_env(session_id):
            if session_id == "session_isolation_1":
                return mock_env1
            elif session_id == "session_isolation_2":
                return mock_env2
            return None

        with patch.object(registry._sandbox_manager, 'get_session_environment', side_effect=get_session_env):
            result1 = await registry.step(parameters1, context1)
            result2 = await registry.step(parameters2, context2)

        # Verify isolation worked
        assert result1.success is True
        assert result1.data["stdout"] == "session1\n"
        assert result1.metadata["container_id"] == "session1_con"  # Truncated to 12 chars

        assert result2.success is True
        assert result2.data["stdout"] == "session2\n"
        assert result2.metadata["container_id"] == "session2_con"  # Truncated to 12 chars

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, registry):
        """Test error handling throughout the Docker system."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        parameters = {"command": "nonexistent_command_xyz"}
        context = {"session_id": "error_test_session"}

        # Mock Docker environment returning error
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock()
        mock_env.execute_command.return_value = CommandResult(
            exit_code=127,
            stdout="",
            stderr="command not found: nonexistent_command_xyz\n",
            execution_time=0.01
        )
        mock_env.get_container_id.return_value = "error_test_container"

        with patch.object(registry._sandbox_manager, 'get_session_environment', return_value=mock_env):
            result = await registry.step(parameters, context)

        assert result.success is False
        assert "Command failed with exit code 127" in result.error
        assert "command not found" in result.error

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
sandbox:
  enabled: true
  image: "saber/updated-sandbox:latest"
  network_mode: "bridge"
"""
        config_file.write_text(config_content)

        # Load new configuration
        registry.load_configuration(str(config_file))

        # Verify configuration was updated
        config = registry.get_configuration()
        assert config.get_execution_timeout() == 90.0
        assert config.get_max_concurrent() == 8
        assert "new_command" in config.get_allowed_commands()

        # Verify sandbox config was updated
        sandbox_config = config.get_sandbox_config()
        assert sandbox_config["image"] == "saber/updated-sandbox:latest"
        assert sandbox_config["network_mode"] == "bridge"

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
            result = await registry.step(parameters)

            assert result.success is False, f"Dangerous command should be blocked: {cmd}"
            assert "Command security validation failed" in result.error

    def test_component_initialization_integration(self, test_config):
        """Test that all components are properly initialized together."""
        with patch("saber.server.execution.sandbox.sandbox_manager.SandboxManager"):
            registry = ExecutionManager(config=test_config)

        # Verify all components exist and are correct types
        assert hasattr(registry, '_configuration')
        assert hasattr(registry, '_security_validator')
        assert hasattr(registry, '_cli_tool')
        assert hasattr(registry, '_sandbox_manager')
        assert hasattr(registry, '_semaphore')

        # Verify component types
        assert isinstance(registry._configuration, ExecutionConfiguration)
        assert isinstance(registry._security_validator, SecurityValidator)
        assert isinstance(registry._cli_tool, DockerCLIExecutor)
        assert isinstance(registry._sandbox_manager, SandboxManager)

    @pytest.mark.asyncio
    async def test_realistic_malware_analysis_scenario(self, registry):
        """Test realistic malware analysis commands in Docker environment."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        # Simulate commands that might be used in malware analysis
        analysis_commands = [
            {"command": "echo 'Analyzing file'"},
            {"command": "echo 'File type: PE32 executable'"},  # Simulating file command
            {"command": "echo 'Strings found: 50'"},           # Simulating strings command
        ]

        # Mock Docker environment
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock()
        mock_env.get_container_id.return_value = "analysis_container"

        results = []

        for i, params in enumerate(analysis_commands):
            expected_output = params["command"].split("'")[1] + "\n"
            mock_env.execute_command.return_value = CommandResult(
                exit_code=0,
                stdout=expected_output,
                stderr="",
                execution_time=0.1
            )

            context = {"session_id": f"analysis_session_{i}"}

            with patch.object(registry._sandbox_manager, 'get_session_environment', return_value=mock_env):
                result = await registry.step(params, context)
                results.append(result)

        # All analysis commands should succeed
        assert all(result.success for result in results)

        # Verify expected outputs
        assert "Analyzing file" in results[0].data["stdout"]
        assert "File type" in results[1].data["stdout"]
        assert "Strings found" in results[2].data["stdout"]
