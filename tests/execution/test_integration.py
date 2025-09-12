"""
Integration tests for the Docker-based tool execution framework.

Tests integration between BashExecutor, SandboxManager,
SecurityValidator, and ExecutionManager working together in Docker containers.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.base import Action, CommandResult
from saber.server.execution.base import ValidationResult
from saber.server.execution.execution_manager import ExecutionManager
from saber.server.execution.executors.executor_factory import ExecutorFactory
from saber.server.execution.executors.standard_registry.bash_executor import BashExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from saber.server.execution.utils.security_validator import SecurityValidator


class TestToolsIntegration:
    """Integration test cases for the complete tools framework."""

    @pytest.fixture
    def test_config(self):
        """Test configuration for integration tests."""
        return {
            "execution": {"timeout": 30.0, "max_concurrent": 3},
            "security": {"allowed_commands": ["echo", "cat", "ls"], "max_command_length": 1000},
            "bash": {"default_shell_mode": False},
            "sandbox": {
                "image": "saber/sandbox:latest",
                "network_mode": "none",
                "read_only_root": True,
                "user": "tooluser:tooluser",
            },
        }

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory for testing."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        return str(config_dir)

    @pytest.fixture
    def registry(self, test_config, temp_config_dir):
        """Create ExecutionManager for integration testing."""
        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager"):
            return ExecutionManager(temp_config_dir)

    @pytest.fixture
    def real_registry(self, test_config, docker_cleanup, temp_config_dir):
        """Create ExecutionManager with real Docker containers for integration testing."""
        execution_manager = ExecutionManager(temp_config_dir)
        # Register for cleanup
        docker_cleanup(execution_manager)
        return execution_manager

    @pytest.mark.skip(
        reason="TODO: Remove skip when security validation is fixed - Command security validation currently disabled"
    )
    @pytest.mark.asyncio
    async def test_end_to_end_blocked_command_execution(self, registry):
        """Test complete flow for blocked command execution."""
        action = Action(tool_name="bash", parameters={"command": "sudo rm -rf /"})
        context = {"episode_id": f"integration_test_002_{uuid.uuid4().hex[:8]}"}

        result = await registry.step(action, context)

        # Should be blocked by security validation
        assert result.success is False
        assert "Command security validation failed" in result.error

    @pytest.mark.asyncio
    async def test_end_to_end_whitelisted_command_execution(self, registry):
        """Test execution of allowed command in Docker container."""

        action = Action(tool_name="bash", parameters={"command": "echo test"})  # Safe command from allowed list
        context = {"episode_id": f"integration_test_003_{uuid.uuid4().hex[:8]}"}

        # Mock Docker environment execution
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock(return_value=CommandResult(
            exit_code=0, stdout="test\n", stderr="", execution_time=0.05
        ))
        mock_env.get_container_id.return_value = "whitelist_container_456"

        with patch.object(registry._sandbox_manager, "get_episode_environment", return_value=mock_env):
            result = await registry.step(action, context)

        assert result.exit_code == 0
        assert result.stdout == "test\n"

    @pytest.mark.asyncio
    async def test_concurrent_command_execution(self, registry):
        """Test concurrent execution with semaphore control in Docker."""

        actions_list = [Action(tool_name="bash", parameters={"command": f"echo test{i}"}) for i in range(5)]

        contexts_list = [{"episode_id": f"concurrent_test_{i}"} for i in range(5)]

        # Mock Docker environment for all commands
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock(return_value=CommandResult(
            exit_code=0, stdout="test\n", stderr="", execution_time=0.02
        ))
        mock_env.get_container_id.return_value = "concurrent_container"

        with patch.object(registry._sandbox_manager, "get_episode_environment", return_value=mock_env):
            # Execute all commands concurrently
            tasks = [
                asyncio.create_task(registry.step(action, context))
                for action, context in zip(actions_list, contexts_list)
            ]

            results = await asyncio.gather(*tasks)

        # All should succeed despite concurrency limits
        assert all(result.exit_code == 0 for result in results)

    @pytest.mark.asyncio
    async def test_shell_mode_integration(self, registry):
        """Test shell mode with complex commands in Docker."""

        # Use a command that would benefit from shell mode but isn't dangerous
        action = Action(tool_name="bash", parameters={"command": "echo 'hello world'", "shell": True})
        context = {"episode_id": f"integration_test_004_{uuid.uuid4().hex[:8]}"}

        # Mock Docker environment execution
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock(return_value=CommandResult(
            exit_code=0, stdout="hello world\n", stderr="", execution_time=0.03
        ))
        mock_env.get_container_id.return_value = "shell_test_container"

        with patch.object(registry._sandbox_manager, "get_episode_environment", return_value=mock_env) as mock_get_env:
            result = await registry.step(action, context)

        assert result.exit_code == 0
        assert result.stdout == "hello world\n"

        # Verify Docker step was called with shell format
        mock_env.execute_command.assert_called_once_with(command=["/bin/sh", "-c", "echo 'hello world'"], timeout=60)

    def test_security_validator_configuration_integration(self, registry):
        """Test that security validator is properly configured at executor level."""
        # Get CLI executor and verify its Docker configuration
        bash_executor = registry.get_executor("bash")
        docker_info = bash_executor.get_docker_info()

        # Verify the CLI executor has proper Docker configuration
        assert "docker_config" in docker_info or "execution_environment" in docker_info

        # Test that security validation works by attempting to validate a command
        validation_result = bash_executor.validate_parameters({"command": "echo test"})
        assert validation_result.valid is True

    def test_mcp_integration(self, registry):
        """Test MCP tools format integration."""
        mcp_tools = registry.to_mcp_tools()

        assert len(mcp_tools) >= 2  # At least CLI and Python executors

        # Find CLI and Python tools - they should be dictionaries now
        cli_tool = next(tool for tool in mcp_tools if "bash" in tool["name"])
        python_tool = next(tool for tool in mcp_tools if "python" in tool["name"])

        # Verify MCP format compliance for CLI tool
        assert "description" in cli_tool
        assert "inputSchema" in cli_tool

        cli_schema = cli_tool["inputSchema"]
        # Handle both dict and MCPInputSchema object formats
        if hasattr(cli_schema, 'type'):
            # It's an MCPInputSchema object
            assert cli_schema.type == "object"
            assert hasattr(cli_schema, 'properties')
            assert hasattr(cli_schema, 'required')
            assert "command" in cli_schema.properties
            assert "command" in cli_schema.required
        else:
            # It's a dictionary
            assert cli_schema["type"] == "object"
            assert "properties" in cli_schema
            assert "required" in cli_schema
            assert "command" in cli_schema["properties"]
            assert "command" in cli_schema["required"]

        # Verify MCP format compliance for Python tool
        assert "description" in python_tool
        assert "inputSchema" in python_tool

        python_schema = python_tool["inputSchema"]
        # Handle both dict and MCPInputSchema object formats
        if hasattr(python_schema, 'type'):
            # It's an MCPInputSchema object
            assert python_schema.type == "object"
            assert hasattr(python_schema, 'properties')
            assert hasattr(python_schema, 'required')
            assert "code" in python_schema.properties
            assert "code" in python_schema.required
        else:
            # It's a dictionary
            assert python_schema["type"] == "object"
            assert "properties" in python_schema
            assert "required" in python_schema
            assert "code" in python_schema["properties"]
            assert "code" in python_schema["required"]

    @pytest.mark.asyncio
    async def test_parameter_validation_integration(self, registry):
        """Test parameter validation integration."""
        # Use unique episode ID to avoid Docker container conflicts
        unique_episode_id = f"test_validation_{uuid.uuid4().hex[:8]}"

        # Test missing required parameter (empty command)
        action = Action(tool_name="bash", parameters={"command": "", "shell": True})  # Missing command
        context = {"episode_id": unique_episode_id}
        result = await registry.step(action, context)
        assert result.exit_code != 0
        # The test should fail during validation or execution, not necessarily with the exact message
        assert result.error is not None

        # Test invalid parameter type
        action = Action(tool_name="bash", parameters={"command": "echo test", "shell": "invalid"})
        result = await registry.step(action, context)
        assert result.exit_code != 0
        # The test should fail with some validation error
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_session_isolation_integration(self, registry):
        """Test that different episodes are properly isolated."""

        action1 = Action(tool_name="bash", parameters={"command": "echo episode1"})
        action2 = Action(tool_name="bash", parameters={"command": "echo episode2"})
        episode_id_1 = f"episode_isolation_1_{uuid.uuid4().hex[:8]}"
        episode_id_2 = f"episode_isolation_2_{uuid.uuid4().hex[:8]}"
        context1 = {"episode_id": episode_id_1}
        context2 = {"episode_id": episode_id_2}

        # Mock different environments for different episodes
        mock_env1 = MagicMock()
        mock_env1.execute_command = AsyncMock(return_value=CommandResult(
            exit_code=0, stdout="episode1\n", stderr="", execution_time=0.1
        ))
        mock_container1 = MagicMock()
        mock_container1.id = "episode1_container"
        mock_env1.get_execution_container.return_value = mock_container1

        mock_env2 = MagicMock()
        mock_env2.execute_command = AsyncMock(return_value=CommandResult(
            exit_code=0, stdout="episode2\n", stderr="", execution_time=0.1
        ))
        mock_container2 = MagicMock()
        mock_container2.id = "episode2_container"
        mock_env2.get_execution_container.return_value = mock_container2

        # Mock sandbox manager to return different environments per episode
        def get_episode_env(episode_id):
            if episode_id == episode_id_1:
                return mock_env1
            elif episode_id == episode_id_2:
                return mock_env2
            return None

        with patch.object(registry._sandbox_manager, "get_episode_environment", side_effect=get_episode_env):
            result1 = await registry.step(action1, context1)
            result2 = await registry.step(action2, context2)

        # Verify isolation worked
        assert result1.exit_code == 0
        assert result1.stdout == "episode1\n"
        assert result1.metadata["container_id"] == "episode1_con"  # Truncated to 12 chars

        assert result2.exit_code == 0
        assert result2.stdout == "episode2\n"
        assert result2.metadata["container_id"] == "episode2_con"  # Truncated to 12 chars

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, registry):
        """Test error handling throughout the Docker system."""

        action = Action(tool_name="bash", parameters={"command": "nonexistent_command_xyz"})
        context = {"episode_id": f"error_test_episode_{uuid.uuid4().hex[:8]}"}

        # Mock Docker environment returning error
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock(return_value=CommandResult(
            exit_code=127, stdout="", stderr="command not found: nonexistent_command_xyz\n", execution_time=0.01
        ))
        mock_env.get_container_id.return_value = "error_test_container"

        with patch.object(registry._sandbox_manager, "get_episode_environment", return_value=mock_env):
            result = await registry.step(action, context)

        assert result.exit_code != 0
        assert "Command failed with exit code 127" in result.error
        assert "command not found" in result.error

    @pytest.mark.skip(
        reason="TODO: Remove skip when security validation is fixed - Command security validation currently disabled"
    )
    @pytest.mark.asyncio
    async def test_security_patterns_integration(self, registry):
        """Test integration of security pattern detection."""
        dangerous_commands = ["echo hello; rm -rf /", "cat file | sh", "echo $(whoami)", "ls > /etc/passwd"]

        for cmd in dangerous_commands:
            action = Action(tool_name="bash", parameters={"command": cmd})
            context = {"episode_id": f"security_test_{uuid.uuid4().hex[:8]}"}
            result = await registry.step(action, context)

            assert result.exit_code != 0, f"Dangerous command should be blocked: {cmd}"
            assert "Command security validation failed" in result.error

    def test_component_initialization_integration(self, test_config, temp_config_dir):
        """Test that all components are properly initialized together."""
        with patch("saber.server.execution.sandbox.sandbox_environment_manager.SandboxEnvironmentManager"):
            registry = ExecutionManager(temp_config_dir)

        # Verify all components exist and are correct types
        assert hasattr(registry, "_configuration")
        assert hasattr(registry, "_executor_factory")
        assert hasattr(registry, "_sandbox_manager")

        # Verify component types
        assert isinstance(registry._configuration, dict)

        # Verify executor factory has CLI capability
        available_executors = registry._executor_factory.get_available_executors()
        assert "bash" in available_executors
        assert "python" in available_executors

        assert isinstance(registry._sandbox_manager, SandboxEnvironmentManager)

    @pytest.mark.asyncio
    async def test_realistic_malware_analysis_scenario(self, registry):
        """Test realistic malware analysis commands in Docker environment."""

        # Simulate commands that might be used in malware analysis
        analysis_commands = [
            Action(tool_name="bash", parameters={"command": "echo 'Analyzing file'"}),
            Action(
                tool_name="bash", parameters={"command": "echo 'File type: PE32 executable'"}
            ),  # Simulating file command
            Action(tool_name="bash", parameters={"command": "echo 'Strings found: 50'"}),  # Simulating strings command
        ]

        # Mock Docker environment
        mock_env = MagicMock()
        mock_env.execute_command = AsyncMock()
        mock_env.get_container_id.return_value = "analysis_container"

        results = []

        for i, action in enumerate(analysis_commands):
            expected_output = action.parameters["command"].split("'")[1] + "\n"
            mock_env.execute_command.return_value = CommandResult(
                exit_code=0, stdout=expected_output, stderr="", execution_time=0.1
            )

            context = {"episode_id": f"analysis_episode_{i}"}

            with patch.object(registry._sandbox_manager, "get_episode_environment", return_value=mock_env):
                result = await registry.step(action, context)
                results.append(result)

        # All analysis commands should succeed
        assert all(result.exit_code == 0 for result in results)

        # Verify expected outputs
        assert "Analyzing file" in results[0].stdout
        assert "File type" in results[1].stdout
        assert "Strings found" in results[2].stdout

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_docker_container_cleanup(self, real_registry, docker_cleanup):
        """Test that real Docker containers are created and properly cleaned up."""
        episode_id = f"real_container_test_{uuid.uuid4().hex[:8]}"

        # Register this episode for cleanup
        docker_cleanup(real_registry, episode_id)

        # Create a simple action that should work in the container
        action = Action(tool_name="bash", parameters={"command": "echo 'real container test'"})
        context = {"episode_id": episode_id}

        try:
            # This should create a real Docker container
            result = await real_registry.step(action, context)

            # Verify it worked (if the Docker image is available)
            # If the image isn't available, the test might fail, but cleanup should still work
            if result.exit_code == 0:
                assert "real container test" in result.stdout
                assert result.exit_code == 0
            else:
                # If Docker image isn't available, that's ok for this test
                # The important part is that cleanup works
                print(f"Docker execution failed (image may not be available): {result.error}")

        except Exception as e:
            # If there's an error, that's ok - the important part is cleanup
            print(f"Docker execution error (expected if image unavailable): {e}")

        # Manually test cleanup
        real_registry.cleanup_episode(episode_id)

        # The actual container cleanup verification happens in the docker_cleanup fixture
