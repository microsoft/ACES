"""
Tests for ExecutionManager.

This module tests the Docker-based execution manager that provides secure command execution
with MCP integration and comprehensive security validation in Docker containers.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path

from saber.server.execution.base import CommandResult, ValidationResult
from saber.server.execution.execution_manager import ExecutionManager, ExecutionConfiguration
from saber.server.execution.executors.cli import CLIExecutor
from saber.server.execution.executors.factory import ExecutorFactory
from saber.server.execution.utils.security_validator import SecurityValidator
from saber.server.execution.exceptions import ExecutionManagerError
from saber.server.execution.sandbox.sandbox_manager import SandboxManager
from saber.server.tasks.base import Action


class TestExecutionConfiguration:
    """Test cases for ExecutionConfiguration."""

    @patch("builtins.open", new_callable=mock_open, read_data="""
execution:
  timeout: 120.0
  max_concurrent: 15
security:
  allowed_commands
    - file
    - strings
    - python3
  max_command_length: 8000
cli:
  default_shell_mode: true
sandbox:
  image: saber/base-sandbox:latest
  network_mode: none
""")
    @patch("yaml.safe_load")
    def test_initialization_with_file(self, mock_yaml_load, mock_file):
        """Test initialization with YAML configuration file."""
        expected_config = {
            "execution": {"timeout": 120.0, "max_concurrent": 15},
            "security": {"allowed_commands": ["file", "strings", "python3"], "max_command_length": 8000},
            "cli": {"default_shell_mode": True}
        }
        mock_yaml_load.return_value = expected_config

        config = ExecutionConfiguration(config_file="test_config.yaml")

        mock_file.assert_called_once_with("test_config.yaml", 'r')
        assert config._config == expected_config

    @patch("builtins.open", side_effect=FileNotFoundError("File not found"))
    def test_initialization_file_not_found(self, mock_file):
        """Test initialization when config file doesn't exist."""
        config = ExecutionConfiguration(config_file="nonexistent.yaml")
        assert config._config == {}

    def test_load_configuration_success(self):
        """Test successful configuration loading."""
        config_data = {"test": "data"}

        with patch("builtins.open", mock_open(read_data="test: data")):
            with patch("yaml.safe_load", return_value=config_data):
                config = ExecutionConfiguration()
                config.load_configuration("test.yaml")

                assert config._config == config_data

    def test_load_configuration_yaml_error(self):
        """Test configuration loading with YAML parsing error."""
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content:")):
            with patch("yaml.safe_load", side_effect=Exception("YAML error")):
                config = ExecutionConfiguration()

                with pytest.raises(Exception, match="YAML error"):
                    config.load_configuration("test.yaml")

    def test_get_generic_section(self):
        """Test getting any configuration section generically."""
        config_dict = {
            "custom_executor": {"setting1": "value1", "setting2": 42},
            "another_section": {"enabled": True}
        }
        config = ExecutionConfiguration(config=config_dict)

        custom_config = config.get_section("custom_executor")
        assert custom_config == {"setting1": "value1", "setting2": 42}

        another_config = config.get_section("another_section")
        assert another_config == {"enabled": True}

        # Test non-existent section
        missing_config = config.get_section("does_not_exist")
        assert missing_config == {}


class TestExecutionManager:
    """Test cases for ExecutionManager."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "execution": {"timeout": 60.0, "max_concurrent": 5},
            "security": {"allowed_commands": ["file", "strings"]},
            "cli": {"default_shell_mode": False},
            "sandbox": {
                "image": "saber/base-sandbox:latest",
                "network_mode": "none",
                "read_only_root": True,
                "user": "tooluser:tooluser"
            }
        }

    @pytest.fixture
    def registry(self, sample_config):
        """Create an ExecutionManager instance for testing."""
        with patch("saber.server.execution.execution_manager.SandboxManager"):
            return ExecutionManager(config=sample_config)

    def test_initialization_with_valid_config(self, sample_config):
        """Test initialization with valid sandbox configuration."""
        with patch("saber.server.execution.execution_manager.SandboxManager") as mock_sandbox:
            registry = ExecutionManager(config=sample_config)

        assert isinstance(registry._configuration, ExecutionConfiguration)
        assert isinstance(registry._security_validator, SecurityValidator)
        assert isinstance(registry._executor_factory, ExecutorFactory)
        assert registry._sandbox_manager == mock_sandbox.return_value
        assert isinstance(registry._semaphore, asyncio.Semaphore)

    def test_initialization_with_config(self, sample_config):
        """Test initialization with configuration."""
        with patch("saber.server.execution.execution_manager.SandboxManager"):
            registry = ExecutionManager(config=sample_config)

        assert registry._configuration.get_execution_timeout() == 60.0
        assert registry._configuration.get_max_concurrent() == 5
        assert "file" in registry._configuration.get_allowed_commands()

        # Test sandbox configuration
        sandbox_config = registry._configuration.get_sandbox_config()
        assert sandbox_config["image"] == "saber/base-sandbox:latest"

    @patch("saber.server.execution.execution_manager.ExecutionConfiguration")
    @patch("saber.server.execution.execution_manager.SandboxManager")
    def test_initialization_with_config_file(self, mock_sandbox, mock_cli_config):
        """Test initialization with configuration file."""
        mock_instance = MagicMock()
        mock_cli_config.return_value = mock_instance
        mock_instance.get_allowed_commands.return_value = []
        mock_instance.get_execution_timeout.return_value = 300.0
        mock_instance.get_max_concurrent.return_value = 10

        registry = ExecutionManager(config_file="test_config.yaml")

        mock_cli_config.assert_called_once_with(None, "test_config.yaml")

    @pytest.mark.asyncio
    async def test_step_success(self, registry):
        """Test successful command execution."""
        action = Action(tool_name="cli", command="echo test", parameters={"shell": False})
        context = {"session_id": "test123"}

        expected_result = CommandResult.success_result(
            data={"stdout": "test\n", "stderr": "", "return_code": 0}
        )

        # Mock the executor factory to return a mock executor
        mock_executor = MagicMock()
        mock_executor.validate_parameters.return_value = ValidationResult.success()
        mock_executor.execute = AsyncMock(return_value=expected_result)

        with patch.object(registry._executor_factory, 'get_executor', return_value=mock_executor):
            with patch.object(registry, '_determine_executor_type', return_value='cli'):
                result = await registry.step(action, context)

        assert result.success is True
        assert result.data["stdout"] == "test\n"
        expected_params = {"command": "echo test", "shell": False}
        mock_executor.execute.assert_called_once_with(expected_params, context)

    @pytest.mark.asyncio
    async def test_step_validation_failure(self, registry):
        """Test command execution with parameter validation failure."""
        action = Action(tool_name="cli", command="", parameters={"invalid": "params"})

        validation_result = ValidationResult.failure(["Missing required parameter 'command'"])

        # Mock the executor factory to return a mock executor
        mock_executor = MagicMock()
        mock_executor.validate_parameters.return_value = validation_result

        with patch.object(registry._executor_factory, 'get_executor', return_value=mock_executor):
            with patch.object(registry, '_determine_executor_type', return_value='cli'):
                result = await registry.step(action)

        assert result.success is False
        assert "Parameter validation failed" in result.error
        assert "Missing required parameter 'command'" in result.error

    @pytest.mark.asyncio
    async def test_step_execution_exception(self, registry):
        """Test command execution with exception during execution."""
        action = Action(tool_name="cli", command="test")

        # Mock the executor factory to return a mock executor
        mock_executor = MagicMock()
        mock_executor.validate_parameters.return_value = ValidationResult.success()
        mock_executor.execute.side_effect = Exception("Execution failed")

        with patch.object(registry._executor_factory, 'get_executor', return_value=mock_executor):
            with patch.object(registry, '_determine_executor_type', return_value='cli'):
                result = await registry.step(action)

        assert result.success is False
        assert "Execution failed" in result.error

    @pytest.mark.asyncio
    async def test_step_concurrency_control(self, registry):
        """Test that concurrency control works with semaphore."""
        action = Action(tool_name="cli", command="sleep 1")

        # Mock the executor to simulate slow execution
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(0.1)
            return CommandResult.success_result(data="done")

        mock_executor = MagicMock()
        mock_executor.validate_parameters.return_value = ValidationResult.success()
        mock_executor.execute.side_effect = slow_execute

        with patch.object(registry._executor_factory, 'get_executor', return_value=mock_executor):
            with patch.object(registry, '_determine_executor_type', return_value='cli'):
                # Start multiple executions
                tasks = [
                    asyncio.create_task(registry.step(action))
                    for _ in range(3)
                ]

                results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(result.success for result in results)

    def test_validate_command(self, registry):
        """Test command validation."""
        with patch.object(registry._security_validator, 'validate_command_string') as mock_validate:
            mock_validate.return_value = ValidationResult.success()

            result = registry.validate_command("ls -la")

            assert result.valid is True
            mock_validate.assert_called_once_with("ls -la")

    def test_get_security_info(self, registry):
        """Test getting security information."""
        expected_info = {"allowed_commands": ["file"], "sandbox_path": None}

        with patch.object(registry._security_validator, 'get_security_info', return_value=expected_info):
            info = registry.get_security_info()

            assert info == expected_info

    def test_to_mcp_tools(self, registry):
        """Test MCP tools conversion."""
        # Mock the executor factory to return predefined tools
        mock_tools = [
            {
                "name": "cli_docker_cli",
                "description": "Execute validated shell commands in Docker containers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                        "shell": {"type": "boolean", "description": "Use shell mode", "default": False}
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "python_python_script",
                "description": "Execute Python scripts in Docker containers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "requirements": {"type": "array", "description": "Python packages to install"}
                    },
                    "required": ["code"]
                }
            }
        ]

        with patch.object(registry._executor_factory, 'get_all_mcp_tools', return_value=mock_tools):
            mcp_tools = registry.to_mcp_tools()

        assert len(mcp_tools) == 2

        # Check CLI tool
        cli_tool = next(tool for tool in mcp_tools if tool["name"] == "cli_docker_cli")
        assert cli_tool["description"] == "Execute validated shell commands in Docker containers"
        assert "inputSchema" in cli_tool
        assert "command" in cli_tool["inputSchema"]["properties"]

        # Check Python tool
        python_tool = next(tool for tool in mcp_tools if tool["name"] == "python_python_script")
        assert python_tool["description"] == "Execute Python scripts in Docker containers"
        assert "code" in python_tool["inputSchema"]["properties"]

    def test_to_mcp_tools_with_cli_config(self, sample_config):
        """Test MCP tools conversion with CLI configuration."""
        sample_config["cli"]["default_shell_mode"] = True

        with patch("saber.server.execution.execution_manager.SandboxManager"):
            registry = ExecutionManager(config=sample_config)

        # Mock the factory's response
        mock_tools = [
            {
                "name": "cli_docker_cli",
                "description": "Execute validated shell commands in Docker containers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                        "shell": {"type": "boolean", "description": "Use shell mode", "default": True}
                    },
                    "required": ["command"]
                }
            }
        ]

        with patch.object(registry._executor_factory, 'get_all_mcp_tools', return_value=mock_tools):
            mcp_tools = registry.to_mcp_tools()

        tool = mcp_tools[0]
        # Should include default value from CLI config
        assert tool["inputSchema"]["properties"]["shell"]["default"] is True

    def test_get_execution_stats(self, registry):
        """Test getting execution statistics."""
        # Mock executor factory info
        mock_executor_info = {
            "available_types": ["cli", "python"],
            "active_instances": ["cli"],
            "registry_size": 2,
            "configurations": {"cli": {}, "python": {}}
        }

        with patch.object(registry._executor_factory, 'get_executor_info', return_value=mock_executor_info):
            stats = registry.get_execution_stats()

        assert "max_concurrent" in stats
        assert "current_available" in stats
        assert "timeout" in stats
        assert "security_config" in stats
        assert "executor_info" in stats
        assert stats["max_concurrent"] == 5  # From sample config
        assert stats["executor_info"] == mock_executor_info

    def test_load_configuration_updates_components(self, registry):
        """Test that loading configuration updates all components."""
        new_config_path = "new_config.yaml"

        with patch.object(registry._configuration, 'load_configuration') as mock_load:
            with patch.object(registry._configuration, 'get_allowed_commands', return_value=["new_cmd"]):
                with patch.object(registry._configuration, 'get_execution_timeout', return_value=120.0):
                    with patch.object(registry._configuration, 'get_max_concurrent', return_value=15):
                        with patch("saber.server.execution.execution_manager.SandboxManager") as mock_sandbox:
                            with patch("saber.server.execution.execution_manager.ExecutorFactory") as mock_factory:
                                registry.load_configuration(new_config_path)

        mock_load.assert_called_once_with(new_config_path)
        # Components should be recreated with new configuration
        mock_sandbox.assert_called()
        mock_factory.assert_called()

    def test_get_configuration(self, registry):
        """Test getting configuration manager."""
        config = registry.get_configuration()
        assert isinstance(config, ExecutionConfiguration)
        assert config == registry._configuration

    @pytest.mark.asyncio
    async def test_step_default_context(self, registry):
        """Test step with default context when none provided."""
        action = Action(tool_name="cli", command="echo test")

        expected_result = CommandResult.success_result(data="test")

        # Mock the executor factory to return a mock executor
        mock_executor = MagicMock()
        mock_executor.validate_parameters.return_value = ValidationResult.success()
        mock_executor.execute = AsyncMock(return_value=expected_result)

        with patch.object(registry._executor_factory, 'get_executor', return_value=mock_executor):
            with patch.object(registry, '_determine_executor_type', return_value='cli'):
                result = await registry.step(action)

        # Should be called with empty context dict
        expected_params = {"command": "echo test"}
        mock_executor.execute.assert_called_once_with(expected_params, {})

    def test_get_executor(self, registry):
        """Test getting specific executor instance."""
        mock_executor = MagicMock()

        with patch.object(registry._executor_factory, 'get_executor', return_value=mock_executor) as mock_get:
            result = registry.get_executor("cli")

        assert result == mock_executor
        mock_get.assert_called_once_with("cli")

    def test_determine_executor_type_explicit(self, registry):
        """Test executor type determination with explicit type."""
        action = Action(tool_name="cli", command="echo test")
        # Since Action doesn't have executor_type field, we'll test a different way
        # by adding it to parameters
        action.parameters["executor_type"] = "python"

        # For this test, let's modify the method to check parameters
        with patch.object(action, '__dict__', {**action.__dict__, 'executor_type': 'python'}):
            result = registry._determine_executor_type(action)
            assert result == "python"

    def test_determine_executor_type_from_parameters(self, registry):
        """Test executor type determination from parameters."""
        action = Action(tool_name="cli", command="", parameters={"code": "print('hello')"})

        result = registry._determine_executor_type(action)
        assert result == "python"

    def test_determine_executor_type_command_analysis(self, registry):
        """Test executor type determination from command analysis."""
        action = Action(tool_name="cli", command="python3 script.py")

        with patch.object(registry._executor_factory, '_analyze_command', return_value='python') as mock_analyze:
            result = registry._determine_executor_type(action)

        assert result == "python"
        mock_analyze.assert_called_once_with("python3 script.py")

    def test_determine_executor_type_default(self, registry):
        """Test executor type determination defaults to CLI."""
        action = Action(tool_name="cli", command="ls -la")

        with patch.object(registry._executor_factory, '_analyze_command', return_value='cli') as mock_analyze:
            result = registry._determine_executor_type(action)

        assert result == "cli"

    @pytest.mark.asyncio
    async def test_step_python_executor(self, registry):
        """Test step with Python executor."""
        action = Action(tool_name="python", command="", parameters={"code": "print('hello')"})
        context = {"session_id": "test123"}

        expected_result = CommandResult.success_result(
            data={"stdout": "hello\n", "stderr": "", "return_code": 0}
        )

        # Mock the executor factory to return a mock Python executor
        mock_executor = MagicMock()
        mock_executor.validate_parameters.return_value = ValidationResult.success()
        mock_executor.execute = AsyncMock(return_value=expected_result)

        with patch.object(registry._executor_factory, 'get_executor', return_value=mock_executor):
            with patch.object(registry, '_determine_executor_type', return_value='python'):
                result = await registry.step(action, context)

        assert result.success is True
        assert result.data["stdout"] == "hello\n"
        expected_params = {"command": "", "code": "print('hello')"}
        mock_executor.execute.assert_called_once_with(expected_params, context)

    def test_cleanup_all_sessions(self, registry):
        """Test cleanup of all sessions."""
        with patch.object(registry._sandbox_manager, 'cleanup_all_sessions') as mock_cleanup_sandbox:
            with patch.object(registry._executor_factory, 'cleanup_all_executors') as mock_cleanup_executors:
                registry.cleanup_all_sessions()

        mock_cleanup_sandbox.assert_called_once()
        mock_cleanup_executors.assert_called_once()

    def test_list_commands(self, registry):
        """Test listing all available commands."""
        mock_commands = [
            {
                "executor_type": "cli",
                "name": "docker_cli",
                "description": "Execute shell commands",
                "domain": "general",
                "security_level": "high",
                "parameters": ["command", "shell"]
            },
            {
                "executor_type": "python",
                "name": "python_script",
                "description": "Execute Python scripts",
                "domain": "python",
                "security_level": "high",
                "parameters": ["code", "requirements"]
            }
        ]

        # Mock executor factory methods
        with patch.object(registry._executor_factory, 'get_available_executors', return_value=['cli', 'python']):
            mock_cli_executor = MagicMock()
            mock_cli_executor._security_command_metadata = {
                'name': 'docker_cli',
                'description': 'Execute shell commands',
                'domain': 'general',
                'security_level': 'high'
            }
            mock_cli_executor.get_parameters.return_value = {'command': MagicMock(), 'shell': MagicMock()}

            mock_python_executor = MagicMock()
            mock_python_executor._security_command_metadata = {
                'name': 'python_script',
                'description': 'Execute Python scripts',
                'domain': 'python',
                'security_level': 'high'
            }
            mock_python_executor.get_parameters.return_value = {'code': MagicMock(), 'requirements': MagicMock()}

            def mock_get_executor(executor_type):
                if executor_type == 'cli':
                    return mock_cli_executor
                elif executor_type == 'python':
                    return mock_python_executor

            with patch.object(registry._executor_factory, 'get_executor', side_effect=mock_get_executor):
                commands = registry.list_commands()

        assert len(commands) == 2
        assert any(cmd['executor_type'] == 'cli' for cmd in commands)
        assert any(cmd['executor_type'] == 'python' for cmd in commands)
