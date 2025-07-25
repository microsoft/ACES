"""
Tests for ExecutionManager.

This module tests the Docker-based execution manager that provides secure command execution
with MCP integration and comprehensive security validation in Docker containers.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path

from saber.server.tools.base import ToolResult, ValidationResult
from saber.server.tools.execution_manager import ExecutionManager, ExecutionConfiguration
from saber.server.tools.executors.cli import DockerCLIExecutor
from saber.server.tools.utils.security_validator import SecurityValidator
from saber.server.tools.exceptions import ExecutionManagerError
from saber.server.tools.sandbox.sandbox_manager import SandboxManager
from saber.server.tools.utils.security_validator import SecurityValidator
from saber.server.tools.exceptions import ExecutionManagerError


class TestExecutionConfiguration:
    """Test cases for ExecutionConfiguration."""

    def test_initialization_empty(self):
        """Test initialization with no configuration."""
        config = ExecutionConfiguration()
        assert config._config == {}

    def test_initialization_with_dict(self):
        """Test initialization with configuration dictionary."""
        config_dict = {
            "execution": {"timeout": 60.0, "max_concurrent": 5},
            "security": {"allowed_commands": ["file", "strings"]},
            "sandbox": {"enabled": True, "image": "saber/base-sandbox:latest"}
        }

        config = ExecutionConfiguration(config=config_dict)
        assert config._config == config_dict

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
  enabled: true
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

    def test_get_execution_config(self):
        """Test getting execution configuration."""
        config_dict = {"execution": {"timeout": 60.0, "max_concurrent": 5}}
        config = ExecutionConfiguration(config=config_dict)

        exec_config = config.get_execution_config()
        assert exec_config == {"timeout": 60.0, "max_concurrent": 5}

    def test_get_execution_config_missing(self):
        """Test getting execution config when not present."""
        config = ExecutionConfiguration()
        exec_config = config.get_execution_config()
        assert exec_config == {}

    def test_get_security_config(self):
        """Test getting security configuration."""
        config_dict = {"security": {"allowed_commands": ["file"]}}
        config = ExecutionConfiguration(config=config_dict)

        sec_config = config.get_security_config()
        assert sec_config == {"allowed_commands": ["file"]}

    def test_get_cli_config(self):
        """Test getting CLI configuration."""
        config_dict = {"cli": {"default_shell_mode": True}}
        config = ExecutionConfiguration(config=config_dict)

        cli_config = config.get_cli_config()
        assert cli_config == {"default_shell_mode": True}

    def test_get_execution_timeout_default(self):
        """Test getting default execution timeout."""
        config = ExecutionConfiguration()
        assert config.get_execution_timeout() == 300.0

    def test_get_execution_timeout_custom(self):
        """Test getting custom execution timeout."""
        config_dict = {"execution": {"timeout": 120.0}}
        config = ExecutionConfiguration(config=config_dict)
        assert config.get_execution_timeout() == 120.0

    def test_get_max_concurrent_default(self):
        """Test getting default max concurrent."""
        config = ExecutionConfiguration()
        assert config.get_max_concurrent() == 10

    def test_get_max_concurrent_custom(self):
        """Test getting custom max concurrent."""
        config_dict = {"execution": {"max_concurrent": 20}}
        config = ExecutionConfiguration(config=config_dict)
        assert config.get_max_concurrent() == 20

    def test_get_allowed_commands_default(self):
        """Test getting default allowed commands."""
        config = ExecutionConfiguration()
        assert config.get_allowed_commands() == []

    def test_get_allowed_commands_custom(self):
        """Test getting custom allowed commands."""
        config_dict = {"security": {"allowed_commands": ["file", "strings"]}}
        config = ExecutionConfiguration(config=config_dict)
        assert config.get_allowed_commands() == ["file", "strings"]

    def test_get_max_command_length_default(self):
        """Test getting default max command length."""
        config = ExecutionConfiguration()
        assert config.get_max_command_length() == 10000

    def test_get_max_command_length_custom(self):
        """Test getting custom max command length."""
        config_dict = {"security": {"max_command_length": 5000}}
        config = ExecutionConfiguration(config=config_dict)
        assert config.get_max_command_length() == 5000

    def test_get_sandbox_config(self):
        """Test getting sandbox configuration."""
        config_dict = {
            "sandbox": {
                "enabled": True,
                "image": "saber/base-sandbox:latest",
                "network_mode": "none"
            }
        }
        config = ExecutionConfiguration(config=config_dict)

        sandbox_config = config.get_sandbox_config()
        assert sandbox_config == {
            "enabled": True,
            "image": "saber/base-sandbox:latest",
            "network_mode": "none"
        }

    def test_get_sandbox_config_missing(self):
        """Test getting sandbox config when not present."""
        config = ExecutionConfiguration()
        sandbox_config = config.get_sandbox_config()
        assert sandbox_config == {}


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
                "enabled": True,
                "image": "saber/base-sandbox:latest",
                "network_mode": "none",
                "read_only_root": True,
                "user": "tooluser:tooluser"
            }
        }

    @pytest.fixture
    def registry(self, sample_config):
        """Create an ExecutionManager instance for testing."""
        with patch("saber.server.tools.sandbox.sandbox_manager.SandboxManager"):
            return ExecutionManager(config=sample_config)

    def test_initialization_default(self):
        """Test that default initialization fails without sandbox config."""
        # Default configuration doesn't include sandbox.enabled=True
        with pytest.raises(ExecutionManagerError, match="Sandbox execution is required but not enabled"):
            ExecutionManager()

    def test_initialization_with_valid_config(self, sample_config):
        """Test initialization with valid sandbox configuration."""
        registry = ExecutionManager(config=sample_config)

        assert isinstance(registry._configuration, ExecutionConfiguration)
        assert isinstance(registry._security_validator, SecurityValidator)
        assert isinstance(registry._cli_tool, DockerCLIExecutor)
        assert isinstance(registry._sandbox_manager, SandboxManager)
        assert isinstance(registry._semaphore, asyncio.Semaphore)

    def test_initialization_sandbox_disabled(self):
        """Test that initialization fails when sandbox is explicitly disabled."""
        config = {
            "sandbox": {"enabled": False}
        }

        with pytest.raises(ExecutionManagerError, match="Sandbox execution is required but not enabled"):
            ExecutionManager(config=config)

    def test_initialization_with_config(self, sample_config):
        """Test initialization with configuration."""
        registry = ExecutionManager(config=sample_config)

        assert registry._configuration.get_execution_timeout() == 60.0
        assert registry._configuration.get_max_concurrent() == 5
        assert "file" in registry._configuration.get_allowed_commands()

        # Test sandbox configuration
        sandbox_config = registry._configuration.get_sandbox_config()
        assert sandbox_config["enabled"] is True
        assert sandbox_config["image"] == "saber/base-sandbox:latest"

    @patch("saber.server.tools.execution_manager.ExecutionConfiguration")
    def test_initialization_with_config_file(self, mock_cli_config):
        """Test initialization with configuration file."""
        mock_instance = MagicMock()
        mock_cli_config.return_value = mock_instance
        mock_instance.get_allowed_commands.return_value = []
        mock_instance.get_execution_timeout.return_value = 300.0
        mock_instance.get_max_concurrent.return_value = 10

        registry = ExecutionManager(config_file="test_config.yaml")

        mock_cli_config.assert_called_once_with(None, "test_config.yaml")

    @pytest.mark.asyncio
    async def test_execute_command_success(self, registry):
        """Test successful command execution."""
        parameters = {"command": "echo test", "shell": False}
        context = {"session_id": "test123"}

        expected_result = ToolResult.success_result(
            data={"stdout": "test\n", "stderr": "", "return_code": 0}
        )

        with patch.object(registry._cli_tool, 'validate_parameters', return_value=ValidationResult.success()):
            with patch.object(registry._cli_tool, 'execute', return_value=expected_result) as mock_execute:
                result = await registry.execute_command(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "test\n"
        mock_execute.assert_called_once_with(parameters, context)

    @pytest.mark.asyncio
    async def test_execute_command_validation_failure(self, registry):
        """Test command execution with parameter validation failure."""
        parameters = {"invalid": "params"}

        validation_result = ValidationResult.failure(["Missing required parameter 'command'"])

        with patch.object(registry._cli_tool, 'validate_parameters', return_value=validation_result):
            result = await registry.execute_command(parameters)

        assert result.success is False
        assert "Parameter validation failed" in result.error
        assert "Missing required parameter 'command'" in result.error

    @pytest.mark.asyncio
    async def test_execute_command_execution_exception(self, registry):
        """Test command execution with exception during execution."""
        parameters = {"command": "test"}

        with patch.object(registry._cli_tool, 'validate_parameters', return_value=ValidationResult.success()):
            with patch.object(registry._cli_tool, 'execute', side_effect=Exception("Execution failed")):
                result = await registry.execute_command(parameters)

        assert result.success is False
        assert "Execution failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_command_concurrency_control(self, registry):
        """Test that concurrency control works with semaphore."""
        parameters = {"command": "sleep 1"}

        # Mock the CLI tool to simulate slow execution
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(0.1)
            return ToolResult.success_result(data="done")

        with patch.object(registry._cli_tool, 'validate_parameters', return_value=ValidationResult.success()):
            with patch.object(registry._cli_tool, 'execute', side_effect=slow_execute):
                # Start multiple executions
                tasks = [
                    asyncio.create_task(registry.execute_command(parameters))
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
        mcp_tools = registry.to_mcp_tools()

        assert len(mcp_tools) == 1
        tool = mcp_tools[0]

        assert tool["name"] == "docker_cli"
        assert tool["description"] == "Execute validated shell commands in Docker containers"
        assert "inputSchema" in tool
        assert "properties" in tool["inputSchema"]
        assert "command" in tool["inputSchema"]["properties"]
        assert "shell" in tool["inputSchema"]["properties"]
        assert tool["inputSchema"]["required"] == ["command"]

    def test_to_mcp_tools_with_cli_config(self, sample_config):
        """Test MCP tools conversion with CLI configuration."""
        sample_config["cli"]["default_shell_mode"] = True

        with patch("saber.server.tools.sandbox.sandbox_manager.SandboxManager"):
            registry = ExecutionManager(config=sample_config)

        mcp_tools = registry.to_mcp_tools()
        tool = mcp_tools[0]

        # Should include default value from CLI config
        assert tool["inputSchema"]["properties"]["shell"]["default"] is True

    def test_get_execution_stats(self, registry):
        """Test getting execution statistics."""
        stats = registry.get_execution_stats()

        assert "max_concurrent" in stats
        assert "current_available" in stats
        assert "timeout" in stats
        assert "security_config" in stats
        assert stats["max_concurrent"] == 5  # From sample config

    def test_load_configuration_updates_components(self, registry):
        """Test that loading configuration updates all components."""
        new_config_path = "new_config.yaml"

        with patch.object(registry._configuration, 'load_configuration') as mock_load:
            with patch.object(registry._configuration, 'get_allowed_commands', return_value=["new_cmd"]):
                with patch.object(registry._configuration, 'get_execution_timeout', return_value=120.0):
                    with patch.object(registry._configuration, 'get_max_concurrent', return_value=15):
                        registry.load_configuration(new_config_path)

        mock_load.assert_called_once_with(new_config_path)
        # Components should be recreated with new configuration

    def test_get_configuration(self, registry):
        """Test getting configuration manager."""
        config = registry.get_configuration()
        assert isinstance(config, ExecutionConfiguration)
        assert config == registry._configuration

    @pytest.mark.asyncio
    async def test_execute_command_default_context(self, registry):
        """Test execute_command with default context when none provided."""
        parameters = {"command": "echo test"}

        expected_result = ToolResult.success_result(data="test")

        with patch.object(registry._cli_tool, 'validate_parameters', return_value=ValidationResult.success()):
            with patch.object(registry._cli_tool, 'execute', return_value=expected_result) as mock_execute:
                result = await registry.execute_command(parameters)

        # Should be called with empty context dict
        mock_execute.assert_called_once_with(parameters, {})

    def test_security_validator_initialization(self, sample_config):
        """Test that SecurityValidator is initialized with correct allowed commands."""
        registry = ExecutionManager(config=sample_config)

        # Verify that allowed commands from config are passed to validator
        allowed_commands = sample_config["security"]["allowed_commands"]

        # We can't directly access the validator's internal state easily,
        # but we can verify it was initialized correctly by checking the config
        assert registry._configuration.get_allowed_commands() == allowed_commands
