"""
Tests for ExecutorFactory.

This module tests the executor factory that manages different types of command executors
with dynamic selection and creation.
"""

import pytest
from unittest.mock import MagicMock, patch

from saber.server.execution.executors.factory import ExecutorFactory
from saber.server.execution.executors.cli import CLIExecutor
from saber.server.execution.executors.curl_executor import CurlExecutor
from saber.server.execution.executors.docker_executor import DockerExecutor
from saber.server.execution.executors.file_io_executor import FileIoExecutor
from saber.server.execution.executors.python_executor import PythonExecutor
from saber.server.execution.executors.sql_executor import SqlExecutor
from saber.server.execution.sandbox.sandbox_manager import SandboxManager


class TestExecutorFactory:
    """Test cases for ExecutorFactory."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxManager."""
        manager = MagicMock(spec=SandboxManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/base-sandbox:latest",
            "network_mode": "none"
        }
        return manager

    @pytest.fixture
    def executor_factory(self, mock_sandbox_manager):
        """Create an ExecutorFactory instance for testing."""
        return ExecutorFactory(sandbox_manager=mock_sandbox_manager)

    def test_initialization(self, mock_sandbox_manager):
        """Test factory initialization."""
        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager)

        assert factory._sandbox_manager == mock_sandbox_manager
        assert len(factory._executor_instances) == 0
        assert "cli" in factory.get_available_executors()
        assert "python" in factory.get_available_executors()
        assert "curl" in factory.get_available_executors()
        assert "sql" in factory.get_available_executors()
        assert "file_io" in factory.get_available_executors()

    def test_initialization_with_config(self, mock_sandbox_manager):
        """Test factory initialization with configuration."""
        from saber.server.execution.execution_manager import ExecutionConfiguration

        config_dict = {
            "executors": {
                "common": {"timeout": 600},
                "python": {"allowed_modules": ["requests", "json"]}
            }
        }
        configuration = ExecutionConfiguration(config=config_dict)

        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager, configuration=configuration)
        assert factory._configuration == configuration

    def test_get_available_executors(self, executor_factory):
        """Test getting list of available executor types."""
        executors = executor_factory.get_available_executors()

        assert isinstance(executors, list)
        assert "cli" in executors
        assert "python" in executors
        assert "curl" in executors
        assert "sql" in executors
        assert "file_io" in executors
        assert len(executors) >= 5

    def test_register_executor(self):
        """Test registering a new executor type."""
        class CustomExecutor(DockerExecutor):
            async def execute(self, parameters, context):
                pass

        initial_count = len(ExecutorFactory._executor_registry)
        ExecutorFactory.register_executor("custom", CustomExecutor)

        assert "custom" in ExecutorFactory._executor_registry
        assert len(ExecutorFactory._executor_registry) == initial_count + 1
        assert ExecutorFactory._executor_registry["custom"] == CustomExecutor

        # Cleanup
        ExecutorFactory.unregister_executor("custom")

    def test_register_executor_invalid_class(self):
        """Test registering an invalid executor class."""
        class InvalidExecutor:
            pass

        with pytest.raises(ValueError, match="must inherit from DockerExecutor"):
            ExecutorFactory.register_executor("invalid", InvalidExecutor)

    def test_unregister_executor(self):
        """Test unregistering an executor type."""
        # Register a temporary executor
        class TempExecutor(DockerExecutor):
            async def execute(self, parameters, context):
                pass

        ExecutorFactory.register_executor("temp", TempExecutor)
        assert "temp" in ExecutorFactory._executor_registry

        # Unregister it
        ExecutorFactory.unregister_executor("temp")
        assert "temp" not in ExecutorFactory._executor_registry

    def test_get_executor_cli(self, executor_factory):
        """Test getting CLI executor."""
        executor = executor_factory.get_executor("cli")

        assert isinstance(executor, CLIExecutor)
        assert executor == executor_factory._executor_instances["cli"]

    def test_get_executor_python(self, executor_factory):
        """Test getting Python executor."""
        executor = executor_factory.get_executor("python")

        assert isinstance(executor, PythonExecutor)
        assert executor == executor_factory._executor_instances["python"]

    def test_get_executor_cached(self, executor_factory):
        """Test that executors are cached and reused."""
        executor1 = executor_factory.get_executor("cli")
        executor2 = executor_factory.get_executor("cli")

        assert executor1 is executor2
        assert len(executor_factory._executor_instances) == 1

    def test_get_executor_force_new(self, executor_factory):
        """Test creating new executor instance with force_new=True."""
        executor1 = executor_factory.get_executor("cli")
        executor2 = executor_factory.get_executor("cli", force_new=True)

        assert executor1 is not executor2
        assert isinstance(executor1, CLIExecutor)
        assert isinstance(executor2, CLIExecutor)

    def test_get_executor_unknown_type(self, executor_factory):
        """Test getting unknown executor type."""
        with pytest.raises(ValueError, match="Unknown executor type: unknown"):
            executor_factory.get_executor("unknown")

    def test_default_executor_configuration(self, executor_factory):
        """Test that executors get default configuration when no specific config provided."""
        # With no configuration, the factory should use built-in defaults
        cli_executor = executor_factory.get_executor("cli")
        python_executor = executor_factory.get_executor("python")

        # Verify executors were created successfully with defaults
        assert isinstance(cli_executor, CLIExecutor)
        assert isinstance(python_executor, PythonExecutor)

    def test_executor_config_extraction(self, mock_sandbox_manager):
        """Test executor configuration extraction with the new configuration system."""
        from saber.server.execution.execution_manager import ExecutionConfiguration

        config_dict = {
            "common": {"timeout": 600},
            "cli": {"default_shell_mode": True}
        }
        configuration = ExecutionConfiguration(config=config_dict)

        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager, configuration=configuration)

        # Test extracting CLI config
        cli_config = configuration.get_section("cli")
        assert cli_config["default_shell_mode"] is True

        # Test extracting common config
        common_config = configuration.get_section("common")
        assert common_config["timeout"] == 600

    def test_analyze_command_python_indicators(self, executor_factory):
        """Test command analysis for Python indicators."""
        python_commands = [
            "python script.py",
            "python3 -c 'print(hello)'",
            "pip install requests",
            "pytest tests/",
            "jupyter notebook"
        ]

        for command in python_commands:
            result = executor_factory._analyze_command(command)
            assert result == "python", f"Command '{command}' should be detected as python"

    def test_analyze_command_cli_default(self, executor_factory):
        """Test command analysis defaults to CLI."""
        cli_commands = [
            "ls -la",
            "grep pattern file.txt",
            "curl http://example.com",
            "docker ps"
        ]

        for command in cli_commands:
            result = executor_factory._analyze_command(command)
            assert result == "cli", f"Command '{command}' should default to cli"

    def test_create_executor_for_command_explicit_type(self, executor_factory):
        """Test creating executor with explicit action type."""
        executor = executor_factory.create_executor_for_command("any command", action_type="python")
        assert isinstance(executor, PythonExecutor)

    def test_create_executor_for_command_analysis(self, executor_factory):
        """Test creating executor based on command analysis."""
        # Python command
        executor = executor_factory.create_executor_for_command("python3 script.py")
        assert isinstance(executor, PythonExecutor)

        # CLI command
        executor = executor_factory.create_executor_for_command("ls -la")
        assert isinstance(executor, CLIExecutor)

    def test_get_all_mcp_tools(self, executor_factory):
        """Test getting MCP tools for all executors."""
        with patch.object(executor_factory, 'get_executor') as mock_get_executor:
            # Mock CLI executor
            mock_cli = MagicMock()
            mock_cli._security_command_metadata = {
                'name': 'docker_cli',
                'description': 'Execute shell commands'
            }
            mock_cli.to_mcp_schema.return_value = {"type": "object", "properties": {"command": {"type": "string"}}}

            # Mock Python executor
            mock_python = MagicMock()
            mock_python._security_command_metadata = {
                'name': 'python_script',
                'description': 'Execute Python scripts'
            }
            mock_python.to_mcp_schema.return_value = {"type": "object", "properties": {"code": {"type": "string"}}}

            def mock_get_executor_side_effect(executor_type):
                if executor_type == "cli":
                    return mock_cli
                elif executor_type == "python":
                    return mock_python
                else:
                    raise ValueError(f"Unknown type: {executor_type}")

            mock_get_executor.side_effect = mock_get_executor_side_effect

            tools = executor_factory.get_all_mcp_tools()

            assert len(tools) == 2

            # Check CLI tool
            cli_tool = next(tool for tool in tools if "cli" in tool["name"])
            assert cli_tool["description"] == "Execute shell commands"
            assert "command" in cli_tool["inputSchema"]["properties"]

            # Check Python tool
            python_tool = next(tool for tool in tools if "python" in tool["name"])
            assert python_tool["description"] == "Execute Python scripts"
            assert "code" in python_tool["inputSchema"]["properties"]

    def test_get_all_mcp_tools_with_error(self, executor_factory):
        """Test getting MCP tools when one executor fails."""
        with patch.object(executor_factory, 'get_executor') as mock_get_executor:
            def mock_get_executor_side_effect(executor_type):
                if executor_type == "cli":
                    mock_cli = MagicMock()
                    mock_cli._security_command_metadata = {'name': 'docker_cli', 'description': 'Execute shell commands'}
                    mock_cli.to_mcp_schema.return_value = {"type": "object"}
                    return mock_cli
                else:
                    raise Exception("Executor failed")

            mock_get_executor.side_effect = mock_get_executor_side_effect

            tools = executor_factory.get_all_mcp_tools()

            # Should return tools for successful executors only
            assert len(tools) >= 1
            assert any("cli" in tool["name"] for tool in tools)

    def test_cleanup_all_executors(self, executor_factory):
        """Test cleanup of all executor instances."""
        # Create some executor instances
        executor_factory.get_executor("cli")
        executor_factory.get_executor("python")

        assert len(executor_factory._executor_instances) == 2

        # Cleanup
        executor_factory.cleanup_all_executors()

        assert len(executor_factory._executor_instances) == 0

    def test_get_executor_info(self, executor_factory):
        """Test getting executor information."""
        # Create one instance to test active instances
        executor_factory.get_executor("cli")

        info = executor_factory.get_executor_info()

        assert "available_types" in info
        assert "active_instances" in info
        assert "registry_size" in info
        assert "configurations" in info

        assert "cli" in info["available_types"]
        assert "python" in info["available_types"]
        assert "cli" in info["active_instances"]
        assert info["registry_size"] >= 2
        assert "cli" in info["configurations"]
        assert "python" in info["configurations"]

    def test_executor_configuration_inheritance(self, mock_sandbox_manager):
        """Test that executor configuration properly inherits from configuration sections."""
        from saber.server.execution.execution_manager import ExecutionConfiguration

        config_dict = {
            "common": {
                "timeout": 900,
                "max_retries": 3
            },
            "python": {
                "allowed_modules": ["requests", "numpy"],
                "timeout": 1200  # Override common timeout for Python
            }
        }
        configuration = ExecutionConfiguration(config=config_dict)

        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager, configuration=configuration)

        # Test CLI config (should get common values only)
        cli_config = configuration.get_section("cli")  # Empty since no CLI section
        common_config = configuration.get_section("common")
        assert common_config["timeout"] == 900
        assert common_config["max_retries"] == 3
        assert "allowed_modules" not in common_config

        # Test Python config (should have specific values)
        python_config = configuration.get_section("python")
        assert python_config["allowed_modules"] == ["requests", "numpy"]
        assert python_config["timeout"] == 1200  # Should override common timeout

    def test_multiple_executor_instances(self, executor_factory):
        """Test that multiple different executors can be created and managed."""
        cli_executor = executor_factory.get_executor("cli")
        python_executor = executor_factory.get_executor("python")

        assert isinstance(cli_executor, CLIExecutor)
        assert isinstance(python_executor, PythonExecutor)
        assert cli_executor is not python_executor
        assert len(executor_factory._executor_instances) == 2
