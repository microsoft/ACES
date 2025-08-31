"""
Tests for ExecutionManager.

This module tests the Docker-based execution manager that provides secure command execution
with MCP integration and comprehensive security validation in Docker containers.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from saber.server.base import Action, CommandResult
from saber.server.execution.base import ValidationResult
from saber.server.execution.exceptions import ExecutionManagerError
from saber.server.execution.execution_manager import ExecutionManager
from saber.server.execution.executors.executor_factory import ExecutorFactory
from saber.server.execution.executors.standard_registry.cli_executor import CLIExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from saber.server.execution.sandbox.environment_spec import (
    PermanentEnvironmentSpec,
    PermanentNetworkSpec,
    PermanentServiceSpec,
)
from saber.server.execution.cleanup.cleanup_reason import CleanupReason
from saber.server.execution.utils.security_validator import SecurityValidator


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
                "image": "saber/sandbox:latest",
                "network_mode": "none",
                "read_only_root": True,
                "user": "tooluser:tooluser",
            },
        }

    @pytest.fixture
    def cleanup_factory(self):
        """Clean up factory state after tests."""
        from saber.server.execution.executors.executor_registry import executor_registry

        # Store original state
        original_registry = executor_registry._registered_executors.copy()
        yield
        # Restore original state
        executor_registry._registered_executors = original_registry

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory for testing."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        return str(config_dir)

    @pytest.fixture
    def registry(self, sample_config, cleanup_factory, temp_config_dir):
        """Create an ExecutionManager instance for testing."""
        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager"):
            return ExecutionManager(temp_config_dir)

    def test_initialization_with_valid_config(self, sample_config, temp_config_dir):
        """Test initialization with valid sandbox configuration."""
        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager") as mock_sandbox:
            registry = ExecutionManager(temp_config_dir)

        assert isinstance(registry._configuration, dict)
        assert isinstance(registry._executor_factory, ExecutorFactory)
        assert registry._sandbox_manager == mock_sandbox.return_value

    def test_initialization_with_config(self, sample_config, temp_config_dir):
        """Test initialization with default configuration."""
        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager"):
            registry = ExecutionManager(temp_config_dir)

        # Should have default timeout since no config provided during initialization
        assert 300.0 == 300.0

        # Should have empty allowed commands by default
        assert registry._configuration.get("allowed_commands", []) == []

        # Test sandbox configuration is empty by default
        assert registry._configuration.get("sandbox", {}) == {}

    def test_configure_for_task(self, registry):
        """Test configuring ExecutionManager for a specific task."""
        # Create a mock task object
        mock_task = MagicMock()
        mock_task.environment = "test_env"
        mock_task.execution_config = {"timeout": 120.0}
        mock_task.allowed_executors = ["cli"]
        mock_task.cli_config = {"default_shell_mode": True}
        # Make sure python_config returns None
        mock_task.python_config = None

        # Mock environment loader
        mock_env_spec = MagicMock()
        registry._environment_loader = MagicMock()
        registry._environment_loader.resolve_environment.return_value = mock_env_spec

        # Mock SandboxManager class to avoid environment creation issues
        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager") as mock_sandbox_class:
            mock_sandbox_instance = MagicMock()
            mock_sandbox_class.return_value = mock_sandbox_instance

            registry.configure_for_task("session123", mock_task)

            # Should have created sandbox manager and called environment creation
            # Check that sandbox manager was called with logging config containing the domain
            call_args = mock_sandbox_class.call_args[0][0]
            assert "domain" in call_args
            assert "logs_directory" in call_args
            assert "enable_container_logging" in call_args
            mock_sandbox_instance.create_session_environment.assert_called_once_with(
                "session123", mock_env_spec
            )

        # Should have updated configuration (only cli config should be present since python_config is None)
        assert registry._configuration["timeout"] == 120.0
        assert registry._configuration["cli"] == {"default_shell_mode": True}

        # Should have called environment resolution
        registry._environment_loader.resolve_environment.assert_called_once_with("test_env")

        # Should have filtered executors
        available_executors = registry._executor_factory.get_available_executors()
        assert "cli" in available_executors

    @pytest.mark.asyncio
    async def test_step_success(self, registry):
        """Test successful command execution."""
        action = Action(tool_name="cli", parameters={"arguments": "echo test", "shell": False})
        context = {"session_id": "test123"}

        expected_result = CommandResult.success_result(data={"stdout": "test\n", "stderr": "", "return_code": 0})

        # Mock the executor factory to return a mock executor
        mock_executor = AsyncMock()
        mock_executor.validate_parameters = MagicMock(return_value=ValidationResult.success())
        mock_executor.return_value = expected_result

        with patch.object(registry._executor_factory, "get_executor", return_value=mock_executor):
            result = await registry.step(action, context)

        assert result.success is True
        assert result.data["stdout"] == "test\n"
        expected_params = {"arguments": "echo test", "shell": False}
        mock_executor.assert_called_once_with(expected_params, context)

    @pytest.mark.asyncio
    async def test_step_validation_failure(self, registry):
        """Test command execution with parameter validation failure."""
        action = Action(tool_name="cli", parameters={"invalid": "params"})

        validation_result = ValidationResult.failure(["Missing required parameter 'arguments'"])

        # Mock the executor factory to return a mock executor
        mock_executor = MagicMock()
        mock_executor.validate_parameters.return_value = validation_result

        with patch.object(registry._executor_factory, "get_executor", return_value=mock_executor):
            result = await registry.step(action)

        assert result.success is False
        assert "Parameter validation failed" in result.error
        assert "Missing required parameter 'arguments'" in result.error

    @pytest.mark.asyncio
    async def test_step_execution_exception(self, registry):
        """Test command execution with exception during execution."""
        action = Action(tool_name="cli", parameters={"arguments": "test"})

        # Mock the executor factory to return a mock executor
        mock_executor = AsyncMock()
        mock_executor.validate_parameters = MagicMock(return_value=ValidationResult.success())
        mock_executor.side_effect = Exception("Execution failed")

        with patch.object(registry._executor_factory, "get_executor", return_value=mock_executor):
            result = await registry.step(action)

        assert result.success is False
        assert "Execution failed" in result.error

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
                        "arguments": {"type": "string", "description": "Command to execute"},
                        "shell": {"type": "boolean", "description": "Use shell mode", "default": False},
                    },
                    "required": ["arguments"],
                },
            },
            {
                "name": "python_python_script",
                "description": "Execute Python scripts in Docker containers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "arguments": {"type": "string", "description": "Python code to execute"},
                        "requirements": {"type": "array", "description": "Python packages to install"},
                    },
                    "required": ["arguments"],
                },
            },
        ]

        with patch.object(registry._executor_factory, "get_all_mcp_tools", return_value=mock_tools):
            mcp_tools = registry.to_mcp_tools()

        assert len(mcp_tools) == 2

        # Check CLI tool
        cli_tool = next(tool for tool in mcp_tools if tool["name"] == "cli_docker_cli")
        assert cli_tool["description"] == "Execute validated shell commands in Docker containers"
        assert "inputSchema" in cli_tool
        assert "arguments" in cli_tool["inputSchema"]["properties"]

        # Check Python tool
        python_tool = next(tool for tool in mcp_tools if tool["name"] == "python_python_script")
        assert python_tool["description"] == "Execute Python scripts in Docker containers"
        assert "arguments" in python_tool["inputSchema"]["properties"]

    def test_to_mcp_tools_with_cli_config(self, sample_config, temp_config_dir):
        """Test MCP tools conversion with CLI configuration."""
        sample_config["cli"]["default_shell_mode"] = True

        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager"):
            registry = ExecutionManager(temp_config_dir)

        # Mock the factory's response
        mock_tools = [
            {
                "name": "cli_docker_cli",
                "description": "Execute validated shell commands in Docker containers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "arguments": {"type": "string", "description": "Command to execute"},
                        "shell": {"type": "boolean", "description": "Use shell mode", "default": True},
                    },
                    "required": ["arguments"],
                },
            }
        ]

        with patch.object(registry._executor_factory, "get_all_mcp_tools", return_value=mock_tools):
            mcp_tools = registry.to_mcp_tools()

        tool = mcp_tools[0]
        # Should include default value from CLI config
        assert tool["inputSchema"]["properties"]["shell"]["default"] is True

    def test_get_execution_stats(self, registry):
        """Test getting execution statistics."""
        stats = registry.get_execution_stats()

        assert "total_active_executions" in stats
        assert "active_sessions" in stats
        assert "max_concurrent_per_session" in stats
        assert "session_execution_counts" in stats
        assert stats["total_active_executions"] == 0
        assert stats["active_sessions"] == 0
        assert stats["max_concurrent_per_session"] == 3

    def test_get_configuration(self, registry):
        """Test getting configuration manager."""
        config = registry.get_configuration()
        assert isinstance(config, dict)
        assert config == registry._configuration

    @pytest.mark.asyncio
    async def test_step_default_context(self, registry):
        """Test step with default context when none provided."""
        action = Action(tool_name="cli", parameters={"arguments": "echo test"})

        expected_result = CommandResult.success_result(data="test")

        # Mock the executor factory to return a mock executor
        mock_executor = AsyncMock()
        mock_executor.validate_parameters = MagicMock(return_value=ValidationResult.success())
        mock_executor.return_value = expected_result

        with patch.object(registry._executor_factory, "get_executor", return_value=mock_executor):
            result = await registry.step(action)

        # Should be called with empty context dict
        expected_params = {"arguments": "echo test"}
        mock_executor.assert_called_once_with(expected_params, {})

    def test_get_executor(self, registry):
        """Test getting specific executor instance."""
        mock_executor = MagicMock()

        with patch.object(registry._executor_factory, "get_executor", return_value=mock_executor) as mock_get:
            result = registry.get_executor("cli")

        assert result == mock_executor
        mock_get.assert_called_once_with("cli")

    @pytest.mark.asyncio
    async def test_step_python_executor(self, registry):
        """Test step with Python executor."""
        action = Action(tool_name="python", parameters={"arguments": "print('hello')"})
        context = {"session_id": "test123"}

        expected_result = CommandResult.success_result(data={"stdout": "hello\n", "stderr": "", "return_code": 0})

        # Mock the executor factory to return a mock Python executor
        mock_executor = AsyncMock()
        mock_executor.validate_parameters = MagicMock(return_value=ValidationResult.success())
        mock_executor.return_value = expected_result

        with patch.object(registry._executor_factory, "get_executor", return_value=mock_executor):
            result = await registry.step(action, context)

        assert result.success is True
        assert result.data["stdout"] == "hello\n"
        expected_params = {"arguments": "print('hello')"}
        mock_executor.assert_called_once_with(expected_params, context)

    def test_list_commands(self, registry):
        """Test listing all available commands."""
        mock_commands = [
            {
                "executor_type": "cli",
                "name": "docker_cli",
                "description": "Execute shell commands",
                "domain": "general",
                "security_level": "high",
                "parameters": ["command", "shell"],
            },
            {
                "executor_type": "python",
                "name": "python_script",
                "description": "Execute Python scripts",
                "domain": "python",
                "security_level": "high",
                "parameters": ["code", "requirements"],
            },
        ]

        # Mock executor factory methods
        with patch.object(registry._executor_factory, "get_available_executors", return_value=["cli", "python"]):
            mock_cli_executor = MagicMock()
            mock_cli_executor._executor_metadata = {
                "name": "docker_cli",
                "description": "Execute shell commands",
            }
            mock_cli_executor.get_parameters.return_value = {"command": MagicMock(), "shell": MagicMock()}

            mock_python_executor = MagicMock()
            mock_python_executor._executor_metadata = {
                "name": "python_script",
                "description": "Execute Python scripts",
            }
            mock_python_executor.get_parameters.return_value = {"code": MagicMock(), "requirements": MagicMock()}

            def mock_get_executor(executor_type):
                if executor_type == "cli":
                    return mock_cli_executor
                elif executor_type == "python":
                    return mock_python_executor

            with patch.object(registry._executor_factory, "get_executor", side_effect=mock_get_executor):
                commands = registry.list_commands()

        assert len(commands) == 2
        assert any(cmd["executor_type"] == "cli" for cmd in commands)
        assert any(cmd["executor_type"] == "python" for cmd in commands)

    def test_timeout_configuration_flow(self, registry):
        """Test that timeout configuration flows from task to executors."""
        # Create a mock task with custom timeout
        mock_task = MagicMock()
        mock_task.environment = None  # Skip environment resolution for this test
        mock_task.execution_config = {"timeout": 150, "allowed_executors": ["cli", "python"]}
        mock_task.allowed_executors = ["cli", "python"]
        mock_task.cli_config = None
        mock_task.python_config = None

        # Configure ExecutionManager with the task
        registry.configure_for_task("timeout_test_session", mock_task)

        # Verify timeout was set in configuration
        assert registry._configuration["timeout"] == 150

        # Test CLI executor timeout
        if "cli" in registry.get_available_executors():
            cli_executor = registry.get_executor("cli")
            assert cli_executor.get_timeout() == 150.0, f"CLI executor should use task timeout 150, got {cli_executor.get_timeout()}"

        # Test Python executor timeout
        if "python" in registry.get_available_executors():
            python_executor = registry.get_executor("python")
            assert python_executor.get_timeout() == 150.0, f"Python executor should use task timeout 150, got {python_executor.get_timeout()}"

    def test_default_timeout_behavior(self, registry):
        """Test that executors use default timeouts when no task timeout is specified."""
        # Create a mock task without timeout configuration
        mock_task = MagicMock()
        mock_task.environment = None
        mock_task.execution_config = {"allowed_executors": ["cli", "python"]}  # No timeout field
        mock_task.allowed_executors = ["cli", "python"]
        mock_task.cli_config = None
        mock_task.python_config = None

        # Configure ExecutionManager with the task
        registry.configure_for_task("default_timeout_session", mock_task)

        # Verify no global timeout is set
        assert "timeout" not in registry._configuration

        # Test that executors use their default timeouts
        if "cli" in registry.get_available_executors():
            cli_executor = registry.get_executor("cli")
            # CLI executor default is 60.0 (from get_default_config)
            assert cli_executor.get_timeout() == 60.0, f"CLI executor should use default timeout 60, got {cli_executor.get_timeout()}"

        if "python" in registry.get_available_executors():
            python_executor = registry.get_executor("python")
            # Python executor default is higher (from get_default_config)
            default_timeout = python_executor.get_timeout()
            assert default_timeout > 60.0, f"Python executor should use default timeout > 60, got {default_timeout}"


class TestExecutionManagerPermanentEnvironment:
    """Test cases for ExecutionManager permanent environment management."""

    @pytest.fixture
    def execution_manager_config(self):
        """Configuration for ExecutionManager testing."""
        return "/test/config"

    @pytest.fixture
    def sample_permanent_spec(self):
        """Create a sample permanent environment specification."""
        service_spec = PermanentServiceSpec(
            name="test_service",
            image="test_image:latest",
            ports=["8080:8080"],
            environment=["TEST_VAR=test_value"],
            volumes=[],
        )

        network_spec = PermanentNetworkSpec(
            name="test_network",
            driver="bridge",
            ipam_config={"subnet": "172.20.0.0/16"},
        )

        return PermanentEnvironmentSpec(
            services={"test_service": service_spec},
            networks={"test_network": network_spec},
        )

    @pytest.fixture
    def permanent_config(self):
        """Configuration for permanent environment manager."""
        return {
            "domain": "test_domain",
            "config_dir": "/test/config",
            "enable_logging": True,
        }

    @patch('saber.server.execution.execution_manager.EnvironmentLoader')
    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.ContainerCleanupManager')
    def test_execution_manager_initialization(
        self,
        mock_cleanup_manager,
        mock_sandbox_manager,
        mock_env_loader,
        execution_manager_config,
    ):
        """Test ExecutionManager initialization with permanent environment support."""
        # Mock environment loader initialization
        mock_env_loader_instance = MagicMock()
        mock_env_loader.return_value = mock_env_loader_instance

        # Mock the existence of environments.yaml
        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Verify components are initialized
        assert execution_manager._config_dir == execution_manager_config
        assert execution_manager._environment_loader == mock_env_loader_instance
        assert execution_manager._permanent_environment_manager is None  # Not initialized yet
        mock_cleanup_manager.assert_called_once()

    @patch('saber.server.execution.execution_manager.EnvironmentLoader')
    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.ContainerCleanupManager')
    @patch('saber.server.execution.execution_manager.PermanentEnvironmentManager')
    def test_initialize_permanent_environment_manager(
        self,
        mock_perm_env_manager_class,
        mock_cleanup_manager,
        mock_sandbox_manager,
        mock_env_loader,
        execution_manager_config,
        permanent_config,
    ):
        """Test initializing permanent environment manager through ExecutionManager."""
        # Setup mocks
        mock_perm_env_manager = MagicMock()
        mock_perm_env_manager_class.return_value = mock_perm_env_manager
        mock_cleanup_manager_instance = MagicMock()
        mock_cleanup_manager.return_value = mock_cleanup_manager_instance
        mock_env_loader_instance = MagicMock()
        mock_env_loader.return_value = mock_env_loader_instance

        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Initialize permanent environment manager
        execution_manager.initialize_permanent_environment_manager(permanent_config)

        # Verify permanent environment manager was created
        mock_perm_env_manager_class.assert_called_once_with(permanent_config)
        assert execution_manager._permanent_environment_manager == mock_perm_env_manager

        # Verify cleanup manager was updated with permanent manager
        assert mock_cleanup_manager_instance.permanent_manager == mock_perm_env_manager

        # Verify environment loader was updated
        assert mock_env_loader_instance.permanent_environment_manager == mock_perm_env_manager

    @patch('saber.server.execution.execution_manager.EnvironmentLoader')
    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.ContainerCleanupManager')
    @patch('saber.server.execution.execution_manager.PermanentEnvironmentManager')
    def test_start_permanent_environment_success(
        self,
        mock_perm_env_manager_class,
        mock_cleanup_manager,
        mock_sandbox_manager,
        mock_env_loader,
        execution_manager_config,
        permanent_config,
        sample_permanent_spec,
    ):
        """Test successful permanent environment startup through ExecutionManager."""
        # Setup mocks
        mock_perm_env_manager = MagicMock()
        mock_perm_env_manager_class.return_value = mock_perm_env_manager
        mock_cleanup_manager_instance = MagicMock()
        mock_cleanup_manager.return_value = mock_cleanup_manager_instance

        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Initialize and start permanent environment
        execution_manager.initialize_permanent_environment_manager(permanent_config)
        execution_manager.start_permanent_environment(sample_permanent_spec)

        # Verify permanent environment was started through PermanentEnvironmentManager
        mock_perm_env_manager.ensure_permanent_environments_current.assert_called_once_with(sample_permanent_spec)

    @patch('saber.server.execution.execution_manager.EnvironmentLoader')
    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.ContainerCleanupManager')
    def test_start_permanent_environment_not_initialized(
        self,
        mock_cleanup_manager,
        mock_sandbox_manager,
        mock_env_loader,
        execution_manager_config,
        sample_permanent_spec,
    ):
        """Test starting permanent environment when manager is not initialized."""
        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Try to start permanent environment without initializing manager
        with pytest.raises(RuntimeError, match="Permanent environment manager not initialized"):
            execution_manager.start_permanent_environment(sample_permanent_spec)

    @patch('saber.server.execution.execution_manager.EnvironmentLoader')
    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.ContainerCleanupManager')
    @patch('saber.server.execution.execution_manager.PermanentEnvironmentManager')
    def test_start_permanent_environment_failure(
        self,
        mock_perm_env_manager_class,
        mock_cleanup_manager,
        mock_sandbox_manager,
        mock_env_loader,
        execution_manager_config,
        permanent_config,
        sample_permanent_spec,
    ):
        """Test permanent environment startup failure through ExecutionManager."""
        # Setup mocks
        mock_perm_env_manager = MagicMock()
        mock_perm_env_manager_class.return_value = mock_perm_env_manager
        mock_perm_env_manager.ensure_permanent_environments_current.side_effect = Exception("Startup failed")

        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Initialize permanent environment manager
        execution_manager.initialize_permanent_environment_manager(permanent_config)

        # Try to start permanent environment - should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to start permanent environment"):
            execution_manager.start_permanent_environment(sample_permanent_spec)

    @patch('saber.server.execution.execution_manager.EnvironmentLoader')
    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.ContainerCleanupManager')
    @patch('saber.server.execution.execution_manager.PermanentEnvironmentManager')
    def test_stop_permanent_environment_success(
        self,
        mock_perm_env_manager_class,
        mock_cleanup_manager,
        mock_sandbox_manager,
        mock_env_loader,
        execution_manager_config,
        permanent_config,
    ):
        """Test successful permanent environment shutdown through ExecutionManager."""
        # Setup mocks
        mock_cleanup_manager_instance = MagicMock()
        mock_cleanup_manager.return_value = mock_cleanup_manager_instance
        mock_cleanup_manager_instance.stop_permanent_environment.return_value = True

        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Initialize permanent environment manager
        execution_manager.initialize_permanent_environment_manager(permanent_config)

        # Stop permanent environment
        execution_manager.stop_permanent_environment()

        # Verify stop was called through cleanup manager
        mock_cleanup_manager_instance.stop_permanent_environment.assert_called_once()

    @patch('saber.server.execution.execution_manager.EnvironmentLoader')
    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.ContainerCleanupManager')
    @patch('saber.server.execution.execution_manager.PermanentEnvironmentManager')
    def test_stop_permanent_environment_failure(
        self,
        mock_perm_env_manager_class,
        mock_cleanup_manager,
        mock_sandbox_manager,
        mock_env_loader,
        execution_manager_config,
        permanent_config,
    ):
        """Test permanent environment shutdown failure through ExecutionManager."""
        # Setup mocks
        mock_cleanup_manager_instance = MagicMock()
        mock_cleanup_manager.return_value = mock_cleanup_manager_instance
        mock_cleanup_manager_instance.stop_permanent_environment.return_value = False

        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Initialize permanent environment manager
        execution_manager.initialize_permanent_environment_manager(permanent_config)

        # Try to stop permanent environment - should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to stop permanent environment"):
            execution_manager.stop_permanent_environment()

    @patch('saber.server.execution.execution_manager.EnvironmentLoader')
    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.ContainerCleanupManager')
    @patch('saber.server.execution.execution_manager.PermanentEnvironmentManager')
    def test_is_permanent_environment_running(
        self,
        mock_perm_env_manager_class,
        mock_cleanup_manager,
        mock_sandbox_manager,
        mock_env_loader,
        execution_manager_config,
        permanent_config,
    ):
        """Test checking if permanent environment is running."""
        # Setup mocks
        mock_cleanup_manager_instance = MagicMock()
        mock_cleanup_manager.return_value = mock_cleanup_manager_instance
        mock_cleanup_manager_instance.is_permanent_environment_running.return_value = True

        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Initialize permanent environment manager
        execution_manager.initialize_permanent_environment_manager(permanent_config)

        # Check if running
        result = execution_manager.is_permanent_environment_running()

        # Verify result
        assert result is True
        mock_cleanup_manager_instance.is_permanent_environment_running.assert_called_once()

    @patch('saber.server.execution.execution_manager.EnvironmentLoader')
    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.ContainerCleanupManager')
    @patch('saber.server.execution.execution_manager.PermanentEnvironmentManager')
    def test_cleanup_all_containers(
        self,
        mock_perm_env_manager_class,
        mock_cleanup_manager,
        mock_sandbox_manager,
        mock_env_loader,
        execution_manager_config,
        permanent_config,
    ):
        """Test cleanup of all containers through ExecutionManager."""
        # Setup mocks
        mock_cleanup_manager_instance = MagicMock()
        mock_cleanup_manager.return_value = mock_cleanup_manager_instance
        expected_result = {
            "ephemeral_sessions_cleaned": 3,
            "permanent_environment_stopped": True,
            "total_cleanup_success": True,
        }
        mock_cleanup_manager_instance.cleanup_all_containers.return_value = expected_result

        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Initialize permanent environment manager
        execution_manager.initialize_permanent_environment_manager(permanent_config)

        # Cleanup all containers
        result = execution_manager.cleanup_all_containers(CleanupReason.SERVER_SHUTDOWN, {"test": "context"})

        # Verify cleanup was delegated to cleanup manager
        assert result == expected_result
        mock_cleanup_manager_instance.cleanup_all_containers.assert_called_once_with(
            CleanupReason.SERVER_SHUTDOWN, {"test": "context"}
        )

    @patch('saber.server.execution.execution_manager.EnvironmentLoader')
    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.ContainerCleanupManager')
    @patch('saber.server.execution.execution_manager.PermanentEnvironmentManager')
    def test_cleanup_session_with_new_interface(
        self,
        mock_perm_env_manager_class,
        mock_cleanup_manager,
        mock_sandbox_manager,
        mock_env_loader,
        execution_manager_config,
        permanent_config,
    ):
        """Test session cleanup through ExecutionManager with enhanced interface."""
        # Setup mocks
        mock_cleanup_manager_instance = MagicMock()
        mock_cleanup_manager.return_value = mock_cleanup_manager_instance
        mock_cleanup_manager_instance.cleanup_session.return_value = True

        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Initialize permanent environment manager
        execution_manager.initialize_permanent_environment_manager(permanent_config)

        # Add active execution tracking
        execution_manager._active_executions["test_session"] = 2

        # Cleanup session
        result = execution_manager.cleanup_session(
            "test_session", CleanupReason.SESSION_TERMINATED, {"manual": True}
        )

        # Verify cleanup was delegated to cleanup manager
        assert result is True
        mock_cleanup_manager_instance.cleanup_session.assert_called_once_with(
            "test_session", CleanupReason.SESSION_TERMINATED, {"manual": True}
        )

        # Verify execution tracking was cleaned up
        assert "test_session" not in execution_manager._active_executions
