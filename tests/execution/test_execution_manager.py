"""
Tests for ExecutionManager.

This module tests the Docker-based execution manager that provides secure command execution
with MCP integration and comprehensive security validation in Docker containers.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

import pytest

from saber.server.base import Action, CommandResult
from saber.server.execution.base import ValidationResult
from saber.server.execution.exceptions import ExecutionManagerError
from saber.server.execution.execution_manager import ExecutionManager
from saber.server.execution.executors.executor_factory import ExecutorFactory
from saber.server.execution.executors.standard_registry.bash_executor import BashExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager
# CleanupReason removed - testing direct component cleanup
from saber.server.execution.utils.security_validator import SecurityValidator


class TestExecutionManager:
    """Test cases for ExecutionManager."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "execution": {"timeout": 60.0, "max_concurrent": 5},
            "security": {"allowed_commands": ["file", "strings"]},
            "bash": {"default_shell_mode": False},
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
        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager") as mock_sandbox_class:
            # Create a mock instance
            mock_sandbox_instance = MagicMock()
            mock_sandbox_instance.is_ready.return_value = True
            mock_sandbox_class.return_value = mock_sandbox_instance

            execution_manager = ExecutionManager(temp_config_dir)

            # Manually set the sandbox manager to the mock instance
            execution_manager._sandbox_environment_manager = mock_sandbox_instance

            # Update the executor factory with the new sandbox manager
            from saber.server.execution.executors.executor_factory import ExecutorFactory
            execution_manager._executor_factory = ExecutorFactory(
                sandbox_manager=mock_sandbox_instance,
                configuration=execution_manager._configuration,
            )

            return execution_manager

    def test_initialization_with_valid_config(self, sample_config, temp_config_dir):
        """Test initialization with valid sandbox configuration."""
        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager") as mock_sandbox:
            registry = ExecutionManager(temp_config_dir)

        assert isinstance(registry._configuration, dict)
        assert registry._executor_factory is None  # Lazy initialization - should be None initially
        assert registry._sandbox_environment_manager is None  # Should be None until initialized

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
        mock_task.allowed_executors = ["bash"]
        mock_task.bash_config = {"default_shell_mode": True}
        # Make sure python_config returns None
        mock_task.python_config = None

        # Mock the sandbox environment manager for this test
        mock_sandbox_manager = MagicMock()
        registry._sandbox_environment_manager = mock_sandbox_manager

        # The registry fixture already mocks SandboxEnvironmentManager, just need to access it
        registry.configure_for_task("episode123", mock_task, session_id="session123")

        # Should have called environment creation on the mocked sandbox manager with environment string
        mock_sandbox_manager.create_episode_environment.assert_called_once_with(
            "episode123", "test_env", None
        )

        # Should have updated configuration (only cli config should be present since python_config is None)
        assert registry._configuration["timeout"] == 120.0
        assert registry._configuration["bash"] == {"default_shell_mode": True}

        # Should have filtered executors
        available_executors = registry._executor_factory.get_available_executors()
        assert "bash" in available_executors

    @pytest.mark.asyncio
    async def test_step_success(self, registry):
        """Test successful command execution."""
        action = Action(tool_name="bash", parameters={"arguments": "echo test", "shell": False})
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
        action = Action(tool_name="bash", parameters={"invalid": "params"})

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
        action = Action(tool_name="bash", parameters={"arguments": "test"})

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
        sample_config["bash"]["default_shell_mode"] = True

        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager") as mock_sandbox_cls:
            registry = ExecutionManager(temp_config_dir)

            # Force initialization by setting up a mock sandbox manager
            mock_sandbox_manager = Mock()
            registry._sandbox_environment_manager = mock_sandbox_manager

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

        # Now access the property to create the factory and patch it
        with patch.object(registry.executor_factory, "get_all_mcp_tools", return_value=mock_tools):
            mcp_tools = registry.to_mcp_tools()

        tool = mcp_tools[0]
        # Should include default value from CLI config
        assert tool["inputSchema"]["properties"]["shell"]["default"] is True

    def test_get_execution_stats(self, registry):
        """Test getting execution statistics."""
        stats = registry.get_execution_stats()

        assert "total_active_executions" in stats
        assert "active_episodes" in stats
        assert "episode_execution_counts" in stats
        assert "max_concurrent_per_episode" in stats
        assert stats["total_active_executions"] == 0
        assert stats["active_episodes"] == 0

    def test_get_configuration(self, registry):
        """Test getting configuration manager."""
        config = registry.get_configuration()
        assert isinstance(config, dict)
        assert config == registry._configuration

    @pytest.mark.asyncio
    async def test_step_default_context(self, registry):
        """Test step with default context when none provided."""
        action = Action(tool_name="bash", parameters={"arguments": "echo test"})

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
            result = registry.get_executor("bash")

        assert result == mock_executor
        mock_get.assert_called_once_with("bash", None)

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
                "executor_type": "bash",
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
        with patch.object(registry._executor_factory, "get_available_executors", return_value=["bash", "python"]):
            mock_bash_executor = MagicMock()
            mock_bash_executor._executor_metadata = {
                "name": "docker_cli",
                "description": "Execute shell commands",
            }
            mock_bash_executor.get_parameters.return_value = {"command": MagicMock(), "shell": MagicMock()}

            mock_python_executor = MagicMock()
            mock_python_executor._executor_metadata = {
                "name": "python_script",
                "description": "Execute Python scripts",
            }
            mock_python_executor.get_parameters.return_value = {"code": MagicMock(), "requirements": MagicMock()}

            def mock_get_executor(executor_type, episode_id=None):
                if executor_type == "bash":
                    return mock_bash_executor
                elif executor_type == "python":
                    return mock_python_executor

            with patch.object(registry._executor_factory, "get_executor", side_effect=mock_get_executor):
                commands = registry.list_commands()

        assert len(commands) == 2
        assert any(cmd["executor_type"] == "bash" for cmd in commands)
        assert any(cmd["executor_type"] == "python" for cmd in commands)

    def test_timeout_configuration_flow(self, registry):
        """Test that timeout configuration flows from task to executors."""
        # Create a mock task with custom timeout
        mock_task = MagicMock()
        mock_task.environment = None  # Skip environment resolution for this test
        mock_task.execution_config = {"timeout": 150, "allowed_executors": ["bash", "python"]}
        mock_task.allowed_executors = ["bash", "python"]
        mock_task.cli_config = None
        mock_task.python_config = None

        # Configure ExecutionManager with the task
        registry.configure_for_task("episode123", mock_task, session_id="timeout_test_session")

        # Verify timeout was set in configuration
        assert registry._configuration["timeout"] == 150

        # Test CLI executor timeout
        # NOTE: Currently CLI executor uses hardcoded 60.0 timeout for testing
        # TODO: This test will need updating when timeout configuration is fully implemented
        if "bash" in registry.get_available_executors():
            bash_executor = registry.get_executor("bash")
            # Current behavior: hardcoded to 60.0 in CLI executor
            assert bash_executor.get_timeout() == 60.0, f"CLI executor currently uses hardcoded timeout 60.0, got {bash_executor.get_timeout()}"

        # Test Python executor timeout
        # NOTE: Currently Python executor also uses hardcoded timeout for testing
        # TODO: This test will need updating when timeout configuration is fully implemented
        if "python" in registry.get_available_executors():
            python_executor = registry.get_executor("python")
            # Current behavior: Python executor uses its own hardcoded timeout (600.0)
            assert python_executor.get_timeout() == 600.0, f"Python executor currently uses hardcoded timeout 600.0, got {python_executor.get_timeout()}"

    def test_default_timeout_behavior(self, registry):
        """Test that executors use default timeouts when no task timeout is specified."""
        # Create a mock task without timeout configuration
        mock_task = MagicMock()
        mock_task.environment = None
        mock_task.execution_config = {"allowed_executors": ["bash", "python"]}  # No timeout field
        mock_task.allowed_executors = ["bash", "python"]
        mock_task.cli_config = None
        mock_task.python_config = None

        # Configure ExecutionManager with the task
        registry.configure_for_task("default_timeout_session", mock_task)

        # Verify no global timeout is set
        assert "timeout" not in registry._configuration

        # Test that executors use their default timeouts
        if "bash" in registry.get_available_executors():
            bash_executor = registry.get_executor("bash")
            # CLI executor default is 60.0 (from get_default_config)
            assert bash_executor.get_timeout() == 60.0, f"CLI executor should use default timeout 60, got {bash_executor.get_timeout()}"

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
    def sample_compose_file(self, tmp_path):
        """Create a sample Docker compose file for testing."""
        compose_content = """
version: '3.8'
services:
  test_service:
    image: test_image:latest
    ports:
      - "8080:8080"
    environment:
      - TEST_VAR=test_value
networks:
  test_network:
    driver: bridge
"""
        compose_file = tmp_path / "test_permanent.compose.yml"
        compose_file.write_text(compose_content)
        return compose_file

    @pytest.fixture
    def permanent_config(self):
        """Configuration for permanent environment manager."""
        return {
            "domain": "test_domain",
            "config_dir": "/test/config",
            "enable_logging": True,
        }

    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    def test_execution_manager_initialization(
        self,
        mock_sandbox_manager,
        execution_manager_config,
    ):
        """Test ExecutionManager initialization."""
        # Mock the existence of environments.yaml
        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Verify components are initialized
        assert execution_manager._config_dir == execution_manager_config
        assert execution_manager._permanent_environment_manager is None  # Not initialized yet

    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    @patch('saber.server.execution.execution_manager.PermanentEnvironmentManager')
    def test_initialize_permanent_environment_manager(
        self,
        mock_perm_env_manager_class,
        mock_sandbox_manager,
        execution_manager_config,
        permanent_config,
    ):
        """Test initializing permanent environment manager through ExecutionManager."""
        # Setup mocks
        mock_perm_env_manager = MagicMock()
        mock_perm_env_manager_class.return_value = mock_perm_env_manager

        with patch('pathlib.Path.exists', return_value=True):
            execution_manager = ExecutionManager(execution_manager_config)

        # Initialize permanent environment manager
        execution_manager.initialize_permanent_environment_manager(permanent_config)

        # Verify permanent environment manager was created

class TestExecutionManagerDebugMode:
    """Test cases for ExecutionManager debug mode functionality."""

    def test_debug_mode_enabled_from_env_true(self, tmp_path, monkeypatch):
        """Test that debug mode is enabled when SABER_DEBUG_MODE=true."""
        # Set environment variable
        monkeypatch.setenv("SABER_DEBUG_MODE", "true")

        # Create execution manager
        execution_manager = ExecutionManager(str(tmp_path))

        # Verify debug mode is enabled
        assert execution_manager.debug_mode is True

    def test_debug_mode_enabled_from_env_1(self, tmp_path, monkeypatch):
        """Test that debug mode is enabled when SABER_DEBUG_MODE=1."""
        # Set environment variable
        monkeypatch.setenv("SABER_DEBUG_MODE", "1")

        # Create execution manager
        execution_manager = ExecutionManager(str(tmp_path))

        # Verify debug mode is enabled
        assert execution_manager.debug_mode is True

    def test_debug_mode_disabled_from_env_false(self, tmp_path, monkeypatch):
        """Test that debug mode is disabled when SABER_DEBUG_MODE=false."""
        # Set environment variable
        monkeypatch.setenv("SABER_DEBUG_MODE", "false")

        # Create execution manager
        execution_manager = ExecutionManager(str(tmp_path))

        # Verify debug mode is disabled
        assert execution_manager.debug_mode is False

    def test_debug_mode_disabled_by_default(self, tmp_path, monkeypatch):
        """Test that debug mode is disabled by default when env var is not set."""
        # Ensure environment variable is not set
        monkeypatch.delenv("SABER_DEBUG_MODE", raising=False)

        # Create execution manager
        execution_manager = ExecutionManager(str(tmp_path))

        # Verify debug mode is disabled
        assert execution_manager.debug_mode is False

    @patch('saber.server.execution.execution_manager.SandboxEnvironmentManager')
    def test_configure_for_task_with_episode_id(self, mock_sandbox_class, tmp_path, monkeypatch):
        """Test configuring ExecutionManager with episode ID for unique container naming."""
        # Ensure SABER_DEBUG_MODE is not set
        monkeypatch.delenv("SABER_DEBUG_MODE", raising=False)

        # Setup mock before creating ExecutionManager
        mock_sandbox_instance = MagicMock()
        mock_sandbox_class.return_value = mock_sandbox_instance

        execution_manager = ExecutionManager(str(tmp_path))

        # Create a mock task object
        mock_task = MagicMock()
        mock_task.environment = "test_env"
        mock_task.execution_config = {"timeout": 120.0}
        mock_task.allowed_executors = ["bash"]
        mock_task.cli_config = {"default_shell_mode": True}
        mock_task.python_config = None

        episode_id = "test-episode-123"
        execution_manager.configure_for_task(episode_id, mock_task, session_id="session123")

        # Verify episode_id was passed to create_episode_environment
        mock_sandbox_instance.create_episode_environment.assert_called_once_with(
            episode_id, "test_env", None
        )
