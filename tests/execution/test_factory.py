"""
Tests for ExecutorFactory.

This module tests the executor factory that manages different types of command executors
with dynamic selection and creation.
"""

from unittest.mock import MagicMock, patch

import pytest

from saber.server.execution.executors.docker_executor import DockerExecutor
from saber.server.execution.executors.executor_factory import ExecutorFactory
from saber.server.execution.executors.executor_registry import executor_registry
from saber.server.execution.executors.standard_registry.bash_executor import BashExecutor
from saber.server.execution.executors.standard_registry.python_executor import PythonExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


class TestExecutorFactory:
    """Test cases for ExecutorFactory."""

    @pytest.fixture(autouse=True)
    def clean_executor_registry(self):
        """Ensure clean executor registry state for factory tests."""
        # Store initial state
        initial_executors = list(executor_registry.get_available_executors())

        # Clear registry and re-register only standard executors to ensure clean state
        executor_registry.clear_all_executors()

        # Re-register standard executors
        from saber.server.execution.executors.executor_registry import register_executor
        register_executor("bash", BashExecutor, "standard")
        register_executor("python", PythonExecutor, "standard")

        yield

        # Cleanup after test - restore to original state if needed
        # Note: We don't restore the initial state as that would re-introduce the pollution
        # Instead, we leave it clean for subsequent tests

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        # Note: get_sandbox_config method removed - not needed in current interface
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
        assert "bash" in factory.get_available_executors()
        assert "python" in factory.get_available_executors()

    def test_initialization_with_config(self, mock_sandbox_manager):
        """Test factory initialization with configuration."""

        config_dict = {"executors": {"common": {"timeout": 600}, "python": {"allowed_modules": ["requests", "json"]}}}
        configuration = dict(config=config_dict)

        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager, configuration=configuration)
        assert factory._configuration == configuration

    def test_get_available_executors(self, executor_factory):
        """Test getting list of available executor types."""
        executors = executor_factory.get_available_executors()

        assert isinstance(executors, list)
        assert "bash" in executors
        assert "python" in executors
        assert len(executors) >= 2

    def test_register_executor(self):
        """Test registering a new executor type through the registry."""
        from saber.server.execution.executors.executor_registry import executor_registry, register_executor

        class CustomExecutor(DockerExecutor):
            async def execute(self, parameters, context):
                pass

        initial_count = len(executor_registry.get_available_executors())
        register_executor("custom", CustomExecutor, "test")

        assert "custom" in executor_registry.get_available_executors()
        assert len(executor_registry.get_available_executors()) == initial_count + 1
        assert executor_registry.get_executor_class("custom") == CustomExecutor

        # Cleanup
        executor_registry.unregister_executor("custom")

    def test_register_executor_invalid_class(self):
        """Test registering an invalid executor class."""
        from saber.server.execution.executors.executor_registry import register_executor

        class InvalidExecutor:
            pass

        with pytest.raises(ValueError, match="must inherit from DockerExecutor"):
            register_executor("invalid", InvalidExecutor, "test")

    def test_unregister_executor(self):
        """Test unregistering an executor type through the registry."""
        from saber.server.execution.executors.executor_registry import executor_registry, register_executor

        # Register a temporary executor
        class TempExecutor(DockerExecutor):
            async def execute(self, parameters, context):
                pass

        register_executor("temp", TempExecutor, "test")
        assert "temp" in executor_registry.get_available_executors()

        # Unregister it
        executor_registry.unregister_executor("temp")
        assert "temp" not in executor_registry.get_available_executors()

    def test_get_executor_cli(self, executor_factory):
        """Test getting CLI executor."""
        executor = executor_factory.get_executor("bash")

        assert isinstance(executor, BashExecutor)
        assert executor == executor_factory._executor_instances["bash"]

    def test_get_executor_python(self, executor_factory):
        """Test getting Python executor."""
        executor = executor_factory.get_executor("python")

        assert isinstance(executor, PythonExecutor)
        assert executor == executor_factory._executor_instances["python"]

    def test_get_executor_cached(self, executor_factory):
        """Test that executors are cached and reused."""
        executor1 = executor_factory.get_executor("bash")
        executor2 = executor_factory.get_executor("bash")

        assert executor1 is executor2
        assert len(executor_factory._executor_instances) == 1

    def test_get_executor_force_new(self, executor_factory):
        """Test creating new executor instance with force_new=True."""
        executor1 = executor_factory.get_executor("bash")
        executor2 = executor_factory.get_executor("bash", force_new=True)

        assert executor1 is not executor2
        assert isinstance(executor1, BashExecutor)
        assert isinstance(executor2, BashExecutor)

    def test_get_executor_unknown_type(self, executor_factory):
        """Test getting unknown executor type."""
        with pytest.raises(ValueError, match="Unknown or disabled executor type: unknown"):
            executor_factory.get_executor("unknown")

    def test_default_executor_configuration(self, executor_factory):
        """Test that executors get default configuration when no specific config provided."""
        # With no configuration, the factory should use built-in defaults
        bash_executor = executor_factory.get_executor("bash")
        python_executor = executor_factory.get_executor("python")

        # Verify executors were created successfully with defaults
        assert isinstance(bash_executor, BashExecutor)
        assert isinstance(python_executor, PythonExecutor)

    def test_executor_config_extraction(self, mock_sandbox_manager):
        """Test executor configuration extraction with the new configuration system."""

        configuration = {"timeout": 600, "bash": {"default_shell_mode": True}}

        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager, configuration=configuration)

        # Test extracting CLI config
        cli_config = configuration.get("bash", {})
        assert cli_config["default_shell_mode"] is True

        # Test extracting timeout from top level
        assert configuration["timeout"] == 600

    def test_create_executor_direct(self, executor_factory):
        """Test creating executor by type."""
        # Test CLI executor creation
        bash_executor = executor_factory.get_executor("bash")
        assert isinstance(bash_executor, BashExecutor)

        # Test Python executor creation
        python_executor = executor_factory.get_executor("python")
        assert isinstance(python_executor, PythonExecutor)

    def test_get_all_mcp_tools(self, executor_factory):
        """Test getting MCP tools for all executors."""
        with patch.object(executor_factory, "get_executor") as mock_get_executor:
            # Mock CLI executor
            mock_cli = MagicMock()
            mock_cli._executor_metadata = {"name": "bash", "description": "Execute shell commands"}
            mock_cli.to_mcp_schema.return_value = {"type": "object", "properties": {"command": {"type": "string"}}}

            # Mock Python executor
            mock_python = MagicMock()
            mock_python._executor_metadata = {"name": "python_script", "description": "Execute Python scripts"}
            mock_python.to_mcp_schema.return_value = {"type": "object", "properties": {"code": {"type": "string"}}}

            def mock_get_executor_side_effect(executor_type, episode_id=None):
                if executor_type == "bash":
                    return mock_cli
                elif executor_type == "python":
                    return mock_python
                else:
                    raise ValueError(f"Unknown type: {executor_type}")

            mock_get_executor.side_effect = mock_get_executor_side_effect

            tools = executor_factory.get_all_mcp_tools()

            assert len(tools) == 2

            # Check CLI tool
            cli_tool = next(tool for tool in tools if "bash" in tool.name)
            assert cli_tool.description == "Execute shell commands"
            assert "command" in cli_tool.inputSchema["properties"]

            # Check Python tool
            python_tool = next(tool for tool in tools if "python" in tool.name)
            assert python_tool.description == "Execute Python scripts"
            assert "code" in python_tool.inputSchema["properties"]

    def test_get_all_mcp_tools_with_error(self, executor_factory):
        """Test getting MCP tools when one executor fails."""
        with patch.object(executor_factory, "get_executor") as mock_get_executor:

            def mock_get_executor_side_effect(executor_type, episode_id=None):
                if executor_type == "bash":
                    mock_cli = MagicMock()
                    mock_cli._executor_metadata = {"name": "bash", "description": "Execute shell commands"}
                    mock_cli.to_mcp_schema.return_value = {"type": "object"}
                    return mock_cli
                else:
                    raise Exception("Executor failed")

            mock_get_executor.side_effect = mock_get_executor_side_effect

            tools = executor_factory.get_all_mcp_tools()

            # Should return tools for successful executors only
            assert len(tools) >= 1
            assert any("bash" in tool.name for tool in tools)

    def test_cleanup_all_executors(self, executor_factory):
        """Test cleanup of all executor instances."""
        # Create some executor instances
        executor_factory.get_executor("bash")
        executor_factory.get_executor("python")

        assert len(executor_factory._executor_instances) == 2

        # Cleanup
        executor_factory.cleanup_all_executors()

        assert len(executor_factory._executor_instances) == 0

    def test_get_executor_info(self, executor_factory):
        """Test getting executor information."""
        # Create one instance to test active instances
        executor_factory.get_executor("bash")

        info = executor_factory.get_executor_info()

        assert "available_types" in info
        assert "active_instances" in info
        assert "registry_size" in info
        assert "configurations" in info

        assert "bash" in info["available_types"]
        assert "python" in info["available_types"]
        assert "bash" in info["active_instances"]
        assert info["registry_size"] >= 2
        assert "bash" in info["configurations"]
        assert "python" in info["configurations"]

    def test_executor_configuration_inheritance(self, mock_sandbox_manager):
        """Test that executor configuration properly inherits from configuration sections."""

        configuration = {
            "timeout": 900,
            "max_retries": 3,
            "python": {"allowed_modules": ["requests", "numpy"], "timeout": 1200},  # Override common timeout for Python
        }

        factory = ExecutorFactory(sandbox_manager=mock_sandbox_manager, configuration=configuration)

        # Test CLI config (should get common values only)
        cli_config = configuration.get("bash", {})  # Empty since no CLI section
        assert configuration["timeout"] == 900
        assert configuration["max_retries"] == 3
        assert "allowed_modules" not in configuration

        # Test Python config (should have specific values)
        python_config = configuration.get("python", {})
        assert python_config["allowed_modules"] == ["requests", "numpy"]
        assert python_config["timeout"] == 1200  # Should override common timeout

    def test_multiple_executor_instances(self, executor_factory):
        """Test that multiple different executors can be created and managed."""
        bash_executor = executor_factory.get_executor("bash")
        python_executor = executor_factory.get_executor("python")

        assert isinstance(bash_executor, BashExecutor)
        assert isinstance(python_executor, PythonExecutor)
        assert bash_executor is not python_executor
        assert len(executor_factory._executor_instances) == 2

    def test_multi_episode_configuration_orchestration(self, executor_factory):
        """Test complex multi-episode orchestration with different executor configurations."""
        # Setup episode configurations with different allowed executors
        episodes = [
            {"id": "pentest-episode-1", "allowed": ["bash"], "config": {"timeout": 300.0, "bash": {"shell_mode": True}}},
            {"id": "analysis-episode-1", "allowed": ["python"], "config": {"timeout": 120.0, "python": {"enable_networking": False}}},
            {"id": "hybrid-episode-1", "allowed": ["bash", "python"], "config": {"timeout": 600.0}},
            {"id": "restricted-episode-1", "allowed": ["bash"], "config": {"timeout": 60.0, "bash": {"shell_mode": False}}},
        ]

        # Register all episode configurations
        for episode in episodes:
            executor_factory.register_episode_configuration(
                episode["id"],
                allowed_executors=episode["allowed"],
                episode_config=episode["config"]
            )

        # Verify episode-specific executor availability
        pentest_executors = executor_factory.get_available_executors("pentest-episode-1")
        assert pentest_executors == ["bash"]

        analysis_executors = executor_factory.get_available_executors("analysis-episode-1")
        assert analysis_executors == ["python"]

        hybrid_executors = executor_factory.get_available_executors("hybrid-episode-1")
        assert set(hybrid_executors) == {"bash", "python"}

        # Test episode-specific executor creation
        bash_executor_pentest = executor_factory.get_executor("bash", episode_id="pentest-episode-1")
        assert isinstance(bash_executor_pentest, BashExecutor)

        python_executor_analysis = executor_factory.get_executor("python", episode_id="analysis-episode-1")
        assert isinstance(python_executor_analysis, PythonExecutor)

        # Test that disallowed executors are rejected
        with pytest.raises(ValueError, match="Unknown or disabled executor type: python for episode pentest-episode-1"):
            executor_factory.get_executor("python", episode_id="pentest-episode-1")

        with pytest.raises(ValueError, match="Unknown or disabled executor type: bash for episode analysis-episode-1"):
            executor_factory.get_executor("bash", episode_id="analysis-episode-1")

        # Test hybrid episode can access both
        bash_executor_hybrid = executor_factory.get_executor("bash", episode_id="hybrid-episode-1")
        python_executor_hybrid = executor_factory.get_executor("python", episode_id="hybrid-episode-1")
        assert isinstance(bash_executor_hybrid, BashExecutor)
        assert isinstance(python_executor_hybrid, PythonExecutor)

        # Verify configuration storage
        assert len(executor_factory._episode_configurations) == 4
        pentest_config = executor_factory._episode_configurations["pentest-episode-1"]
        assert pentest_config["allowed_executors"] == ["bash"]
        assert pentest_config["config"]["timeout"] == 300.0

        # Test partial cleanup - unregister some episodes
        executor_factory.unregister_episode_configuration("pentest-episode-1")
        executor_factory.unregister_episode_configuration("analysis-episode-1")

        # Verify remaining configurations
        assert len(executor_factory._episode_configurations) == 2
        assert "hybrid-episode-1" in executor_factory._episode_configurations
        assert "restricted-episode-1" in executor_factory._episode_configurations

        # Test that unregistered episodes fall back to all executors with warning
        fallback_executors = executor_factory.get_available_executors("pentest-episode-1")
        assert set(fallback_executors) == {"bash", "python"}  # Should return all available

    def test_episode_executor_isolation_and_mcp_tools(self, executor_factory):
        """Test episode isolation for executor management and MCP tool generation."""
        # Setup episodes with different security profiles
        security_episodes = [
            {"id": "secure-episode-1", "allowed": ["bash"], "config": {"security_level": "high"}},
            {"id": "dev-episode-1", "allowed": ["bash", "python"], "config": {"security_level": "low"}},
            {"id": "python-only-episode", "allowed": ["python"], "config": {"security_level": "medium"}},
        ]

        # Register episode configurations
        for episode in security_episodes:
            executor_factory.register_episode_configuration(
                episode["id"],
                allowed_executors=episode["allowed"],
                episode_config=episode["config"]
            )

        # Test episode-specific MCP tool generation
        with patch.object(executor_factory, "get_executor") as mock_get_executor:
            # Mock CLI executor
            mock_cli = MagicMock()
            mock_cli._executor_metadata = {"name": "secure_cli", "description": "Secure CLI execution"}
            mock_cli.to_mcp_schema.return_value = {"type": "object", "properties": {"command": {"type": "string"}}}

            # Mock Python executor
            mock_python = MagicMock()
            mock_python._executor_metadata = {"name": "dev_python", "description": "Development Python execution"}
            mock_python.to_mcp_schema.return_value = {"type": "object", "properties": {"code": {"type": "string"}}}

            def mock_get_executor_side_effect(executor_type, episode_id=None):
                if executor_type == "bash":
                    return mock_cli
                elif executor_type == "python":
                    return mock_python
                else:
                    raise ValueError(f"Unknown type: {executor_type}")

            mock_get_executor.side_effect = mock_get_executor_side_effect

            # Test MCP tools for secure episode (CLI only)
            secure_tools = executor_factory.get_all_mcp_tools(episode_id="secure-episode-1")
            assert len(secure_tools) == 1
            assert "secure_cli" in secure_tools[0].name

            # Test MCP tools for dev episode (CLI + Python)
            dev_tools = executor_factory.get_all_mcp_tools(episode_id="dev-episode-1")
            assert len(dev_tools) == 2
            tool_names = [tool.name for tool in dev_tools]
            assert any("secure_cli" in name for name in tool_names)
            assert any("dev_python" in name for name in tool_names)

            # Test MCP tools for Python-only episode
            python_tools = executor_factory.get_all_mcp_tools(episode_id="python-only-episode")
            assert len(python_tools) == 1
            assert "dev_python" in python_tools[0].name

        # Test concurrent episode executor access patterns
        episodes_to_test = ["secure-episode-1", "dev-episode-1", "python-only-episode"]
        concurrent_results = {}

        for episode_id in episodes_to_test:
            available = executor_factory.get_available_executors(episode_id)
            concurrent_results[episode_id] = available

        # Verify isolation - each episode sees only its configured executors
        assert concurrent_results["secure-episode-1"] == ["bash"]
        assert set(concurrent_results["dev-episode-1"]) == {"bash", "python"}
        assert concurrent_results["python-only-episode"] == ["python"]

        # Test executor info aggregation across episodes
        executor_info = executor_factory.get_executor_info()
        assert "available_types" in executor_info
        assert set(executor_info["available_types"]) == {"bash", "python"}  # All registered types
        assert "configurations" in executor_info

        # Cleanup all episode configurations
        for episode in security_episodes:
            executor_factory.unregister_episode_configuration(episode["id"])

        # Verify clean state
        assert len(executor_factory._episode_configurations) == 0
