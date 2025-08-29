"""
Integration test to verify custom executor loading and MCP integration.

This test demonstrates that custom executors can be loaded from external files
and become available through the SABER execution framework.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from saber.server.execution.execution_manager import ExecutionManager
from saber.server.execution.executors.executor_factory import ExecutorFactory
from saber.server.execution.executors.executor_registry import get_executor_info


class TestCustomExecutorIntegration:
    """Integration tests for custom executor loading."""

    @pytest.fixture
    def cleanup_factory(self):
        """Clean up factory state after tests."""
        # No factory-level cleanup needed since we use the global registry
        yield

    def test_execution_manager_loads_custom_executors(self, cleanup_factory):
        """Test that ExecutionManager loads custom executors from config directory."""
        # Create a temporary directory with a custom executor
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a custom executor file
            executor_file = temp_path / "test_integration_executor.py"
            executor_code = '''
from typing import Any, Dict, Optional
from saber.server.base import CommandResult
from saber.server.execution.base import Parameter, ParameterType
from saber.server.execution.executors.executor_registry import register_executor
from saber.server.execution.executors.docker_executor import DockerExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager

class TestIntegrationExecutor(DockerExecutor):
    """Test executor for integration testing."""

    def __init__(self, sandbox_manager: SandboxEnvironmentManager, docker_config: Optional[Dict[str, Any]] = None):
        super().__init__(sandbox_manager, docker_config)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"timeout": 30.0, "test_param": "integration_test"}

    def setup_parameters(self, config: Dict[str, Any]) -> None:
        self.add_parameter(
            Parameter(
                name="test_input",
                type=ParameterType.STRING,
                description="Test input for integration testing",
                required=True,
            )
        )

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        return CommandResult.success_result(
            output=f"Integration test executed with: {parameters.get('test_input', 'none')}"
        )

    def to_mcp_schema(self) -> Dict[str, Any]:
        self._executor_metadata = {
            "name": "integration_test",
            "description": "Test executor for integration testing",
        }
        return super().to_mcp_schema()

register_executor("integration_test", TestIntegrationExecutor, "test")
'''
            executor_file.write_text(executor_code)

            # Mock SandboxEnvironmentManager to avoid Docker dependencies in tests
            with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager") as mock_sandbox:
                mock_sandbox_instance = MagicMock()
                mock_sandbox.return_value = mock_sandbox_instance

                # Create ExecutionManager with the temp directory as config_dir
                execution_manager = ExecutionManager(config_dir=str(temp_dir))

                # Check that the custom executor was loaded and is available
                available_executors = execution_manager._executor_factory.get_available_executors()
                assert "integration_test" in available_executors

                # Verify it's in the executor registry
                custom_info = get_executor_info()
                assert "integration_test" in custom_info
                assert custom_info["integration_test"]["class_name"] == "TestIntegrationExecutor"

                # Check that it appears in MCP tools
                mcp_tools = execution_manager.to_mcp_tools()
                integration_tools = [tool for tool in mcp_tools if "integration_test" in tool["name"]]
                assert len(integration_tools) == 1
                assert integration_tools[0]["description"] == "Test executor for integration testing"

    def test_pentest_demo_executor_loading(self, cleanup_factory):
        """Test loading the pentest demo executors."""
        pentest_demo_path = Path("/home/ms_test/repos/saber_vibin/examples/pentest_demo/server")

        # Skip if pentest demo doesn't exist
        if not pentest_demo_path.exists():
            pytest.skip("Pentest demo directory not found")

        # Mock SandboxEnvironmentManager to avoid Docker dependencies
        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager") as mock_sandbox:
            mock_sandbox_instance = MagicMock()
            mock_sandbox.return_value = mock_sandbox_instance

            # Create ExecutionManager with pentest demo config directory
            execution_manager = ExecutionManager(config_dir=str(pentest_demo_path))

        # Check that custom executors were loaded
        available_executors = execution_manager._executor_factory.get_available_executors()
        # Should have the built-in executors plus any custom ones
        assert "cli" in available_executors
        assert "python" in available_executors

        # Check if pentest executors were loaded (if file exists)
        pentest_executor_file = pentest_demo_path / "pentest_executors.py"
        if pentest_executor_file.exists():
            custom_info = get_executor_info()
            # The pentest_executors.py file should register nmap and sqlmap
            expected_executors = ["nmap", "sqlmap"]
            for executor_name in expected_executors:
                if executor_name in custom_info:
                    assert executor_name in available_executors

            # Check MCP tools include custom executors
            mcp_tools = execution_manager.to_mcp_tools()
            tool_names = [tool["name"] for tool in mcp_tools]

            # Should have at least the built-in tools
            cli_tools = [name for name in tool_names if "cli" in name]
            python_tools = [name for name in tool_names if "python" in name]
            assert len(cli_tools) > 0
            assert len(python_tools) > 0

    def test_custom_executor_mcp_schema_generation(self, cleanup_factory):
        """Test that custom executors generate proper MCP schemas."""
        # Create a temporary directory with a custom executor
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a custom executor with specific parameters
            executor_file = temp_path / "schema_test_executor.py"
            executor_code = """
from typing import Any, Dict, Optional
from saber.server.base import CommandResult
from saber.server.execution.base import Parameter, ParameterType
from saber.server.execution.executors.executor_registry import register_executor
from saber.server.execution.executors.docker_executor import DockerExecutor

class SchemaTestExecutor(DockerExecutor):
    def __init__(self, sandbox_manager, docker_config=None):
        super().__init__(sandbox_manager, docker_config)

    @classmethod
    def get_default_config(cls):
        return {"timeout": 45.0}

    def setup_parameters(self, config):
        self.add_parameter(
            Parameter(
                name="required_param",
                type=ParameterType.STRING,
                description="A required string parameter",
                required=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="optional_param",
                type=ParameterType.INTEGER,
                description="An optional integer parameter",
                required=False,
                default=42,
                min_value=1,
                max_value=100,
            )
        )
        self.add_parameter(
            Parameter(
                name="boolean_param",
                type=ParameterType.BOOLEAN,
                description="A boolean parameter",
                required=False,
                default=True,
            )
        )

    async def execute(self, parameters, context):
        return CommandResult.success_result(output="Schema test executed")

    def to_mcp_schema(self):
        self._executor_metadata = {
            "name": "schema_test",
            "description": "Test executor for MCP schema validation",
        }
        return super().to_mcp_schema()

register_executor("schema_test", SchemaTestExecutor)
"""
            executor_file.write_text(executor_code)

            # Mock SandboxEnvironmentManager
            with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager") as mock_sandbox:
                mock_sandbox_instance = MagicMock()
                mock_sandbox.return_value = mock_sandbox_instance

                # Create ExecutionManager and load the custom executor
                execution_manager = ExecutionManager(config_dir=str(temp_dir))

                # Get MCP tools and find our custom executor
                mcp_tools = execution_manager.to_mcp_tools()
                schema_tools = [tool for tool in mcp_tools if "schema_test" in tool["name"]]
                assert len(schema_tools) == 1

                tool = schema_tools[0]
                assert tool["description"] == "Test executor for MCP schema validation"

                # Check the schema structure
                input_schema = tool["inputSchema"]
                assert input_schema["type"] == "object"

                properties = input_schema["properties"]
                required = input_schema["required"]

                # Check required parameter
                assert "required_param" in properties
                assert "required_param" in required
                assert properties["required_param"]["type"] == "string"
                assert properties["required_param"]["description"] == "A required string parameter"

                # Check optional integer parameter with constraints
                assert "optional_param" in properties
                assert "optional_param" not in required
                assert properties["optional_param"]["type"] == "integer"
                assert properties["optional_param"]["minimum"] == 1
                assert properties["optional_param"]["maximum"] == 100

                # Check boolean parameter
                assert "boolean_param" in properties
                assert properties["boolean_param"]["type"] == "boolean"

    def test_factory_integration_with_custom_executors(self, cleanup_factory):
        """Test that ExecutorFactory properly integrates custom executors."""
        # Create a temporary custom executor
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            executor_file = temp_path / "factory_test_executor.py"
            executor_code = """
from saber.server.execution.executors.executor_registry import register_executor
from saber.server.execution.executors.docker_executor import DockerExecutor

class FactoryTestExecutor(DockerExecutor):
    @classmethod
    def get_default_config(cls):
        return {"timeout": 60.0, "custom_setting": "factory_test"}

    def setup_parameters(self, config):
        pass

    async def execute(self, parameters, context):
        from saber.server.base import CommandResult
        return CommandResult.success_result(output="Factory test")

    def to_mcp_schema(self):
        self._executor_metadata = {
            "name": "factory_test",
            "description": "Test executor for factory integration",
        }
        return super().to_mcp_schema()

register_executor("factory_test", FactoryTestExecutor)
"""
            executor_file.write_text(executor_code)

            with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager") as mock_sandbox:
                mock_sandbox_instance = MagicMock()
                mock_sandbox.return_value = mock_sandbox_instance

                # Create ExecutionManager
                execution_manager = ExecutionManager(config_dir=str(temp_dir))

                # Test factory methods work with custom executor
                factory = execution_manager._executor_factory

                # Check it's in available executors
                available = factory.get_available_executors()
                assert "factory_test" in available

                # Check we can get the executor instance
                executor_instance = factory.get_executor("factory_test")
                assert executor_instance is not None
                assert executor_instance.__class__.__name__ == "FactoryTestExecutor"

                # Check executor info includes custom executor
                executor_info = factory.get_executor_info()
                assert "factory_test" in executor_info["available_types"]
                assert "factory_test" in executor_info["configurations"]
                assert executor_info["configurations"]["factory_test"]["custom_setting"] == "factory_test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
