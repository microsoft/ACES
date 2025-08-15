"""
Tests for custom executor registry functionality.

These tests demonstrate how the custom executor registration system works
and validate that external executors can be properly registered and used.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from saber.server.base import CommandResult
from saber.server.execution.base import Parameter, ParameterType
from saber.server.execution.custom_executor_registry import (
    CustomExecutorRegistry,
    register_custom_executor,
    register_executor_from_file,
    load_custom_executors_from_directory,
    get_custom_executor_info,
)
from saber.server.execution.executors.docker_executor import DockerExecutor
from saber.server.execution.executors.factory import ExecutorFactory
from saber.server.execution.sandbox.sandbox_manager import SandboxManager


class TestCustomExecutor(DockerExecutor):
    """Test executor for unit testing."""

    def __init__(self, sandbox_manager: SandboxManager, docker_config: Dict[str, Any] = None):
        super().__init__(sandbox_manager, docker_config)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"timeout": 30.0, "test_param": "test_value"}

    def setup_parameters(self, config: Dict[str, Any]) -> None:
        self.add_parameter(
            Parameter(
                name="test_input",
                type=ParameterType.STRING,
                description="Test input parameter",
                required=True,
            )
        )

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        return CommandResult.success_result(
            output=f"Test executed with input: {parameters.get('test_input', 'none')}"
        )

    def to_mcp_schema(self) -> Dict[str, Any]:
        self._security_command_metadata = {
            "name": "test_executor",
            "description": "Test executor for unit testing",
        }
        return super().to_mcp_schema()


class TestCustomExecutorRegistry:
    """Test cases for CustomExecutorRegistry."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxManager."""
        mock = MagicMock(spec=SandboxManager)
        return mock

    @pytest.fixture
    def registry(self):
        """Create a fresh CustomExecutorRegistry for testing."""
        return CustomExecutorRegistry()

    @pytest.fixture
    def cleanup_factory(self):
        """Clean up factory state after tests."""
        # Store original state
        original_registry = ExecutorFactory._executor_registry.copy()
        yield
        # Restore original state
        ExecutorFactory._executor_registry = original_registry

    def test_register_executor_class_success(self, registry, mock_sandbox_manager, cleanup_factory):
        """Test successful executor class registration."""
        registry.register_executor_class("test", TestCustomExecutor)

        assert "test" in registry._custom_executors
        assert registry._custom_executors["test"] == TestCustomExecutor
        assert "test" in ExecutorFactory._executor_registry

    def test_register_executor_class_invalid_class(self, registry, cleanup_factory):
        """Test registering invalid executor class."""
        class InvalidExecutor:
            pass

        with pytest.raises(ValueError, match="must inherit from DockerExecutor"):
            registry.register_executor_class("invalid", InvalidExecutor)

    def test_register_executor_class_duplicate(self, registry, cleanup_factory):
        """Test registering duplicate executor type."""
        registry.register_executor_class("test", TestCustomExecutor)

        with pytest.raises(RuntimeError, match="already registered"):
            registry.register_executor_class("test", TestCustomExecutor)

    def test_register_executor_from_file_success(self, registry, cleanup_factory):
        """Test registering executor from file."""
        # Create a temporary Python file with an executor
        executor_code = '''
from typing import Any, Dict
from saber.server.base import CommandResult
from saber.server.execution.base import Parameter, ParameterType
from saber.server.execution.executors.docker_executor import DockerExecutor

class FileTestExecutor(DockerExecutor):
    def __init__(self, sandbox_manager, docker_config=None):
        super().__init__(sandbox_manager, docker_config)

    @classmethod
    def get_default_config(cls):
        return {"timeout": 60.0}

    def setup_parameters(self, config):
        self.add_parameter(
            Parameter(
                name="input",
                type=ParameterType.STRING,
                description="Test input",
                required=True,
            )
        )

    async def execute(self, parameters, context):
        return CommandResult.success_result(output="File executor test")

    def to_mcp_schema(self):
        self._security_command_metadata = {
            "name": "file_test",
            "description": "File-based test executor",
        }
        return super().to_mcp_schema()
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(executor_code)
            temp_file = f.name

        try:
            registry.register_executor_from_file("filetest", temp_file, "FileTestExecutor")

            assert "filetest" in registry._custom_executors
            assert "filetest" in ExecutorFactory._executor_registry
        finally:
            Path(temp_file).unlink()

    def test_register_executor_from_file_not_found(self, registry, cleanup_factory):
        """Test registering executor from non-existent file."""
        with pytest.raises(FileNotFoundError):
            registry.register_executor_from_file("test", "/nonexistent/file.py", "TestExecutor")

    def test_register_executor_from_file_class_not_found(self, registry, cleanup_factory):
        """Test registering executor from file with missing class."""
        executor_code = "# Empty file"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(executor_code)
            temp_file = f.name

        try:
            with pytest.raises(ImportError, match="Class 'MissingExecutor' not found"):
                registry.register_executor_from_file("test", temp_file, "MissingExecutor")
        finally:
            Path(temp_file).unlink()

    def test_load_executors_from_directory(self, registry, cleanup_factory):
        """Test loading executors from directory."""
        # Create a temporary directory with executor files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a valid executor file
            executor_file = temp_path / "custom_executor.py"
            executor_code = '''
from typing import Any, Dict
from saber.server.base import CommandResult
from saber.server.execution.base import Parameter, ParameterType
from saber.server.execution.executors.docker_executor import DockerExecutor

class DirTestExecutor(DockerExecutor):
    def __init__(self, sandbox_manager, docker_config=None):
        super().__init__(sandbox_manager, docker_config)

    @classmethod
    def get_default_config(cls):
        return {"timeout": 30.0}

    def setup_parameters(self, config):
        self.add_parameter(
            Parameter(
                name="input",
                type=ParameterType.STRING,
                description="Test input",
                required=True,
            )
        )

    async def execute(self, parameters, context):
        return CommandResult.success_result(output="Directory executor test")

    def to_mcp_schema(self):
        self._security_command_metadata = {
            "name": "dir_test",
            "description": "Directory-based test executor",
        }
        return super().to_mcp_schema()

# Register the executor using the global registry for testing
from saber.server.execution.custom_executor_registry import custom_executor_registry
custom_executor_registry.register_executor_class("dirtest", DirTestExecutor)
'''
            executor_file.write_text(executor_code)

            # Create an invalid file
            invalid_file = temp_path / "invalid_executor.py"
            invalid_file.write_text("invalid python code !!!")

            # Load executors from directory
            results = registry.load_executors_from_directory(str(temp_dir))

            # Check results
            assert len(results) == 2
            assert str(executor_file) in results
            assert str(invalid_file) in results
            assert results[str(executor_file)] == "loaded_successfully"
            assert "Failed to load" in results[str(invalid_file)]

            # Check that the valid executor was registered in the global registry
            from saber.server.execution.custom_executor_registry import custom_executor_registry
            assert "dirtest" in custom_executor_registry._custom_executors

    def test_unregister_executor(self, registry, cleanup_factory):
        """Test unregistering executor."""
        registry.register_executor_class("test", TestCustomExecutor)
        assert "test" in registry._custom_executors

        registry.unregister_executor("test")
        assert "test" not in registry._custom_executors
        assert "test" not in ExecutorFactory._executor_registry

    def test_list_custom_executors(self, registry, cleanup_factory):
        """Test listing custom executors."""
        registry.register_executor_class("test", TestCustomExecutor)

        executors = registry.list_custom_executors()

        assert "test" in executors
        assert executors["test"]["class_name"] == "TestCustomExecutor"
        assert executors["test"]["source"] == "external"
        assert "default_config" in executors["test"]

    def test_clear_all_custom_executors(self, registry, cleanup_factory):
        """Test clearing all custom executors."""
        registry.register_executor_class("test1", TestCustomExecutor)
        registry.register_executor_class("test2", TestCustomExecutor, "test_source")

        assert len(registry._custom_executors) == 2

        registry.clear_all_custom_executors()

        assert len(registry._custom_executors) == 0


class TestHookFunctions:
    """Test the global hook functions."""

    @pytest.fixture
    def cleanup_registry(self):
        """Clean up registry state after tests."""
        from saber.server.execution.custom_executor_registry import custom_executor_registry

        # Store original state
        original_executors = custom_executor_registry._custom_executors.copy()
        original_sources = custom_executor_registry._registration_sources.copy()
        original_factory = ExecutorFactory._executor_registry.copy()

        yield

        # Restore original state
        custom_executor_registry._custom_executors = original_executors
        custom_executor_registry._registration_sources = original_sources
        ExecutorFactory._executor_registry = original_factory

    def test_register_custom_executor_hook(self, cleanup_registry):
        """Test the register_custom_executor hook function."""
        register_custom_executor("hooktest", TestCustomExecutor)

        # Check that it was registered
        info = get_custom_executor_info()
        assert "hooktest" in info
        assert info["hooktest"]["class_name"] == "TestCustomExecutor"

    def test_register_executor_from_file_hook(self, cleanup_registry):
        """Test the register_executor_from_file hook function."""
        # Create a temporary executor file
        executor_code = '''
from typing import Any, Dict
from saber.server.base import CommandResult
from saber.server.execution.base import Parameter, ParameterType
from saber.server.execution.executors.docker_executor import DockerExecutor

class HookFileExecutor(DockerExecutor):
    def __init__(self, sandbox_manager, docker_config=None):
        super().__init__(sandbox_manager, docker_config)

    @classmethod
    def get_default_config(cls):
        return {"timeout": 45.0}

    def setup_parameters(self, config):
        self.add_parameter(
            Parameter(
                name="input",
                type=ParameterType.STRING,
                description="Test input",
                required=True,
            )
        )

    async def execute(self, parameters, context):
        return CommandResult.success_result(output="Hook file executor test")

    def to_mcp_schema(self):
        self._security_command_metadata = {
            "name": "hook_file_test",
            "description": "Hook file test executor",
        }
        return super().to_mcp_schema()
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(executor_code)
            temp_file = f.name

        try:
            register_executor_from_file("hookfile", temp_file, "HookFileExecutor")

            # Check that it was registered
            info = get_custom_executor_info()
            assert "hookfile" in info
            assert info["hookfile"]["class_name"] == "HookFileExecutor"
        finally:
            Path(temp_file).unlink()

    def test_load_custom_executors_from_directory_hook(self, cleanup_registry):
        """Test the load_custom_executors_from_directory hook function."""
        # Create a temporary directory with an executor file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            executor_file = temp_path / "hook_dir_executor.py"
            executor_code = '''
from saber.server.execution.custom_executor_registry import register_custom_executor
from typing import Any, Dict
from saber.server.base import CommandResult
from saber.server.execution.base import Parameter, ParameterType
from saber.server.execution.executors.docker_executor import DockerExecutor

class HookDirExecutor(DockerExecutor):
    def __init__(self, sandbox_manager, docker_config=None):
        super().__init__(sandbox_manager, docker_config)

    @classmethod
    def get_default_config(cls):
        return {"timeout": 25.0}

    def setup_parameters(self, config):
        self.add_parameter(
            Parameter(
                name="input",
                type=ParameterType.STRING,
                description="Test input",
                required=True,
            )
        )

    async def execute(self, parameters, context):
        return CommandResult.success_result(output="Hook directory executor test")

    def to_mcp_schema(self):
        self._security_command_metadata = {
            "name": "hook_dir_test",
            "description": "Hook directory test executor",
        }
        return super().to_mcp_schema()

# Register the executor
register_custom_executor("hookdir", HookDirExecutor)
'''
            executor_file.write_text(executor_code)

            # Load executors from directory
            results = load_custom_executors_from_directory(str(temp_dir))

            # Check results
            assert str(executor_file) in results
            assert results[str(executor_file)] == "loaded_successfully"

            # Check that the executor was registered
            info = get_custom_executor_info()
            assert "hookdir" in info
            assert info["hookdir"]["class_name"] == "HookDirExecutor"

    def test_get_custom_executor_info_hook(self, cleanup_registry):
        """Test the get_custom_executor_info hook function."""
        # Register a test executor
        register_custom_executor("infotest", TestCustomExecutor)

        # Get info
        info = get_custom_executor_info()

        assert "infotest" in info
        assert info["infotest"]["class_name"] == "TestCustomExecutor"
        assert "default_config" in info["infotest"]
        assert "docstring" in info["infotest"]
