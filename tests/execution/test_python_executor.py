"""
Tests for Python executor.

This module tests the secure Python script executor that executes Python code
in Docker containers with dependency management and security validation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.base import CommandResult
from saber.server.execution.base import ParameterType, ValidationResult, ExecutionContext, PythonParameters
from saber.server.execution.models import ExecutorConfig, PythonExecutorConfig
from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.executors.standard_registry.python_executor import PythonExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


class TestPythonExecutor:
    """Test cases for Python executor."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {
            "image": "saber/sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser",
        }
        return manager

    @pytest.fixture
    def python_executor(self, mock_sandbox_manager):
        """Create a Python executor instance for testing."""
        config = PythonExecutorConfig(
            timeout=60.0,
            allowed_modules=["os", "sys", "json"],
            script_templates={},
        )
        return PythonExecutor(sandbox_manager=mock_sandbox_manager, config=config)

    @pytest.fixture
    def mock_docker_environment(self):
        """Create a mock Docker execution environment."""
        env = MagicMock()
        # Mock the new interface
        container_mock = MagicMock()
        container_mock.id = "container123456789"
        env.get_execution_container.return_value = container_mock
        env.execute_command = MagicMock()  # Not async anymore
        return env

    def test_initialization(self, mock_sandbox_manager):
        """Test Python executor initialization."""
        config = PythonExecutorConfig(
            timeout=120.0,
            allowed_modules=["os", "sys", "json"],
            script_templates={},
        )
        executor = PythonExecutor(sandbox_manager=mock_sandbox_manager, config=config)

        assert executor.get_timeout() == 120.0

        # Check that parameters were added
        params = executor.get_parameters()
        assert "code" in params
        assert "template" in params
        assert "working_dir" in params

        # Verify parameter definitions
        code_param = params["code"]
        assert code_param.name == "code"
        assert code_param.type == ParameterType.STRING
        assert code_param.required is True

    def test_initialization_without_sandbox_manager(self):
        """Test that initialization fails without sandbox manager."""
        config = PythonExecutorConfig(timeout=60.0, allowed_modules=["os"])
        with pytest.raises(SandboxExecutionError, match="sandbox_manager is required"):
            PythonExecutor(sandbox_manager=None, config=config)

    def test_validate_python_code_valid(self, python_executor):
        """Test validation of valid Python code."""
        code = """
print("Hello, World!")
x = 1 + 2
print(f"Result: {x}")
"""
        result = python_executor.validate_python_code(code)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_python_code_empty(self, python_executor):
        """Test validation of empty Python code."""
        result = python_executor.validate_python_code("")
        assert result.valid is False
        assert "Python code cannot be empty" in result.errors

    def test_validate_python_code_syntax_error(self, python_executor):
        """Test validation of Python code with syntax error."""
        code = "print('hello'"  # Missing closing parenthesis
        result = python_executor.validate_python_code(code)
        assert result.valid is False
        assert any("syntax error" in error.lower() for error in result.errors)

    def test_validate_python_code_dangerous_constructs(self, python_executor):
        """Test validation warns about dangerous constructs."""
        code = """
import os
exec("print('hello')")
eval("1+1")
"""
        result = python_executor.validate_python_code(code)
        assert result.valid is True  # Valid syntax but has warnings
        assert len(result.warnings) > 0
        assert any("exec(" in warning for warning in result.warnings)
        assert any("eval(" in warning for warning in result.warnings)

    def test_validate_python_code_restricted_imports(self, python_executor):
        """Test validation warns about non-whitelisted imports."""
        code = """
import socket
import subprocess
from urllib import request
"""
        result = python_executor.validate_python_code(code)
        assert result.valid is True  # Valid syntax but has warnings
        assert len(result.warnings) > 0
        assert any("socket" in warning for warning in result.warnings)

    def test_build_python_script_simple(self, python_executor):
        """Test building simple Python script."""
        parameters = PythonParameters(code="print('Hello, World!')")

        script = python_executor._build_python_script(parameters)

        assert "#!/usr/bin/env python3" in script
        assert "print('Hello, World!')" in script
        assert "import sys" in script
        assert "import os" in script

    def test_build_python_script_with_template(self, python_executor):
        """Test building Python script with template."""
        # Configure executor with templates
        python_executor._script_templates = {
            "data_analysis": """
import pandas as pd
import numpy as np

# Data analysis template
{user_code}

# End of analysis
"""
        }

        parameters = PythonParameters(code="df = pd.DataFrame({'x': [1, 2, 3]})", template="data_analysis")

        script = python_executor._build_python_script(parameters)

        assert "import pandas as pd" in script
        assert "import numpy as np" in script
        assert "df = pd.DataFrame({'x': [1, 2, 3]})" in script

    def test_parse_python_output_success(self, python_executor):
        """Test parsing successful Python execution output."""
        result = python_executor.parse_python_output(
            stdout="Hello, World!\n", stderr="", return_code=0, script_path="/tmp/script.py"
        )

        assert result.success is True
        assert result.data["stdout"] == "Hello, World!\n"
        assert result.data["stderr"] == ""
        assert result.data["return_code"] == 0
        assert result.data["script_path"] == "/tmp/script.py"
        assert result.metadata["command_type"] == "python_script"

    def test_parse_python_output_failure(self, python_executor):
        """Test parsing failed Python execution output."""
        result = python_executor.parse_python_output(
            stdout="",
            stderr="NameError: name 'undefined_var' is not defined",
            return_code=1,
            script_path="/tmp/script.py",
        )

        assert result.success is False
        assert "Python script failed with exit code 1" in result.error
        assert "NameError" in result.error

    def test_get_python_environment(self, python_executor):
        """Test getting Python environment information."""
        with patch.object(python_executor, "get_episode_environment") as mock_get_env:
            mock_env = MagicMock()
            mock_get_env.return_value = mock_env

            env_info = python_executor.get_python_environment("test_episode")

            assert env_info["episode_id"] == "test_episode"
            assert env_info["container_ready"] is True

    @pytest.mark.asyncio
    async def test_execute_success(self, python_executor, mock_sandbox_manager, mock_docker_environment):
        """Test successful Python script execution."""
        # Mock environment creation
        mock_sandbox_manager.get_episode_environment.return_value = mock_docker_environment
        python_executor._sandbox_manager = mock_sandbox_manager

        # Mock script creation and execution
        create_result = MagicMock()
        create_result.exit_code = 0
        create_result.stdout = ""
        create_result.stderr = ""
        create_result.execution_time = 0.1

        execute_result = MagicMock()
        execute_result.exit_code = 0
        execute_result.stdout = "Hello, World!\n"
        execute_result.stderr = ""
        execute_result.execution_time = 1.0

        mock_docker_environment.execute_command = AsyncMock(side_effect=[create_result, execute_result])

        parameters = PythonParameters(code="print('Hello, World!')")
        context = ExecutionContext(episode_id="test123")

        result = await python_executor(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "Hello, World!\n"
        assert result.metadata["episode_id"] == "test123"
        assert result.metadata["container_id"] == "container123"  # First 12 chars: container123

    @pytest.mark.asyncio
    async def test_execute_validation_failure(self, python_executor):
        """Test execution with invalid Python code."""
        parameters = PythonParameters(code="")  # Empty code
        context = ExecutionContext(episode_id="test123")

        result = await python_executor(parameters, context)

        assert result.success is False
        assert "Python code validation failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_episode_id(self, python_executor, mock_sandbox_manager):
        """Test execution without episode ID."""
        # Mock to raise an error when accessing with empty episode_id
        mock_sandbox_manager.get_episode_environment.return_value = None
        python_executor._sandbox_manager = mock_sandbox_manager

        parameters = PythonParameters(code="print('hello')")
        context = ExecutionContext(episode_id="")  # Empty episode_id

        result = await python_executor(parameters, context)

        # Should fail due to empty/missing episode environment
        assert result.success is False

    def test_validate_parameters_valid(self, python_executor):
        """Test parameter validation with valid parameters."""
        parameters = PythonParameters(code="print('hello')", working_dir="/workspace")

        result = python_executor.validate_parameters(parameters)
        assert result.valid is True


class TestPythonExecutorTargetContainer:
    """Tests for target_container execution path in PythonExecutor."""

    @pytest.fixture
    def mock_sandbox_manager_no_env(self) -> MagicMock:
        """Create a SandboxEnvironmentManager that returns None for episode environments."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {
            "image": "saber/sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser",
        }
        manager.get_episode_environment.return_value = None
        return manager

    @pytest.fixture
    def python_executor_with_target_container(self, mock_sandbox_manager_no_env: MagicMock) -> PythonExecutor:
        """Create a PythonExecutor with target_container configured and no episode environment."""
        config = ExecutorConfig(timeout=30.0, target_container="saber-excytin-sandbox")
        return PythonExecutor(sandbox_manager=mock_sandbox_manager_no_env, config=config)

    @pytest.fixture
    def mock_sandbox_manager_with_env_for_tc(self) -> MagicMock:
        """Create a SandboxEnvironmentManager that returns a valid environment."""
        env = MagicMock()
        container_mock = MagicMock()
        container_mock.id = "container123456789"
        env.get_execution_container.return_value = container_mock
        env.execute_command = MagicMock()

        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {
            "image": "saber/sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser",
        }
        manager.get_episode_environment.return_value = env
        return manager

    @pytest.fixture
    def python_executor_with_target_container_and_env(
        self, mock_sandbox_manager_with_env_for_tc: MagicMock
    ) -> PythonExecutor:
        """Create a PythonExecutor with target_container configured AND a valid episode environment."""
        config = ExecutorConfig(timeout=30.0, target_container="saber-excytin-sandbox")
        return PythonExecutor(sandbox_manager=mock_sandbox_manager_with_env_for_tc, config=config)

    @pytest.mark.asyncio
    async def test_execute_with_target_container_no_episode_env(
        self, python_executor_with_target_container: PythonExecutor
    ) -> None:
        """Test execution via direct ComposeOrchestrator when no episode environment exists."""
        command_result = CommandResult(exit_code=0, stdout="hello\n", stderr="", execution_time=0.3)

        with patch(
            "saber.server.execution.sandbox.compose_orchestrator.ComposeOrchestrator"
        ) as mock_orch_cls:
            mock_orchestrator = MagicMock()
            mock_orchestrator.execute_command = AsyncMock(return_value=command_result)
            mock_orch_cls.return_value = mock_orchestrator

            parameters = PythonParameters(code="print('hello')")
            context = ExecutionContext(episode_id="ep_test")

            result = await python_executor_with_target_container(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "hello\n"
        # PythonExecutor calls execute_command twice: create script file + run script
        assert mock_orchestrator.execute_command.call_count == 2
        for call in mock_orchestrator.execute_command.call_args_list:
            assert call.kwargs["target_container"] == "saber-excytin-sandbox"
        assert result.metadata["container_id"] == "direct:saber-excytin-sandbox"

    @pytest.mark.asyncio
    async def test_execute_with_target_container_and_episode_env(
        self,
        python_executor_with_target_container_and_env: PythonExecutor,
        mock_sandbox_manager_with_env_for_tc: MagicMock,
    ) -> None:
        """Test that target_container is passed through to environment.execute_command when both exist."""
        create_result = CommandResult(exit_code=0, stdout="", stderr="", execution_time=0.1)
        execute_result = CommandResult(exit_code=0, stdout="env output\n", stderr="", execution_time=0.5)

        env = mock_sandbox_manager_with_env_for_tc.get_episode_environment.return_value
        env.execute_command = AsyncMock(side_effect=[create_result, execute_result])

        parameters = PythonParameters(code="print('hello')")
        context = ExecutionContext(episode_id="ep_with_env")

        result = await python_executor_with_target_container_and_env(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "env output\n"
        assert env.execute_command.call_count == 2
        for call in env.execute_command.call_args_list:
            assert call.kwargs["target_container"] == "saber-excytin-sandbox"
        # container_id comes from environment, not from direct path
        assert result.metadata["container_id"] == "container123"

    @pytest.mark.asyncio
    async def test_execute_no_environment_no_target_container_raises(
        self, mock_sandbox_manager_no_env: MagicMock
    ) -> None:
        """Test error result when neither environment nor target_container exists."""
        config = ExecutorConfig(timeout=30.0)  # No target_container
        executor = PythonExecutor(sandbox_manager=mock_sandbox_manager_no_env, config=config)

        parameters = PythonParameters(code="print('hello')")
        context = ExecutionContext(episode_id="ep_nothing")

        result = await executor(parameters, context)

        assert result.success is False
        assert "No sandbox environment and no target_container configured" in result.error
