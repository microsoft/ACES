"""
Tests for Python executor.

This module tests the secure Python script executor that executes Python code
in Docker containers with dependency management and security validation.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from saber.server.execution.base import CommandResult, ParameterType, ValidationResult
from saber.server.execution.executors.python_executor import PythonExecutor
from saber.server.execution.sandbox.sandbox_manager import SandboxManager
from saber.server.execution.exceptions import SandboxExecutionError


class TestPythonExecutor:
    """Test cases for Python executor."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxManager."""
        manager = MagicMock(spec=SandboxManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/python-sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser"
        }
        return manager

    @pytest.fixture
    def python_executor(self, mock_sandbox_manager):
        """Create a Python executor instance for testing."""
        return PythonExecutor(sandbox_manager=mock_sandbox_manager, timeout=60.0)

    @pytest.fixture
    def mock_docker_environment(self):
        """Create a mock Docker execution environment."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        env = MagicMock()
        # Mock the new interface
        container_mock = MagicMock()
        container_mock.id = "container123456789"
        env.get_execution_container.return_value = container_mock
        env.execute_command = MagicMock()  # Not async anymore
        return env

    def test_initialization(self, mock_sandbox_manager):
        """Test Python executor initialization."""
        executor = PythonExecutor(sandbox_manager=mock_sandbox_manager, timeout=120.0)

        assert executor.get_timeout() == 120.0

        # Check that parameters were added
        params = executor.get_parameters()
        assert "code" in params
        assert "requirements" in params
        assert "template" in params
        assert "working_dir" in params

        # Verify parameter definitions
        code_param = params["code"]
        assert code_param.name == "code"
        assert code_param.type == ParameterType.STRING
        assert code_param.required is True

        requirements_param = params["requirements"]
        assert requirements_param.name == "requirements"
        assert requirements_param.type == ParameterType.ARRAY
        assert requirements_param.required is False
        assert requirements_param.default == []

    def test_initialization_without_sandbox_manager(self):
        """Test that initialization fails without sandbox manager."""
        with pytest.raises(SandboxExecutionError, match="sandbox_manager is required"):
            PythonExecutor(sandbox_manager=None, timeout=60.0)

    def test_security_command_metadata(self, python_executor):
        """Test that security command metadata is properly set."""
        metadata = python_executor._security_command_metadata

        assert metadata["domain"] == "python"
        assert metadata["name"] == "python_script"
        assert metadata["description"] == "Execute Python scripts in Docker containers with dependency management"
        assert metadata["author"] == "SABER Team"
        assert metadata["security_level"] == "high"
        assert metadata["requires_validation"] is True

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
        parameters = {"code": "print('Hello, World!')"}
        context = {}

        script = python_executor.build_python_script(parameters, context)

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

        parameters = {"code": "df = pd.DataFrame({'x': [1, 2, 3]})", "template": "data_analysis"}
        context = {}

        script = python_executor.build_python_script(parameters, context)

        assert "import pandas as pd" in script
        assert "import numpy as np" in script
        assert "df = pd.DataFrame({'x': [1, 2, 3]})" in script

    @pytest.mark.asyncio
    async def test_install_requirements_empty(self, python_executor, mock_docker_environment):
        """Test installing empty requirements list."""
        result = await python_executor.install_requirements(mock_docker_environment, [])

        assert result.success is True
        assert "No requirements to install" in result.data["message"]

    @pytest.mark.asyncio
    async def test_install_requirements_success(self, python_executor, mock_docker_environment):
        """Test successful package installation."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        # Mock successful pip install
        pip_result = CommandResult(
            exit_code=0,
            stdout="Successfully installed requests-2.28.1",
            stderr="",
            execution_time=5.0
        )
        mock_docker_environment.execute_command.return_value = pip_result

        result = await python_executor.install_requirements(mock_docker_environment, ["requests"])

        assert result.success is True
        assert "requests" in result.data["installed_packages"]
        mock_docker_environment.execute_command.assert_called_once_with(
            command=["pip", "install", "requests"]
        )

    @pytest.mark.asyncio
    async def test_install_requirements_failure(self, python_executor, mock_docker_environment):
        """Test failed package installation."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        # Mock failed pip install
        pip_result = CommandResult(
            exit_code=1,
            stdout="",
            stderr="ERROR: Could not find a version that satisfies the requirement invalid-package",
            execution_time=2.0
        )
        mock_docker_environment.execute_command.return_value = pip_result

        result = await python_executor.install_requirements(mock_docker_environment, ["invalid-package"])

        assert result.success is False
        assert "Package installation failed" in result.error

    def test_parse_python_output_success(self, python_executor):
        """Test parsing successful Python execution output."""
        result = python_executor.parse_python_output(
            stdout="Hello, World!\n",
            stderr="",
            return_code=0,
            script_path="/tmp/script.py"
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
            script_path="/tmp/script.py"
        )

        assert result.success is False
        assert "Python script failed with exit code 1" in result.error
        assert "NameError" in result.error

    def test_get_python_environment(self, python_executor):
        """Test getting Python environment information."""
        with patch.object(python_executor, 'get_session_environment') as mock_get_env:
            mock_env = MagicMock()
            mock_get_env.return_value = mock_env

            env_info = python_executor.get_python_environment("test_session")

            assert env_info["session_id"] == "test_session"
            assert env_info["container_ready"] is True

    @pytest.mark.asyncio
    async def test_execute_success(self, python_executor, mock_sandbox_manager, mock_docker_environment):
        """Test successful Python script execution."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        # Mock environment creation
        mock_sandbox_manager.get_session_environment.return_value = mock_docker_environment
        python_executor._sandbox_manager = mock_sandbox_manager

        # Mock script creation and execution
        create_result = CommandResult(exit_code=0, stdout="", stderr="", execution_time=0.1)
        execute_result = CommandResult(
            exit_code=0,
            stdout="Hello, World!\n",
            stderr="",
            execution_time=1.0
        )

        mock_docker_environment.execute_command.side_effect = [create_result, execute_result]

        parameters = {"code": "print('Hello, World!')"}
        context = {"session_id": "test123"}

        result = await python_executor.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "Hello, World!\n"
        assert result.metadata["session_id"] == "test123"
        assert result.metadata["container_id"] == "container123"  # First 12 chars: container123

    @pytest.mark.asyncio
    async def test_execute_with_requirements(self, python_executor, mock_sandbox_manager, mock_docker_environment):
        """Test Python script execution with package requirements."""
        from saber.server.execution.sandbox.docker_environment import CommandResult

        # Mock environment
        mock_sandbox_manager.get_session_environment.return_value = mock_docker_environment
        python_executor._sandbox_manager = mock_sandbox_manager

        # Mock pip install, script creation, and execution
        pip_result = CommandResult(exit_code=0, stdout="Successfully installed requests", stderr="", execution_time=5.0)
        create_result = CommandResult(exit_code=0, stdout="", stderr="", execution_time=0.1)
        execute_result = CommandResult(exit_code=0, stdout="Request sent\n", stderr="", execution_time=2.0)

        mock_docker_environment.execute_command.side_effect = [pip_result, create_result, execute_result]

        parameters = {
            "code": "import requests; print('Request sent')",
            "requirements": ["requests"]
        }
        context = {"session_id": "test123"}

        result = await python_executor.execute(parameters, context)

        assert result.success is True
        assert result.data["stdout"] == "Request sent\n"
        assert result.metadata["requirements_installed"] == ["requests"]

    @pytest.mark.asyncio
    async def test_execute_validation_failure(self, python_executor):
        """Test execution with invalid Python code."""
        parameters = {"code": ""}  # Empty code
        context = {"session_id": "test123"}

        result = await python_executor.execute(parameters, context)

        assert result.success is False
        assert "Python code validation failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_session_id(self, python_executor):
        """Test execution without session ID."""
        parameters = {"code": "print('hello')"}
        context = {}  # Missing session_id

        result = await python_executor.execute(parameters, context)

        assert result.success is False
        assert "session_id required" in result.error

    def test_validate_parameters_valid(self, python_executor):
        """Test parameter validation with valid parameters."""
        parameters = {
            "code": "print('hello')",
            "requirements": ["requests"],
            "working_dir": "/workspace"
        }

        result = python_executor.validate_parameters(parameters)
        assert result.valid is True

    def test_validate_parameters_invalid_requirements(self, python_executor):
        """Test parameter validation with invalid requirements."""
        parameters = {
            "code": "print('hello')",
            "requirements": "not_a_list"  # Should be a list
        }

        result = python_executor.validate_parameters(parameters)
        assert result.valid is False
        assert "must be an array" in str(result.errors) or "requirements must be a list" in str(result.errors)

    def test_validate_parameters_invalid_requirement_item(self, python_executor):
        """Test parameter validation with invalid requirement item."""
        parameters = {
            "code": "print('hello')",
            "requirements": ["requests", 123]  # 123 is not a string
        }

        result = python_executor.validate_parameters(parameters)
        assert result.valid is False
        assert "requirement must be string" in str(result.errors)

    def test_get_security_info(self, python_executor):
        """Test getting security information."""
        info = python_executor.get_docker_info()

        assert info["execution_environment"] == "docker_container"
        assert "timeout" in info
        assert "docker_config" in info
