"""
Test configuration and utilities for SABER test suite.

This module provides common test fixtures, mock objects, and utilities
that can be shared across multiple test modules.
"""

import asyncio
import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from unittest.mock import patch, Mock

from saber.server.tools.base import ToolResult, Parameter, ParameterType
from saber.server.tools.executors.base_executors import ToolExecutor
from saber.server.tools.executors.cli import DockerCLIExecutor
from saber.server.tools.utils.security_validator import SecurityValidator
from saber.server.tools.execution_manager import ExecutionManager

logger = logging.getLogger(__name__)


class MockExecutor(ToolExecutor):
    """
    Mock executor for testing and development.

    Returns predefined responses for testing purposes.
    """

    def __init__(self, mock_response: Any = None, mock_error: Optional[str] = None,
                 delay: float = 0.0, timeout: Optional[float] = None):
        """
        Initialize mock executor.

        Args:
            mock_response: Response to return on success
            mock_error: Error message to return on failure
            delay: Artificial delay in seconds
            timeout: Execution timeout in seconds
        """
        super().__init__(timeout)
        self._mock_response = mock_response
        self._mock_error = mock_error
        self._delay = delay

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """
        Execute mock tool.

        Args:
            parameters: Tool parameters (logged but not used)
            context: Execution context (logged but not used)

        Returns:
            ToolResult with mock response
        """
        # Add artificial delay if specified
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        # Log execution for debugging
        logger.debug(f"Mock tool executed with parameters: {parameters}")

        # Return error or success based on configuration
        if self._mock_error:
            return ToolResult.error_result(self._mock_error)
        else:
            return ToolResult.success_result(self._mock_response or {"status": "success", "parameters": parameters})


class TestCommandLineExecutor(DockerCLIExecutor):
    """Test implementation of DockerCLIExecutor for security testing."""

    def __init__(self, command="echo", **kwargs):
        super().__init__(**kwargs)
        self._command = command

    def build_command(self, parameters, context):
        """Simple command builder for testing."""
        text = parameters.get("text", "hello")
        return [self._command, text]

    def parse_output(self, stdout, stderr, return_code):
        """Simple output parser for testing."""
        if return_code != 0:
            return ToolResult.error_result(f"Command failed: {stderr}")
        return ToolResult.success_result({"output": stdout.strip()})


class TestEchoTool(DockerCLIExecutor):
    """Test tool for configuration testing."""

    _security_tool_metadata = {
        "domain": "malware",
        "name": "test_echo_tool",
        "description": "Test echo tool for configuration testing",
        "author": "Test Suite"
    }

    def __init__(self):
        super().__init__(command="echo", allowed_commands=["echo"])
        self.add_parameter(Parameter(
            name="text",
            type=ParameterType.STRING,
            description="Text to echo",
            required=True
        ))

    def build_command(self, parameters, context):
        return ["echo", parameters["text"]]

    def parse_output(self, stdout, stderr, return_code):
        if return_code != 0:
            return ToolResult.error_result(f"Command failed: {stderr}")
        return ToolResult.success_result({"output": stdout.strip()})


class TestDangerousTool(DockerCLIExecutor):
    """Test tool that should be blocked by security configuration."""

    _security_tool_metadata = {
        "domain": "malware",
        "name": "test_dangerous_tool",
        "description": "Dangerous tool for security testing",
        "author": "Test Suite"
    }

    def __init__(self):
        # This should fail if security is properly configured - rm is blocked and not allowed
        super().__init__()
        self._command = "rm"

    def build_command(self, parameters, context):
        return ["rm", "-rf", "/"]

    def parse_output(self, stdout, stderr, return_code):
        return ToolResult.success_result({"output": "Should never execute"})


# Test fixtures
@pytest.fixture
def test_config_path():
    """Path to test configuration file."""
    # Look for config in the appropriate subdirectory based on test location
    test_file_path = Path(__file__).parent

    # Check if we're in a subdirectory and adjust accordingly
    if (test_file_path / "tools" / "config" / "test_tool_config.yaml").exists():
        return test_file_path / "tools" / "config" / "test_tool_config.yaml"
    elif (test_file_path / "config" / "test_tool_config.yaml").exists():
        return test_file_path / "config" / "test_tool_config.yaml"
    else:
        # Fallback to original location
        return test_file_path / "config" / "test_tool_config.yaml"


@pytest.fixture
def test_config_dict(test_config_path):
    """Load test configuration as dictionary."""
    with open(test_config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def registry_with_config(test_config_dict):
    """Create ExecutionManager with test configuration."""
    return ExecutionManager(config=test_config_dict)


@pytest.fixture
def security_validator():
    """Create a SecurityValidator instance for testing."""
    return SecurityValidator()


@pytest.fixture
def temp_sandbox_dir():
    """Create a temporary directory for sandbox testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
