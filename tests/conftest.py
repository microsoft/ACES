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


# Task Framework Test Fixtures

@pytest.fixture
def sample_task_yaml():
    """Sample YAML content for task configuration testing."""
    return """
domain: "malware_classification"
tasks:
  - task_id: "malware_family_analysis"
    title: "Malware Family Classification and Analysis"
    description: "Analyze malware sample to determine family, capabilities, and threat level"
    initial_context:
      sample_path: "/data/samples/unknown_sample.exe"
      analysis_timeout: 300
    subtasks:
      - subtask_id: "static_analysis"
        title: "Static Analysis"
        description: "Perform static analysis of the malware sample"
        objective: "Extract basic file properties, strings, and structural information"
        completion_conditions: ["file unknown_sample.exe", "strings unknown_sample.exe"]
        depends_on: []
      - subtask_id: "dynamic_analysis"
        title: "Dynamic Analysis"
        description: "Execute sample in sandboxed environment"
        objective: "Observe runtime behavior and system interactions"
        completion_conditions: ["sandbox_run unknown_sample.exe"]
        depends_on: ["static_analysis"]
"""


@pytest.fixture
def temp_tasks_file(tmp_path, sample_task_yaml):
    """Create a temporary YAML file with task configuration."""
    tasks_file = tmp_path / "test_tasks.yaml"
    tasks_file.write_text(sample_task_yaml)
    return str(tasks_file)


@pytest.fixture
def sample_subtask_data():
    """Sample subtask data for testing."""
    return {
        "subtask_id": "test_subtask",
        "task_id": "test_task",
        "title": "Test SubTask",
        "description": "A test subtask for unit testing",
        "objective": "Complete the test objectives",
        "completion_conditions": ["test_command", "another_command"],
        "depends_on": ["prerequisite_subtask"]
    }


@pytest.fixture
def sample_action():
    """Sample action for episode testing."""
    from saber.server.tasks.episodes.episode import Action
    return Action(
        tool_name="docker_cli_executor",
        parameters={"command": "file sample.exe"},
        command="file sample.exe"
    )


@pytest.fixture
def sample_tool_result():
    """Sample tool result for testing."""
    return ToolResult.success_result({
        "output": "sample.exe: PE32 executable",
        "file_type": "PE32"
    })
