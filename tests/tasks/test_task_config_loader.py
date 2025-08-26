"""
Unit tests for TaskConfigLoader.

Tests YAML task configuration loading and parsing.
Simplified after removing dependency validation and progression logic.
"""

import os
import tempfile
from pathlib import Path

import pytest

from saber.server.tasks.subtask import SubTask
from saber.server.tasks.task import Task
from saber.server.tasks.task_config_loader import TaskConfigLoader


class TestTaskConfigLoader:
    """Test cases for TaskConfigLoader functionality."""

    def test_task_config_loader_init(self):
        """Test TaskConfigLoader initialization."""
        loader = TaskConfigLoader("test_domain")
        assert loader.domain == "test_domain"

    def test_load_tasks_from_file_basic(self):
        """Test loading basic tasks from YAML file."""
        yaml_content = """
domain: test_domain
tasks:
  - task_id: test_task
    title: Test Task
    description: A simple test task
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = TaskConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert isinstance(tasks, dict)
            assert "test_task" in tasks
            task = tasks["test_task"]
            assert task.task_id == "test_task"
            assert task.title == "Test Task"
            assert task.description == "A simple test task"
            assert task.subtasks == []
            assert task.domain == "test_domain"
        finally:
            os.unlink(temp_path)

    def test_load_tasks_with_subtasks(self):
        """Test loading tasks with subtasks from YAML."""
        yaml_content = """
domain: test_domain
tasks:
  - task_id: complex_task
    title: Complex Task
    description: A task with multiple subtasks
    initial_context:
      timeout: 300
    subtasks:
      - subtask_id: subtask1
        title: First Subtask
        description: First step of the task
        objective: Complete first step
      - subtask_id: subtask2
        title: Second Subtask
        description: Second step of the task
        objective: Complete second step
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = TaskConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "complex_task" in tasks
            task = tasks["complex_task"]
            assert task.task_id == "complex_task"
            assert task.title == "Complex Task"
            assert len(task.subtasks) == 2

            # Check first subtask
            subtask1 = task.subtasks[0]
            assert isinstance(subtask1, SubTask)
            assert subtask1.subtask_id == "subtask1"
            assert subtask1.title == "First Subtask"
            assert subtask1.objective == "Complete first step"

            # Check second subtask
            subtask2 = task.subtasks[1]
            assert isinstance(subtask2, SubTask)
            assert subtask2.subtask_id == "subtask2"
            assert subtask2.title == "Second Subtask"
            assert subtask2.objective == "Complete second step"

            # Check task context
            assert task.initial_context["timeout"] == 300
        finally:
            os.unlink(temp_path)

    def test_load_tasks_missing_file(self):
        """Test loading from non-existent file."""
        loader = TaskConfigLoader("test_domain")

        with pytest.raises(Exception):  # Should raise some kind of file not found error
            loader.load_tasks_from_file("/nonexistent/path/task.yaml")

    def test_load_tasks_invalid_yaml(self):
        """Test loading invalid YAML."""
        yaml_content = """
domain: test_domain
tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    subtasks: [
      - subtask_id: subtask1
        title: First Subtask
        # Missing closing bracket - invalid YAML
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = TaskConfigLoader("test_domain")
            with pytest.raises(Exception):  # Should raise YAML parsing error
                loader.load_tasks_from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_tasks_missing_domain(self):
        """Test loading YAML missing domain field."""
        yaml_content = """
tasks:
  - task_id: test_task
    title: Test Task
    description: A test task without domain
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = TaskConfigLoader("test_domain")
            with pytest.raises(Exception):  # Should raise validation error
                loader.load_tasks_from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_tasks_with_execution_timeout_config(self):
        """Test loading tasks with execution timeout configuration."""
        yaml_content = """
domain: test_domain
tasks:
  - task_id: timeout_task
    title: Task with Custom Timeout
    description: A task with custom execution timeout
    environment: test_env
    execution_config:
      allowed_executors: ["cli", "python"]
      timeout: 180
    episode_config:
      max_steps: 25
    subtasks:
      - subtask_id: test_subtask
        title: Test Subtask
        description: Test subtask with timeout
        objective: Complete with custom timeout
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = TaskConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "timeout_task" in tasks
            task = tasks["timeout_task"]

            # Verify basic task properties
            assert task.task_id == "timeout_task"
            assert task.title == "Task with Custom Timeout"
            assert task.environment == "test_env"

            # Verify execution config contains timeout
            assert task.execution_config is not None
            assert "timeout" in task.execution_config
            assert task.execution_config["timeout"] == 180
            assert task.execution_config["allowed_executors"] == ["cli", "python"]

            # Verify episode config
            assert task.episode_config is not None
            assert task.episode_config["max_steps"] == 25

            # Verify subtask
            assert len(task.subtasks) == 1
            subtask = task.subtasks[0]
            assert subtask.subtask_id == "test_subtask"
            assert subtask.title == "Test Subtask"

        finally:
            os.unlink(temp_path)

    def test_load_tasks_without_execution_timeout_config(self):
        """Test loading tasks without execution timeout (should use defaults)."""
        yaml_content = """
domain: test_domain
tasks:
  - task_id: default_timeout_task
    title: Task with Default Timeout
    description: A task without custom timeout configuration
    execution_config:
      allowed_executors: ["cli"]
    subtasks:
      - subtask_id: test_subtask
        title: Test Subtask
        description: Test subtask
        objective: Complete with default timeout
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = TaskConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "default_timeout_task" in tasks
            task = tasks["default_timeout_task"]

            # Verify execution config exists but has no timeout
            assert task.execution_config is not None
            assert "timeout" not in task.execution_config
            assert task.execution_config["allowed_executors"] == ["cli"]

        finally:
            os.unlink(temp_path)
