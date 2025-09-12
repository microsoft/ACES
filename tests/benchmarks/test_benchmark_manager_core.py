"""
Unit tests for BenchmarkManager - Core Functionality.

Tests initialization, task loading, basic operations, and task/subtask retrieval.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from saber.server.benchmarks.exceptions import (
    InvalidTaskDefinitionException,
    SubTaskNotFoundException,
    TaskNotFoundException,
)
from saber.server.benchmarks.subtask import SubTask
from saber.server.benchmarks.task import Task
from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.benchmark_manager import BenchmarkManager


class TestBenchmarkManagerCore:
    """Test cases for BenchmarkManager core functionality."""

    def _create_benchmark_manager_from_temp_config_dir(self, temp_config_dir):
        """Helper to create BenchmarkManager from temp directory fixture."""
        return BenchmarkManager("malware_classification", temp_config_dir)

    def test_load_tasks_from_yaml_success(self, tmp_path, temp_config_dir_helper, sample_task_yaml):
        """Test successful loading of tasks from YAML."""
        temp_config_dir = temp_config_dir_helper(tmp_path, sample_task_yaml)
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        # Clear tasks and reload
        manager.tasks = {}
        manager.load_tasks_from_yaml()

        assert len(manager.tasks) == 1
        assert "malware_family_analysis" in manager.tasks

    @patch.object(BenchmarkConfigLoader, "load_tasks_from_file")
    def test_load_tasks_from_yaml_delegates_to_config_loader(self, mock_load, temp_config_dir):
        """Test that loading delegates to BenchmarkConfigLoader."""
        mock_task = Mock(spec=Task)
        mock_task.prompt_template_file = "test_prompt.md"
        mock_task.task_id = "test_task"
        mock_tasks = {"test_task": mock_task}
        mock_load.return_value = mock_tasks

        # Also mock the prompt generator validation to prevent template validation errors
        with patch.object(BenchmarkManager, 'validate_all_task_templates'):
            manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        # Should be called with the full path to tasks.yaml
        expected_path = str(Path(temp_config_dir) / "tasks.yaml")
        mock_load.assert_called_with(expected_path)
        assert manager.tasks == mock_tasks

    def test_load_tasks_invalid_file_raises_exception(self):
        """Test that loading invalid file raises exception."""
        from saber.server.benchmarks.prompt_generator import TemplateValidationError
        with pytest.raises(TemplateValidationError):
            BenchmarkManager("malware_classification", "/nonexistent/file.yaml")

    def test_get_task_success(self, temp_config_dir):
        """Test successful task retrieval."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        task = manager.get_task("malware_family_analysis")

        assert isinstance(task, Task)
        assert task.task_id == "malware_family_analysis"

    def test_get_task_not_found(self, temp_config_dir):
        """Test task retrieval when task doesn't exist."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        with pytest.raises(TaskNotFoundException) as exc_info:
            manager.get_task("nonexistent_task")

        assert exc_info.value.task_id == "nonexistent_task"
        assert "nonexistent_task" in str(exc_info.value)

    def test_get_subtask_success(self, temp_config_dir):
        """Test successful subtask retrieval."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        subtask = manager.get_subtask("malware_family_analysis", "static_analysis")

        assert isinstance(subtask, SubTask)
        assert subtask.subtask_id == "static_analysis"
        assert subtask.task_id == "malware_family_analysis"

    def test_get_subtask_task_not_found(self, temp_config_dir):
        """Test subtask retrieval when parent task doesn't exist."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        with pytest.raises(TaskNotFoundException):
            manager.get_subtask("nonexistent_task", "some_subtask")

    def test_get_subtask_subtask_not_found(self, temp_config_dir):
        """Test subtask retrieval when subtask doesn't exist."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        with pytest.raises(SubTaskNotFoundException) as exc_info:
            manager.get_subtask("malware_family_analysis", "nonexistent_subtask")

        assert exc_info.value.task_id == "malware_family_analysis"
        assert exc_info.value.subtask_id == "nonexistent_subtask"

    def test_list_tasks(self, temp_config_dir):
        """Test listing all available tasks."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        task_list = manager.list_tasks()

        assert isinstance(task_list, list)
        assert len(task_list) == 1

        task_info = task_list[0]
        assert task_info["task_id"] == "malware_family_analysis"
        assert task_info["title"] == "Malware Family Classification and Analysis"
        assert task_info["description"] is not None
        assert task_info["subtask_count"] == 2

    def test_list_tasks_empty(self, tmp_path, temp_config_dir_helper):
        """Test listing tasks when no tasks are defined."""
        # Create tasks.yaml in the tmp directory
        yaml_content = """
domain: "malware_classification"

benchmark_config:
  episode_attempts: 1

global_defaults:
  execution_config:
    timeout: 300
  episode_config:
    max_steps: 50

executors:
  - bash_executor
  - python_executor

tasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("malware_classification", config_dir)

        task_list = manager.list_tasks()
        assert task_list == []

    def test_list_tasks_multiple_tasks(self, tmp_path, temp_config_dir_helper):
        """Test listing multiple tasks."""
        # Create tasks.yaml in the tmp directory
        yaml_content = """
domain: "malware_classification"

benchmark_config:
  episode_attempts: 2

global_defaults:
  execution_config:
    timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "task1"
    title: "First Task"
    description: "First test task"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
        - python_executor
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        points: 100
    subtasks:
      - subtask_id: "subtask1"
        title: "SubTask 1"
        description: "First subtask"
        objective: "Complete first"
  - task_id: "task2"
    title: "Second Task"
    description: "Second test task"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
        - python_executor
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        points: 100
    subtasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("malware_classification", config_dir)

        task_list = manager.list_tasks()
        assert len(task_list) == 2

        task_ids = [task["task_id"] for task in task_list]
        assert "task1" in task_ids
        assert "task2" in task_ids

        # Check subtask counts
        task1_info = next(task for task in task_list if task["task_id"] == "task1")
        task2_info = next(task for task in task_list if task["task_id"] == "task2")
        assert task1_info["subtask_count"] == 1
        assert task2_info["subtask_count"] == 0

    def test_domain_consistency(self, temp_config_dir):
        """Test that domain is consistently used across components."""
        domain = "malware_classification"
        manager = BenchmarkManager(domain, temp_config_dir)

        assert manager.domain == domain
        assert manager.config_loader.domain == domain

        # Check that loaded tasks have correct domain
        for task in manager.tasks.values():
            assert task.domain == domain

    def test_task_manager_logging_behavior(self, temp_config_dir, caplog):
        """Test that BenchmarkManager provides appropriate logging."""
        import logging

        with caplog.at_level(logging.INFO):
            manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        # Check for initialization logs
        assert any(
            "Initializing BenchmarkManager for domain 'malware_classification'" in record.message
            for record in caplog.records
        )
        assert any("BenchmarkManager initialization complete" in record.message for record in caplog.records)
        assert any("Loaded 1 tasks" in record.message for record in caplog.records)

    def test_task_manager_file_path_handling(self, temp_config_dir):
        """Test that BenchmarkManager correctly handles file path types."""
        # Test with string path
        manager1 = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)
        assert isinstance(manager1.tasks_file_path, Path)

        # Test with Path object
        path_obj = Path(temp_config_dir)
        manager2 = BenchmarkManager("malware_classification", path_obj)
        assert isinstance(manager2.tasks_file_path, Path)
        # The tasks_file_path should be the tasks.yaml file within the directory
        assert manager2.tasks_file_path == path_obj / "tasks.yaml"


class TestBenchmarkManagerBenchmarkConfig:
    """Test cases for BenchmarkManager benchmark configuration functionality."""

    def test_get_benchmark_config_success(self, tmp_path, temp_config_dir_helper):
        """Test successful retrieval of benchmark configuration."""
        yaml_content = """
domain: "webapp_pentest"

benchmark_config:
  episode_attempts: 5
  max_duration_minutes: 30

global_defaults:
  execution_config:
    timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
        - python_executor
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        points: 100
    subtasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("webapp_pentest", config_dir)
        config = manager.get_benchmark_config()

        assert config["episode_attempts"] == 5
        assert config["max_duration_minutes"] == 30

    def test_start_benchmark_success(self, tmp_path, temp_config_dir_helper):
        """Test successful benchmark session start."""
        yaml_content = """
domain: "webapp_pentest"

benchmark_config:
  episode_attempts: 3

global_defaults:
  execution_config:
    timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "task1"
    title: "Task 1"
    description: "First task"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
        - python_executor
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        points: 100
    subtasks: []
  - task_id: "task2"
    title: "Task 2"
    description: "Second task"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
        - python_executor
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        points: 100
    subtasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("webapp_pentest", config_dir)

        # Use current architecture - get benchmark info
        benchmark_info = manager.get_benchmark_info()
        result = benchmark_info.to_dict()

        assert result["domain"] == "webapp_pentest"
        assert result["total_tasks"] == 2
        assert "task1" in [task["task_id"] for task in result["tasks"]]
        assert "task2" in [task["task_id"] for task in result["tasks"]]

    def test_list_benchmark_tasks_with_episode_attempts(self, tmp_path, temp_config_dir_helper):
        """Test listing tasks with episode attempts information."""
        yaml_content = """
domain: "webapp_pentest"

benchmark_config:
  episode_attempts: 3

global_defaults:
  execution_config:
    timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "task_default"
    title: "Task with Default"
    description: "Uses domain default"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
        - python_executor
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        points: 100
    subtasks: []
  - task_id: "task_override"
    title: "Task with Override"
    description: "Overrides domain default"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
        - python_executor
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        points: 100
    benchmark_config:
      episode_attempts: 10
    subtasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("webapp_pentest", config_dir)
        tasks = manager.list_benchmark_tasks()

        assert len(tasks) == 2

        task_default = next(t for t in tasks if t["task_id"] == "task_default")
        task_override = next(t for t in tasks if t["task_id"] == "task_override")

        assert task_default["episode_attempts"] == 3  # Domain default
        assert task_override["episode_attempts"] == 10  # Task override
        assert "benchmark_config" in task_default
        assert "benchmark_config" in task_override

    def test_task_get_episode_attempts(self, tmp_path, temp_config_dir_helper):
        """Test that tasks correctly return episode attempts."""
        yaml_content = """
domain: "webapp_pentest"

benchmark_config:
  episode_attempts: 7

global_defaults:
  execution_config:
    timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
        - python_executor
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        points: 100
    benchmark_config:
      episode_attempts: 15
    subtasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("webapp_pentest", config_dir)
        task = manager.get_task("test_task")

        assert task.get_episode_attempts() == 15

    def test_missing_benchmark_config_fails(self, sample_yaml_missing_benchmark_config, tmp_path, temp_config_dir_helper):
        """Test that missing benchmark_config section causes failure."""
        config_dir = temp_config_dir_helper(tmp_path, sample_yaml_missing_benchmark_config)

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            BenchmarkManager("malware_classification", config_dir)

        # Updated error message reflects the new global defaults system
        assert "Missing required 'episode_attempts' in benchmark configuration" in str(exc_info.value)
        assert "global_defaults.benchmark_config" in str(exc_info.value)

    def test_missing_episode_attempts_fails(self, sample_yaml_missing_episode_attempts, tmp_path, temp_config_dir_helper):
        """Test that missing episode_attempts causes failure."""
        config_dir = temp_config_dir_helper(tmp_path, sample_yaml_missing_episode_attempts)

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            BenchmarkManager("malware_classification", config_dir)

        assert "Missing required 'episode_attempts'" in str(exc_info.value)

    def test_invalid_episode_attempts_fails(self, sample_yaml_invalid_episode_attempts, tmp_path, temp_config_dir_helper):
        """Test that invalid episode_attempts value causes failure."""
        config_dir = temp_config_dir_helper(tmp_path, sample_yaml_invalid_episode_attempts)

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            BenchmarkManager("malware_classification", config_dir)

        assert "episode_attempts must be a positive integer" in str(exc_info.value)
