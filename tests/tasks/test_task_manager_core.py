"""
Unit tests for TaskManager - Core Functionality.

Tests initialization, task loading, basic operations, and task/subtask retrieval.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from saber.server.tasks.task_manager import TaskManager
from saber.server.tasks.task import Task
from saber.server.tasks.subtask import SubTask
from saber.server.tasks.task_config_loader import TaskConfigLoader
from saber.server.tasks.exceptions import (
    TaskNotFoundException,
    SubTaskNotFoundException,
    InvalidTaskDefinitionException
)


class TestTaskManagerCore:
    """Test cases for TaskManager core functionality."""

    def _create_task_manager_from_temp_config_dir(self, temp_config_dir):
        """Helper to create TaskManager from temp directory fixture."""
        return TaskManager("malware_classification", temp_config_dir)

    def test_load_tasks_from_yaml_success(self, temp_config_dir):
        """Test successful loading of tasks from YAML."""
        manager = self._create_task_manager_from_temp_config_dir(temp_config_dir)

        # Clear tasks and reload
        manager.tasks = {}
        manager.load_tasks_from_yaml()

        assert len(manager.tasks) == 1
        assert "malware_family_analysis" in manager.tasks

    @patch.object(TaskConfigLoader, 'load_tasks_from_file')
    def test_load_tasks_from_yaml_delegates_to_config_loader(self, mock_load, temp_config_dir):
        """Test that loading delegates to TaskConfigLoader."""
        mock_tasks = {"test_task": Mock(spec=Task)}
        mock_load.return_value = mock_tasks

        manager = self._create_task_manager_from_temp_config_dir(temp_config_dir)

        # Should be called with the full path to tasks.yaml
        expected_path = str(Path(temp_config_dir) / "tasks.yaml")
        mock_load.assert_called_with(expected_path)
        assert manager.tasks == mock_tasks

    def test_load_tasks_invalid_file_raises_exception(self):
        """Test that loading invalid file raises exception."""
        with pytest.raises(InvalidTaskDefinitionException):
            TaskManager("malware_classification", "/nonexistent/file.yaml")

    def test_get_task_success(self, temp_config_dir):
        """Test successful task retrieval."""
        manager = self._create_task_manager_from_temp_config_dir(temp_config_dir)

        task = manager.get_task("malware_family_analysis")

        assert isinstance(task, Task)
        assert task.task_id == "malware_family_analysis"

    def test_get_task_not_found(self, temp_config_dir):
        """Test task retrieval when task doesn't exist."""
        manager = self._create_task_manager_from_temp_config_dir(temp_config_dir)

        with pytest.raises(TaskNotFoundException) as exc_info:
            manager.get_task("nonexistent_task")

        assert exc_info.value.task_id == "nonexistent_task"
        assert "nonexistent_task" in str(exc_info.value)

    def test_get_subtask_success(self, temp_config_dir):
        """Test successful subtask retrieval."""
        manager = self._create_task_manager_from_temp_config_dir(temp_config_dir)

        subtask = manager.get_subtask("malware_family_analysis", "static_analysis")

        assert isinstance(subtask, SubTask)
        assert subtask.subtask_id == "static_analysis"
        assert subtask.task_id == "malware_family_analysis"

    def test_get_subtask_task_not_found(self, temp_config_dir):
        """Test subtask retrieval when parent task doesn't exist."""
        manager = self._create_task_manager_from_temp_config_dir(temp_config_dir)

        with pytest.raises(TaskNotFoundException):
            manager.get_subtask("nonexistent_task", "some_subtask")

    def test_get_subtask_subtask_not_found(self, temp_config_dir):
        """Test subtask retrieval when subtask doesn't exist."""
        manager = self._create_task_manager_from_temp_config_dir(temp_config_dir)

        with pytest.raises(SubTaskNotFoundException) as exc_info:
            manager.get_subtask("malware_family_analysis", "nonexistent_subtask")

        assert exc_info.value.task_id == "malware_family_analysis"
        assert exc_info.value.subtask_id == "nonexistent_subtask"

    def test_list_tasks(self, temp_config_dir):
        """Test listing all available tasks."""
        manager = self._create_task_manager_from_temp_config_dir(temp_config_dir)

        task_list = manager.list_tasks()

        assert isinstance(task_list, list)
        assert len(task_list) == 1

        task_info = task_list[0]
        assert task_info["task_id"] == "malware_family_analysis"
        assert task_info["title"] == "Malware Family Classification and Analysis"
        assert task_info["description"] is not None
        assert task_info["subtask_count"] == 2

    def test_list_tasks_empty(self, tmp_path):
        """Test listing tasks when no tasks are defined."""
        # Create tasks.yaml in the tmp directory
        tasks_yaml = tmp_path / "tasks.yaml"
        tasks_yaml.write_text("""
domain: "malware_classification"
tasks: []
""")

        manager = TaskManager("malware_classification", str(tmp_path))

        task_list = manager.list_tasks()
        assert task_list == []

    def test_list_tasks_multiple_tasks(self, tmp_path):
        """Test listing multiple tasks."""
        # Create tasks.yaml in the tmp directory
        tasks_yaml = tmp_path / "tasks.yaml"
        tasks_yaml.write_text("""
domain: "malware_classification"
tasks:
  - task_id: "task1"
    title: "First Task"
    description: "First test task"
    subtasks:
      - subtask_id: "subtask1"
        title: "SubTask 1"
        description: "First subtask"
        objective: "Complete first"
  - task_id: "task2"
    title: "Second Task"
    description: "Second test task"
    subtasks: []
""")

        manager = TaskManager("malware_classification", str(tmp_path))

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
        manager = TaskManager(domain, temp_config_dir)

        assert manager.domain == domain
        assert manager.config_loader.domain == domain

        # Check that loaded tasks have correct domain
        for task in manager.tasks.values():
            assert task.domain == domain

    def test_task_manager_logging_behavior(self, temp_config_dir, caplog):
        """Test that TaskManager provides appropriate logging."""
        import logging

        with caplog.at_level(logging.INFO):
            manager = self._create_task_manager_from_temp_config_dir(temp_config_dir)

        # Check for initialization logs
        assert any("Initializing TaskManager for domain 'malware_classification'" in record.message
                  for record in caplog.records)
        assert any("TaskManager initialization complete" in record.message
                  for record in caplog.records)
        assert any("Loaded 1 tasks" in record.message
                  for record in caplog.records)

    def test_task_manager_file_path_handling(self, temp_config_dir):
        """Test that TaskManager correctly handles file path types."""
        # Test with string path
        manager1 = self._create_task_manager_from_temp_config_dir(temp_config_dir)
        assert isinstance(manager1.tasks_file_path, Path)

        # Test with Path object
        path_obj = Path(temp_config_dir)
        manager2 = TaskManager("malware_classification", path_obj)
        assert isinstance(manager2.tasks_file_path, Path)
        # The tasks_file_path should be the tasks.yaml file within the directory
        assert manager2.tasks_file_path == path_obj / "tasks.yaml"
