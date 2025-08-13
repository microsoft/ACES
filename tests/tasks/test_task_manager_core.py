"""
Unit tests for TaskManager - Core Functionality.

Tests initialization, task loading, basic operations, and task/subtask retrieval.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from saber.server.tasks.task_manager import TaskManager
from saber.server.tasks.core.task import Task
from saber.server.tasks.core.subtask import SubTask
from saber.server.tasks.core.task_config_loader import TaskConfigLoader
from saber.server.tasks.episodes.episode_manager import EpisodeManager
from saber.server.tasks.exceptions import (
    TaskNotFoundException,
    SubTaskNotFoundException,
    EpisodeNotFoundException,
    InvalidTaskDefinitionException
)


class TestTaskManagerCore:
    """Test cases for TaskManager core functionality."""

    def test_load_tasks_from_yaml_success(self, temp_tasks_file):
        """Test successful loading of tasks from YAML."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # Clear tasks and reload
        manager.tasks = {}
        manager.load_tasks_from_yaml()

        assert len(manager.tasks) == 1
        assert "malware_family_analysis" in manager.tasks

    @patch.object(TaskConfigLoader, 'load_tasks_from_file')
    def test_load_tasks_from_yaml_delegates_to_config_loader(self, mock_load, temp_tasks_file):
        """Test that loading delegates to TaskConfigLoader."""
        mock_tasks = {"test_task": Mock(spec=Task)}
        mock_load.return_value = mock_tasks

        manager = TaskManager("malware_classification", temp_tasks_file)

        mock_load.assert_called_with(temp_tasks_file)
        assert manager.tasks == mock_tasks

    def test_load_tasks_invalid_file_raises_exception(self):
        """Test that loading invalid file raises exception."""
        with pytest.raises(InvalidTaskDefinitionException):
            TaskManager("malware_classification", "/nonexistent/file.yaml")

    def test_get_task_success(self, temp_tasks_file):
        """Test successful task retrieval."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        task = manager.get_task("malware_family_analysis")

        assert isinstance(task, Task)
        assert task.task_id == "malware_family_analysis"

    def test_get_task_not_found(self, temp_tasks_file):
        """Test task retrieval when task doesn't exist."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        with pytest.raises(TaskNotFoundException) as exc_info:
            manager.get_task("nonexistent_task")

        assert exc_info.value.task_id == "nonexistent_task"
        assert "nonexistent_task" in str(exc_info.value)

    def test_get_task_environment_spec_none(self, temp_tasks_file):
        """Test getting environment spec for task without environment configuration."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # The default test task doesn't have an environment spec
        env_spec = manager.get_task_environment_spec("malware_family_analysis")

        assert env_spec is None

    def test_get_task_environment_spec_exists(self, tmp_path):
        """Test getting environment spec for task with environment configuration."""
        # Create environments configuration file first
        environments_yaml = """
# Environment Template Definitions
containers:
  execution_sandbox:
    image: "ubuntu:latest"
    working_dir: "/workspace"

  test_db:
    image: "mysql:5.7"
    environment:
      - "MYSQL_ROOT_PASSWORD=test"

networks:
  test_network:
    driver: "bridge"
    internal: false

environments:
  excytin_db1:
    network: "test_network"
    execution: "execution_sandbox"
    services:
      - name: "database"
        container: "test_db"
"""

        # Create a task configuration with environment spec
        task_yaml_with_env = """
domain: "malware_classification"
tasks:
  - task_id: "test_task_with_env"
    title: "Test Task with Environment"
    description: "Test task that includes environment configuration"
    environment: "excytin_db1"
    initial_context:
      sample_path: "/data/test.exe"
    subtasks:
      - subtask_id: "analysis"
        title: "Analysis"
        description: "Analyze sample"
        objective: "Complete analysis"
        completion_conditions: ["test_command"]
        depends_on: []
"""

        # Create both files
        environments_file = tmp_path / "environments.yaml"
        environments_file.write_text(environments_yaml)

        tasks_file = tmp_path / "test_tasks_with_env.yaml"
        tasks_file.write_text(task_yaml_with_env)

        # Create manager with both files
        manager = TaskManager("malware_classification", str(tasks_file), str(environments_file))

        # Get environment spec
        env_spec = manager.get_task_environment_spec("test_task_with_env")

        # Should have environment spec since task has environment configuration
        assert env_spec is not None
        # The exact structure depends on environment loader, but it should be an EnvironmentSpec
        from saber.server.execution.sandbox.environment_spec import EnvironmentSpec
        assert isinstance(env_spec, EnvironmentSpec)
        assert env_spec.execution_service == "execution_sandbox"
        assert env_spec.network.name == "test_network"

    def test_get_task_environment_spec_task_not_found(self, temp_tasks_file):
        """Test getting environment spec for non-existent task."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        with pytest.raises(TaskNotFoundException) as exc_info:
            manager.get_task_environment_spec("nonexistent_task")

        assert exc_info.value.task_id == "nonexistent_task"

    def test_get_subtask_success(self, temp_tasks_file):
        """Test successful subtask retrieval."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        subtask = manager.get_subtask("malware_family_analysis", "static_analysis")

        assert isinstance(subtask, SubTask)
        assert subtask.subtask_id == "static_analysis"
        assert subtask.task_id == "malware_family_analysis"

    def test_get_subtask_task_not_found(self, temp_tasks_file):
        """Test subtask retrieval when parent task doesn't exist."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        with pytest.raises(TaskNotFoundException):
            manager.get_subtask("nonexistent_task", "some_subtask")

    def test_get_subtask_subtask_not_found(self, temp_tasks_file):
        """Test subtask retrieval when subtask doesn't exist."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        with pytest.raises(SubTaskNotFoundException) as exc_info:
            manager.get_subtask("malware_family_analysis", "nonexistent_subtask")

        assert exc_info.value.task_id == "malware_family_analysis"
        assert exc_info.value.subtask_id == "nonexistent_subtask"

    def test_list_tasks(self, temp_tasks_file):
        """Test listing all available tasks."""
        manager = TaskManager("malware_classification", temp_tasks_file)

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
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("""
domain: "malware_classification"
tasks: []
""")

        manager = TaskManager("malware_classification", str(empty_yaml))

        task_list = manager.list_tasks()
        assert task_list == []

    def test_list_tasks_multiple_tasks(self, tmp_path):
        """Test listing multiple tasks."""
        multi_task_yaml = tmp_path / "multi_tasks.yaml"
        multi_task_yaml.write_text("""
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

        manager = TaskManager("malware_classification", str(multi_task_yaml))

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

    def test_get_episode_info_with_active_episode(self, temp_tasks_file):
        """Test getting episode info when episode exists."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # Start an episode
        episode = manager.start_episode("test_session", "malware_family_analysis")

        # Mock the episode manager's progress info method
        expected_info = {
            "episode_id": episode.episode_id,
            "task_id": "malware_family_analysis",
            "state": "active",
            "total_steps": 1  # Initial step
        }

        with patch.object(manager.episode_manager, '_get_episode_progress_info', return_value=expected_info):
            episode_info = manager.get_episode_info("test_session")
            assert episode_info == expected_info

    def test_get_episode_info_no_active_episode(self, temp_tasks_file):
        """Test getting episode info when no episode exists."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        episode_info = manager.get_episode_info("nonexistent_session")
        assert episode_info == {"error": "No active episode for session"}

    def test_get_current_episode_exists(self, temp_tasks_file):
        """Test getting current episode when it exists."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        episode = manager.start_episode("test_session", "malware_family_analysis")
        current = manager.get_current_episode("test_session")

        assert current == episode

    def test_get_current_episode_not_exists(self, temp_tasks_file):
        """Test getting current episode when it doesn't exist."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        current = manager.get_current_episode("nonexistent_session")
        assert current is None

    def test_domain_consistency(self, temp_tasks_file):
        """Test that domain is consistently used across components."""
        domain = "malware_classification"
        manager = TaskManager(domain, temp_tasks_file)

        assert manager.domain == domain
        assert manager.config_loader.domain == domain

        # Check that loaded tasks have correct domain
        for task in manager.tasks.values():
            assert task.domain == domain

    def test_list_tasks(self, temp_tasks_file):
        """Test listing all available tasks."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        tasks_list = manager.list_tasks()

        assert isinstance(tasks_list, list)
        assert len(tasks_list) > 0

        # Check structure of task info
        task_info = tasks_list[0]
        assert "task_id" in task_info
        assert "title" in task_info
        assert "description" in task_info
        assert "subtask_count" in task_info

        # Verify one of the expected tasks
        malware_task = next((t for t in tasks_list if t["task_id"] == "malware_family_analysis"), None)
        assert malware_task is not None
        assert isinstance(malware_task["subtask_count"], int)

    def test_task_manager_logging_behavior(self, temp_tasks_file, caplog):
        """Test that TaskManager provides appropriate logging."""
        import logging

        with caplog.at_level(logging.INFO):
            manager = TaskManager("malware_classification", temp_tasks_file)

        # Check for initialization logs
        assert any("Initializing TaskManager for domain 'malware_classification'" in record.message
                  for record in caplog.records)
        assert any("TaskManager initialization complete" in record.message
                  for record in caplog.records)
        assert any("Loaded 1 tasks" in record.message
                  for record in caplog.records)

    def test_task_manager_file_path_handling(self, temp_tasks_file):
        """Test that TaskManager correctly handles file path types."""
        # Test with string path
        manager1 = TaskManager("malware_classification", temp_tasks_file)
        assert isinstance(manager1.tasks_file_path, Path)

        # Test with Path object
        path_obj = Path(temp_tasks_file)
        manager2 = TaskManager("malware_classification", path_obj)
        assert isinstance(manager2.tasks_file_path, Path)
        assert manager2.tasks_file_path == path_obj
