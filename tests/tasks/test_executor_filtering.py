"""
Tests for executor filtering feature in TaskConfigLoader.
"""
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from saber.server.tasks.task_config_loader import TaskConfigLoader
from saber.server.tasks.exceptions import InvalidTaskDefinitionException


class TestExecutorFiltering:
    """Test cases for executor filtering functionality."""

    @pytest.fixture
    def sample_tasks_with_executors(self):
        """Sample tasks configuration with executor restrictions."""
        return {
            "domain": "test_domain",
            "executors": ["cli", "python"],
            "tasks": [
                {
                    "task_id": "test_task",
                    "title": "Test Task",
                    "description": "Test description",
                    "subtasks": [
                        {
                            "subtask_id": "test_subtask",
                            "title": "Test Subtask",
                            "description": "Test subtask description",
                            "objective": "Test objective",
                        }
                    ],
                }
            ],
        }

    @pytest.fixture
    def sample_tasks_without_executors(self):
        """Sample tasks configuration without executor restrictions."""
        return {
            "domain": "test_domain",
            "tasks": [
                {
                    "task_id": "test_task",
                    "title": "Test Task",
                    "description": "Test description",
                    "subtasks": [
                        {
                            "subtask_id": "test_subtask",
                            "title": "Test Subtask",
                            "description": "Test subtask description",
                            "objective": "Test objective",
                        }
                    ],
                }
            ],
        }

    def test_load_tasks_with_executor_restrictions(self, sample_tasks_with_executors):
        """Test loading tasks with executor restrictions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_tasks_with_executors, f)
            temp_path = f.name

        try:
            loader = TaskConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)
            allowed_executors = loader.get_allowed_executors()

            assert len(tasks) == 1
            assert "test_task" in tasks
            assert allowed_executors == ["cli", "python"]

        finally:
            Path(temp_path).unlink()

    def test_load_tasks_without_executor_restrictions(self, sample_tasks_without_executors):
        """Test loading tasks without executor restrictions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_tasks_without_executors, f)
            temp_path = f.name

        try:
            loader = TaskConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)
            allowed_executors = loader.get_allowed_executors()

            assert len(tasks) == 1
            assert "test_task" in tasks
            assert allowed_executors is None

        finally:
            Path(temp_path).unlink()

    def test_invalid_executors_field_type(self):
        """Test that invalid executors field type raises exception."""
        invalid_config = {
            "domain": "test_domain",
            "executors": "not_a_list",  # Should be a list
            "tasks": [
                {
                    "task_id": "test_task",
                    "title": "Test Task",
                    "description": "Test description",
                    "subtasks": [
                        {
                            "subtask_id": "test_subtask",
                            "title": "Test Subtask",
                            "description": "Test subtask description",
                            "objective": "Test objective",
                        }
                    ],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name

        try:
            loader = TaskConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="Executors must be a list"):
                loader.load_tasks_from_file(temp_path)

        finally:
            Path(temp_path).unlink()

    def test_empty_executors_list(self):
        """Test loading tasks with empty executors list."""
        config_with_empty_executors = {
            "domain": "test_domain",
            "executors": [],  # Empty list
            "tasks": [
                {
                    "task_id": "test_task",
                    "title": "Test Task",
                    "description": "Test description",
                    "subtasks": [
                        {
                            "subtask_id": "test_subtask",
                            "title": "Test Subtask",
                            "description": "Test subtask description",
                            "objective": "Test objective",
                        }
                    ],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_with_empty_executors, f)
            temp_path = f.name

        try:
            loader = TaskConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)
            allowed_executors = loader.get_allowed_executors()

            assert len(tasks) == 1
            assert "test_task" in tasks
            assert allowed_executors == []

        finally:
            Path(temp_path).unlink()

    def test_get_allowed_executors_before_loading(self):
        """Test getting allowed executors before loading any configuration."""
        loader = TaskConfigLoader("test_domain")
        allowed_executors = loader.get_allowed_executors()
        assert allowed_executors is None
