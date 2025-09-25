"""
Tests for executor filtering feature in BenchmarkConfigLoader.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException
from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader


class TestExecutorFiltering:
    """Test cases for executor filtering functionality."""

    @pytest.fixture
    def sample_tasks_with_executors(self):
        """Sample tasks configuration with executor restrictions."""
        return {
            "domain": "test_domain",
            "benchmark_config": {
                "episode_attempts": 2
            },
            "global_defaults": {
                "execution_config": {
                    "timeout": 300
                },
                "episode_config": {
                    "max_steps": 50
                }
            },
            "executors": ["bash", "python"],
            "tasks": [
                {
                    "task_id": "test_task",
                    "title": "Test Task",
                    "description": "Test description",
                    "prompts": {"instruction": "test_task_prompt.md", "assistant": "test_task_prompt.md", "submit": "test_task_prompt.md"},
                    "execution_config": {
                        "allowed_executors": ["bash", "python"]
                    },
                    "evaluation_config": {
                        "strategy": "static",
                        "criteria": {
                "expected_answers": ["task_completion"]
            },
                        "scoring": {
                            "points": 100
                        }
                    },
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
            "benchmark_config": {
                "episode_attempts": 1
            },
            "global_defaults": {
                "execution_config": {
                    "timeout": 300,
                    "allowed_executors": ["bash", "python"]
                },
                "episode_config": {
                    "max_steps": 50
                }
            },
            "tasks": [
                {
                                    "prompts": {
                                        "instruction": "test_task_prompt.md",
                                        "assistant": "test_task_prompt.md",
                                        "submit": "test_task_prompt.md"
                                    },
                    "task_id": "test_task",
                    "title": "Test Task",
                    "description": "Test description",
                    "prompts": {"instruction": "test_task_prompt.md", "assistant": "test_task_prompt.md", "submit": "test_task_prompt.md"},
                    "evaluation_config": {
                        "strategy": "static",
                        "criteria": {
                "expected_answers": ["task_completion"]
            },
                        "scoring": {
                            "points": 100
                        }
                    },
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
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)
            allowed_executors = loader.get_allowed_executors()

            assert len(tasks) == 1
            assert "test_task" in tasks
            assert allowed_executors == ["bash", "python"]

        finally:
            Path(temp_path).unlink()

    def test_load_tasks_without_executor_restrictions(self, sample_tasks_without_executors):
        """Test loading tasks without executor restrictions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_tasks_without_executors, f)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
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
            "benchmark_config": {
                "episode_attempts": 1
            },
            "global_defaults": {
                "execution_config": {
                    "timeout": 300
                },
                "episode_config": {
                    "max_steps": 50
                }
            },
            "executors": "not_a_list",  # Should be a list
            "tasks": [
                {
                                    "prompts": {
                                        "instruction": "test_task_prompt.md",
                                        "assistant": "test_task_prompt.md",
                                        "submit": "test_task_prompt.md"
                                    },
                    "task_id": "test_task",
                    "title": "Test Task",
                    "description": "Test description",
                    "prompts": {"instruction": "test_task_prompt.md", "assistant": "test_task_prompt.md", "submit": "test_task_prompt.md"},
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
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="Executors must be a list"):
                loader.load_tasks_from_file(temp_path)

        finally:
            Path(temp_path).unlink()

    def test_empty_executors_list(self):
        """Test loading tasks with empty executors list."""
        config_with_empty_executors = {
            "domain": "test_domain",
            "benchmark_config": {
                "episode_attempts": 1
            },
            "global_defaults": {
                "execution_config": {
                    "timeout": 300
                },
                "episode_config": {
                    "max_steps": 50
                }
            },
            "executors": [],  # Empty list
            "tasks": [
                {
                                    "prompts": {
                                        "instruction": "test_task_prompt.md",
                                        "assistant": "test_task_prompt.md",
                                        "submit": "test_task_prompt.md"
                                    },
                    "task_id": "test_task",
                    "title": "Test Task",
                    "description": "Test description",
                    "prompts": {"instruction": "test_task_prompt.md", "assistant": "test_task_prompt.md", "submit": "test_task_prompt.md"},
                    "execution_config": {
                        "allowed_executors": []  # Empty list should trigger validation error
                    },
                    "evaluation_config": {
                        "strategy": "static",
                        "criteria": {
                "expected_answers": ["task_completion"]
            },
                        "scoring": {
                            "points": 100
                        }
                    },
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
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="allowed_executors must be a non-empty list"):
                loader.load_tasks_from_file(temp_path)

        finally:
            Path(temp_path).unlink()

    def test_get_allowed_executors_before_loading(self):
        """Test getting allowed executors before loading any configuration."""
        loader = BenchmarkConfigLoader("test_domain")
        allowed_executors = loader.get_allowed_executors()
        assert allowed_executors is None
