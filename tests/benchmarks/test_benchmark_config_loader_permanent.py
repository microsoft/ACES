"""
Unit tests for BenchmarkConfigLoader permanent environment support.

Tests the new permanent_environment field handling in task configurations.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.task import Task


class TestBenchmarkConfigLoaderPermanentSupport:
    """Test BenchmarkConfigLoader permanent environment functionality."""

    @pytest.fixture
    def sample_tasks_yaml(self):
        """Create sample tasks.yaml with permanent environment support."""
        return {
            "domain": "test_domain",
            "benchmark_config": {
                "episode_attempts": 3
            },
            "global_defaults": {
                "execution_config": {
                    "timeout": 300
                },
                "episode_config": {
                    "max_steps": 50
                }
            },
            "executors": ["bash", "python"],  # Domain-level executor configuration
            "permanent_environment": {
                "services": [
                    {
                        "name": "postgres",
                        "container": "postgres:13",
                        "image": "postgres:13",
                        "ports": ["5432:5432"],
                        "environment": {
                            "POSTGRES_PASSWORD": "test123"
                        }
                    },
                    {
                        "name": "web",
                        "container": "nginx:alpine",
                        "image": "nginx:alpine",
                        "ports": ["80:80"]
                    }
                ],
                "networks": ["test_network", "extra_network"]
            },
            "tasks": [
                {
                    "task_id": "test_task_with_permanent",
                    "title": "Test Task with Permanent Environment",
                    "description": "Task that uses permanent environment",
                    "prompt_template_file": "test_task_with_permanent_prompt.md",
                    "sandbox_environment": "test_sandbox",
                    "permanent_environment": "default",  # Reference to the global permanent environment
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
                            "subtask_id": "main",
                            "title": "Main Subtask",
                            "description": "Main subtask",
                            "objective": "Complete the main objective",
                            "contexts": [
                                {
                                    "name": "main",
                                    "description": "Main context",
                                    "default": True,
                                    "commands": [
                                        {
                                            "prompt": "Do something",
                                            "action": "action1"
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    "task_id": "test_task_no_permanent",
                    "title": "Test Task without Permanent Environment",
                    "description": "Task that doesn't use permanent environment",
                    "prompt_template_file": "test_task_no_permanent_prompt.md",
                    "sandbox_environment": "test_sandbox",
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
                            "subtask_id": "main",
                            "title": "Main Subtask",
                            "description": "Main subtask",
                            "objective": "Complete the main objective",
                            "contexts": [
                                {
                                    "name": "main",
                                    "description": "Main context",
                                    "default": True,
                                    "commands": [
                                        {
                                            "prompt": "Do something",
                                            "action": "action1"
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    @pytest.fixture
    def temp_tasks_file(self, sample_tasks_yaml):
        """Create temporary tasks.yaml file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_tasks_yaml, f)
            return f.name

    def test_load_task_with_permanent_environment(self, temp_tasks_file):
        """Test loading task with permanent environment specified."""
        loader = BenchmarkConfigLoader("test_domain")

        # Load tasks from file
        tasks = loader.load_tasks_from_file(temp_tasks_file)

        # Get task with permanent environment
        task = tasks["test_task_with_permanent"]

        # Verify task properties
        assert isinstance(task, Task)
        assert task.title == "Test Task with Permanent Environment"
        assert task.environment == "test_sandbox"  # This is the sandbox environment
        assert "bash" in task.allowed_executors
        assert "python" in task.allowed_executors

        # Verify permanent environment is available at the loader level
        assert loader.get_permanent_environment() is not None

    def test_load_task_without_permanent_environment(self, temp_tasks_file):
        """Test loading task without permanent environment."""
        loader = BenchmarkConfigLoader("test_domain")

        # Load tasks from file
        tasks = loader.load_tasks_from_file(temp_tasks_file)

        # Get task without permanent environment
        task = tasks["test_task_no_permanent"]

        # Verify task properties
        assert isinstance(task, Task)
        assert task.title == "Test Task without Permanent Environment"
        assert task.environment == "test_sandbox"
        assert "bash" in task.allowed_executors

    def test_load_all_tasks(self, temp_tasks_file):
        """Test loading all tasks including those with permanent environments."""
        loader = BenchmarkConfigLoader("test_domain")

        # Load all tasks
        all_tasks = loader.load_tasks_from_file(temp_tasks_file)

        # Verify we get all tasks
        assert len(all_tasks) == 2
        task_names = list(all_tasks.keys())
        assert "test_task_with_permanent" in task_names
        assert "test_task_no_permanent" in task_names

        # Verify permanent environment is available at loader level
        assert loader.get_permanent_environment() is not None

    def test_task_validation_with_permanent_environment(self, sample_tasks_yaml):
        """Test task validation with permanent environment fields."""
        # Create task with both sandbox and permanent environments
        enhanced_task = {
            "task_id": "enhanced_task",
            "title": "Enhanced Task",
            "description": "Task with both environment types",
            "prompt_template_file": "enhanced_task_prompt.md",
            "sandbox_environment": "test_sandbox",
            "permanent_environment": "default",
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
                    "subtask_id": "main",
                    "title": "Main Subtask",
                    "description": "Main subtask",
                    "objective": "Complete the main objective",
                    "contexts": [
                        {
                            "name": "main",
                            "description": "Main context",
                            "default": True,
                            "commands": [
                                {
                                    "prompt": "Do something",
                                    "action": "action1"
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        config = sample_tasks_yaml.copy()
        config["tasks"].append(enhanced_task)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_file)
            task = tasks["enhanced_task"]

            # Verify environment settings
            assert task.environment == "test_sandbox"
            # Permanent environment is managed at loader level
            assert loader.get_permanent_environment() is not None

        finally:
            Path(temp_file).unlink()

    def test_task_with_only_permanent_environment(self, sample_tasks_yaml):
        """Test task that only specifies permanent environment."""
        # Create task with only permanent environment
        permanent_only_task = {
            "task_id": "permanent_only_task",
            "title": "Permanent Only Task",
            "description": "Task with only permanent environment",
            "prompt_template_file": "permanent_only_task_prompt.md",
            "permanent_environment": "default",
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
                    "subtask_id": "main",
                    "title": "Main Subtask",
                    "description": "Main subtask",
                    "objective": "Complete the main objective",
                    "contexts": [
                        {
                            "name": "main",
                            "description": "Main context",
                            "default": True,
                            "commands": [
                                {
                                    "prompt": "Do something",
                                    "action": "action1"
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        config = sample_tasks_yaml.copy()
        config["tasks"].append(permanent_only_task)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_file)
            task = tasks["permanent_only_task"]

            # Verify environment settings
            assert task.environment is None  # No sandbox environment specified
            # Permanent environment is managed at loader level
            assert loader.get_permanent_environment() is not None

        finally:
            Path(temp_file).unlink()
