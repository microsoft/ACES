"""
Unit tests for benchmark config validation.

Targets remaining uncovered lines in BenchmarkConfigLoader.
"""

import tempfile
from pathlib import Path

import pytest

from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException


class TestBenchmarkConfigValidation:
    """Test benchmark config validation paths."""

    def test_benchmark_config_not_dict(self):
        """Test error when benchmark_config is not a dict."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
    benchmark_config: "should_be_dict"
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="benchmark_config must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_benchmark_config_invalid_episode_attempts(self):
        """Test error when episode_attempts is not a positive integer."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
    benchmark_config:
      episode_attempts: 0
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="episode_attempts must be a positive integer"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_benchmark_config_missing_episode_attempts_after_merge(self):
        """Test error when episode_attempts is missing after config merge."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Don't provide benchmark_config at global level
            loader = BenchmarkConfigLoader("test_domain")
            loader.benchmark_config = {}  # Empty benchmark config
            with pytest.raises(InvalidTaskDefinitionException, match="episode_attempts"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()
