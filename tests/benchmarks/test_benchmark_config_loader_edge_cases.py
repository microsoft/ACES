"""
Unit tests for BenchmarkConfigLoader edge cases and error paths.

Tests deep_merge_dicts, YAML parsing errors, and uncovered code paths.
"""

import tempfile
from pathlib import Path

import pytest

from saber.server.benchmarks.benchmark_config_loader import (
    BenchmarkConfigLoader,
    deep_merge_dicts,
)
from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException


class TestDeepMergeDicts:
    """Test deep_merge_dicts utility function."""

    def test_deep_merge_none_override(self):
        """Test merging when override is None."""
        base = {"key1": "value1", "key2": {"nested": "value2"}}
        result = deep_merge_dicts(base, None)

        assert result == base
        assert result is not base  # Should return a copy

    def test_deep_merge_empty_base(self):
        """Test merging with empty base dictionary."""
        base = {}
        override = {"key1": "value1", "key2": {"nested": "value2"}}

        result = deep_merge_dicts(base, override)

        assert result == override

    def test_deep_merge_nested_dicts(self):
        """Test deep merging of nested dictionaries."""
        base = {
            "level1": {
                "level2": {
                    "key1": "base_value1",
                    "key2": "base_value2"
                },
                "other": "base_other"
            }
        }
        override = {
            "level1": {
                "level2": {
                    "key1": "override_value1"  # Override this
                }
            }
        }

        result = deep_merge_dicts(base, override)

        assert result["level1"]["level2"]["key1"] == "override_value1"
        assert result["level1"]["level2"]["key2"] == "base_value2"  # Preserved
        assert result["level1"]["other"] == "base_other"  # Preserved


class TestBenchmarkConfigLoaderEdgeCases:
    """Test edge cases and error paths."""

    def test_load_shared_config_yaml_error(self):
        """Test error handling when shared.yaml has YAML syntax errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()
            subdir = tasks_dir / "subdir"
            subdir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create invalid shared.yaml
            shared_yaml = subdir / "shared.yaml"
            shared_content = """
initial_context:
  database: [
    invalid_yaml_syntax
"""
            with open(shared_yaml, "w") as f:
                f.write(shared_content)

            # Create task file that tries to use shared config
            task_file = subdir / "task.yaml"
            task_content = """
tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    inherit_shared: true
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="YAML parsing error.*shared"):
                loader.load_tasks_from_directory(str(tasks_dir))

    def test_load_shared_config_not_dict(self):
        """Test error when shared.yaml root is not a dictionary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()
            subdir = tasks_dir / "subdir"
            subdir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create shared.yaml that's a list instead of dict
            shared_yaml = subdir / "shared.yaml"
            shared_content = """
- item1
- item2
"""
            with open(shared_yaml, "w") as f:
                f.write(shared_content)

            # Create task file
            task_file = subdir / "task.yaml"
            task_content = """
tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    inherit_shared: true
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="Shared config YAML root must be a dictionary"):
                loader.load_tasks_from_directory(str(tasks_dir))

    def test_load_task_file_not_dict(self):
        """Test error when task file root is not a dictionary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create task file that's a list
            task_file = tasks_dir / "tasks.yaml"
            task_content = """
- not_a_dict
- another_item
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="Task file YAML root must be a dictionary"):
                loader.load_tasks_from_directory(str(tasks_dir))

    def test_load_task_file_tasks_not_list(self):
        """Test error when tasks section is not a list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create task file with invalid tasks section
            task_file = tasks_dir / "tasks.yaml"
            task_content = """
tasks: "should_be_list"
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="'tasks' section must be a list"):
                loader.load_tasks_from_directory(str(tasks_dir))

    def test_global_defaults_section_not_dict(self):
        """Test error when global_defaults section value is not a dict."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"
  execution_config: "should_be_dict"

benchmark_config:
  episode_attempts: 3

tasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="global_defaults.*must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_global_prompts_not_dict(self):
        """Test error when global prompts is not a dictionary."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts: "should_be_dict"

benchmark_config:
  episode_attempts: 3

tasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="global_defaults.prompts must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_global_prompt_value_not_string(self):
        """Test error when global prompt value is not a string."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: 123
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"

benchmark_config:
  episode_attempts: 3

tasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="must be a non-empty string"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_submission_criteria_not_dict(self):
        """Test error when submission criteria is not a dict."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"
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
    submission_evaluation_config:
      strategy: "static"
      criteria: "should_be_dict"
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="Missing or invalid.*criteria"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_submission_scoring_not_dict(self):
        """Test error when submission scoring is not a dict."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"
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
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring: "should_be_dict"
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="scoring must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_submission_invalid_max_score(self):
        """Test error when submission max_score is invalid."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"
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
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="max_score must be a positive number"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_step_criteria_not_dict(self):
        """Test error when step evaluation criteria is not a dict."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"
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
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    step_evaluation_config:
      strategy: "static"
      criteria: "should_be_dict"
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="Missing or invalid.*criteria"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_negative_weight(self):
        """Test error when step weight is negative."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"
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
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    step_evaluation_config:
      strategy: "static"
      criteria:
        expected_outputs: ["test"]
      scoring:
        max_score: 1.0
        weight: -0.5
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="weight must be a non-negative number"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()
