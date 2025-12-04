"""
Unit tests for BenchmarkConfigLoader prompt handling.

Tests prompt inheritance, validation, and template path validation.
"""

import tempfile
from pathlib import Path

import pytest

from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException


class TestBenchmarkConfigLoaderPrompts:
    """Test prompt configuration handling."""

    def test_prompts_inheritance_from_global(self):
        """Test that tasks inherit prompts from global_defaults."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "global_instruction.md"
    assistant: "global_assistant.md"
    submit: "global_submit.md"
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
    description: A test task using global prompts
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
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            task = tasks["test_task"]
            assert task.prompts["instruction"] == "global_instruction.md"
            assert task.prompts["assistant"] == "global_assistant.md"
            assert task.prompts["submit"] == "global_submit.md"
        finally:
            Path(temp_path).unlink()

    def test_prompts_task_override(self):
        """Test that task-specific prompts override global defaults."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "global_instruction.md"
    assistant: "global_assistant.md"
    submit: "global_submit.md"
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
    description: A test task with custom prompts
    prompts:
      instruction: "task_instruction.md"
      assistant: "task_assistant.md"
      submit: "task_submit.md"
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
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            task = tasks["test_task"]
            assert task.prompts["instruction"] == "task_instruction.md"
            assert task.prompts["assistant"] == "task_assistant.md"
            assert task.prompts["submit"] == "task_submit.md"
        finally:
            Path(temp_path).unlink()

    def test_prompts_partial_override(self):
        """Test partial prompt override (some from global, some from task)."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "global_instruction.md"
    assistant: "global_assistant.md"
    submit: "global_submit.md"
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
    description: A test task with partial prompt override
    prompts:
      instruction: "task_instruction.md"
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
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            task = tasks["test_task"]
            assert task.prompts["instruction"] == "task_instruction.md"  # Overridden
            assert task.prompts["assistant"] == "global_assistant.md"  # Inherited
            assert task.prompts["submit"] == "global_submit.md"  # Inherited
        finally:
            Path(temp_path).unlink()

    def test_prompts_missing_required_type(self):
        """Test error when required prompt type is missing."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "global_instruction.md"
    assistant: "global_assistant.md"
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
    description: A test task with missing submit prompt
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
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="missing.*submit.*prompt"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_prompts_not_dict(self):
        """Test error when prompts field is not a dictionary."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "global_instruction.md"
    assistant: "global_assistant.md"
    submit: "global_submit.md"
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
    description: A test task with invalid prompts
    prompts: "should_be_dict"
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
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="prompts must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_prompts_invalid_type(self):
        """Test error when global prompts contains invalid prompt type."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "global_instruction.md"
    assistant: "global_assistant.md"
    submit: "global_submit.md"
    invalid_type: "invalid.md"
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10

benchmark_config:
  episode_attempts: 3

tasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="invalid prompt type.*invalid_type"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_template_path_missing_md_extension(self):
        """Test error when template path doesn't end with .md."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "global_instruction.md"
    assistant: "global_assistant.md"
    submit: "global_submit.md"
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
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "system.txt"
        judge_user_template: "user.md"
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="must end with .md extension"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_template_path_with_backslashes(self):
        """Test error when template path contains backslashes."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "global_instruction.md"
    assistant: "global_assistant.md"
    submit: "global_submit.md"
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
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "templates\\\\system.md"
        judge_user_template: "user.md"
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="must use forward slashes"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_template_path_with_traversal(self):
        """Test error when template path contains path traversal (..)."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "global_instruction.md"
    assistant: "global_assistant.md"
    submit: "global_submit.md"
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
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "../../../etc/passwd.md"
        judge_user_template: "user.md"
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="cannot contain.*path traversal"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_template_path_empty(self):
        """Test error when template path is empty."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "global_instruction.md"
    assistant: "global_assistant.md"
    submit: "global_submit.md"
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
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: ""
        judge_user_template: "user.md"
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="llm_judge requires.*judge_system_template"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()
