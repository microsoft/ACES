"""
Unit tests for BenchmarkConfigLoader validation logic.

Tests error handling, validation, and edge cases.
"""

import tempfile
from pathlib import Path

import pytest

from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException


class TestBenchmarkConfigLoaderValidation:
    """Test validation logic for BenchmarkConfigLoader."""

    def test_load_tasks_file_not_directory(self):
        """Test error when tasks directory path points to a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file instead of directory
            tasks_file = Path(temp_dir) / "tasks"
            tasks_file.touch()

            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="Tasks path is not a directory"):
                loader.load_tasks_from_directory(str(tasks_file))

    def test_domain_mismatch_in_file(self):
        """Test error when YAML domain doesn't match loader domain."""
        yaml_content = """
domain: different_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_template.md
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
            with pytest.raises(InvalidTaskDefinitionException, match="Domain mismatch.*expected.*test_domain.*got.*different_domain"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_yaml_root_not_dict(self):
        """Test error when YAML root is not a dictionary."""
        yaml_content = """
- this_is_a_list
- not_a_dict
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="YAML root must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_tasks_not_list(self):
        """Test error when tasks field is not a list."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"

benchmark_config:
  episode_attempts: 3

tasks: "this_should_be_a_list"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="Tasks must be a list"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_executors_not_list(self):
        """Test error when executors field is not a list."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"

benchmark_config:
  episode_attempts: 3

executors: "not_a_list"

tasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="Executors must be a list"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_missing_required_task_field(self):
        """Test error when task is missing required fields."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"

benchmark_config:
  episode_attempts: 3

tasks:
  - title: Missing task_id
    description: This task is missing task_id
    prompt_template_file: test_template.md
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
            with pytest.raises(InvalidTaskDefinitionException, match="Missing required field.*task_id"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_missing_executors_in_execution_config(self):
        """Test error when execution_config is missing executors section."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_template.md
    execution_config:
      some_other_config: true
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
            with pytest.raises(InvalidTaskDefinitionException, match="missing required.*executors"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_executors_not_dict(self):
        """Test error when executors in execution_config is not a dict."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_template.md
    execution_config:
      executors: ["not", "a", "dict"]
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
            with pytest.raises(InvalidTaskDefinitionException, match="executors must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_executor_config_not_dict(self):
        """Test error when specific executor config is not a dict."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_template.md
    execution_config:
      executors:
        bash: "should_be_dict"
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
            with pytest.raises(InvalidTaskDefinitionException, match="executor config.*must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_executor_missing_timeout(self):
        """Test error when executor config is missing timeout."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_template.md
    execution_config:
      executors:
        bash:
          some_config: true
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
            with pytest.raises(InvalidTaskDefinitionException, match="missing required.*timeout"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_executor_invalid_timeout(self):
        """Test error when executor timeout is not a positive integer."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
    continue: "test_continue.md"

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_template.md
    execution_config:
      executors:
        bash:
          timeout: -10
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
            with pytest.raises(InvalidTaskDefinitionException, match="timeout must be positive int"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_missing_max_steps(self):
        """Test error when episode_config is missing max_steps."""
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

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_template.md
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      some_other_config: true
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
            with pytest.raises(InvalidTaskDefinitionException, match="missing required.*max_steps"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_invalid_max_steps(self):
        """Test error when max_steps is not a positive integer."""
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

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_template.md
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 0
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
            with pytest.raises(InvalidTaskDefinitionException, match="max_steps must be positive int"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_benchmark_config_not_dict(self):
        """Test error when task benchmark_config is not a dict."""
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
    prompt_template_file: test_template.md
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

    def test_missing_submission_evaluation_config(self):
        """Test error when submission_evaluation_config is missing."""
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
    prompt_template_file: test_template.md
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="missing required.*submission_evaluation_config"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_invalid_submission_strategy(self):
        """Test error when submission evaluation strategy is invalid."""
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
    prompt_template_file: test_template.md
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "invalid_strategy"
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
            with pytest.raises(InvalidTaskDefinitionException, match="Invalid submission evaluation strategy"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_static_submission_missing_expected_answers(self):
        """Test error when static submission strategy is missing expected_answers."""
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
    prompt_template_file: test_template.md
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        some_other_field: true
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="static strategy requires.*expected_answers"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_llm_judge_submission_missing_model(self):
        """Test error when llm_judge submission strategy is missing model."""
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
    prompt_template_file: test_template.md
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        judge_system_template: "system.md"
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
            with pytest.raises(InvalidTaskDefinitionException, match="llm_judge.*requires.*model"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_llm_judge_submission_missing_system_template(self):
        """Test error when llm_judge is missing judge_system_template."""
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
    prompt_template_file: test_template.md
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

    def test_llm_judge_submission_missing_user_template(self):
        """Test error when llm_judge is missing judge_user_template."""
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
    prompt_template_file: test_template.md
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
        judge_system_template: "system.md"
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="llm_judge requires.*judge_user_template"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()
