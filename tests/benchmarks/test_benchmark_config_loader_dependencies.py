"""
Unit tests for BenchmarkConfigLoader dependency and orchestration features.

Tests dependency configuration, role validation, and initial files handling.
"""

import tempfile
from pathlib import Path

import pytest

from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException


class TestBenchmarkConfigLoaderDependencies:
    """Test dependency configuration handling."""

    def test_get_dependency_config_default(self):
        """Test that get_dependency_config returns sensible defaults."""
        loader = BenchmarkConfigLoader("test_domain")

        # Set minimal YAML data to trigger defaults
        loader.yaml_data = {"domain": "test_domain"}
        loader.global_defaults = {}

        dep_config = loader.get_dependency_config()

        assert dep_config["wait_seconds"] == 10.0
        assert dep_config["retry_interval"] == 0.5
        assert dep_config["max_retry_interval"] == 2.0

    def test_get_dependency_config_from_global_defaults(self):
        """Test dependency config from global_defaults."""
        yaml_content = """
domain: test_domain

global_defaults:
  prompts:
    instruction: "test_instruction.md"
    assistant: "test_assistant.md"
    submit: "test_submit.md"
  dependency_config:
    wait_seconds: 20.0
    retry_interval: 1.0
    max_retry_interval: 5.0
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
            loader.load_tasks_from_file(temp_path)

            dep_config = loader.get_dependency_config()

            assert dep_config["wait_seconds"] == 20.0
            assert dep_config["retry_interval"] == 1.0
            assert dep_config["max_retry_interval"] == 5.0
        finally:
            Path(temp_path).unlink()

    def test_validate_dependency_roles_missing_root_role(self):
        """Test error when root task (with dependents) is missing role."""
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
  - task_id: root_task
    title: Root Task
    description: A root task without role
    is_template: true
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

  - task_id: dependent_task
    title: Dependent Task
    description: A task that depends on root_task
    dependency_template: root_task
    role: "dependent_role"
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
            with pytest.raises(InvalidTaskDefinitionException, match="Templates must have a 'role' defined"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validate_dependency_roles_dependent_task_missing_role(self):
        """Test error when dependent task is missing role."""
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
  - task_id: root_task
    title: Root Task
    description: A root task with role
    role: "root_role"
    is_template: true
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

  - task_id: dependent_task
    title: Dependent Task
    description: A task without role
    dependency_template: root_task
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
            with pytest.raises(InvalidTaskDefinitionException, match="'role' is required when 'dependency_template' is set"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validate_dependency_roles_nonexistent_dependency(self):
        """Test error when task depends on non-existent task."""
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
  - task_id: dependent_task
    title: Dependent Task
    description: A task that depends on non-existent task
    dependency_template: nonexistent_task
    role: "dependent_role"
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
            with pytest.raises(InvalidTaskDefinitionException, match="references non-existent template"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestBenchmarkConfigLoaderInitialFiles:
    """Test initial_files configuration handling."""

    def test_initial_files_not_dict(self):
        """Test error when initial_files is not a dictionary."""
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
    initial_files: "should_be_dict"
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
            with pytest.raises(InvalidTaskDefinitionException, match="initial_files must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_initial_files_invalid_mapping(self):
        """Test error when initial_files entries are not string->string mappings."""
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
    initial_files:
      "/dest/path": 123
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
            with pytest.raises(InvalidTaskDefinitionException, match="must be string -> string mappings"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_initial_files_valid(self):
        """Test valid initial_files configuration."""
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
    initial_files:
      "/dest/file1.txt": "source/file1.txt"
      "/dest/file2.txt": "source/file2.txt"
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
            assert task.initial_files is not None
            assert task.initial_files["/dest/file1.txt"] == "source/file1.txt"
            assert task.initial_files["/dest/file2.txt"] == "source/file2.txt"
        finally:
            Path(temp_path).unlink()


class TestBenchmarkConfigLoaderDirectory:
    """Test directory-based loading edge cases."""

    def test_directory_duplicate_task_ids_across_files(self):
        """Test error when duplicate task_id appears in different files."""
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

            # Create first task file
            task_file1 = tasks_dir / "file1.yaml"
            task_content1 = """
tasks:
  - task_id: duplicate_task
    title: Task in File 1
    description: A task
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file1, "w") as f:
                f.write(task_content1)

            # Create second task file with duplicate task_id
            task_file2 = tasks_dir / "file2.yaml"
            task_content2 = """
tasks:
  - task_id: duplicate_task
    title: Task in File 2
    description: A duplicate task
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file2, "w") as f:
                f.write(task_content2)

            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="Duplicate task_id.*duplicate_task"):
                loader.load_tasks_from_directory(str(tasks_dir))

    def test_directory_shared_config_not_inherited(self):
        """Test that shared config is not inherited without inherit_shared flag."""
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

            # Create shared.yaml
            shared_yaml = subdir / "shared.yaml"
            shared_content = """
initial_context:
  shared_value: "should_not_be_inherited"
"""
            with open(shared_yaml, "w") as f:
                f.write(shared_content)

            # Create task without inherit_shared flag
            task_file = subdir / "task.yaml"
            task_content = """
tasks:
  - task_id: no_inherit_task
    title: No Inherit Task
    description: A task that doesn't inherit shared
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
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            task = tasks["no_inherit_task"]
            # Should not have inherited shared_value since inherit_shared not set
            assert "shared_value" not in task.initial_context

    def test_directory_empty_task_file(self):
        """Test handling of empty task file (no tasks)."""
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

            # Create empty task file
            task_file = tasks_dir / "empty.yaml"
            task_content = """
tasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            # Create another task file with actual tasks
            task_file2 = tasks_dir / "real.yaml"
            task_content2 = """
tasks:
  - task_id: real_task
    title: Real Task
    description: A real task
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file2, "w") as f:
                f.write(task_content2)

            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            # Should only get the one real task
            assert len(tasks) == 1
            assert "real_task" in tasks
