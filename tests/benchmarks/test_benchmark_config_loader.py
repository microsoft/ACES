"""
Unit tests for BenchmarkConfigLoader.

Tests YAML task configuration loading and parsing.
Simplified after removing dependency validation and progression logic.
"""

import os
import tempfile
from pathlib import Path

import pytest

from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException
from saber.server.benchmarks.subtask import SubTask
from saber.server.benchmarks.task import Task
from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader


class TestBenchmarkConfigLoader:
    """Test cases for BenchmarkConfigLoader functionality."""

    def test_benchmark_config_loader_init(self):
        """Test BenchmarkConfigLoader initialization."""
        loader = BenchmarkConfigLoader("test_domain")
        assert loader.domain == "test_domain"

    def test_load_tasks_from_file_basic(self):
        """Test loading basic tasks from YAML file."""
        yaml_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["cli"]
    timeout: 30
  episode_config:
    max_steps: 5
  benchmark_config:
    episode_attempts: 3

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A simple test task
    prompt_template_file: test_template.md
    execution_config:
      timeout: 300
      allowed_executors:
        - "test_executor"
    episode_config:
      max_steps: 10
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
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

            assert isinstance(tasks, dict)
            assert "test_task" in tasks
            task = tasks["test_task"]
            assert task.task_id == "test_task"
            assert task.title == "Test Task"
            assert task.description == "A simple test task"
            assert task.subtasks == []
            assert task.domain == "test_domain"
        finally:
            os.unlink(temp_path)

    def test_load_tasks_with_subtasks(self):
        """Test loading tasks with subtasks from YAML."""
        yaml_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["cli"]
    timeout: 30
  episode_config:
    max_steps: 6
  benchmark_config:
    episode_attempts: 2

benchmark_config:
  episode_attempts: 2

tasks:
  - task_id: complex_task
    title: Complex Task
    description: A task with multiple subtasks
    prompt_template_file: complex_task_template.md
    execution_config:
      timeout: 300
      allowed_executors:
        - "test_executor"
    episode_config:
      max_steps: 10
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0
    initial_context:
      timeout: 300
    subtasks:
      - subtask_id: subtask1
        title: First Subtask
        description: First step of the task
        objective: Complete first step
      - subtask_id: subtask2
        title: Second Subtask
        description: Second step of the task
        objective: Complete second step
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "complex_task" in tasks
            task = tasks["complex_task"]
            assert task.task_id == "complex_task"
            assert task.title == "Complex Task"
            assert len(task.subtasks) == 2

            # Check first subtask
            subtask1 = task.subtasks[0]
            assert isinstance(subtask1, SubTask)
            assert subtask1.subtask_id == "subtask1"
            assert subtask1.title == "First Subtask"
            assert subtask1.objective == "Complete first step"

            # Check second subtask
            subtask2 = task.subtasks[1]
            assert isinstance(subtask2, SubTask)
            assert subtask2.subtask_id == "subtask2"
            assert subtask2.title == "Second Subtask"
            assert subtask2.objective == "Complete second step"

            # Check task context
            assert task.initial_context["timeout"] == 300
        finally:
            os.unlink(temp_path)

    def test_load_tasks_missing_file(self):
        """Test loading from non-existent file."""
        loader = BenchmarkConfigLoader("test_domain")

        with pytest.raises(Exception):  # Should raise some kind of file not found error
            loader.load_tasks_from_file("/nonexistent/path/task.yaml")

    def test_load_tasks_invalid_yaml(self):
        """Test loading invalid YAML."""
        yaml_content = """
domain: test_domain
tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
  prompt_template_file: test_task_template.md
    subtasks: [
      - subtask_id: subtask1
        title: First Subtask
        # Missing closing bracket - invalid YAML
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(Exception):  # Should raise YAML parsing error
                loader.load_tasks_from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_tasks_missing_domain(self):
        """Test loading YAML missing domain field."""
        yaml_content = """
tasks:
  - task_id: test_task
    title: Test Task
    description: A test task without domain
  prompt_template_file: test_task_template.md
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(Exception):  # Should raise validation error
                loader.load_tasks_from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_tasks_with_execution_timeout_config(self):
        """Test loading tasks with execution timeout configuration."""
        yaml_content = """
domain: test_domain

benchmark_config:
  episode_attempts: 4

tasks:
  - task_id: timeout_task
    title: Task with Custom Timeout
    description: A task with custom execution timeout
    prompt_template_file: timeout_task_template.md
    environment: test_env
    execution_config:
      allowed_executors: ["cli", "python"]
      timeout: 180
    episode_config:
      max_steps: 25
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0
    subtasks:
      - subtask_id: test_subtask
        title: Test Subtask
        description: Test subtask with timeout
        objective: Complete with custom timeout
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "timeout_task" in tasks
            task = tasks["timeout_task"]

            # Verify basic task properties
            assert task.task_id == "timeout_task"
            assert task.title == "Task with Custom Timeout"
            assert task.environment == "test_env"

            # Verify execution config contains timeout
            assert task.execution_config is not None
            assert "timeout" in task.execution_config
            assert task.execution_config["timeout"] == 180
            assert task.execution_config["allowed_executors"] == ["cli", "python"]

            # Verify episode config
            assert task.episode_config is not None
            assert task.episode_config["max_steps"] == 25

            # Verify subtask
            assert len(task.subtasks) == 1
            subtask = task.subtasks[0]
            assert subtask.subtask_id == "test_subtask"
            assert subtask.title == "Test Subtask"

        finally:
            os.unlink(temp_path)

    def test_load_tasks_without_execution_timeout_config(self):
        """Test loading tasks without execution timeout (should use defaults)."""
        yaml_content = """
domain: test_domain

benchmark_config:
  episode_attempts: 1

global_defaults:
  execution_config:
    allowed_executors: ["cli"]
    timeout: 45  # Required default timeout now enforced
  benchmark_config:
    episode_attempts: 1
  episode_config:
    max_steps: 15  # Provide required episode max_steps default

tasks:
  - task_id: default_timeout_task
    title: Task with Default Timeout
    description: A task without custom timeout configuration
    prompt_template_file: default_timeout_task_template.md
    # No task-level timeout -> should inherit from global_defaults
    execution_config:
      allowed_executors: ["cli"]  # Inherit timeout only
    episode_config:
      max_steps: 15
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0
    subtasks:
      - subtask_id: test_subtask
        title: Test Subtask
        description: Test subtask
        objective: Complete with default timeout
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "default_timeout_task" in tasks
            task = tasks["default_timeout_task"]
            # Verify execution config inherited timeout from global defaults
            assert task.execution_config is not None
            assert task.execution_config["timeout"] == 45
            assert task.execution_config["allowed_executors"] == ["cli"]
            # Verify episode config inherited from global defaults
            assert task.episode_config["max_steps"] == 15

        finally:
            os.unlink(temp_path)


class TestBenchmarkConfigLoaderBenchmarkConfig:
    """Test cases for BenchmarkConfigLoader benchmark configuration functionality."""

    def test_load_benchmark_config_success(self):
        """Test successful loading of benchmark configuration."""
        yaml_content = """
domain: test_domain

benchmark_config:
  episode_attempts: 5
  max_duration_minutes: 30
  parallel_tasks: false

global_defaults:
  execution_config:
    timeout: 30
    allowed_executors: ["cli"]
  episode_config:
    max_steps: 40

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_task_template.md
    execution_config:
      allowed_executors: ["cli"]  # Explicit allowed executors (timeout via global)
    episode_config:
      max_steps: 40
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
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
            config = loader.load_benchmark_config()

            assert config["episode_attempts"] == 5
            assert config["max_duration_minutes"] == 30
            assert config["parallel_tasks"] is False
            # Verify task inherited episode_config.max_steps from global defaults
            task = tasks["test_task"]
            assert task.episode_config["max_steps"] == 40

        finally:
            os.unlink(temp_path)

    def test_task_level_benchmark_config_override(self):
        """Test task-level benchmark configuration override."""
        yaml_content = """
domain: test_domain

benchmark_config:
  episode_attempts: 3

global_defaults:
  execution_config:
    timeout: 50
    allowed_executors: ["cli"]
  episode_config:
    max_steps: 60

tasks:
  - task_id: task_default
    title: Task with Default
    description: Uses domain default
    prompt_template_file: task_default_template.md
    execution_config:
      allowed_executors: ["cli"]  # Inherit timeout 50
    episode_config:
      max_steps: 60  # Inherit via global defaults (explicit for clarity)
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0
    subtasks: []
  - task_id: task_override
    title: Task with Override
    description: Overrides domain default
    prompt_template_file: task_override_template.md
    benchmark_config:
      episode_attempts: 10
    execution_config:
      timeout: 75  # Override timeout
      allowed_executors: ["cli"]
    episode_config:
      max_steps: 80  # Override
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
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

            task_default = tasks["task_default"]
            task_override = tasks["task_override"]

            assert task_default.get_episode_attempts() == 3  # Domain default
            assert task_override.get_episode_attempts() == 10  # Task override

        finally:
            os.unlink(temp_path)

    def test_missing_benchmark_config_fails(self):
        """Test that missing benchmark_config section fails."""
        yaml_content = """
domain: test_domain
tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_task_template.md
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException) as exc_info:
                loader.load_tasks_from_file(temp_path)

            # Updated error message reflects the new global defaults system
            assert "Missing required 'episode_attempts' in benchmark configuration" in str(exc_info.value)
            assert "global_defaults.benchmark_config" in str(exc_info.value)

        finally:
            os.unlink(temp_path)

    def test_missing_episode_attempts_fails(self):
        """Test that missing episode_attempts fails."""
        yaml_content = """
domain: test_domain

benchmark_config:
  other_setting: true

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_task_template.md
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException) as exc_info:
                loader.load_tasks_from_file(temp_path)

            assert "Missing required 'episode_attempts'" in str(exc_info.value)

        finally:
            os.unlink(temp_path)

    def test_invalid_episode_attempts_fails(self):
        """Test that invalid episode_attempts values fail."""
        # Test zero
        yaml_content_zero = """
domain: test_domain

benchmark_config:
  episode_attempts: 0

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_task_template.md
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content_zero)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException) as exc_info:
                loader.load_tasks_from_file(temp_path)

            assert "episode_attempts must be a positive integer" in str(exc_info.value)

        finally:
            os.unlink(temp_path)

        # Test negative
        yaml_content_negative = """
domain: test_domain

benchmark_config:
  episode_attempts: -1

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_task_template.md
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content_negative)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException) as exc_info:
                loader.load_tasks_from_file(temp_path)

            assert "episode_attempts must be a positive integer" in str(exc_info.value)

        finally:
            os.unlink(temp_path)

    def test_invalid_task_level_episode_attempts_fails(self):
        """Test that invalid task-level episode_attempts fail."""
        yaml_content = """
domain: test_domain

benchmark_config:
  episode_attempts: 3

global_defaults:
  execution_config:
    timeout: 40
    allowed_executors: ["cli"]
  episode_config:
    max_steps: 30

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_task_template.md
    benchmark_config:
      episode_attempts: -5
    execution_config:
      allowed_executors: ["cli"]
    episode_config:
      max_steps: 30
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException) as exc_info:
                loader.load_tasks_from_file(temp_path)

            assert "Task 'test_task' episode_attempts must be a positive integer" in str(exc_info.value)

        finally:
            os.unlink(temp_path)

    def test_global_defaults_configuration(self):
        """Test global defaults configuration loading and merging."""
        yaml_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["cli", "python"]
    timeout: 60
  episode_config:
    max_steps: 100
  benchmark_config:
    episode_attempts: 3

benchmark_config:
  episode_attempts: 5  # Domain-level override

tasks:
  - task_id: test_task_minimal
    title: Test Task with Minimal Config
    description: A task that should inherit global defaults
    prompt_template_file: test_task_minimal_template.md
    execution_config:
      allowed_executors: ["cli", "python"]
      timeout: 60
    episode_config:
      max_steps: 100
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0
    subtasks: []

  - task_id: test_task_with_overrides
    title: Test Task with Overrides
    description: A task that overrides some defaults
    prompt_template_file: test_task_with_overrides_template.md
    execution_config:
      timeout: 30  # Override global default
      allowed_executors: ["cli", "python"]
    episode_config:
      max_steps: 50  # Override global default
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0
    benchmark_config:
      episode_attempts: 10  # Override domain-level config
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            # Test global defaults are loaded
            global_defaults = loader.get_global_defaults()
            assert global_defaults["execution_config"]["timeout"] == 60
            assert global_defaults["execution_config"]["allowed_executors"] == ["cli", "python"]
            assert global_defaults["episode_config"]["max_steps"] == 100
            assert global_defaults["benchmark_config"]["episode_attempts"] == 3

            # Test minimal task inherits all global defaults
            minimal_task = tasks["test_task_minimal"]
            assert minimal_task.execution_config["timeout"] == 60
            assert minimal_task.execution_config["allowed_executors"] == ["cli", "python"]
            assert minimal_task.episode_config["max_steps"] == 100
            assert minimal_task.benchmark_config["episode_attempts"] == 5  # Domain-level override

            # Test task with overrides
            override_task = tasks["test_task_with_overrides"]
            assert override_task.execution_config["timeout"] == 30  # Task override
            assert override_task.execution_config["allowed_executors"] == ["cli", "python"]  # From global defaults
            assert override_task.episode_config["max_steps"] == 50  # Task override
            assert override_task.benchmark_config["episode_attempts"] == 10  # Task override

        finally:
            os.unlink(temp_path)

    def test_global_defaults_missing_section(self):
        """Test that missing global_defaults section works correctly."""
        yaml_content = """
domain: test_domain

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task without global defaults
    prompt_template_file: test_task_template.md
    execution_config:
      timeout: 90
      allowed_executors: ["cli"]
    episode_config:
      max_steps: 55
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test_answer"]
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

            # Test that empty global defaults work
            global_defaults = loader.get_global_defaults()
            assert global_defaults == {}

            # Test task loads correctly without global defaults
            task = tasks["test_task"]
            assert task.execution_config["timeout"] == 90
            assert task.execution_config["allowed_executors"] == ["cli"]
            # Episode config explicitly provided at task-level
            assert task.episode_config["max_steps"] == 55
            assert task.benchmark_config["episode_attempts"] == 3

        finally:
            os.unlink(temp_path)

    def test_global_defaults_invalid_structure(self):
        """Test validation of global_defaults structure."""
        yaml_content = """
domain: test_domain

global_defaults:
  invalid_section:
    some_value: true
  execution_config:
    timeout: 60

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_task_template.md
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException) as exc_info:
                loader.load_tasks_from_file(temp_path)

            assert "Invalid section 'invalid_section' in global_defaults" in str(exc_info.value)

        finally:
            os.unlink(temp_path)

    def test_global_defaults_only_no_domain_benchmark_config(self):
        """Test that global defaults can provide benchmark_config when domain-level is missing."""
        yaml_content = """
domain: test_domain

global_defaults:
  benchmark_config:
    episode_attempts: 3
  execution_config:
    timeout: 120
    allowed_executors: ["cli"]
  episode_config:
    max_steps: 45

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task using only global benchmark defaults
    prompt_template_file: test_task_template.md
    execution_config:
      allowed_executors: ["cli"]
    episode_config:
      max_steps: 45
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test_answer"]
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

            # Should work with only global defaults
            task = tasks["test_task"]
            assert task.benchmark_config["episode_attempts"] == 3

        finally:
            os.unlink(temp_path)
