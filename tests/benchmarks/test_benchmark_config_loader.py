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
    allowed_executors: ["bash"]
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
    allowed_executors: ["bash"]
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
      allowed_executors: ["bash", "python"]
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
            assert task.execution_config["allowed_executors"] == ["bash", "python"]

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
    allowed_executors: ["bash"]
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
      allowed_executors: ["bash"]  # Inherit timeout only
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
            assert task.execution_config["allowed_executors"] == ["bash"]
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
    allowed_executors: ["bash"]
  episode_config:
    max_steps: 40

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_task_template.md
    execution_config:
      allowed_executors: ["bash"]  # Explicit allowed executors (timeout via global)
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
    allowed_executors: ["bash"]
  episode_config:
    max_steps: 60

tasks:
  - task_id: task_default
    title: Task with Default
    description: Uses domain default
    prompt_template_file: task_default_template.md
    execution_config:
      allowed_executors: ["bash"]  # Inherit timeout 50
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
      allowed_executors: ["bash"]
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
    allowed_executors: ["bash"]
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
      allowed_executors: ["bash"]
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
    allowed_executors: ["bash", "python"]
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
      allowed_executors: ["bash", "python"]
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
      allowed_executors: ["bash", "python"]
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
            assert global_defaults["execution_config"]["allowed_executors"] == ["bash", "python"]
            assert global_defaults["episode_config"]["max_steps"] == 100
            assert global_defaults["benchmark_config"]["episode_attempts"] == 3

            # Test minimal task inherits all global defaults
            minimal_task = tasks["test_task_minimal"]
            assert minimal_task.execution_config["timeout"] == 60
            assert minimal_task.execution_config["allowed_executors"] == ["bash", "python"]
            assert minimal_task.episode_config["max_steps"] == 100
            assert minimal_task.benchmark_config["episode_attempts"] == 5  # Domain-level override

            # Test task with overrides
            override_task = tasks["test_task_with_overrides"]
            assert override_task.execution_config["timeout"] == 30  # Task override
            assert override_task.execution_config["allowed_executors"] == ["bash", "python"]  # From global defaults
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
      allowed_executors: ["bash"]
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
            assert task.execution_config["allowed_executors"] == ["bash"]
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
    allowed_executors: ["bash"]
  episode_config:
    max_steps: 45

tasks:
  - task_id: test_task
    title: Test Task
    description: A test task using only global benchmark defaults
    prompt_template_file: test_task_template.md
    execution_config:
      allowed_executors: ["bash"]
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


class TestBenchmarkConfigLoaderDirectory:
    """Test cases for directory-based task loading functionality."""

    def test_load_tasks_from_directory_basic(self):
        """Test basic directory loading with global.yaml and task files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create tasks directory structure
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
  episode_config:
    max_steps: 5
  benchmark_config:
    episode_attempts: 3

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create a task file
            task_file = tasks_dir / "test_tasks.yaml"
            task_content = """
tasks:
  - task_id: dir_test_task
    title: Directory Test Task
    description: A test task loaded from directory
    prompt_template_file: test_template.md
    execution_config:
      timeout: 60
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
            with open(task_file, "w") as f:
                f.write(task_content)

            # Test directory loading
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            assert isinstance(tasks, dict)
            assert "dir_test_task" in tasks
            task = tasks["dir_test_task"]
            assert task.task_id == "dir_test_task"
            assert task.title == "Directory Test Task"
            assert task.domain == "test_domain"
            # Test inheritance from global defaults
            assert task.execution_config["allowed_executors"] == ["bash"]  # from global
            assert task.execution_config["timeout"] == 60  # overridden in task
            assert task.benchmark_config["episode_attempts"] == 3  # from global

    def test_load_tasks_from_directory_with_shared_config(self):
        """Test directory loading with shared configuration in subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create tasks directory structure
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create subdirectory for incident tasks
            incident_dir = tasks_dir / "incident_1"
            incident_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
  episode_config:
    max_steps: 5
  benchmark_config:
    episode_attempts: 3

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create shared.yaml in subdirectory
            shared_yaml = incident_dir / "shared.yaml"
            shared_content = """
initial_context:
  database_connection:
    host: "test-db.example.com"
    port: 5432
    database: "test_database"
    username: "test_user"
    password: "test_password"
  shared_context: |
    You are working on incident 1.
    Database connection details are provided.
"""
            with open(shared_yaml, "w") as f:
                f.write(shared_content)

            # Create task file in subdirectory
            task_file = incident_dir / "incident_1_tasks.yaml"
            task_content = """
tasks:
  - task_id: incident_1_task_1
    title: Incident 1 Task 1
    description: A task with shared config
    prompt_template_file: test_template.md
    execution_config:
      timeout: 120
    episode_config:
      max_steps: 8
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "shared_answer"
      scoring:
        max_score: 1.0
    subtasks: []
  - task_id: incident_1_task_2
    title: Incident 1 Task 2
    description: Another task with shared config
    prompt_template_file: test_template.md
    initial_context:
      database_connection:
        host: "override-db.example.com"  # This should override shared
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "override_answer"
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            # Test directory loading with shared config
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            assert len(tasks) == 2

            # Test first task inherits shared config
            task1 = tasks["incident_1_task_1"]
            assert task1.task_id == "incident_1_task_1"
            assert "database_connection" in task1.initial_context
            assert task1.initial_context["database_connection"]["host"] == "test-db.example.com"
            assert task1.initial_context["database_connection"]["port"] == 5432
            assert "shared_context" in task1.initial_context
            assert "incident 1" in task1.initial_context["shared_context"]

            # Test second task overrides shared config
            task2 = tasks["incident_1_task_2"]
            assert task2.task_id == "incident_1_task_2"
            assert "database_connection" in task2.initial_context
            assert task2.initial_context["database_connection"]["host"] == "override-db.example.com"
            # Should still inherit other shared config
            assert "shared_context" in task2.initial_context

    def test_load_tasks_from_directory_multiple_files(self):
        """Test loading tasks from multiple files in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
  episode_config:
    max_steps: 5
  benchmark_config:
    episode_attempts: 2

benchmark_config:
  episode_attempts: 2
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create first task file
            task_file1 = tasks_dir / "batch_1.yaml"
            task_content1 = """
tasks:
  - task_id: batch_1_task_1
    title: Batch 1 Task 1
    description: First task in batch 1
    prompt_template_file: test_template.md
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer1"]
      scoring:
        max_score: 1.0
    subtasks: []
  - task_id: batch_1_task_2
    title: Batch 1 Task 2
    description: Second task in batch 1
    prompt_template_file: test_template.md
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer2"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file1, "w") as f:
                f.write(task_content1)

            # Create second task file
            task_file2 = tasks_dir / "batch_2.yaml"
            task_content2 = """
tasks:
  - task_id: batch_2_task_1
    title: Batch 2 Task 1
    description: First task in batch 2
    prompt_template_file: test_template.md
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer3"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file2, "w") as f:
                f.write(task_content2)

            # Test loading from multiple files
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            assert len(tasks) == 3
            assert "batch_1_task_1" in tasks
            assert "batch_1_task_2" in tasks
            assert "batch_2_task_1" in tasks

            # Verify all tasks have global defaults
            for task_id, task in tasks.items():
                assert task.domain == "test_domain"
                assert task.execution_config["allowed_executors"] == ["bash"]
                assert task.execution_config["timeout"] == 30
                assert task.benchmark_config["episode_attempts"] == 2

    def test_load_tasks_from_directory_missing_global(self):
        """Test directory loading fails gracefully when global.yaml is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create only a task file, no global.yaml
            task_file = tasks_dir / "test_tasks.yaml"
            task_content = """
tasks:
  - task_id: orphan_task
    title: Orphan Task
    description: A task without global config
    prompt_template_file: test_template.md
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")

            # Should raise exception for missing global.yaml
            with pytest.raises(InvalidTaskDefinitionException, match="Missing required global.yaml"):
                loader.load_tasks_from_directory(str(tasks_dir))

    def test_load_tasks_from_directory_nonexistent_path(self):
        """Test directory loading with nonexistent directory path."""
        loader = BenchmarkConfigLoader("test_domain")

        with pytest.raises(InvalidTaskDefinitionException, match="Tasks directory not found"):
            loader.load_tasks_from_directory("/nonexistent/tasks/directory")

    def test_load_tasks_from_directory_empty_directory(self):
        """Test directory loading with empty directory (no task files)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml but no task files
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
  episode_config:
    max_steps: 5
  benchmark_config:
    episode_attempts: 3

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            loader = BenchmarkConfigLoader("test_domain")

            # Should raise exception when no task files found
            with pytest.raises(InvalidTaskDefinitionException, match="No task files found"):
                loader.load_tasks_from_directory(str(tasks_dir))


class TestBenchmarkConfigLoaderGlobalConfig:
    """Test cases for global configuration loading and inheritance."""

    def test_global_config_inheritance_basic(self):
        """Test basic inheritance of global configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml with comprehensive defaults
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash", "python"]
    timeout: 60
    max_memory: "1GB"
  episode_config:
    max_steps: 10
    step_timeout: 300
  benchmark_config:
    episode_attempts: 5
    timeout_strategy: "fail_fast"

benchmark_config:
  episode_attempts: 5
  timeout_strategy: "fail_fast"
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create task that should inherit all defaults
            task_file = tasks_dir / "minimal_task.yaml"
            task_content = """
tasks:
  - task_id: minimal_task
    title: Minimal Task
    description: Task with minimal config relying on global defaults
    prompt_template_file: test_template.md
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["inherited"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            task = tasks["minimal_task"]

            # Verify inheritance of execution config
            assert task.execution_config["allowed_executors"] == ["bash", "python"]
            assert task.execution_config["timeout"] == 60
            assert task.execution_config["max_memory"] == "1GB"

            # Verify inheritance of episode config
            assert task.episode_config["max_steps"] == 10
            assert task.episode_config["step_timeout"] == 300

            # Verify evaluation config is as specified (no global inheritance for eval config)
            assert task.evaluation_config["strategy"] == "static"
            assert task.evaluation_config["criteria"]["expected_answers"] == ["inherited"]

            # Verify benchmark config inheritance
            assert task.benchmark_config["episode_attempts"] == 5
            assert task.benchmark_config["timeout_strategy"] == "fail_fast"

    def test_global_config_overrides(self):
        """Test that task-specific config properly overrides global defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
    retries: 3
  episode_config:
    max_steps: 5
  benchmark_config:
    episode_attempts: 2

benchmark_config:
  episode_attempts: 2
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create task that overrides specific global settings
            task_file = tasks_dir / "override_task.yaml"
            task_content = """
tasks:
  - task_id: override_task
    title: Override Task
    description: Task that overrides global settings
    prompt_template_file: test_template.md
    execution_config:
      allowed_executors: ["python", "bash"]  # Override global
      timeout: 120  # Override global
      # retries: 3 should be inherited from global
    episode_config:
      max_steps: 15  # Override global
      custom_setting: "task_specific"  # New setting
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["override"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            task = tasks["override_task"]

            # Verify overrides worked
            assert task.execution_config["allowed_executors"] == ["python", "bash"]
            assert task.execution_config["timeout"] == 120
            # Verify inheritance still works for non-overridden values
            assert task.execution_config["retries"] == 3

            # Verify episode config override and inheritance
            assert task.episode_config["max_steps"] == 15
            assert task.episode_config["custom_setting"] == "task_specific"

            # Verify benchmark config inheritance (not overridden)
            assert task.benchmark_config["episode_attempts"] == 2

    def test_global_config_deep_merge(self):
        """Test deep merging of nested configuration objects from global defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml with nested structures
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
    security:
      sandboxed: true
      network_access: false
      file_system: "restricted"
  episode_config:
    max_steps: 15
    step_timeout: 120

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create task that partially overrides nested structures
            task_file = tasks_dir / "deep_merge_task.yaml"
            task_content = """
tasks:
  - task_id: deep_merge_task
    title: Deep Merge Task
    description: Task testing deep merge behavior
    prompt_template_file: test_template.md
    execution_config:
      timeout: 60  # Override top-level
      security:
        network_access: true  # Override just this nested value
        # sandboxed and file_system should be inherited from global
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["deep_merge"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

        task = tasks["deep_merge_task"]

        # Verify deep merge in execution_config (nested dicts are merged recursively)
        assert task.execution_config["timeout"] == 60  # Overridden
        assert task.execution_config["allowed_executors"] == ["bash"]  # Inherited
        assert task.execution_config["security"]["network_access"] is True  # Overridden
        # With deep merge, these should be inherited from global defaults
        assert task.execution_config["security"]["sandboxed"] is True  # Inherited via deep merge
        assert task.execution_config["security"]["file_system"] == "restricted"  # Inherited via deep merge

        # Verify episode_config inheritance
        assert task.episode_config["max_steps"] == 15  # Inherited from global
        assert task.episode_config["step_timeout"] == 120  # Inherited from global

    def test_global_config_missing_domain(self):
        """Test behavior when global.yaml is missing required domain field."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml without domain
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create minimal task file
            task_file = tasks_dir / "test_task.yaml"
            task_content = """
tasks:
  - task_id: test_task
    title: Test Task
    description: A test task
    prompt_template_file: test_template.md
    evaluation_config:
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

            # Should raise exception for missing domain in global config
            with pytest.raises(InvalidTaskDefinitionException, match="Domain mismatch.*expected.*test_domain.*got.*None"):
                loader.load_tasks_from_directory(str(tasks_dir))
class TestBenchmarkConfigLoaderSharedConfig:
    """Test cases for shared configuration loading and merging."""

    def test_shared_config_basic_inheritance(self):
        """Test basic inheritance of shared configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()
            incident_dir = tasks_dir / "incident_test"
            incident_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
  episode_config:
    max_steps: 20
  benchmark_config:
    episode_attempts: 3

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create shared.yaml in subdirectory
            shared_yaml = incident_dir / "shared.yaml"
            shared_content = """
initial_context:
  database_connection:
    host: "shared-db.example.com"
    port: 5432
    database: "shared_database"
    username: "shared_user"
    password: "shared_password"
  common_variables:
    api_endpoint: "https://api.example.com"
    timeout_seconds: 300
    retry_count: 5
  shared_instructions: |
    This is shared context for all tasks in this directory.
    Database and API details are provided in the configuration.
"""
            with open(shared_yaml, "w") as f:
                f.write(shared_content)

            # Create task file that should inherit shared config
            task_file = incident_dir / "shared_tasks.yaml"
            task_content = """
tasks:
  - task_id: shared_task_1
    title: Shared Task 1
    description: Task that inherits shared config
    prompt_template_file: test_template.md
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["shared_1"]
      scoring:
        max_score: 1.0
    subtasks: []
  - task_id: shared_task_2
    title: Shared Task 2
    description: Another task with shared config
    prompt_template_file: test_template.md
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["shared_2"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            assert len(tasks) == 2

            # Verify both tasks inherit shared config
            for task_id in ["shared_task_1", "shared_task_2"]:
                task = tasks[task_id]

                # Check database_connection inheritance
                assert "database_connection" in task.initial_context
                db_config = task.initial_context["database_connection"]
                assert db_config["host"] == "shared-db.example.com"
                assert db_config["port"] == 5432
                assert db_config["database"] == "shared_database"
                assert db_config["username"] == "shared_user"
                assert db_config["password"] == "shared_password"

                # Check common_variables inheritance
                assert "common_variables" in task.initial_context
                vars_config = task.initial_context["common_variables"]
                assert vars_config["api_endpoint"] == "https://api.example.com"
                assert vars_config["timeout_seconds"] == 300
                assert vars_config["retry_count"] == 5

                # Check initial_context inheritance
                assert "shared_instructions" in task.initial_context
                assert "shared context" in task.initial_context["shared_instructions"]

    def test_shared_config_override_behavior(self):
        """Test that task-specific config overrides shared config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()
            incident_dir = tasks_dir / "incident_override"
            incident_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
  episode_config:
    max_steps: 20
  benchmark_config:
    episode_attempts: 3

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create shared.yaml
            shared_yaml = incident_dir / "shared.yaml"
            shared_content = """
initial_context:
  database_connection:
    host: "shared-db.example.com"
    port: 5432
    database: "shared_database"
    ssl_enabled: true
  api_config:
    endpoint: "https://shared-api.example.com"
    version: "v1"
    rate_limit: 100
"""
            with open(shared_yaml, "w") as f:
                f.write(shared_content)

            # Create task file with overrides
            task_file = incident_dir / "override_tasks.yaml"
            task_content = """
tasks:
  - task_id: override_task
    title: Override Task
    description: Task that overrides shared config
    prompt_template_file: test_template.md
    initial_context:
      database_connection:
        host: "override-db.example.com"  # Override shared
        port: 3306  # Override shared
        # database and ssl_enabled should be inherited
      api_config:
        endpoint: "https://custom-api.example.com"  # Override shared
        # version and rate_limit should be inherited
      custom_config:
        task_specific: "value"  # New config not in shared
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["override"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            task = tasks["override_task"]

            # Verify database_connection override and inheritance with deep merge
            db_config = task.initial_context["database_connection"]
            assert db_config["host"] == "override-db.example.com"  # Overridden
            assert db_config["port"] == 3306  # Overridden
            assert db_config["database"] == "shared_database"  # Inherited via deep merge
            assert db_config["ssl_enabled"] is True  # Inherited via deep merge

            # Verify api_config override and inheritance with deep merge
            api_config = task.initial_context["api_config"]
            assert api_config["endpoint"] == "https://custom-api.example.com"  # Overridden
            assert api_config["version"] == "v1"  # Inherited via deep merge
            assert api_config["rate_limit"] == 100  # Inherited via deep merge

            # Verify new task-specific config
            assert "custom_config" in task.initial_context
            assert task.initial_context["custom_config"]["task_specific"] == "value"

    def test_shared_config_nested_directories(self):
        """Test shared config behavior with nested directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create nested directory structure
            incident_dir = tasks_dir / "incident_nested"
            incident_dir.mkdir()
            sub_dir = incident_dir / "sub_scenario"
            sub_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
  episode_config:
    max_steps: 20
  benchmark_config:
    episode_attempts: 3

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create shared.yaml in subdirectory
            sub_shared_yaml = sub_dir / "shared.yaml"
            sub_shared_content = """
initial_context:
  sub_config:
    level: "sub_scenario"
    shared_value: "sub_shared"
  database_connection:
    host: "sub-db.example.com"
    port: 5432
    database: "sub_database"
"""
            with open(sub_shared_yaml, "w") as f:
                f.write(sub_shared_content)

            # Create task in subdirectory
            task_file = sub_dir / "nested_tasks.yaml"
            task_content = """
tasks:
  - task_id: nested_task
    title: Nested Task
    description: Task in nested directory with shared config
    prompt_template_file: test_template.md
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["nested"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            task = tasks["nested_task"]

            # Verify inheritance from subdirectory shared config
            assert "sub_config" in task.initial_context
            assert task.initial_context["sub_config"]["level"] == "sub_scenario"
            assert task.initial_context["sub_config"]["shared_value"] == "sub_shared"

            # Verify database_connection inheritance
            assert "database_connection" in task.initial_context
            db_config = task.initial_context["database_connection"]
            assert db_config["host"] == "sub-db.example.com"
            assert db_config["port"] == 5432
            assert db_config["database"] == "sub_database"

    def test_shared_config_no_shared_file(self):
        """Test behavior when directory has no shared.yaml file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()
            no_shared_dir = tasks_dir / "no_shared"
            no_shared_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
  episode_config:
    max_steps: 20
  benchmark_config:
    episode_attempts: 3

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create task file in directory without shared.yaml
            task_file = no_shared_dir / "no_shared_tasks.yaml"
            task_content = """
tasks:
  - task_id: no_shared_task
    title: No Shared Task
    description: Task in directory without shared.yaml
    prompt_template_file: test_template.md
    initial_context:
      custom_config:
        task_only: "value"
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["no_shared"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            task = tasks["no_shared_task"]

            # Should still work without shared config
            assert task.task_id == "no_shared_task"
            assert task.domain == "test_domain"
            # Should have global defaults
            assert task.execution_config["allowed_executors"] == ["bash"]
        # Should have task-specific config in initial_context
        assert "custom_config" in task.initial_context
        assert task.initial_context["custom_config"]["task_only"] == "value"

    def test_shared_config_malformed_yaml(self):
        """Test behavior with malformed shared.yaml file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()
            bad_shared_dir = tasks_dir / "bad_shared"
            bad_shared_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: test_domain

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
  benchmark_config:
    episode_attempts: 3

benchmark_config:
  episode_attempts: 3
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create malformed shared.yaml
            shared_yaml = bad_shared_dir / "shared.yaml"
            malformed_content = """
shared_config:
  database_connection:
    host: "bad-db.example.com"
    port: [invalid yaml structure
  # Missing closing bracket
"""
            with open(shared_yaml, "w") as f:
                f.write(malformed_content)

            # Create task file
            task_file = bad_shared_dir / "bad_shared_task.yaml"
            task_content = """
tasks:
  - task_id: bad_shared_task
    title: Bad Shared Task
    description: Task with malformed shared.yaml
    prompt_template_file: test_template.md
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["bad_shared"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("test_domain")

            # Should raise YAML parsing exception
            with pytest.raises(Exception):  # Could be yaml.YAMLError or similar
                loader.load_tasks_from_directory(str(tasks_dir))


class TestBenchmarkConfigLoaderIntegration:
    """Integration tests for complete directory-based loading with global + shared configs."""

    def test_complete_directory_structure_integration(self):
        """Test complete real-world directory structure with all features."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create multiple incident directories
            incident_1_dir = tasks_dir / "incident_1"
            incident_1_dir.mkdir()
            incident_2_dir = tasks_dir / "incident_2"
            incident_2_dir.mkdir()

            # Create comprehensive global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: security_assessment

permanent_environment: security_lab

global_defaults:
  execution_config:
    allowed_executors: ["bash", "python"]
    timeout: 120
    max_memory: "2GB"
    security:
      sandboxed: true
      network_access: true
  episode_config:
    max_steps: 20
    step_timeout: 600
    allow_interrupt: true
  evaluation_config:
    strategy: "llm_judge"
    default_timeout: 300
    scoring:
      max_score: 10.0
      partial_credit: true
  benchmark_config:
    episode_attempts: 3
    timeout_strategy: "graceful"
    concurrent_limit: 2

benchmark_config:
  episode_attempts: 3
  timeout_strategy: "graceful"
  concurrent_limit: 2
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create incident_1 shared.yaml
            incident_1_shared = incident_1_dir / "shared.yaml"
            incident_1_shared_content = """
shared_config:
  database_connection:
    host: "incident1-db.security.com"
    port: 5432
    database: "incident_1_forensics"
    username: "analyst"
    password: "secure_password_1"
    ssl_required: true
  infrastructure:
    vpc_id: "vpc-incident1"
    subnet_ids: ["subnet-1a", "subnet-1b"]
    security_groups: ["sg-analysis", "sg-forensics"]
  initial_context: |
    INCIDENT 1: Data Breach Investigation

    You are investigating a potential data breach in the customer database.
    Database connection details and infrastructure information are provided.

    Your goal is to identify the attack vector and assess the data exposure.
"""
            with open(incident_1_shared, "w") as f:
                f.write(incident_1_shared_content)

            # Create incident_2 shared.yaml
            incident_2_shared = incident_2_dir / "shared.yaml"
            incident_2_shared_content = """
shared_config:
  database_connection:
    host: "incident2-db.security.com"
    port: 3306
    database: "incident_2_logs"
    username: "investigator"
    password: "secure_password_2"
    ssl_required: false
  infrastructure:
    vpc_id: "vpc-incident2"
    subnet_ids: ["subnet-2a"]
    security_groups: ["sg-monitoring"]
  initial_context: |
    INCIDENT 2: Malware Analysis

    Suspicious activity detected in the network monitoring systems.
    Log database and infrastructure details are available.

    Your task is to analyze the malware behavior and containment strategy.
"""
            with open(incident_2_shared, "w") as f:
                f.write(incident_2_shared_content)

            # Create incident_1 task files
            incident_1_task_1 = incident_1_dir / "data_breach_analysis.yaml"
            incident_1_task_1_content = """
tasks:
  - task_id: incident_1_task_database_forensics
    title: Database Forensics Analysis
    description: Analyze database logs for unauthorized access
    prompt_template_file: database_forensics.md
    execution_config:
      timeout: 300  # Override global for this complex task
    episode_config:
      max_steps: 25  # Override global
    evaluation_config:
      judge_prompts:
        system: "cybersecurity_incident_system.md"
        user: "cybersecurity_incident_user.md"
      criteria:
        attack_vector_identified: true
        data_exposure_assessed: true
        timeline_reconstructed: true
      scoring:
        max_score: 15.0  # Override global
    forensics_tools:
      - "sql_analyzer"
      - "log_parser"
      - "timeline_reconstructor"
    subtasks:
      - subtask_id: "analyze_access_logs"
        title: "Analyze Database Access Logs"
        description: "Review database access logs for suspicious activity"
        evaluation_config:
          criteria:
            suspicious_queries_found: true
          scoring:
            max_score: 5.0
      - subtask_id: "identify_compromised_accounts"
        title: "Identify Compromised User Accounts"
        description: "Determine which user accounts were compromised"
        evaluation_config:
          criteria:
            compromised_accounts_listed: true
          scoring:
            max_score: 5.0
"""
            with open(incident_1_task_1, "w") as f:
                f.write(incident_1_task_1_content)

            # Create incident_2 task file
            incident_2_task_1 = incident_2_dir / "malware_analysis.yaml"
            incident_2_task_1_content = """
tasks:
  - task_id: incident_2_task_malware_behavior
    title: Malware Behavior Analysis
    description: Analyze malware samples and network behavior
    prompt_template_file: malware_analysis.md
    database_connection:
      # Override shared database for this specific task
      host: "malware-analysis-db.security.com"
      database: "malware_samples"
    execution_config:
      security:
        sandboxed: true  # Extra security for malware analysis
        network_access: false  # Override global - no network for malware
    evaluation_config:
      criteria:
        malware_family_identified: true
        attack_vector_determined: true
        iocs_extracted: true
      scoring:
        max_score: 12.0
    analysis_tools:
      - "dynamic_analyzer"
      - "static_analyzer"
      - "network_tracer"
    subtasks:
      - subtask_id: "static_analysis"
        title: "Static Malware Analysis"
        description: "Perform static analysis on malware samples"
        evaluation_config:
          criteria:
            file_hashes_computed: true
            strings_extracted: true
          scoring:
            max_score: 4.0
"""
            with open(incident_2_task_1, "w") as f:
                f.write(incident_2_task_1_content)

            # Create standalone task file (no shared config)
            standalone_task = tasks_dir / "standalone_security.yaml"
            standalone_content = """
tasks:
  - task_id: standalone_vulnerability_scan
    title: Standalone Vulnerability Scan
    description: General vulnerability assessment
    prompt_template_file: vuln_scan.md
    evaluation_config:
      criteria:
        vulnerabilities_found: true
        risk_assessment_complete: true
      scoring:
        max_score: 8.0
    subtasks: []
"""
            with open(standalone_task, "w") as f:
                f.write(standalone_content)

            # Load all tasks
            loader = BenchmarkConfigLoader("security_assessment")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            # Verify correct number of tasks loaded
            assert len(tasks) == 3
            expected_task_ids = [
                "incident_1_task_database_forensics",
                "incident_2_task_malware_behavior",
                "standalone_vulnerability_scan"
            ]
            for task_id in expected_task_ids:
                assert task_id in tasks

            # Test incident_1 task
            incident_1_task = tasks["incident_1_task_database_forensics"]
            assert incident_1_task.domain == "security_assessment"
            assert incident_1_task.title == "Database Forensics Analysis"

            # Verify global inheritance
            assert incident_1_task.execution_config["allowed_executors"] == ["bash", "python"]
            assert incident_1_task.execution_config["max_memory"] == "2GB"
            assert incident_1_task.execution_config["security"]["sandboxed"] is True

            # Verify global overrides
            assert incident_1_task.execution_config["timeout"] == 300  # Task override
            assert incident_1_task.episode_config["max_steps"] == 25  # Task override
            assert incident_1_task.evaluation_config["scoring"]["max_score"] == 15.0  # Task override

            # Verify shared config inheritance
            assert "database_connection" in incident_1_task.additional_config
            db_config = incident_1_task.additional_config["database_connection"]
            assert db_config["host"] == "incident1-db.security.com"
            assert db_config["database"] == "incident_1_forensics"
            assert db_config["ssl_required"] is True

            assert "infrastructure" in incident_1_task.additional_config
            infra_config = incident_1_task.additional_config["infrastructure"]
            assert infra_config["vpc_id"] == "vpc-incident1"
            assert "subnet-1a" in infra_config["subnet_ids"]

            assert "initial_context" in incident_1_task.additional_config
            assert "INCIDENT 1" in incident_1_task.additional_config["initial_context"]

            # Verify task-specific config
            assert "forensics_tools" in incident_1_task.additional_config
            assert "sql_analyzer" in incident_1_task.additional_config["forensics_tools"]

            # Verify subtasks
            assert len(incident_1_task.subtasks) == 2
            subtask_1 = incident_1_task.subtasks[0]
            assert subtask_1.subtask_id == "analyze_access_logs"
            assert subtask_1.evaluation_config["scoring"]["max_score"] == 5.0

            # Test incident_2 task
            incident_2_task = tasks["incident_2_task_malware_behavior"]

            # Verify different shared config
            db_config_2 = incident_2_task.additional_config["database_connection"]
            assert db_config_2["host"] == "malware-analysis-db.security.com"  # Task override
            assert db_config_2["database"] == "malware_samples"  # Task override
            # Should still inherit some shared values
            assert db_config_2["username"] == "investigator"  # From shared

            # Verify execution config override for security
            assert incident_2_task.execution_config["security"]["network_access"] is False  # Task override
            assert incident_2_task.execution_config["security"]["sandboxed"] is True  # Inherited from global

            # Verify different initial context
            assert "INCIDENT 2" in incident_2_task.additional_config["initial_context"]
            assert "Malware Analysis" in incident_2_task.additional_config["initial_context"]

            # Test standalone task (no shared config)
            standalone_task = tasks["standalone_vulnerability_scan"]
            assert standalone_task.domain == "security_assessment"

            # Should have global defaults
            assert standalone_task.execution_config["allowed_executors"] == ["bash", "python"]
            assert standalone_task.benchmark_config["episode_attempts"] == 3

            # Should not have shared config
            assert "database_connection" not in standalone_task.additional_config
            assert "infrastructure" not in standalone_task.additional_config

    def test_complex_precedence_order(self):
        """Test complex configuration precedence: task > shared > global."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()
            precedence_dir = tasks_dir / "precedence_test"
            precedence_dir.mkdir()

            # Create global.yaml with base config
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: precedence_test

global_defaults:
  execution_config:
    timeout: 60
    retries: 3
    priority: "low"
  test_config:
    level: "global"
    global_only: true
    shared_value: "from_global"
    task_value: "from_global"

benchmark_config:
  episode_attempts: 2
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create shared.yaml that overrides some global values
            shared_yaml = precedence_dir / "shared.yaml"
            shared_content = """
shared_config:
  execution_config:
    retries: 5  # Override global
    priority: "medium"  # Override global
    # timeout: 60 should be inherited from global
  test_config:
    level: "shared"  # Override global
    shared_only: true  # New value
    shared_value: "from_shared"  # Override global
    # task_value and global_only should be inherited from global
"""
            with open(shared_yaml, "w") as f:
                f.write(shared_content)

            # Create task that overrides both shared and global
            task_file = precedence_dir / "precedence_task.yaml"
            task_content = """
tasks:
  - task_id: precedence_task
    title: Precedence Test Task
    description: Task testing configuration precedence
    prompt_template_file: test_template.md
    execution_config:
      priority: "high"  # Override shared (which overrode global)
      # timeout and retries should come from shared/global
    test_config:
      level: "task"  # Override shared (which overrode global)
      task_only: true  # New value
      task_value: "from_task"  # Override shared/global
      # shared_value, shared_only, and global_only should be inherited
    evaluation_config:
      strategy: "static"
      criteria:
        precedence_test: true
      scoring:
        max_score: 1.0
    subtasks: []
"""
            with open(task_file, "w") as f:
                f.write(task_content)

            loader = BenchmarkConfigLoader("precedence_test")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            task = tasks["precedence_task"]

            # Test execution_config precedence
            exec_config = task.execution_config
            assert exec_config["timeout"] == 60  # From global (not overridden)
            assert exec_config["retries"] == 5  # From shared (overrode global)
            assert exec_config["priority"] == "high"  # From task (overrode shared)

            # Test test_config precedence
            test_config = task.additional_config["test_config"]
            assert test_config["level"] == "task"  # Task overrode shared which overrode global
            assert test_config["global_only"] is True  # From global only
            assert test_config["shared_only"] is True  # From shared only
            assert test_config["task_only"] is True  # From task only
            assert test_config["shared_value"] == "from_shared"  # Shared overrode global
            assert test_config["task_value"] == "from_task"  # Task overrode shared/global

    def test_large_scale_directory_structure(self):
        """Test performance and correctness with larger directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_content = """
domain: large_scale_test

global_defaults:
  execution_config:
    allowed_executors: ["bash"]
    timeout: 30
  benchmark_config:
    episode_attempts: 2

benchmark_config:
  episode_attempts: 2
"""
            with open(global_yaml, "w") as f:
                f.write(global_content)

            # Create multiple directories with tasks
            expected_tasks = []
            for incident_num in range(1, 6):  # 5 incidents
                incident_dir = tasks_dir / f"incident_{incident_num}"
                incident_dir.mkdir()

                # Create shared.yaml for each incident
                shared_yaml = incident_dir / "shared.yaml"
                shared_content = f"""
shared_config:
  incident_id: {incident_num}
  database_name: "incident_{incident_num}_db"
"""
                with open(shared_yaml, "w") as f:
                    f.write(shared_content)

                # Create multiple task files per incident
                for batch_num in range(1, 4):  # 3 batches per incident
                    task_file = incident_dir / f"batch_{batch_num}.yaml"

                    # Create 2 tasks per batch
                    tasks_content = "tasks:\n"
                    for task_num in range(1, 3):  # 2 tasks per batch
                        task_id = f"incident_{incident_num}_batch_{batch_num}_task_{task_num}"
                        expected_tasks.append(task_id)

                        tasks_content += f"""  - task_id: {task_id}
    title: Incident {incident_num} Batch {batch_num} Task {task_num}
    description: Auto-generated task for scale testing
    prompt_template_file: test_template.md
    evaluation_config:
      strategy: "static"
      criteria:
        scale_test: true
      scoring:
        max_score: 1.0
    subtasks: []
"""

                    with open(task_file, "w") as f:
                        f.write(tasks_content)

            # Load all tasks
            loader = BenchmarkConfigLoader("large_scale_test")
            tasks = loader.load_tasks_from_directory(str(tasks_dir))

            # Should load all 30 tasks (5 incidents × 3 batches × 2 tasks)
            assert len(tasks) == 30
            assert len(expected_tasks) == 30

            # Verify all expected tasks are present
            for expected_task_id in expected_tasks:
                assert expected_task_id in tasks

                # Extract incident number from task_id
                incident_num = int(expected_task_id.split('_')[1])

                task = tasks[expected_task_id]
                assert task.domain == "large_scale_test"

                # Verify shared config inheritance
                assert "incident_id" in task.additional_config
                assert task.additional_config["incident_id"] == incident_num
                assert task.additional_config["database_name"] == f"incident_{incident_num}_db"

                # Verify global inheritance
                assert task.execution_config["allowed_executors"] == ["bash"]
                assert task.execution_config["timeout"] == 30
                assert task.benchmark_config["episode_attempts"] == 2
