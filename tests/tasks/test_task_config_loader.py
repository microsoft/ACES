"""
Unit tests for TaskConfigLoader.

Tests YAML parsing, validation, task/subtask creation, and error handling.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from saber.server.tasks.core.task_config_loader import TaskConfigLoader
from saber.server.tasks.core.task import Task
from saber.server.tasks.core.subtask import SubTask
from saber.server.tasks.exceptions import InvalidTaskDefinitionException


class TestTaskConfigLoader:
    """Test cases for TaskConfigLoader functionality."""

    def test_init(self):
        """Test TaskConfigLoader initialization."""
        loader = TaskConfigLoader("malware_classification")
        assert loader.domain == "malware_classification"

    def test_load_tasks_from_file_success(self, temp_tasks_file):
        """Test successful loading of tasks from YAML file."""
        loader = TaskConfigLoader("malware_classification")
        tasks = loader.load_tasks_from_file(temp_tasks_file)

        assert len(tasks) == 1
        assert "malware_family_analysis" in tasks

        task = tasks["malware_family_analysis"]
        assert isinstance(task, Task)
        assert task.task_id == "malware_family_analysis"
        assert task.domain == "malware_classification"
        assert task.title == "Malware Family Classification and Analysis"
        assert len(task.subtasks) == 2

    def test_load_tasks_file_not_found(self):
        """Test error handling when YAML file doesn't exist."""
        loader = TaskConfigLoader("malware_classification")

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            loader.load_tasks_from_file("/nonexistent/path/tasks.yaml")

        assert "Tasks file not found" in str(exc_info.value)
        assert exc_info.value.file_path == "/nonexistent/path/tasks.yaml"

    def test_load_tasks_invalid_yaml(self, tmp_path):
        """Test error handling for invalid YAML syntax."""
        invalid_yaml_file = tmp_path / "invalid.yaml"
        invalid_yaml_file.write_text("invalid: yaml: content: [")

        loader = TaskConfigLoader("malware_classification")

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            loader.load_tasks_from_file(str(invalid_yaml_file))

        assert "YAML parsing error" in str(exc_info.value)

    def test_load_tasks_not_dict(self, tmp_path):
        """Test error handling when YAML root is not a dictionary."""
        invalid_yaml_file = tmp_path / "not_dict.yaml"
        invalid_yaml_file.write_text("- this\n- is\n- a\n- list")

        loader = TaskConfigLoader("malware_classification")

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            loader.load_tasks_from_file(str(invalid_yaml_file))

        assert "YAML root must be a dictionary" in str(exc_info.value)

    def test_load_tasks_domain_mismatch(self, tmp_path):
        """Test error handling for domain mismatch."""
        wrong_domain_yaml = tmp_path / "wrong_domain.yaml"
        wrong_domain_yaml.write_text("""
domain: "wrong_domain"
tasks: []
""")

        loader = TaskConfigLoader("malware_classification")

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            loader.load_tasks_from_file(str(wrong_domain_yaml))

        assert "Domain mismatch" in str(exc_info.value)
        assert "expected 'malware_classification', got 'wrong_domain'" in str(exc_info.value)

    def test_load_tasks_tasks_not_list(self, tmp_path):
        """Test error handling when tasks field is not a list."""
        invalid_tasks_yaml = tmp_path / "invalid_tasks.yaml"
        invalid_tasks_yaml.write_text("""
domain: "malware_classification"
tasks: "not a list"
""")

        loader = TaskConfigLoader("malware_classification")

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            loader.load_tasks_from_file(str(invalid_tasks_yaml))

        assert "Tasks must be a list" in str(exc_info.value)

    def test_parse_task_missing_required_field(self, tmp_path):
        """Test error handling for missing required task fields."""
        missing_field_yaml = tmp_path / "missing_field.yaml"
        missing_field_yaml.write_text("""
domain: "malware_classification"
tasks:
  - title: "Missing task_id"
    description: "Task without required field"
""")

        loader = TaskConfigLoader("malware_classification")

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            loader.load_tasks_from_file(str(missing_field_yaml))

        assert "Missing required field: task_id" in str(exc_info.value)

    def test_parse_task_with_initial_context(self, tmp_path):
        """Test parsing task with initial context."""
        task_yaml = tmp_path / "task_with_context.yaml"
        task_yaml.write_text("""
domain: "malware_classification"
tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task"
    initial_context:
      sample_path: "/data/sample.exe"
      timeout: 300
    subtasks: []
""")

        loader = TaskConfigLoader("malware_classification")
        tasks = loader.load_tasks_from_file(str(task_yaml))

        task = tasks["test_task"]
        assert task.initial_context == {
            "sample_path": "/data/sample.exe",
            "timeout": 300
        }

    def test_parse_task_without_initial_context(self, tmp_path):
        """Test parsing task without initial context."""
        task_yaml = tmp_path / "task_no_context.yaml"
        task_yaml.write_text("""
domain: "malware_classification"
tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task"
    subtasks: []
""")

        loader = TaskConfigLoader("malware_classification")
        tasks = loader.load_tasks_from_file(str(task_yaml))

        task = tasks["test_task"]
        assert task.initial_context == {}

    def test_parse_subtask_success(self, tmp_path):
        """Test successful subtask parsing."""
        subtask_yaml = tmp_path / "subtask_test.yaml"
        subtask_yaml.write_text("""
domain: "malware_classification"
tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task"
    subtasks:
      - subtask_id: "test_subtask"
        title: "Test SubTask"
        description: "A test subtask"
        objective: "Complete the test"
        completion_conditions: ["command1", "command2"]
""")

        loader = TaskConfigLoader("malware_classification")
        tasks = loader.load_tasks_from_file(str(subtask_yaml))

        task = tasks["test_task"]
        assert len(task.subtasks) == 1

        subtask = task.subtasks[0]
        assert isinstance(subtask, SubTask)
        assert subtask.subtask_id == "test_subtask"
        assert subtask.task_id == "test_task"
        assert subtask.title == "Test SubTask"
        assert subtask.completion_conditions == ["command1", "command2"]
        assert subtask.depends_on == []

    def test_parse_subtask_missing_required_field(self, tmp_path):
        """Test error handling for missing required subtask fields."""
        missing_subtask_field_yaml = tmp_path / "missing_subtask_field.yaml"
        missing_subtask_field_yaml.write_text("""
domain: "malware_classification"
tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task"
    subtasks:
      - subtask_id: "test_subtask"
        title: "Test SubTask"
        # Missing description, objective
""")

        loader = TaskConfigLoader("malware_classification")

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            loader.load_tasks_from_file(str(missing_subtask_field_yaml))

        assert "Missing required subtask field: description" in str(exc_info.value)

    def test_parse_subtask_default_values(self, tmp_path):
        """Test subtask parsing with default values for optional fields."""
        minimal_subtask_yaml = tmp_path / "minimal_subtask.yaml"
        minimal_subtask_yaml.write_text("""
domain: "malware_classification"
tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task"
    subtasks:
      - subtask_id: "test_subtask"
        title: "Test SubTask"
        description: "A test subtask"
        objective: "Complete the test"
""")

        loader = TaskConfigLoader("malware_classification")
        tasks = loader.load_tasks_from_file(str(minimal_subtask_yaml))

        subtask = tasks["test_task"].subtasks[0]
        assert subtask.completion_conditions == []
        assert subtask.depends_on == []

    def test_validate_all_dependencies_success(self, temp_tasks_file):
        """Test successful dependency validation."""
        loader = TaskConfigLoader("malware_classification")
        # Should not raise any exception
        tasks = loader.load_tasks_from_file(temp_tasks_file)

        # Verify the dependency structure from the test file
        task = tasks["malware_family_analysis"]
        static_analysis = next(st for st in task.subtasks if st.subtask_id == "static_analysis")
        dynamic_analysis = next(st for st in task.subtasks if st.subtask_id == "dynamic_analysis")

        assert static_analysis.depends_on == []
        assert dynamic_analysis.depends_on == ["static_analysis"]

    def test_validate_all_dependencies_invalid(self, tmp_path):
        """Test dependency validation with invalid dependencies."""
        invalid_deps_yaml = tmp_path / "invalid_deps.yaml"
        invalid_deps_yaml.write_text("""
domain: "malware_classification"
tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task"
    subtasks:
      - subtask_id: "subtask1"
        title: "SubTask 1"
        description: "First subtask"
        objective: "Do something"
        depends_on: ["nonexistent_subtask"]
      - subtask_id: "subtask2"
        title: "SubTask 2"
        description: "Second subtask"
        objective: "Do something else"
        depends_on: ["another_nonexistent"]
""")

        loader = TaskConfigLoader("malware_classification")

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            loader.load_tasks_from_file(str(invalid_deps_yaml))

        error_msg = str(exc_info.value)
        assert "Dependency validation errors" in error_msg
        assert "nonexistent_subtask" in error_msg
        assert "another_nonexistent" in error_msg

    def test_multiple_tasks_loading(self, tmp_path):
        """Test loading multiple tasks from single YAML file."""
        multi_task_yaml = tmp_path / "multi_tasks.yaml"
        multi_task_yaml.write_text("""
domain: "malware_classification"
tasks:
  - task_id: "task1"
    title: "First Task"
    description: "First test task"
    subtasks: []
  - task_id: "task2"
    title: "Second Task"
    description: "Second test task"
    subtasks: []
""")

        loader = TaskConfigLoader("malware_classification")
        tasks = loader.load_tasks_from_file(str(multi_task_yaml))

        assert len(tasks) == 2
        assert "task1" in tasks
        assert "task2" in tasks
        assert tasks["task1"].title == "First Task"
        assert tasks["task2"].title == "Second Task"

    def test_complex_task_structure(self, tmp_path):
        """Test loading complex task with multiple subtasks and dependencies."""
        complex_yaml = tmp_path / "complex_task.yaml"
        complex_yaml.write_text("""
domain: "malware_classification"
tasks:
  - task_id: "complex_analysis"
    title: "Complex Malware Analysis"
    description: "Multi-stage analysis workflow"
    initial_context:
      priority: "high"
      analyst: "security_team"
    subtasks:
      - subtask_id: "init"
        title: "Initialization"
        description: "Setup analysis environment"
        objective: "Prepare for analysis"
        completion_conditions: ["setup_env"]
        depends_on: []
      - subtask_id: "static"
        title: "Static Analysis"
        description: "Static analysis phase"
        objective: "Extract static indicators"
        completion_conditions: ["file_analysis", "string_extraction"]
        depends_on: ["init"]
      - subtask_id: "dynamic"
        title: "Dynamic Analysis"
        description: "Dynamic analysis phase"
        objective: "Observe runtime behavior"
        completion_conditions: ["sandbox_execution"]
        depends_on: ["init", "static"]
      - subtask_id: "report"
        title: "Report Generation"
        description: "Generate analysis report"
        objective: "Compile findings"
        completion_conditions: ["generate_report"]
        depends_on: ["static", "dynamic"]
""")

        loader = TaskConfigLoader("malware_classification")
        tasks = loader.load_tasks_from_file(str(complex_yaml))

        task = tasks["complex_analysis"]
        assert len(task.subtasks) == 4
        assert task.initial_context["priority"] == "high"

        # Verify dependency structure
        init_task = next(st for st in task.subtasks if st.subtask_id == "init")
        static_task = next(st for st in task.subtasks if st.subtask_id == "static")
        dynamic_task = next(st for st in task.subtasks if st.subtask_id == "dynamic")
        report_task = next(st for st in task.subtasks if st.subtask_id == "report")

        assert init_task.depends_on == []
        assert static_task.depends_on == ["init"]
        assert dynamic_task.depends_on == ["init", "static"]
        assert report_task.depends_on == ["static", "dynamic"]
