"""
Unit tests for BenchmarkManager - Core Functionality.

Tests initialization, task loading, basic operations, and task/subtask retrieval.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from saber.server.benchmarks.exceptions import (
    InvalidTaskDefinitionException,
    SubTaskNotFoundException,
    TaskNotFoundException,
)
from saber.server.benchmarks.prompt_generator import TemplateValidationError
from saber.server.benchmarks.subtask import SubTask
from saber.server.benchmarks.task import Task
from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.benchmark_manager import BenchmarkManager


class TestBenchmarkManagerCore:
    """Test cases for BenchmarkManager core functionality."""

    def _create_benchmark_manager_from_temp_config_dir(self, temp_config_dir):
        """Helper to create BenchmarkManager from temp directory fixture."""
        return BenchmarkManager("malware_classification", temp_config_dir)

    def test_load_tasks_from_directory_success(self, tmp_path, temp_config_dir_helper, sample_task_yaml):
        """Test successful loading of tasks from directory structure."""
        temp_config_dir = temp_config_dir_helper(tmp_path, sample_task_yaml)
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        # Clear tasks and reload
        manager.tasks = {}
        manager.load_tasks_from_directory()

        assert len(manager.tasks) == 1
        assert "malware_family_analysis" in manager.tasks

    @patch.object(BenchmarkConfigLoader, "load_tasks_from_directory")
    def test_load_tasks_from_directory_delegates_to_config_loader(self, mock_load, temp_config_dir):
        """Test that loading delegates to BenchmarkConfigLoader."""
        mock_task = Mock(spec=Task)
        mock_task.prompts={"instruction": "test_prompt.md", "assistant": "test_prompt.md", "submit": "test_prompt.md"}
        mock_task.task_id = "test_task"
        mock_task.submission_evaluation_config = None  # No LLM judge config
        mock_task.depends_on_task_id = None  # No dependencies
        mock_tasks = {"test_task": mock_task}
        mock_load.return_value = mock_tasks

        # Also mock the prompt generator validation to prevent template validation errors
        with patch.object(BenchmarkManager, 'validate_all_task_templates'):
            manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        # Should be called with the full path to tasks directory
        expected_path = str(Path(temp_config_dir) / "tasks")
        mock_load.assert_called_with(expected_path)
        assert manager.tasks == mock_tasks

    def test_load_tasks_invalid_file_raises_exception(self):
        """Test that loading invalid file raises exception."""
        from saber.server.benchmarks.prompt_generator import TemplateValidationError
        with pytest.raises(TemplateValidationError):
            BenchmarkManager("malware_classification", "/nonexistent/file.yaml")

    def test_get_task_success(self, temp_config_dir):
        """Test successful task retrieval."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        task = manager.get_task("malware_family_analysis")

        assert isinstance(task, Task)
        assert task.task_id == "malware_family_analysis"

    def test_get_task_not_found(self, temp_config_dir):
        """Test task retrieval when task doesn't exist."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        with pytest.raises(TaskNotFoundException) as exc_info:
            manager.get_task("nonexistent_task")

        assert exc_info.value.task_id == "nonexistent_task"
        assert "nonexistent_task" in str(exc_info.value)

    def test_get_subtask_success(self, temp_config_dir):
        """Test successful subtask retrieval."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        subtask = manager.get_subtask("malware_family_analysis", "static_analysis")

        assert isinstance(subtask, SubTask)
        assert subtask.subtask_id == "static_analysis"
        assert subtask.task_id == "malware_family_analysis"

    def test_get_subtask_task_not_found(self, temp_config_dir):
        """Test subtask retrieval when parent task doesn't exist."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        with pytest.raises(TaskNotFoundException):
            manager.get_subtask("nonexistent_task", "some_subtask")

    def test_get_subtask_subtask_not_found(self, temp_config_dir):
        """Test subtask retrieval when subtask doesn't exist."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        with pytest.raises(SubTaskNotFoundException) as exc_info:
            manager.get_subtask("malware_family_analysis", "nonexistent_subtask")

        assert exc_info.value.task_id == "malware_family_analysis"
        assert exc_info.value.subtask_id == "nonexistent_subtask"

    def test_list_tasks(self, temp_config_dir):
        """Test listing all available tasks."""
        manager = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        task_list = manager.list_tasks()

        assert isinstance(task_list, list)
        assert len(task_list) == 1

        task_info = task_list[0]
        assert task_info["task_id"] == "malware_family_analysis"
        assert task_info["title"] == "Malware Family Classification and Analysis"
        assert task_info["description"] is not None
        assert task_info["subtask_count"] == 2

    def test_list_tasks_empty(self, tmp_path, temp_config_dir_helper):
        """Test listing tasks when no tasks are defined."""
        # Create tasks.yaml in the tmp directory
        yaml_content = """
domain: "malware_classification"

benchmark_config:
  episode_attempts: 1

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

executors:
  - bash_executor
  - python_executor

tasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("malware_classification", config_dir)

        task_list = manager.list_tasks()
        assert task_list == []

    def test_list_tasks_multiple_tasks(self, tmp_path, temp_config_dir_helper):
        """Test listing multiple tasks."""
        # Create tasks.yaml in the tmp directory
        yaml_content = """
domain: "malware_classification"

benchmark_config:
  episode_attempts: 2

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "task1"
    title: "First Task"
    description: "First test task"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      executors:
        bash_executor:
          timeout: 300
        python_executor:
          timeout: 300
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        max_score: 1.0

    subtasks:
      - subtask_id: "subtask1"
        title: "SubTask 1"
        description: "First subtask"
        objective: "Complete first"
  - task_id: "task2"
    title: "Second Task"
    description: "Second test task"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      executors:
        bash_executor:
          timeout: 300
        python_executor:
          timeout: 300
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        max_score: 1.0

    subtasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("malware_classification", config_dir)

        task_list = manager.list_tasks()
        assert len(task_list) == 2

        task_ids = [task["task_id"] for task in task_list]
        assert "task1" in task_ids
        assert "task2" in task_ids

        # Check subtask counts
        task1_info = next(task for task in task_list if task["task_id"] == "task1")
        task2_info = next(task for task in task_list if task["task_id"] == "task2")
        assert task1_info["subtask_count"] == 1
        assert task2_info["subtask_count"] == 0

    def test_domain_consistency(self, temp_config_dir):
        """Test that domain is consistently used across components."""
        domain = "malware_classification"
        manager = BenchmarkManager(domain, temp_config_dir)

        assert manager.domain == domain
        assert manager.config_loader.domain == domain

        # Check that loaded tasks have correct domain
        for task in manager.tasks.values():
            assert task.domain == domain

    def test_task_manager_logging_behavior(self, temp_config_dir, caplog):
        """Test that BenchmarkManager provides appropriate logging."""
        import logging

        with caplog.at_level(logging.INFO):
            self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)

        structured_events = [getattr(record, "_structured", {}) for record in caplog.records]

        # Check for initialization logs with structured payloads
        assert any(event.get("event") == "benchmark_manager_init" for event in structured_events)

        # Ensure load operation completed and reported task count
        assert any(
            event.get("event") == "operation_completed"
            and event.get("operation") == "benchmark_manager_load_tasks"
            for event in structured_events
        )
        assert any(
            event.get("operation") == "benchmark_manager_load_tasks" and event.get("task_count") == 1
            for event in structured_events
        )

    def test_task_manager_file_path_handling(self, temp_config_dir):
        """Test that BenchmarkManager correctly handles file path types."""
        # Test with string path
        manager1 = self._create_benchmark_manager_from_temp_config_dir(temp_config_dir)
        assert isinstance(manager1.tasks_dir_path, Path)

        # Test with Path object
        path_obj = Path(temp_config_dir)
        manager2 = BenchmarkManager("malware_classification", path_obj)
        assert isinstance(manager2.tasks_dir_path, Path)
        # The tasks_dir_path should be the tasks directory within the config directory
        assert manager2.tasks_dir_path == path_obj / "tasks"


class TestBenchmarkManagerBenchmarkConfig:
    """Test cases for BenchmarkManager benchmark configuration functionality."""

    def test_get_benchmark_config_success(self, tmp_path, temp_config_dir_helper):
        """Test successful retrieval of benchmark configuration."""
        yaml_content = """
domain: "webapp_pentest"

benchmark_config:
  episode_attempts: 5
  max_duration_minutes: 30

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      executors:
        bash_executor:
          timeout: 300
        python_executor:
          timeout: 300
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        max_score: 1.0

    subtasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("webapp_pentest", config_dir)
        config = manager.get_benchmark_config()

        assert config["episode_attempts"] == 5
        assert config["max_duration_minutes"] == 30

    def test_start_benchmark_success(self, tmp_path, temp_config_dir_helper):
        """Test successful benchmark session start with multi-prompt validation."""
        yaml_content = """
domain: "webapp_pentest"

benchmark_config:
  episode_attempts: 3

global_defaults:
  prompts:
    instruction: "test_task_prompt.md"
    assistant: "test_task_prompt.md"
    submit: "test_task_prompt.md"
  execution_config:
    timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "task1"
    title: "Task 1"
    description: "First task"
    execution_config:
      executors:
        bash_executor:
          timeout: 300
        python_executor:
          timeout: 300
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        max_score: 1.0

    subtasks: []
  - task_id: "task2"
    title: "Task 2"
    description: "Second task"
    execution_config:
      executors:
        bash_executor:
          timeout: 300
        python_executor:
          timeout: 300
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        max_score: 1.0

    subtasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("webapp_pentest", config_dir)

        # Use current architecture - get benchmark info
        benchmark_info = manager.get_benchmark_info()
        result = benchmark_info.to_dict()

        assert result["domain"] == "webapp_pentest"
        assert result["total_tasks"] == 2
        assert "task1" in [task["task_id"] for task in result["tasks"]]
        assert "task2" in [task["task_id"] for task in result["tasks"]]

        # Enhanced: Validate that each task has proper prompt structure
        for task_dict in result["tasks"]:
            # Each task should have generated prompts in TaskInfo structure
            # Note: These tasks use legacy prompt_template_file, so they should have
            # instruction_prompt, assistant_prompt, and submit_prompt all containing
            # the same content from the single template (backward compatibility)
            assert "instruction_prompt" in task_dict
            assert "assistant_prompt" in task_dict
            assert "submit_prompt" in task_dict

            # All prompts should be non-empty strings
            assert isinstance(task_dict["instruction_prompt"], str)
            assert isinstance(task_dict["assistant_prompt"], str)
            assert isinstance(task_dict["submit_prompt"], str)
            assert len(task_dict["instruction_prompt"]) > 0
            assert len(task_dict["assistant_prompt"]) > 0
            assert len(task_dict["submit_prompt"]) > 0

            # Deprecated fields should not exist
            assert "initial_prompt" not in task_dict
            assert "prompt" not in task_dict

    def test_list_benchmark_tasks_with_episode_attempts(self, tmp_path, temp_config_dir_helper):
        """Test listing tasks with episode attempts information."""
        yaml_content = """
domain: "webapp_pentest"

benchmark_config:
  episode_attempts: 3

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "task_default"
    title: "Task with Default"
    description: "Uses domain default"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      executors:
        bash_executor:
          timeout: 300
        python_executor:
          timeout: 300
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        max_score: 1.0

    subtasks: []
  - task_id: "task_override"
    title: "Task with Override"
    description: "Overrides domain default"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      executors:
        bash_executor:
          timeout: 300
        python_executor:
          timeout: 300
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        max_score: 1.0
    benchmark_config:
      episode_attempts: 10
    subtasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("webapp_pentest", config_dir)
        tasks = manager.list_benchmark_tasks()

        assert len(tasks) == 2

        task_default = next(t for t in tasks if t["task_id"] == "task_default")
        task_override = next(t for t in tasks if t["task_id"] == "task_override")

        assert task_default["episode_attempts"] == 3  # Domain default
        assert task_override["episode_attempts"] == 10  # Task override
        assert "benchmark_config" in task_default
        assert "benchmark_config" in task_override

    def test_task_get_episode_attempts(self, tmp_path, temp_config_dir_helper):
        """Test that tasks correctly return episode attempts."""
        yaml_content = """
domain: "webapp_pentest"

benchmark_config:
  episode_attempts: 7

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      executors:
        bash_executor:
          timeout: 300
        python_executor:
          timeout: 300
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        max_score: 1.0
    benchmark_config:
      episode_attempts: 15
    subtasks: []
"""
        config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        manager = BenchmarkManager("webapp_pentest", config_dir)
        task = manager.get_task("test_task")

        assert task.get_episode_attempts() == 15

    def test_missing_benchmark_config_fails(self, sample_yaml_missing_benchmark_config, tmp_path, temp_config_dir_helper):
        """Test that missing benchmark_config section causes failure."""
        config_dir = temp_config_dir_helper(tmp_path, sample_yaml_missing_benchmark_config)

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            BenchmarkManager("malware_classification", config_dir)

        # Updated error message reflects the new global defaults system
        assert "Missing required 'episode_attempts' in benchmark configuration" in str(exc_info.value)
        assert "global_defaults.benchmark_config" in str(exc_info.value)

    def test_missing_episode_attempts_fails(self, sample_yaml_missing_episode_attempts, tmp_path, temp_config_dir_helper):
        """Test that missing episode_attempts causes failure."""
        config_dir = temp_config_dir_helper(tmp_path, sample_yaml_missing_episode_attempts)

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            BenchmarkManager("malware_classification", config_dir)

        assert "Missing required 'episode_attempts'" in str(exc_info.value)

    def test_invalid_episode_attempts_fails(self, sample_yaml_invalid_episode_attempts, tmp_path, temp_config_dir_helper):
        """Test that invalid episode_attempts value causes failure."""
        config_dir = temp_config_dir_helper(tmp_path, sample_yaml_invalid_episode_attempts)

        with pytest.raises(InvalidTaskDefinitionException) as exc_info:
            BenchmarkManager("malware_classification", config_dir)

        assert "episode_attempts must be a positive integer" in str(exc_info.value)


class TestBenchmarkManagerJudgeRenderer:
    """Test cases for BenchmarkManager judge prompt renderer functionality.

    CLIENT-SIDE EVALUATION: Judge rendering is now done client-side.
    These tests verify that the renderer injection is no longer performed.
    """

    def test_create_renderer_success(self, tmp_path, temp_config_dir_helper):
        """Test that LLM judge tasks load correctly (no renderer injection in client-side eval)."""
        # Create a basic config with LLM judge task
        yaml_content = """
domain: "test"

benchmark_config:
  episode_attempts: 1

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor

tasks:
  - task_id: "test_llm_task"
    title: "Test LLM Task"
    description: "Test description"
    domain: "test"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        golden_answer: "Expected answer for test task"
        judge_system_template: "judge/system.md"
        judge_user_template: "judge/user.md"
"""

        config_dir = temp_config_dir_helper(tmp_path, yaml_content)
        manager = BenchmarkManager("test", config_dir)

        # Get the task - should load successfully
        task = manager.get_task("test_llm_task")
        assert task.submission_evaluation_config["strategy"] == "llm_judge"

        # CLIENT-SIDE EVALUATION: judge_prompt_renderer should NOT be injected
        assert "judge_prompt_renderer" not in task.submission_evaluation_config

    def test_create_renderer_calls_correct_method(self, tmp_path, temp_config_dir_helper):
        """Test that render_judge_prompt_for_episode is deprecated (client-side eval)."""
        # Create a basic config with LLM judge task
        yaml_content = """
domain: "test"

benchmark_config:
  episode_attempts: 1

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor

tasks:
  - task_id: "test_llm_task"
    title: "Test LLM Task"
    description: "Test description"
    domain: "test"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        golden_answer: "Expected answer for test task"
        judge_system_template: "judge/system.md"
        judge_user_template: "judge/user.md"
"""

        config_dir = temp_config_dir_helper(tmp_path, yaml_content)
        manager = BenchmarkManager("test", config_dir)

        # Verify that calling deprecated method raises NotImplementedError
        task = manager.get_task("test_llm_task")
        mock_episode = Mock()

        with pytest.raises(NotImplementedError) as exc_info:
            manager.render_judge_prompt_for_episode("test_llm_task", mock_episode)

        assert "client-side" in str(exc_info.value).lower()

    def test_create_renderer_handles_attribute_error(self, tmp_path, temp_config_dir_helper):
        """Test that LLM judge tasks don't have renderer injected (client-side eval)."""
        # Create a basic config with LLM judge task
        yaml_content = """
domain: "test"

benchmark_config:
  episode_attempts: 1

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor

tasks:
  - task_id: "test_llm_task"
    title: "Test LLM Task"
    description: "Test description"
    domain: "test"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        golden_answer: "Expected answer for test task"
        judge_system_template: "judge/system.md"
        judge_user_template: "judge/user.md"
"""

        config_dir = temp_config_dir_helper(tmp_path, yaml_content)
        manager = BenchmarkManager("test", config_dir)

        # Verify no renderer is injected
        task = manager.get_task("test_llm_task")
        assert "judge_prompt_renderer" not in task.submission_evaluation_config

    def test_no_renderer_for_static_evaluation(self, tmp_path, temp_config_dir_helper):
        """Test that static evaluation tasks don't get judge prompt renderer."""
        # Create a basic config with static task
        yaml_content = """
domain: "test"

benchmark_config:
  episode_attempts: 1

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

allowed_executors:
  - bash_executor

tasks:
  - task_id: "test_static_task"
    title: "Test Static Task"
    description: "Test description"
    domain: "test"
    prompt_template_file: "test_task_prompt.md"
    execution_config:
      allowed_executors:
        - bash_executor
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer1"]
      scoring:
        max_score: 1.0
"""

        config_dir = temp_config_dir_helper(tmp_path, yaml_content)
        manager = BenchmarkManager("test", config_dir)

        # Get the task - should not have renderer injected
        task = manager.get_task("test_static_task")
        assert task.submission_evaluation_config["strategy"] == "static"
        assert "judge_prompt_renderer" not in task.submission_evaluation_config


class TestBenchmarkManagerMultiPrompt:
    """Test cases for BenchmarkManager multi-prompt functionality."""

    def test_multi_prompt_task_loading(self, tmp_path, temp_config_dir_helper, sample_multi_prompt_task_yaml):
        """Test successful loading of tasks with multi-prompt configuration."""
        temp_config_dir = temp_config_dir_helper(tmp_path, sample_multi_prompt_task_yaml)
        manager = BenchmarkManager("cybersecurity", temp_config_dir)

        assert len(manager.tasks) == 1
        assert "multi_prompt_security_task" in manager.tasks

        task = manager.tasks["multi_prompt_security_task"]

        # Verify multi-prompt structure is set
        assert hasattr(task, 'prompts')
        assert isinstance(task.prompts, dict)

        # Verify all three prompt types are present
        assert 'instruction' in task.prompts
        assert 'assistant' in task.prompts
        assert 'submit' in task.prompts

        assert task.prompts['instruction'] == "instructions/security_analysis_instruction.md"
        assert task.prompts['assistant'] == "assistants/security_analysis_assistant.md"
        assert task.prompts['submit'] == "submits/security_analysis_submit.md"

    def test_get_benchmark_info_with_multi_prompts(self, tmp_path, temp_config_dir_helper, sample_multi_prompt_task_yaml):
        """Test that get_benchmark_info generates all three prompts for multi-prompt tasks."""
        temp_config_dir = temp_config_dir_helper(tmp_path, sample_multi_prompt_task_yaml)
        manager = BenchmarkManager("cybersecurity", temp_config_dir)

        benchmark_info = manager.get_benchmark_info()

        assert benchmark_info.domain == "cybersecurity"
        assert benchmark_info.total_tasks == 1
        assert len(benchmark_info.tasks) == 1

        task_info = benchmark_info.tasks[0]
        assert task_info.task_id == "multi_prompt_security_task"

        # Verify all three prompts are generated and non-empty
        assert hasattr(task_info, 'instruction_prompt')
        assert hasattr(task_info, 'assistant_prompt')
        assert hasattr(task_info, 'submit_prompt')

        assert task_info.instruction_prompt is not None
        assert task_info.assistant_prompt is not None
        assert task_info.submit_prompt is not None

        assert len(task_info.instruction_prompt) > 0
        assert len(task_info.assistant_prompt) > 0
        assert len(task_info.submit_prompt) > 0

        # Verify prompts contain expected content
        assert "security analyst" in task_info.instruction_prompt.lower()
        assert "assistant" in task_info.assistant_prompt.lower()
        assert "submit" in task_info.submit_prompt.lower()

    def test_multi_prompt_task_info_structure(self, tmp_path, temp_config_dir_helper, sample_multi_prompt_task_yaml):
        """Test that TaskInfo objects have the correct multi-prompt structure."""
        temp_config_dir = temp_config_dir_helper(tmp_path, sample_multi_prompt_task_yaml)
        manager = BenchmarkManager("cybersecurity", temp_config_dir)

        benchmark_info = manager.get_benchmark_info()
        task_info = benchmark_info.tasks[0]

        # Convert to dict to verify structure
        task_dict = task_info.model_dump()

        # Verify multi-prompt fields exist
        assert 'instruction_prompt' in task_dict
        assert 'assistant_prompt' in task_dict
        assert 'submit_prompt' in task_dict

        # Verify deprecated single prompt field does NOT exist
        assert 'initial_prompt' not in task_dict
        assert 'prompt' not in task_dict

    def test_prompt_generator_multi_prompt_rendering(self, tmp_path, temp_config_dir_helper, sample_multi_prompt_task_yaml):
        """Test that PromptGenerator correctly renders all three prompt types."""
        temp_config_dir = temp_config_dir_helper(tmp_path, sample_multi_prompt_task_yaml)
        manager = BenchmarkManager("cybersecurity", temp_config_dir)

        task_id = "multi_prompt_security_task"
        task = manager.get_task(task_id)

        # Test direct prompt rendering
        rendered_prompts = manager.prompt_generator.render_agent_prompts_for_task(task)

        # Verify all three prompt types are rendered
        expected_prompt_types = {'instruction', 'assistant', 'submit'}
        actual_prompt_types = set(rendered_prompts.keys())

        assert expected_prompt_types == actual_prompt_types, f"Expected {expected_prompt_types}, got {actual_prompt_types}"

        # Verify each prompt is rendered and non-empty
        for prompt_type, prompt_content in rendered_prompts.items():
            assert prompt_content is not None
            assert len(prompt_content) > 0
            assert isinstance(prompt_content, str)

    def test_multi_prompt_fails_fast_on_missing_prompts(self, tmp_path, temp_config_dir_helper):
        """Test that the system fails fast when any prompt type is missing."""
        # Create configuration with missing assistant prompt
        incomplete_yaml = """
domain: "incomplete_domain"

benchmark_config:
  episode_attempts: 1

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
  episode_config:
    max_steps: 50

executors:
  - bash_executor

tasks:
  - task_id: "incomplete_task"
    title: "Incomplete Task"
    description: "A task missing one of the required prompt types"
    prompts:
      instruction: "instructions/test_instruction.md"
      # Missing assistant and submit prompts
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0

    subtasks: []
"""

        temp_config_dir = temp_config_dir_helper(tmp_path, incomplete_yaml)

        # Should fail fast during BenchmarkManager initialization (template validation)
        with pytest.raises(TemplateValidationError) as exc_info:
            BenchmarkManager("incomplete_domain", temp_config_dir)

        # Verify the error message indicates template validation failure
        error_message = str(exc_info.value)
        assert "Template validation failed" in error_message


class TestBenchmarkManagerOrchestrationRoles:
    """Test cases for orchestrated task role configuration."""

    def test_orchestrated_task_uses_yaml_roles(self, tmp_path, temp_config_dir_helper):
        """Test that orchestrated tasks use roles from YAML instead of hardcoded values."""
        yaml_content = """
domain: test_domain

global_defaults:
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10
  prompts:
    instruction: "instructions/default.md"
    assistant: "assistants/default.md"
    submit: "submits/default.md"

benchmark_config:
  episode_attempts: 1

executors:
  - bash

tasks:
  - task_id: "blue_defend"
    role: blue
    title: "Blue Team Defense"
    description: "Defender task"
    prompts:
      instruction: "instructions/blue.md"
      assistant: "assistants/blue.md"
      submit: "submits/blue.md"
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["defended"]
      scoring:
        max_score: 1.0
    subtasks: []

  - task_id: "red_attack"
    role: red
    depends_on_task_id: "blue_defend"
    title: "Red Team Attack"
    description: "Attacker task"
    prompts:
      instruction: "instructions/red.md"
      assistant: "assistants/red.md"
      submit: "submits/red.md"
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["attacked"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
        temp_config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        # Create template files to satisfy validation
        prompts_dir = Path(temp_config_dir) / "prompts"
        for subdir in ["instructions", "assistants", "submits"]:
            (prompts_dir / subdir).mkdir(parents=True, exist_ok=True)
            for template in ["default.md", "blue.md", "red.md"]:
                (prompts_dir / subdir / template).write_text("# Template")

        manager = BenchmarkManager("test_domain", temp_config_dir)
        tasks = manager.list_benchmark_tasks()

        # With dependencies, we should see both tasks listed
        # The blue task should have role="blue" and red should have role="red"
        assert len(tasks) >= 1

        # Find tasks by ID
        blue_task = next((t for t in tasks if t.get("task_id") == "blue_defend"), None)
        red_task = next((t for t in tasks if t.get("task_id") == "red_attack"), None)

        # At minimum, the blue task should exist
        assert blue_task is not None

        # Check that tasks were loaded with roles from YAML
        assert manager.tasks["blue_defend"].role == "blue"
        assert manager.tasks["red_attack"].role == "red"

    def test_orchestrated_task_root_validation(self, tmp_path, temp_config_dir_helper):
        """Test that orchestration fails if root task lacks role."""
        yaml_content = """
domain: test_domain

global_defaults:
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10
  prompts:
    instruction: "instructions/default.md"
    assistant: "assistants/default.md"
    submit: "submits/default.md"

benchmark_config:
  episode_attempts: 1

executors:
  - bash

tasks:
  - task_id: "root_task"
    # Missing role field - should fail
    title: "Root Task"
    description: "Root without role"
    prompts:
      instruction: "instructions/root.md"
      assistant: "assistants/root.md"
      submit: "submits/root.md"
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    subtasks: []

  - task_id: "dependent_task"
    role: dependent
    depends_on_task_id: "root_task"
    title: "Dependent Task"
    description: "Depends on root"
    prompts:
      instruction: "instructions/dependent.md"
      assistant: "assistants/dependent.md"
      submit: "submits/dependent.md"
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
        temp_config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        # Create template files
        templates_dir = Path(temp_config_dir) / "templates"
        for subdir in ["instructions", "assistants", "submits"]:
            (templates_dir / subdir).mkdir(parents=True, exist_ok=True)
            for template in ["default.md", "root.md", "dependent.md"]:
                (templates_dir / subdir / template).write_text("# Template")

        # Should fail during initialization when validating dependencies
        with pytest.raises(InvalidTaskDefinitionException, match="Root task 'root_task' must have a 'role' defined"):
            BenchmarkManager("test_domain", temp_config_dir)

    def test_orchestrated_task_custom_semantic_roles(self, tmp_path, temp_config_dir_helper):
        """Test orchestration with custom semantic role names."""
        yaml_content = """
domain: test_domain

global_defaults:
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10
  prompts:
    instruction: "instructions/default.md"
    assistant: "assistants/default.md"
    submit: "submits/default.md"

benchmark_config:
  episode_attempts: 1

executors:
  - bash

tasks:
  - task_id: "defender_task"
    role: defender
    title: "System Defender"
    description: "Defend the system"
    prompts:
      instruction: "instructions/defender.md"
      assistant: "assistants/defender.md"
      submit: "submits/defender.md"
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["defended"]
      scoring:
        max_score: 1.0
    subtasks: []

  - task_id: "attacker_task"
    role: attacker
    depends_on_task_id: "defender_task"
    title: "System Attacker"
    description: "Attack the system"
    prompts:
      instruction: "instructions/attacker.md"
      assistant: "assistants/attacker.md"
      submit: "submits/attacker.md"
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["attacked"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
        temp_config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        # Create template files
        prompts_dir = Path(temp_config_dir) / "prompts"
        for subdir in ["instructions", "assistants", "submits"]:
            (prompts_dir / subdir).mkdir(parents=True, exist_ok=True)
            for template in ["default.md", "defender.md", "attacker.md"]:
                (prompts_dir / subdir / template).write_text("# Template")

        manager = BenchmarkManager("test_domain", temp_config_dir)
        tasks = manager.list_benchmark_tasks()

        # Check that tasks were loaded with custom semantic roles
        assert manager.tasks["defender_task"].role == "defender"
        assert manager.tasks["attacker_task"].role == "attacker"
        assert manager.tasks["attacker_task"].depends_on_task_id == "defender_task"

    def test_single_task_without_role_not_orchestrated(self, tmp_path, temp_config_dir_helper):
        """Test that single tasks without roles are not treated as orchestrated."""
        yaml_content = """
domain: test_domain

global_defaults:
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 10
  prompts:
    instruction: "instructions/default.md"
    assistant: "assistants/default.md"
    submit: "submits/default.md"

benchmark_config:
  episode_attempts: 1

executors:
  - bash

tasks:
  - task_id: "single_task"
    # No role field - single episode task
    title: "Single Task"
    description: "Standalone task"
    prompts:
      instruction: "instructions/single.md"
      assistant: "assistants/single.md"
      submit: "submits/single.md"
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["completed"]
      scoring:
        max_score: 1.0
    subtasks: []
"""
        temp_config_dir = temp_config_dir_helper(tmp_path, yaml_content)

        # Create template files
        prompts_dir = Path(temp_config_dir) / "prompts"
        for subdir in ["instructions", "assistants", "submits"]:
            (prompts_dir / subdir).mkdir(parents=True, exist_ok=True)
            for template in ["default.md", "single.md"]:
                (prompts_dir / subdir / template).write_text("# Template")

        manager = BenchmarkManager("test_domain", temp_config_dir)
        tasks = manager.list_benchmark_tasks()

        # Should have 1 task
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "single_task"

        # Verify the task has no role (it's a single episode task)
        assert manager.tasks["single_task"].role is None
        assert manager.tasks["single_task"].depends_on_task_id is None
