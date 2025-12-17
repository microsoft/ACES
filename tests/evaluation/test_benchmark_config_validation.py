"""
Unit tests for evaluation configuration validation in BenchmarkConfigLoader.
"""

import os
import tempfile
import pytest
import yaml

from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException


DEFAULT_PROMPTS = {
    "instruction": "instructions/default.md",
    "assistant": "assistants/default.md",
    "submit": "submits/default.md",
    "continue": "continues/default.md",
}


def _normalize_yaml_with_prompts(raw_yaml: str, prompts: dict[str, str] | None = None) -> str:
  """Upgrade legacy benchmark YAML to include required prompt mappings."""

  data = yaml.safe_load(raw_yaml)
  if not isinstance(data, dict):
    return raw_yaml

  prompts = prompts or DEFAULT_PROMPTS

  # Ensure global defaults exist and have prompts
  global_defaults = data.setdefault("global_defaults", {})
  global_defaults.setdefault("prompts", prompts.copy())

  # Ensure each task has explicit prompts mapping
  for task in data.get("tasks", []):
    if not isinstance(task, dict):
      continue
    template = task.pop("prompt_template_file", None)
    if template:
      task.setdefault(
        "prompts",
        {
          "instruction": template,
          "assistant": template,
          "submit": template,
          "continue": template,
        },
      )
    else:
      task.setdefault("prompts", prompts.copy())

  return yaml.safe_dump(data, sort_keys=False)


class TestBenchmarkConfigLoaderEvaluation:
    """Test cases for evaluation configuration validation in BenchmarkConfigLoader."""

    def test_valid_static_evaluation_config(self):
        """Test loading task with valid static evaluation configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(_normalize_yaml_with_prompts("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "static_eval_task"
    title: "Static Evaluation Task"
    description: "Task with static evaluation"
    prompts:
      instruction: "test_task_prompt.md"
      assistant: "test_task_prompt.md"
      submit: "test_task_prompt.md"
      continue: "test_continue.md"
    execution_config:
      executors:
        test_executor:
          timeout: 300
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "flag{correct_answer}"
      scoring:
        max_score: 1.0

    subtasks: []
"""))
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "static_eval_task" in tasks
            task = tasks["static_eval_task"]
            assert task.submission_evaluation_config is not None
            assert task.submission_evaluation_config["strategy"] == "static"
            assert task.submission_evaluation_config["criteria"]["expected_answers"] == ["flag{correct_answer}"]
            assert task.submission_evaluation_config["scoring"]["max_score"] == 1.0

        finally:
            os.unlink(temp_path)

    def test_valid_llm_evaluation_config(self):
        """Test loading task with valid LLM evaluation configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(_normalize_yaml_with_prompts("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "llm_eval_task"
    title: "LLM Evaluation Task"
    description: "Task with LLM evaluation"
    prompt_template_file: "test_template.j2"
    execution_config:
      executors:
        test_executor:
          timeout: 300
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        golden_answer: "The malware is a banking trojan"
        model: "gpt-4"
        judge_system_template: "test_system.md"
        judge_user_template: "test_user.md"
      scoring:
        max_score: 100.0
    subtasks: []
"""))
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "llm_eval_task" in tasks
            task = tasks["llm_eval_task"]
            assert task.submission_evaluation_config["strategy"] == "llm_judge"
            assert task.submission_evaluation_config["criteria"]["golden_answer"] == "The malware is a banking trojan"
            assert task.submission_evaluation_config["criteria"]["model"] == "gpt-4"

        finally:
            os.unlink(temp_path)

    def test_missing_evaluation_config(self):
        """Test that missing evaluation_config causes validation failure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(_normalize_yaml_with_prompts("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "no_eval_task"
    title: "No Evaluation Task"
    description: "Task without evaluation config"
    prompt_template_file: "test_template.j2"
    execution_config:
      executors:
        test_executor:
          timeout: 300
    episode_config:
      max_steps: 10
    subtasks: []
"""))
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="missing required submission_evaluation_config"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_invalid_evaluation_strategy(self):
        """Test that custom/unknown strategies are now accepted.

        Strategy validation was relaxed to allow domains to define custom strategies
        that are registered at the client-side (inspect_ai scoring registry).
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(_normalize_yaml_with_prompts("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "custom_strategy_task"
    title: "Custom Strategy Task"
    description: "Task with custom evaluation strategy"
    prompt_template_file: "test_template.j2"
    execution_config:
      executors:
        test_executor:
          timeout: 300
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "custom_strategy"
      criteria: {}
      scoring:
        max_score: 1.0

    subtasks: []
"""))
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            # Custom strategies are now allowed (validation deferred to client-side)
            tasks = loader.load_tasks_from_file(temp_path)
            assert len(tasks) == 1

        finally:
            os.unlink(temp_path)

    def test_missing_criteria_section(self):
        """Test that missing criteria section causes validation failure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(_normalize_yaml_with_prompts("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "missing_criteria_task"
    title: "Missing Criteria Task"
    description: "Task without criteria section"
    prompt_template_file: "test_template.j2"
    execution_config:
      executors:
        test_executor:
          timeout: 300
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      scoring:
        max_score: 1.0

    subtasks: []
"""))
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="Missing or invalid criteria in submission_evaluation_config"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_invalid_max_score(self):
        """Test that invalid max_score causes validation failure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(_normalize_yaml_with_prompts("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "invalid_max_score_task"
    title: "Invalid Max Score Task"
    description: "Task with invalid max_score"
    prompt_template_file: "test_template.j2"
    execution_config:
      executors:
        test_executor:
          timeout: 300
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer"]
      scoring:
        max_score: -1.0
    subtasks: []
"""))
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="max_score must be a positive number"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_static_missing_expected_answers(self):
        """Test that static strategy without expected_answers fails validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(_normalize_yaml_with_prompts("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "static_no_answers_task"
    title: "Static No Answers Task"
    description: "Static task without expected_answers"
    prompt_template_file: "test_template.j2"
    execution_config:
      executors:
        test_executor:
          timeout: 300
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria: {}
      scoring:
        max_score: 1.0

    subtasks: []
"""))
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="static strategy requires 'expected_answers'"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_llm_optional_golden_answer(self):
        """Test that LLM strategy works with or without golden_answer."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(_normalize_yaml_with_prompts("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "llm_no_golden_task"
    title: "LLM No Golden Task"
    description: "LLM task without golden_answer"
    prompt_template_file: "test_template.j2"
    execution_config:
      executors:
        test_executor:
          timeout: 300
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "system_template.md"
        judge_user_template: "user_template.md"
      scoring:
        max_score: 1.0

    subtasks: []
"""))
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            # This should NOT raise an exception now that golden_answer is optional
            tasks = loader.load_tasks_from_file(temp_path)
            assert len(tasks) == 1
            assert "llm_no_golden_task" in tasks
            assert tasks["llm_no_golden_task"].task_id == "llm_no_golden_task"

        finally:
            os.unlink(temp_path)

    def test_llm_missing_model(self):
        """Test that LLM strategy without model fails validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(_normalize_yaml_with_prompts("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "llm_no_model_task"
    title: "LLM No Model Task"
    description: "LLM task without model"
    prompt_template_file: "test_template.j2"
    execution_config:
      executors:
        test_executor:
          timeout: 300
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        golden_answer: "The correct answer"
      scoring:
        max_score: 1.0

    subtasks: []
"""))
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="llm_judge strategy requires 'model'"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)
