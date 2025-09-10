"""
Unit tests for evaluation configuration validation in BenchmarkConfigLoader.
"""

import os
import tempfile
import pytest

from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException


class TestBenchmarkConfigLoaderEvaluation:
    """Test cases for evaluation configuration validation in BenchmarkConfigLoader."""

    def test_valid_static_evaluation_config(self):
        """Test loading task with valid static evaluation configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "static_eval_task"
    title: "Static Evaluation Task"
    description: "Task with static evaluation"
    prompt_template_file: "test_template.j2"
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
          - "flag{correct_answer}"
      scoring:
        max_score: 1.0
    subtasks: []
""")
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "static_eval_task" in tasks
            task = tasks["static_eval_task"]
            assert task.evaluation_config is not None
            assert task.evaluation_config["strategy"] == "static"
            assert task.evaluation_config["criteria"]["expected_answers"] == ["flag{correct_answer}"]
            assert task.evaluation_config["scoring"]["max_score"] == 1.0

        finally:
            os.unlink(temp_path)

    def test_valid_llm_evaluation_config(self):
        """Test loading task with valid LLM evaluation configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "llm_eval_task"
    title: "LLM Evaluation Task"
    description: "Task with LLM evaluation"
    prompt_template_file: "test_template.j2"
    execution_config:
      timeout: 300
      allowed_executors:
        - "test_executor"
    episode_config:
      max_steps: 10
    evaluation_config:
      strategy: "llm_judge"
      criteria:
        golden_answer: "The malware is a banking trojan"
        model: "gpt-4"
      scoring:
        max_score: 100.0
    subtasks: []
""")
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "llm_eval_task" in tasks
            task = tasks["llm_eval_task"]
            assert task.evaluation_config["strategy"] == "llm_judge"
            assert task.evaluation_config["criteria"]["golden_answer"] == "The malware is a banking trojan"
            assert task.evaluation_config["criteria"]["model"] == "gpt-4"

        finally:
            os.unlink(temp_path)

    def test_missing_evaluation_config(self):
        """Test that missing evaluation_config causes validation failure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "no_eval_task"
    title: "No Evaluation Task"
    description: "Task without evaluation config"
    prompt_template_file: "test_template.j2"
    execution_config:
      timeout: 300
      allowed_executors:
        - "test_executor"
    episode_config:
      max_steps: 10
    subtasks: []
""")
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="missing required evaluation_config"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_invalid_evaluation_strategy(self):
        """Test that invalid evaluation strategy causes validation failure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "invalid_strategy_task"
    title: "Invalid Strategy Task"
    description: "Task with invalid evaluation strategy"
    prompt_template_file: "test_template.j2"
    execution_config:
      timeout: 300
      allowed_executors:
        - "test_executor"
    episode_config:
      max_steps: 10
    evaluation_config:
      strategy: "invalid_strategy"
      criteria: {}
      scoring:
        max_score: 1.0
    subtasks: []
""")
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="Invalid or missing evaluation strategy"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_missing_criteria_section(self):
        """Test that missing criteria section causes validation failure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "missing_criteria_task"
    title: "Missing Criteria Task"
    description: "Task without criteria section"
    prompt_template_file: "test_template.j2"
    execution_config:
      timeout: 300
      allowed_executors:
        - "test_executor"
    episode_config:
      max_steps: 10
    evaluation_config:
      strategy: "static"
      scoring:
        max_score: 1.0
    subtasks: []
""")
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="Missing or invalid criteria section"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_invalid_max_score(self):
        """Test that invalid max_score causes validation failure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "invalid_max_score_task"
    title: "Invalid Max Score Task"
    description: "Task with invalid max_score"
    prompt_template_file: "test_template.j2"
    execution_config:
      timeout: 300
      allowed_executors:
        - "test_executor"
    episode_config:
      max_steps: 10
    evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer"]
      scoring:
        max_score: -1.0
    subtasks: []
""")
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
            f.write("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "static_no_answers_task"
    title: "Static No Answers Task"
    description: "Static task without expected_answers"
    prompt_template_file: "test_template.j2"
    execution_config:
      timeout: 300
      allowed_executors:
        - "test_executor"
    episode_config:
      max_steps: 10
    evaluation_config:
      strategy: "static"
      criteria: {}
      scoring:
        max_score: 1.0
    subtasks: []
""")
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="static strategy requires 'expected_answers'"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_llm_missing_golden_answer(self):
        """Test that LLM strategy without golden_answer fails validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "llm_no_golden_task"
    title: "LLM No Golden Task"
    description: "LLM task without golden_answer"
    prompt_template_file: "test_template.j2"
    execution_config:
      timeout: 300
      allowed_executors:
        - "test_executor"
    episode_config:
      max_steps: 10
    evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
      scoring:
        max_score: 1.0
    subtasks: []
""")
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="llm_judge strategy requires 'golden_answer'"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_llm_missing_model(self):
        """Test that LLM strategy without model fails validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
domain: "test_domain"

benchmark_config:
  episode_attempts: 1

tasks:
  - task_id: "llm_no_model_task"
    title: "LLM No Model Task"
    description: "LLM task without model"
    prompt_template_file: "test_template.j2"
    execution_config:
      timeout: 300
      allowed_executors:
        - "test_executor"
    episode_config:
      max_steps: 10
    evaluation_config:
      strategy: "llm_judge"
      criteria:
        golden_answer: "The correct answer"
      scoring:
        max_score: 1.0
    subtasks: []
""")
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="llm_judge strategy requires 'model'"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)
