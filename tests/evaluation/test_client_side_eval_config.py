"""
Unit tests for new client-side evaluation configuration format.

Tests the new dual-config format (submission_evaluation_config + step_evaluation_config),
nested subtask scoring structure, and validation methods added in the client-side
evaluation migration.
"""

import os
import tempfile
import pytest

from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException


class TestSubtaskNestedScoringFormat:
    """Test cases for nested scoring structure in subtasks."""

    def test_subtask_with_nested_scoring_structure(self):
        """Test that subtasks with scoring: { max_score: X } are correctly parsed."""
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
  episode_attempts: 1

tasks:
  - task_id: test_task_nested_scoring
    title: Test Task with Nested Scoring
    description: Task with nested scoring structure in subtasks
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0

    subtasks:
      - subtask_id: checkpoint_1
        title: First Checkpoint
        description: First step
        objective: Complete first step
        scoring:
          max_score: 0.3
      - subtask_id: checkpoint_2
        title: Second Checkpoint
        description: Second step
        objective: Complete second step
        scoring:
          max_score: 0.7
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "test_task_nested_scoring" in tasks
            task = tasks["test_task_nested_scoring"]

            # Verify subtasks loaded correctly
            assert len(task.subtasks) == 2

            # Check first subtask
            subtask1 = task.subtasks[0]
            assert subtask1.subtask_id == "checkpoint_1"
            assert subtask1.subtask_max_score == 0.3

            # Check second subtask
            subtask2 = task.subtasks[1]
            assert subtask2.subtask_id == "checkpoint_2"
            assert subtask2.subtask_max_score == 0.7

            # Verify total score
            total_score = sum(s.subtask_max_score for s in task.subtasks)
            assert total_score == 1.0

        finally:
            os.unlink(temp_path)

    def test_subtask_without_scoring_defaults_to_zero(self):
        """Test that subtasks without scoring field default to max_score of 0.0."""
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
  episode_attempts: 1

tasks:
  - task_id: test_task_no_scoring
    title: Test Task without Scoring
    description: Task with subtasks missing scoring field
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0

    subtasks:
      - subtask_id: checkpoint_1
        title: First Checkpoint
        description: First step
        objective: Complete first step
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            task = tasks["test_task_no_scoring"]
            subtask = task.subtasks[0]

            # Should default to 1.0 when scoring is missing (DEFAULT_MAX_SCORE)
            assert subtask.subtask_max_score == 1.0

        finally:
            os.unlink(temp_path)

    def test_subtask_invalid_scoring_not_dict(self):
        """Test that subtask with scoring as non-dict raises error."""
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
  episode_attempts: 1

tasks:
  - task_id: test_task_invalid_scoring
    title: Test Task with Invalid Scoring
    description: Task with invalid scoring structure
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0

    subtasks:
      - subtask_id: checkpoint_1
        title: First Checkpoint
        description: First step
        objective: Complete first step
        scoring: "invalid_string"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="scoring must be a dictionary"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_subtask_scoring_missing_max_score_field(self):
        """Test that subtask with scoring dict but no max_score defaults to 0.0."""
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
  episode_attempts: 1

tasks:
  - task_id: test_task_empty_scoring
    title: Test Task with Empty Scoring
    description: Task with empty scoring dict
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "test_answer"
      scoring:
        max_score: 1.0

    subtasks:
      - subtask_id: checkpoint_1
        title: First Checkpoint
        description: First step
        objective: Complete first step
        scoring: {}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            task = tasks["test_task_empty_scoring"]
            subtask = task.subtasks[0]

            # Should default to 1.0 when max_score is missing from scoring dict (DEFAULT_MAX_SCORE)
            assert subtask.subtask_max_score == 1.0

        finally:
            os.unlink(temp_path)


class TestDualConfigFormat:
    """Test cases for dual config format (submission + step evaluation configs)."""

    def test_task_with_both_submission_and_step_configs(self):
        """Test loading task with both submission and step evaluation configs."""
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
  episode_attempts: 1

tasks:
  - task_id: dual_config_task
    title: Task with Dual Configs
    description: Task with both submission and step evaluation
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        golden_answer: "Expected answer"
        judge_system_template: "judge/submission/system.md"
        judge_user_template: "judge/submission/user.md"
      scoring:
        max_score: 1.0
    step_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "judge/step/system.md"
        judge_user_template: "judge/step/user.md"
        steps_per_message: 10
    subtasks: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "dual_config_task" in tasks
            task = tasks["dual_config_task"]

            # Verify submission config
            assert task.submission_evaluation_config is not None
            assert task.submission_evaluation_config["strategy"] == "llm_judge"
            assert task.submission_evaluation_config["criteria"]["model"] == "gpt-4"
            assert task.submission_evaluation_config["criteria"]["judge_system_template"] == "judge/submission/system.md"
            assert task.submission_evaluation_config["criteria"]["judge_user_template"] == "judge/submission/user.md"

            # Verify step config
            assert task.step_evaluation_config is not None
            assert task.step_evaluation_config["strategy"] == "llm_judge"
            assert task.step_evaluation_config["criteria"]["model"] == "gpt-4"
            assert task.step_evaluation_config["criteria"]["judge_system_template"] == "judge/step/system.md"
            assert task.step_evaluation_config["criteria"]["judge_user_template"] == "judge/step/user.md"
            assert task.step_evaluation_config["criteria"]["steps_per_message"] == 10

        finally:
            os.unlink(temp_path)

    def test_task_with_only_submission_config(self):
        """Test that task with only submission config (no step config) loads correctly."""
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
  episode_attempts: 1

tasks:
  - task_id: submission_only_task
    title: Task with Only Submission Config
    description: Task without step evaluation
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "correct_answer"
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

            assert "submission_only_task" in tasks
            task = tasks["submission_only_task"]

            # Verify submission config exists
            assert task.submission_evaluation_config is not None
            assert task.submission_evaluation_config["strategy"] == "static"

            # Verify step config is None or empty
            assert task.step_evaluation_config is None or task.step_evaluation_config == {}

        finally:
            os.unlink(temp_path)


class TestStepEvaluationConfigValidation:
    """Test cases for step_evaluation_config validation."""

    def test_step_config_with_invalid_strategy(self):
        """Test that step config with invalid strategy fails validation."""
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
  episode_attempts: 1

tasks:
  - task_id: invalid_step_strategy
    title: Invalid Step Strategy
    description: Task with invalid step strategy
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer"]
      scoring:
        max_score: 1.0
    step_evaluation_config:
      strategy: "invalid_strategy"
      criteria:
        model: "gpt-4"
    subtasks: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="Invalid step evaluation strategy"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_step_config_llm_missing_model(self):
        """Test that step config with llm_judge but no model fails validation."""
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
  episode_attempts: 1

tasks:
  - task_id: step_no_model
    title: Step Config No Model
    description: Step config missing model
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer"]
      scoring:
        max_score: 1.0
    step_evaluation_config:
      strategy: "llm_judge"
      criteria:
        judge_system_template: "system.md"
        judge_user_template: "user.md"
    subtasks: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="step evaluation llm_judge requires 'model'"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_step_config_llm_missing_system_template(self):
        """Test that step config with llm_judge but no system template fails validation."""
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
  episode_attempts: 1

tasks:
  - task_id: step_no_system_template
    title: Step Config No System Template
    description: Step config missing system template
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer"]
      scoring:
        max_score: 1.0
    step_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_user_template: "user.md"
    subtasks: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="step evaluation requires 'judge_system_template'"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_step_config_llm_missing_user_template(self):
        """Test that step config with llm_judge but no user template fails validation."""
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
  episode_attempts: 1

tasks:
  - task_id: step_no_user_template
    title: Step Config No User Template
    description: Step config missing user template
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer"]
      scoring:
        max_score: 1.0
    step_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "system.md"
    subtasks: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="step evaluation requires 'judge_user_template'"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_step_config_invalid_steps_per_message(self):
        """Test that step config with invalid steps_per_message fails validation."""
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
  episode_attempts: 1

tasks:
  - task_id: step_invalid_steps_per_message
    title: Invalid Steps Per Message
    description: Step config with invalid steps_per_message
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["answer"]
      scoring:
        max_score: 1.0
    step_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "system.md"
        judge_user_template: "user.md"
        steps_per_message: -5
    subtasks: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="steps_per_message must be a positive integer"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)


class TestSubmissionTemplatePathValidation:
    """Test cases for template path validation in submission evaluation config."""

    def test_submission_llm_missing_system_template(self):
        """Test that submission config with llm_judge but no system template fails."""
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
  episode_attempts: 1

tasks:
  - task_id: submission_no_system_template
    title: Submission No System Template
    description: Submission config missing system template
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
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

            with pytest.raises(InvalidTaskDefinitionException, match="llm_judge requires 'judge_system_template'"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_submission_llm_missing_user_template(self):
        """Test that submission config with llm_judge but no user template fails."""
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
  episode_attempts: 1

tasks:
  - task_id: submission_no_user_template
    title: Submission No User Template
    description: Submission config missing user template
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
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

            with pytest.raises(InvalidTaskDefinitionException, match="llm_judge requires 'judge_user_template'"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_template_path_without_md_extension(self):
        """Test that template paths must end with .md extension."""
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
  episode_attempts: 1

tasks:
  - task_id: template_no_extension
    title: Template No Extension
    description: Template path without .md extension
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "judge/submission/system.txt"
        judge_user_template: "judge/submission/user.md"
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
            os.unlink(temp_path)

    def test_template_path_with_backslashes(self):
        """Test that template paths cannot use backslashes."""
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
  episode_attempts: 1

tasks:
  - task_id: template_backslashes
    title: Template Backslashes
    description: Template path with backslashes
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: 'judge\\\\submission\\\\system.md'
        judge_user_template: "judge/submission/user.md"
      scoring:
        max_score: 1.0

    subtasks: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="must use forward slashes, not backslashes"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_template_path_with_traversal(self):
        """Test that template paths cannot contain '..' path traversal."""
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
  episode_attempts: 1

tasks:
  - task_id: template_traversal
    title: Template Traversal
    description: Template path with path traversal
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "../../../etc/passwd.md"
        judge_user_template: "judge/submission/user.md"
      scoring:
        max_score: 1.0

    subtasks: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="cannot contain '..' path traversal"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)

    def test_template_path_empty_or_whitespace(self):
        """Test that template paths cannot be empty or just whitespace."""
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
  episode_attempts: 1

tasks:
  - task_id: template_empty
    title: Template Empty
    description: Template path is empty string
    prompts:
      instruction: "test_instruction.md"
      assistant: "test_assistant.md"
      submit: "test_submit.md"
    execution_config:
    episode_config:
      max_steps: 10
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "   "
        judge_user_template: "judge/submission/user.md"
      scoring:
        max_score: 1.0

    subtasks: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")

            with pytest.raises(InvalidTaskDefinitionException, match="cannot be empty or whitespace"):
                loader.load_tasks_from_file(temp_path)

        finally:
            os.unlink(temp_path)


class TestCompleteNewFormatIntegration:
    """Integration test for complete new format with all features."""

    def test_complete_new_format_task(self):
        """Test a complete task with all new format features."""
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
    max_steps: 20

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: complete_new_format_task
    title: Complete New Format Task
    description: Task demonstrating all new format features
    prompts:
      instruction: "task_instruction.md"
      assistant: "task_assistant.md"
      submit: "task_submit.md"
    execution_config:
      executors:
        bash:
          timeout: 600
        python:
          timeout: 600
    episode_config:
      max_steps: 15
    submission_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "openai/azure/gpt-4.1"
        golden_answer: "The complete solution"
        judge_system_template: "judge/submission/cybench_evaluation_system.md"
        judge_user_template: "judge/submission/cybench_evaluation_user.md"
      scoring:
        max_score: 1.0
    step_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "openai/azure/gpt-4.1"
        judge_system_template: "judge/step/system.md"
        judge_user_template: "judge/step/user.md"
        steps_per_message: 10
    subtasks:
      - subtask_id: checkpoint_1
        title: Initial Setup
        description: Set up the environment
        objective: Environment is configured correctly
        scoring:
          max_score: 0.15
        hints:
          - "Check the configuration files"
      - subtask_id: checkpoint_2
        title: Data Collection
        description: Collect necessary data
        objective: All data is collected
        scoring:
          max_score: 0.20
        hints:
          - "Look in the logs directory"
      - subtask_id: checkpoint_3
        title: Analysis
        description: Analyze the collected data
        objective: Analysis is complete
        scoring:
          max_score: 0.35
      - subtask_id: checkpoint_4
        title: Final Report
        description: Generate the final report
        objective: Report is generated
        scoring:
          max_score: 0.30
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            tasks = loader.load_tasks_from_file(temp_path)

            assert "complete_new_format_task" in tasks
            task = tasks["complete_new_format_task"]

            # Verify basic task properties
            assert task.task_id == "complete_new_format_task"
            assert task.title == "Complete New Format Task"

            # Verify submission evaluation config
            assert task.submission_evaluation_config is not None
            assert task.submission_evaluation_config["strategy"] == "llm_judge"
            assert task.submission_evaluation_config["criteria"]["model"] == "openai/azure/gpt-4.1"
            assert task.submission_evaluation_config["criteria"]["golden_answer"] == "The complete solution"
            assert task.submission_evaluation_config["criteria"]["judge_system_template"] == "judge/submission/cybench_evaluation_system.md"
            assert task.submission_evaluation_config["criteria"]["judge_user_template"] == "judge/submission/cybench_evaluation_user.md"
            assert task.submission_evaluation_config["scoring"]["max_score"] == 1.0

            # Verify step evaluation config
            assert task.step_evaluation_config is not None
            assert task.step_evaluation_config["strategy"] == "llm_judge"
            assert task.step_evaluation_config["criteria"]["model"] == "openai/azure/gpt-4.1"
            assert task.step_evaluation_config["criteria"]["judge_system_template"] == "judge/step/system.md"
            assert task.step_evaluation_config["criteria"]["judge_user_template"] == "judge/step/user.md"
            assert task.step_evaluation_config["criteria"]["steps_per_message"] == 10

            # Verify subtasks with nested scoring
            assert len(task.subtasks) == 4

            subtask1 = task.subtasks[0]
            assert subtask1.subtask_id == "checkpoint_1"
            assert subtask1.subtask_max_score == 0.15
            assert subtask1.hints == ["Check the configuration files"]

            subtask2 = task.subtasks[1]
            assert subtask2.subtask_id == "checkpoint_2"
            assert subtask2.subtask_max_score == 0.20
            assert subtask2.hints == ["Look in the logs directory"]

            subtask3 = task.subtasks[2]
            assert subtask3.subtask_id == "checkpoint_3"
            assert subtask3.subtask_max_score == 0.35

            subtask4 = task.subtasks[3]
            assert subtask4.subtask_id == "checkpoint_4"
            assert subtask4.subtask_max_score == 0.30

            # Verify total subtask scores sum correctly
            total_score = sum(s.subtask_max_score for s in task.subtasks)
            assert total_score == 1.0

            # Verify execution and episode configs
            assert "executors" in task.execution_config
            assert "bash" in task.execution_config["executors"]
            assert task.execution_config["executors"]["bash"]["timeout"] == 600

            assert task.episode_config["max_steps"] == 15

        finally:
            os.unlink(temp_path)
