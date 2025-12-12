"""
Unit tests for BenchmarkConfigLoader subtask and step evaluation handling.

Tests subtask parsing, step evaluation validation, and transcript configuration.
"""

import tempfile
from pathlib import Path

import pytest

from saber.server.benchmarks.benchmark_config_loader import BenchmarkConfigLoader
from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException


class TestBenchmarkConfigLoaderSubtasks:
    """Test subtask configuration handling."""

    def test_subtask_missing_required_field(self):
        """Test error when subtask is missing required fields."""
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
    subtasks:
      - title: Missing subtask_id
        description: This subtask is missing subtask_id
        objective: Complete the objective
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="Missing required subtask field"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_subtask_scoring_not_dict(self):
        """Test error when subtask scoring is not a dictionary."""
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
    subtasks:
      - subtask_id: subtask_1
        title: Subtask 1
        description: A subtask
        objective: Complete the objective
        scoring: "should_be_dict"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="scoring must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_step_evaluation_invalid_strategy(self):
        """Test error when step evaluation strategy is invalid."""
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
    step_evaluation_config:
      strategy: "invalid_strategy"
      criteria:
        expected_outputs: ["test"]
      scoring:
        max_score: 1.0
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
            Path(temp_path).unlink()

    def test_step_evaluation_scoring_not_dict(self):
        """Test error when step evaluation scoring is not a dict."""
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
    step_evaluation_config:
      strategy: "static"
      criteria:
        expected_outputs: ["test"]
      scoring: "should_be_dict"
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="scoring must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_step_evaluation_invalid_max_score(self):
        """Test error when step evaluation max_score is invalid."""
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
    step_evaluation_config:
      strategy: "static"
      criteria:
        expected_outputs: ["test"]
      scoring:
        max_score: -1.0
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="max_score must be a non-negative number"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_step_evaluation_invalid_weight(self):
        """Test error when step evaluation weight is invalid."""
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
    step_evaluation_config:
      strategy: "static"
      criteria:
        expected_outputs: ["test"]
      scoring:
        max_score: 1.0
        weight: 1.5
    subtasks: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            loader = BenchmarkConfigLoader("test_domain")
            with pytest.raises(InvalidTaskDefinitionException, match="weight must not exceed 1.0"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_step_evaluation_static_missing_expected_outputs(self):
        """Test error when static step evaluation is missing expected_outputs."""
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
    step_evaluation_config:
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
            with pytest.raises(InvalidTaskDefinitionException, match="must have.*expected_outputs"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_step_evaluation_tool_call_missing_expected_tools(self):
        """Test error when tool_call step evaluation is missing expected_tools."""
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
    step_evaluation_config:
      strategy: "tool_call"
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
            with pytest.raises(InvalidTaskDefinitionException, match="must have.*expected_tools"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_step_evaluation_llm_judge_missing_model(self):
        """Test error when llm_judge step evaluation is missing model."""
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
    step_evaluation_config:
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
            with pytest.raises(InvalidTaskDefinitionException, match="llm_judge requires.*model"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_step_evaluation_llm_judge_invalid_steps_per_message(self):
        """Test error when steps_per_message is invalid."""
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
    step_evaluation_config:
      strategy: "llm_judge"
      criteria:
        model: "gpt-4"
        judge_system_template: "system.md"
        judge_user_template: "user.md"
        steps_per_message: 0
      scoring:
        max_score: 1.0
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
            Path(temp_path).unlink()


class TestBenchmarkConfigLoaderTranscriptConfig:
    """Test transcript configuration handling."""

    def test_transcript_config_not_dict(self):
        """Test error when transcript_config is not a dictionary."""
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
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
      transcript_config: "should_be_dict"
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
            with pytest.raises(InvalidTaskDefinitionException, match="transcript_config must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_transcript_websocket_not_dict(self):
        """Test error when websocket config is not a dictionary."""
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
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
      transcript_config:
        websocket: "should_be_dict"
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
            with pytest.raises(InvalidTaskDefinitionException, match="websocket must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_transcript_websocket_invalid_numeric_field(self):
        """Test error when websocket numeric field is invalid."""
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
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
      transcript_config:
        websocket:
          connection_timeout: -5
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
            with pytest.raises(InvalidTaskDefinitionException, match="connection_timeout.*must be positive number"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_transcript_websocket_invalid_boolean_field(self):
        """Test error when websocket boolean field is invalid."""
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
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
      transcript_config:
        websocket:
          reconnect_enabled: "not_boolean"
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
            with pytest.raises(InvalidTaskDefinitionException, match="reconnect_enabled.*must be boolean"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_transcript_websocket_push_not_dict(self):
        """Test error when websocket push config is not a dictionary."""
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
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
      transcript_config:
        websocket:
          push: "should_be_dict"
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
            with pytest.raises(InvalidTaskDefinitionException, match="push must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_transcript_websocket_pull_not_dict(self):
        """Test error when websocket pull config is not a dictionary."""
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
    execution_config:
      executors:
        bash:
          timeout: 30
    episode_config:
      max_steps: 10
      transcript_config:
        websocket:
          pull: "should_be_dict"
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
            with pytest.raises(InvalidTaskDefinitionException, match="pull must be a dictionary"):
                loader.load_tasks_from_file(temp_path)
        finally:
            Path(temp_path).unlink()
