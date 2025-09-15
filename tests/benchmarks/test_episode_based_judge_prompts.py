"""
Tests for episode-based judge prompt functionality.

This test module ?? enhanced judge prompt generation that uses complete
episode objects instead of just submission strings, enabling rich template
rendering with step-by-step execution data.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from typing import Dict, Any

from saber.server.base import Episode, EpisodeState, Step, Action
from saber.server.benchmarks.benchmark_manager import BenchmarkManager
from saber.server.benchmarks.prompt_generator import PromptGenerator, JudgePromptContext, JudgePromptPayload
from saber.server.benchmarks.task import Task
from saber.server.evaluation.exceptions import EvaluationConfigError


@pytest.fixture
def mock_episode_no_submission():
    """Create a complete episode without submission."""
    episode = Mock(spec=Episode)
    episode.episode_id = "no-submission-789"
    episode.state = EpisodeState.COMPLETED
    episode.is_complete = True
    episode.submission = None
    episode.start_time = datetime.now()
    episode.end_time = datetime.now()
    episode.steps = []
    return episode


@pytest.fixture
def mock_task_llm_judge():
    """Create a mock Task with LLM judge configuration."""
    task = Mock(spec=Task)
    task.id = "security-task-1"
    task.task_id = "security-task-1"  # New format
    task.title = "Find Hidden Files"
    task.description = "Locate and extract hidden credentials"
    task.domain = "cybersecurity"
    task.task_name = "Locate and extract hidden credentials"  # Legacy
    task.expected_output = "FLAG{found_the_secret}"
    task.evaluation_config = {
        "strategy": "llm_judge",
        "criteria": {
            "judge_system_template": "security_system.md",
            "judge_user_template": "security_user.md",
            "model": "gpt-4",
            "golden_answer": "FLAG{found_the_secret}"
        }
    }
    # Legacy attributes for backward compatibility
    task.llm_judge = Mock()
    task.llm_judge.model = "gpt-4"
    task.llm_judge.judge_template = "cybersecurity_incident_user.md"
    task.llm_judge.system_prompt = "You are an expert cybersecurity evaluator..."
    task.llm_judge.custom_criteria = []
    return task


class TestEpisodeBasedJudgePrompts:
    """Test episode-based judge prompt functionality."""

    @pytest.fixture
    def mock_episode_complete(self):
        """Create a complete episode with multiple steps."""
        # Create mock steps
        step1 = Mock(spec=Step)
        step1.step_number = 1
        step1.timestamp = datetime.utcnow()
        step1.action = Mock(spec=Action)
        step1.action.tool_name = "cli"
        step1.action.parameters = {"arguments": "ls -la /home/user"}
        step1.response = {"output": "drwxr-xr-x user user Documents\n-rw-r--r-- user user secret.txt", "exit_code": 0}
        step1.done = False

        step2 = Mock(spec=Action)
        step2.step_number = 2
        step2.timestamp = datetime.utcnow()
        step2.action = Mock(spec=Action)
        step2.action.tool_name = "cli"
        step2.action.parameters = {"arguments": "cat /home/user/secret.txt"}
        step2.response = {"output": "FLAG{found_the_secret}", "exit_code": 0}
        step2.done = True

        episode = Mock(spec=Episode)
        episode.episode_id = "test-episode-123"
        episode.task_id = "security-task-1"
        episode.session_id = "session-abc"
        episode.state = EpisodeState.COMPLETED
        episode.is_complete = True
        episode.submission = "I found the flag: FLAG{found_the_secret}"
        episode.steps = [step1, step2]
        episode.start_time = datetime.utcnow()
        episode.end_time = datetime.utcnow()
        episode.duration = 45.2
        episode.metadata = {"agent": "test_agent"}
        episode.context = {"environment": "test"}
        episode.max_steps = 10
        episode.completion_reason = "agent_completed"
        episode.get_executed_commands.return_value = ["ls -la /home/user", "cat /home/user/secret.txt"]

        return episode

    @pytest.fixture
    def mock_episode_incomplete(self):
        """Create an incomplete episode."""
        episode = Mock(spec=Episode)
        episode.episode_id = "incomplete-episode-456"
        episode.state = EpisodeState.ACTIVE
        episode.is_complete = False
        episode.submission = None
        return episode

    @pytest.fixture
    def mock_episode_no_submission(self):
        """Create a complete episode without submission."""
        episode = Mock(spec=Episode)
        episode.episode_id = "no-submission-789"
        episode.state = EpisodeState.COMPLETED
        episode.is_complete = True
        episode.submission = ""  # Empty submission
        return episode

    @pytest.fixture
    def mock_task_non_llm(self):
        """Create a task not configured for LLM judge evaluation."""
        task = Mock(spec=Task)
        task.task_id = "static-task"
        task.evaluation_config = {
            "strategy": "static",
            "criteria": {"expected_answers": ["correct_answer"]}
        }
        return task

    @pytest.fixture
    def prompt_generator(self, tmp_path):
        """Create PromptGenerator with temporary prompts directory."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        # Create judge templates directory
        judge_dir = prompts_dir / "judge"
        judge_dir.mkdir()

        # Create mock judge templates
        (judge_dir / "security_system.md").write_text(
            "You are a cybersecurity expert evaluating agent performance."
        )
        (judge_dir / "security_user.md").write_text(
            "**Task:** {{ question }}\n"
            "**Expected:** {{ golden_answer }}\n"
            "**Submission:** {{ submission }}\n"
            "**Episode:** {{ episode.episode_id }}\n"
            "**Duration:** {{ episode.duration }} seconds\n"
            "**Steps:** {{ episode.steps | length }}\n"
            "{% for step in episode.steps %}"
            "Step {{ step.step_number }}: {{ step.action.tool_name }}\n"
            "{% endfor %}"
        )

        return PromptGenerator(str(prompts_dir))

    def test_judge_prompt_context_creation_with_episode(self, mock_episode_complete, mock_task_llm_judge):
        """Test JudgePromptContext creation with episode data."""
        context = JudgePromptContext(
            question="Find the hidden flag",
            golden_answer="FLAG{found_the_secret}",
            episode=mock_episode_complete,
            task=mock_task_llm_judge,
            evaluation_config=mock_task_llm_judge.evaluation_config["criteria"],
            model="gpt-4",
            domain="cybersecurity",
            task_id="security-task-1"
        )

        context_dict = context.to_dict()

        # Test basic fields
        assert context_dict["question"] == "Find the hidden flag"
        assert context_dict["golden_answer"] == "FLAG{found_the_secret}"
        assert context_dict["submission"] == "I found the flag: FLAG{found_the_secret}"
        assert context_dict["episode_id"] == "test-episode-123"

        # Test episode data
        episode_data = context_dict["episode"]
        assert episode_data["episode_id"] == "test-episode-123"
        assert episode_data["state"] == "completed"
        assert episode_data["duration"] == 45.2
        assert len(episode_data["steps"]) == 2

        # Test step data structure
        step1_data = episode_data["steps"][0]
        assert step1_data["step_number"] == 1
        assert step1_data["action"]["tool_name"] == "cli"
        assert step1_data["action"]["parameters"]["arguments"] == "ls -la /home/user"
        assert step1_data["response"]["exit_code"] == 0

    def test_judge_prompt_context_helper_methods(self, mock_episode_complete, mock_task_llm_judge):
        """Test helper methods in JudgePromptContext."""
        context = JudgePromptContext(
            question="Find the hidden flag",
            golden_answer="FLAG{found_the_secret}",
            episode=mock_episode_complete,
            task=mock_task_llm_judge,
            evaluation_config=mock_task_llm_judge.evaluation_config["criteria"],
            model="gpt-4",
            domain="cybersecurity",
            task_id="security-task-1"
        )

        context_dict = context.to_dict()
        episode_helpers = context_dict["episode"]

        # Test helper methods
        assert episode_helpers["get_step_count"]() == 2
        assert len(episode_helpers["get_last_n_steps"](1)) == 1
        assert len(episode_helpers["get_first_n_steps"](1)) == 1

        # Test commands summary
        commands_summary = episode_helpers["get_commands_summary"](100)
        assert "Step 1: ls -la /home/user" in commands_summary
        assert "Step 2: cat /home/user/secret.txt" in commands_summary

    def test_render_judge_prompt_success(self, prompt_generator, mock_episode_complete, mock_task_llm_judge):
        """Test successful judge prompt rendering with episode data."""
        result = prompt_generator.render_judge_prompt_for_episode(mock_task_llm_judge, mock_episode_complete)

        assert isinstance(result, JudgePromptPayload)
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "system"
        assert result.messages[1]["role"] == "user"
        assert result.model == "gpt-4"
        assert result.task_id == "security-task-1"
        assert result.episode_id == "test-episode-123"

        # Check that episode data is in the user prompt
        user_content = result.messages[1]["content"]
        assert "test-episode-123" in user_content
        assert "45.2 seconds" in user_content
        assert "**Steps:** 2" in user_content
        assert "Step 1: cli" in user_content
        assert "Step 2: cli" in user_content

    def test_render_judge_prompt_incomplete_episode_fails(self, prompt_generator, mock_episode_incomplete, mock_task_llm_judge):
        """Test that incomplete episodes fail fast."""
        with pytest.raises(EvaluationConfigError) as exc_info:
            prompt_generator.render_judge_prompt_for_episode(mock_task_llm_judge, mock_episode_incomplete)

        assert "incomplete episode" in str(exc_info.value).lower()
        assert "incomplete-episode-456" in str(exc_info.value)
        assert "active" in str(exc_info.value).lower()

    def test_render_judge_prompt_no_submission_fails(self, prompt_generator, mock_episode_no_submission, mock_task_llm_judge):
        """Test that episodes without submission fail fast."""
        with pytest.raises(EvaluationConfigError) as exc_info:
            prompt_generator.render_judge_prompt_for_episode(mock_task_llm_judge, mock_episode_no_submission)

        assert "without submission data" in str(exc_info.value).lower()
        assert "no-submission-789" in str(exc_info.value)

    def test_render_judge_prompt_non_llm_task_fails(self, prompt_generator, mock_episode_complete, mock_task_non_llm):
        """Test that non-LLM judge tasks fail fast."""
        with pytest.raises(EvaluationConfigError) as exc_info:
            prompt_generator.render_judge_prompt_for_episode(mock_task_non_llm, mock_episode_complete)

        assert "not configured for LLM judge evaluation" in str(exc_info.value)
        assert "static" in str(exc_info.value)

    def test_benchmark_manager_render_judge_prompt_for_episode(self):
        """Test BenchmarkManager's new render_judge_prompt_for_episode method."""
        # Create mock objects
        mock_task = Mock(spec=Task)
        mock_task.task_id = "security-task-1"
        mock_task.description = "Locate and extract hidden credentials"
        mock_task.domain = "cybersecurity"
        mock_task.evaluation_config = {
            "strategy": "llm_judge",
            "criteria": {
                "judge_system_template": "security_system.md",
                "judge_user_template": "security_user.md",
                "model": "gpt-4",
                "golden_answer": "FLAG{found_the_secret}"
            }
        }
        mock_episode = Mock(spec=Episode)
        mock_episode.submission = "Task completed successfully"
        mock_episode.is_complete = True
        mock_prompt_generator = Mock()
        mock_expected_result = Mock()

        mock_prompt_generator.render_judge_prompt_for_episode.return_value = mock_expected_result

        # Create BenchmarkManager with mocked components
        with patch('saber.server.benchmarks.benchmark_manager.BenchmarkConfigLoader'), \
             patch('saber.server.benchmarks.benchmark_manager.PromptGenerator', return_value=mock_prompt_generator), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch.object(BenchmarkManager, 'load_tasks_from_directory'), \
             patch.object(BenchmarkManager, '__init__', return_value=None):

            manager = BenchmarkManager("test_domain", "/tmp/config")
            manager.prompt_generator = mock_prompt_generator
            manager.tasks = {"task1": mock_task}

            result = manager.render_judge_prompt_for_episode("task1", mock_episode)

            assert result == mock_expected_result
            mock_prompt_generator.render_judge_prompt_for_episode.assert_called_once_with(mock_task, mock_episode)

    def test_benchmark_manager_render_judge_prompt_task_not_found(self):
        """Test BenchmarkManager fails fast on missing task."""
        mock_episode = Mock(spec=Episode)

        with patch('saber.server.benchmarks.benchmark_manager.BenchmarkConfigLoader'), \
             patch('saber.server.benchmarks.benchmark_manager.PromptGenerator'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch.object(BenchmarkManager, 'load_tasks_from_directory'):

            manager = BenchmarkManager("test_domain", "/tmp/config")
            manager.tasks = {}  # No tasks

            with pytest.raises(Exception):  # Should raise TaskNotFoundException
                manager.render_judge_prompt_for_episode("nonexistent_task", mock_episode)

    def test_injected_judge_prompt_renderer_signature(self, mock_task_llm_judge, tmp_path):
        """Test that injected judge prompt renderers use episode-based signature."""
        # Create temporary template files
        prompts_dir = tmp_path / "prompts"
        judge_dir = prompts_dir / "judge"
        judge_dir.mkdir(parents=True)
        (judge_dir / "security_system.md").write_text("System prompt")
        (judge_dir / "security_user.md").write_text("User prompt")

        with patch('saber.server.benchmarks.benchmark_manager.BenchmarkConfigLoader'), \
             patch.object(BenchmarkManager, 'load_tasks_from_directory'):

            manager = BenchmarkManager("test_domain", str(tmp_path))
            manager.tasks = {"task1": mock_task_llm_judge}
            manager._inject_judge_prompt_renderers()

            # Get the injected renderer
            renderer = mock_task_llm_judge.evaluation_config["judge_prompt_renderer"]

            # Test that it accepts episode object
            mock_episode = Mock(spec=Episode)
            mock_episode.submission = "Task completed successfully"
            mock_episode.is_complete = True
            mock_episode.episode_id = "test-episode"
            mock_episode.state = EpisodeState.COMPLETED
            mock_episode.start_time = datetime.utcnow()
            mock_episode.end_time = datetime.utcnow()
            mock_episode.duration = 45.2
            mock_episode.completion_reason = "agent_completed"
            mock_episode.metadata = {"agent": "test_agent"}
            mock_episode.context = {"environment": "test"}
            mock_episode.max_steps = 10
            mock_episode.steps = []
            result = renderer(mock_episode)

            # Verify the renderer works and returns a result
            assert result is not None


class TestJudgePromptTemplateFeatures:
    """Test advanced template features for episode-based judge prompts."""

    @pytest.fixture
    def complex_episode(self):
        """Create an episode with various step types and failures."""
        steps = []

        # Successful step
        step1 = Mock(spec=Step)
        step1.step_number = 1
        step1.timestamp = datetime.utcnow()
        step1.action = Mock(spec=Action)
        step1.action.tool_name = "cli"
        step1.action.parameters = {"arguments": "ls -la"}
        step1.response = {"output": "file1.txt\nfile2.txt", "exit_code": 0}
        step1.done = False
        steps.append(step1)

        # Failed step
        step2 = Mock(spec=Step)
        step2.step_number = 2
        step2.timestamp = datetime.utcnow()
        step2.action = Mock(spec=Action)
        step2.action.tool_name = "cli"
        step2.action.parameters = {"arguments": "cat nonexistent.txt"}
        step2.response = {"output": "cat: nonexistent.txt: No such file", "exit_code": 1}
        step2.done = False
        steps.append(step2)

        # Long output step
        step3 = Mock(spec=Step)
        step3.step_number = 3
        step3.timestamp = datetime.utcnow()
        step3.action = Mock(spec=Action)
        step3.action.tool_name = "python"
        step3.action.parameters = {"arguments": "print('very long output ' * 100)"}
        step3.response = {"output": "very long output " * 100, "exit_code": 0}
        step3.done = True
        steps.append(step3)

        episode = Mock(spec=Episode)
        episode.episode_id = "complex-episode"
        episode.state = EpisodeState.COMPLETED
        episode.is_complete = True
        episode.submission = "Task completed successfully"
        episode.steps = steps
        episode.duration = 120.5
        episode.start_time = datetime.utcnow()
        episode.end_time = datetime.utcnow()
        episode.completion_reason = "agent_completed"
        episode.metadata = {"agent": "test_agent"}
        episode.context = {"environment": "test"}
        episode.max_steps = 10
        episode.get_executed_commands.return_value = ["ls -la", "cat nonexistent.txt", "print('very long output ' * 100)"]

        return episode

    def test_template_step_limiting(self, tmp_path, complex_episode, mock_task_llm_judge):
        """Test template features for limiting step display."""
        prompts_dir = tmp_path / "prompts"
        judge_dir = prompts_dir / "judge"
        judge_dir.mkdir(parents=True)

        # Template that uses step limiting
        (judge_dir / "step_limit_user.md").write_text(
            "Total steps: {{ episode.steps | length }}\n"
            "Last 2 steps:\n"
            "{% for step in episode.get_last_n_steps(2) %}"
            "- Step {{ step.step_number }}: {{ step.action.tool_name }}\n"
            "{% endfor %}"
            "Failed steps:\n"
            "{% for step in episode.get_failed_steps() %}"
            "- Step {{ step.step_number }} FAILED\n"
            "{% endfor %}"
        )

        (judge_dir / "step_limit_system.md").write_text("System prompt")

        # Update task to use the new templates
        mock_task_llm_judge.evaluation_config["criteria"]["judge_user_template"] = "step_limit_user.md"
        mock_task_llm_judge.evaluation_config["criteria"]["judge_system_template"] = "step_limit_system.md"

        prompt_generator = PromptGenerator(str(prompts_dir))
        result = prompt_generator.render_judge_prompt_for_episode(mock_task_llm_judge, complex_episode)

        user_content = result.messages[1]["content"]

        # Should show total steps
        assert "Total steps: 3" in user_content

        # Should show last 2 steps (steps 2 and 3)
        assert "Step 2: cli" in user_content
        assert "Step 3: python" in user_content
        assert "Step 1: cli" not in user_content  # Should be excluded

        # Should show failed step
        assert "Step 2 FAILED" in user_content

    def test_template_string_truncation(self, tmp_path, complex_episode, mock_task_llm_judge):
        """Test template string truncation features."""
        prompts_dir = tmp_path / "prompts"
        judge_dir = prompts_dir / "judge"
        judge_dir.mkdir(parents=True)

        # Template with string truncation
        (judge_dir / "truncate_user.md").write_text(
            "Commands summary (limited to 50 chars):\n"
            "{{ episode.get_commands_summary(50) }}\n"
            "Long output truncated:\n"
            "{% for step in episode.steps %}"
            "Step {{ step.step_number }}: {{ step.response.output[:20] }}{% if step.response.output|length > 20 %}...{% endif %}\n"
            "{% endfor %}"
        )

        (judge_dir / "truncate_system.md").write_text("System prompt")

        mock_task_llm_judge.evaluation_config["criteria"]["judge_user_template"] = "truncate_user.md"
        mock_task_llm_judge.evaluation_config["criteria"]["judge_system_template"] = "truncate_system.md"

        prompt_generator = PromptGenerator(str(prompts_dir))
        result = prompt_generator.render_judge_prompt_for_episode(mock_task_llm_judge, complex_episode)

        user_content = result.messages[1]["content"]

        # Commands summary should be truncated with ellipsis
        assert "..." in user_content

        # Long output should be truncated
        lines = user_content.split('\n')
        step3_line = [line for line in lines if "Step 3:" in line][0]
        assert len(step3_line) < 100  # Should be much shorter than the original very long output
        assert "very long output ver..." in step3_line  # Should be truncated with ellipsis
