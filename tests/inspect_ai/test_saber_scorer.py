"""
Unit tests for SABER scorer module.

Tests the client-side evaluation logic including:
- Utility functions
- Template rendering
- Metrics calculation
- Submission scoring (static and LLM)
- Subtask scoring (static, tool_call, and LLM)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import SampleScore, Score
from inspect_ai.solver import TaskState
from jinja2 import TemplateError

from saber.inspect_ai.core.saber_scorer import (
    EpisodeContextForTemplate,
    StepContextForTemplate,
    TemplateStringLoader,
    _parse_llm_step_evaluations,
    _score_all_subtasks,
    _score_submission,
    _score_subtasks_llm_batch,
    clean_dict,
    per_task_submission_scores,
    per_task_subtask_scores,
    saber_score,
    submission_score,
    subtask_score,
    subtask_score_metrics,
)
from saber.inspect_ai.core.scoring.standard_submission import (
    score_submission_llm,
    score_submission_static,
)
from saber.models.constants import MetadataKeys, StepEvaluationStrategy, SubmissionEvaluationStrategy
from saber.models.rest.evaluation import (
    EpisodeStepData,
    EpisodeStepsResponse,
    EpisodeSubmissionResponse,
    SubmissionEvaluationCriteriaResponse,
    SubtaskEvaluationCriteriaResponse,
    TaskEvaluationContext,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def task_context():
    """Create a sample TaskEvaluationContext for tests."""
    return TaskEvaluationContext(
        task_id="task1",
        title="Test Task",
        description="Test task description",
        domain="test",
    )


@pytest.fixture
def submission_criteria_static(task_context):
    """Create SubmissionEvaluationCriteriaResponse with static strategy."""
    return SubmissionEvaluationCriteriaResponse(
        session_id="session1",
        episode_id="ep1",
        task_id="task1",
        strategy=SubmissionEvaluationStrategy.STATIC,
        criteria={"expected_answers": ["42"]},
        scoring={"max_score": 1.0},
        task_context=task_context,
    )


@pytest.fixture
def submission_criteria_llm(task_context):
    """Create SubmissionEvaluationCriteriaResponse with LLM strategy."""
    return SubmissionEvaluationCriteriaResponse(
        session_id="session1",
        episode_id="ep1",
        task_id="task1",
        strategy=SubmissionEvaluationStrategy.LLM_JUDGE,
        criteria={
            "judge_system_template": "You are a judge",
            "judge_user_template": "Is {{ submission }} correct?",
            "model": "openai/gpt-4",
            "golden_answer": "Paris",
        },
        scoring={"max_score": 1.0},
        task_context=task_context,
    )


@pytest.fixture
def subtask_criteria_static(task_context):
    """Create SubtaskEvaluationCriteriaResponse with static strategy."""
    return SubtaskEvaluationCriteriaResponse(
        session_id="session1",
        episode_id="ep1",
        task_id="task1",
        subtask_id="subtask1",
        title="Test Subtask",
        description="Test subtask description",
        objective="Test objective",
        strategy=StepEvaluationStrategy.STATIC,
        criteria={"expected_outputs": ["config.yaml"]},
        max_score=1.0,
        weight=1.0,
        task_context=task_context,
    )


@pytest.fixture
def subtask_criteria_tool_call(task_context):
    """Create SubtaskEvaluationCriteriaResponse with tool_call strategy."""
    return SubtaskEvaluationCriteriaResponse(
        session_id="session1",
        episode_id="ep1",
        task_id="task1",
        subtask_id="subtask1",
        title="Test Subtask",
        description="Test subtask description",
        objective="Test objective",
        strategy=StepEvaluationStrategy.TOOL_CALL,
        criteria={"expected_tools": ["python"]},
        max_score=1.0,
        weight=1.0,
        task_context=task_context,
    )


@pytest.fixture
def subtask_criteria_llm(task_context):
    """Create SubtaskEvaluationCriteriaResponse with LLM strategy."""
    return SubtaskEvaluationCriteriaResponse(
        session_id="session1",
        episode_id="ep1",
        task_id="task1",
        subtask_id="subtask1",
        title="Test Subtask",
        description="Test subtask description",
        objective="Test objective",
        strategy=StepEvaluationStrategy.LLM_JUDGE,
        criteria={
            "judge_system_template": "Evaluate steps",
            "judge_user_template": "Check if objective completed: {{ subtask.objective }}",
            "model": "openai/gpt-4",
        },
        max_score=1.0,
        weight=1.0,
        task_context=task_context,
    )


# ============================================================================
# Test Utility Functions
# ============================================================================


class TestCleanDict:
    """Test cases for clean_dict utility function."""

    def test_clean_dict_removes_none_values(self):
        """Test that None values are removed from dict."""
        payload = {"key1": "value1", "key2": None, "key3": "value3"}
        result = clean_dict(payload)
        assert result == {"key1": "value1", "key3": "value3"}

    def test_clean_dict_removes_false_values(self):
        """Test that False values are removed from dict."""
        payload = {"key1": True, "key2": False, "key3": "value"}
        result = clean_dict(payload)
        assert result == {"key1": True, "key3": "value"}

    def test_clean_dict_nested_dicts(self):
        """Test that nested dicts are cleaned recursively."""
        payload = {"outer": {"inner1": "value", "inner2": None}, "key": False}
        result = clean_dict(payload)
        assert result == {"outer": {"inner1": "value"}}

    def test_clean_dict_with_lists(self):
        """Test that lists are cleaned recursively."""
        payload = {"list": [1, None, 3, False, 5]}
        result = clean_dict(payload)
        assert result == {"list": [1, 3, 5]}

    def test_clean_dict_non_dict_returns_unchanged(self):
        """Test that non-dict values are returned unchanged."""
        assert clean_dict("string") == "string"
        assert clean_dict(42) == 42
        assert clean_dict(True) is True


# ============================================================================
# Test Template Context Helpers
# ============================================================================


class TestEpisodeContextForTemplate:
    """Test cases for EpisodeContextForTemplate."""

    def test_episode_context_get_step_count(self):
        """Test get_step_count returns correct count."""
        steps = [{"step": 1}, {"step": 2}, {"step": 3}]
        episode = EpisodeContextForTemplate(steps)
        assert episode.get_step_count() == 3

    def test_episode_context_empty_steps(self):
        """Test episode context with empty steps."""
        episode = EpisodeContextForTemplate([])
        assert episode.get_step_count() == 0


class TestStepContextForTemplate:
    """Test cases for StepContextForTemplate."""

    def test_step_context_basic_fields(self):
        """Test step context creates correct fields."""
        step_data = {
            "step_number": 1,
            "tool_name": "bash",
            "tool_input": {"command": "ls"},
            "tool_output": "file1.txt file2.txt",
            "done": False,
        }
        step = StepContextForTemplate(step_data)
        assert step.step_number == 1
        assert step.done is False
        assert step.action.tool_name == "bash"
        assert step.action.parameters == {"command": "ls"}
        assert step.response == "file1.txt file2.txt"

    def test_step_context_with_optional_fields(self):
        """Test step context with assistant_message and reasoning."""
        step_data = {
            "step_number": 2,
            "tool_name": "python",
            "tool_input": {"code": "print('hello')"},
            "tool_output": "hello",
            "assistant_message": "Running code",
            "reasoning": "Need to test output",
        }
        step = StepContextForTemplate(step_data)
        assert step.action.assistant_message == "Running code"
        assert step.action.reasoning == "Need to test output"

    def test_step_context_minimal_data(self):
        """Test step context with minimal required data."""
        step_data = {"step_number": 1, "tool_name": "test", "tool_input": {}, "tool_output": ""}
        step = StepContextForTemplate(step_data)
        assert step.step_number == 1
        assert step.action.tool_name == "test"
        assert step.response == ""


# ============================================================================
# Test Template Loader
# ============================================================================


class TestTemplateStringLoader:
    """Test cases for TemplateStringLoader."""

    def test_template_loader_get_existing_template(self):
        """Test getting an existing template."""
        templates = {"test_template": "Hello {{ name }}"}
        loader = TemplateStringLoader(templates)
        source, filename, uptodate = loader.get_source(None, "test_template")
        assert source == "Hello {{ name }}"
        assert filename is None
        assert uptodate() is True

    def test_template_loader_missing_template_raises_error(self):
        """Test that missing template raises TemplateError."""
        loader = TemplateStringLoader({})
        with pytest.raises(TemplateError, match="Template not found: missing"):
            loader.get_source(None, "missing")

    def test_template_loader_multiple_templates(self):
        """Test loader with multiple templates."""
        templates = {"template1": "Content 1", "template2": "Content 2"}
        loader = TemplateStringLoader(templates)
        source1, _, _ = loader.get_source(None, "template1")
        source2, _, _ = loader.get_source(None, "template2")
        assert source1 == "Content 1"
        assert source2 == "Content 2"


# ============================================================================
# Test Metrics
# ============================================================================


class TestSaberScoreMetric:
    """Test cases for saber_score metric."""

    def test_saber_score_single_sample(self):
        """Test saber_score with a single sample."""
        metric_fn = saber_score()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=0.75, answer="test"),
                sample_metadata={},
            )
        ]
        result = metric_fn(sample_scores)
        assert result == 0.75

    def test_saber_score_multiple_samples(self):
        """Test saber_score averages across multiple samples."""
        metric_fn = saber_score()
        sample_scores = [
            SampleScore(sample_id="1", score=Score(value=0.5, answer=""), sample_metadata={}),
            SampleScore(sample_id="2", score=Score(value=1.0, answer=""), sample_metadata={}),
            SampleScore(sample_id="3", score=Score(value=0.75, answer=""), sample_metadata={}),
        ]
        result = metric_fn(sample_scores)
        assert result == pytest.approx(0.75)

    def test_saber_score_empty_list(self):
        """Test saber_score returns 0.0 for empty list."""
        metric_fn = saber_score()
        result = metric_fn([])
        assert result == 0.0


class TestSubmissionScoreMetric:
    """Test cases for submission_score metric."""

    def test_submission_score_from_metadata(self):
        """Test submission_score extracts from metadata."""
        metric_fn = submission_score()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBMISSION_SCORE: 0.8}),
                sample_metadata={},
            )
        ]
        result = metric_fn(sample_scores)
        assert result == {"submission_score": 0.8}

    def test_submission_score_multiple_samples(self):
        """Test submission_score averages across samples."""
        metric_fn = submission_score()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBMISSION_SCORE: 1.0}),
                sample_metadata={},
            ),
            SampleScore(
                sample_id="2",
                score=Score(value=0.5, answer="", metadata={MetadataKeys.SUBMISSION_SCORE: 0.5}),
                sample_metadata={},
            ),
        ]
        result = metric_fn(sample_scores)
        assert result == {"submission_score": 0.75}

    def test_submission_score_missing_metadata(self):
        """Test submission_score with missing metadata returns empty dict."""
        metric_fn = submission_score()
        sample_scores = [
            SampleScore(sample_id="1", score=Score(value=1.0, answer="", metadata={}), sample_metadata={})
        ]
        result = metric_fn(sample_scores)
        assert result == {}

    def test_submission_score_empty_list(self):
        """Test submission_score returns empty dict for empty list."""
        metric_fn = submission_score()
        result = metric_fn([])
        assert result == {}


class TestSubtaskScoreMetric:
    """Test cases for subtask_score metric."""

    def test_subtask_score_from_metadata(self):
        """Test subtask_score extracts from metadata."""
        metric_fn = subtask_score()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.5, answer="", metadata={MetadataKeys.SUBTASK_SCORE: 0.5}),
                sample_metadata={},
            )
        ]
        result = metric_fn(sample_scores)
        assert result == {"subtask_score": 0.5}

    def test_subtask_score_averages_multiple(self):
        """Test subtask_score averages across samples."""
        metric_fn = subtask_score()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBTASK_SCORE: 0.6}),
                sample_metadata={},
            ),
            SampleScore(
                sample_id="2",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBTASK_SCORE: 0.4}),
                sample_metadata={},
            ),
        ]
        result = metric_fn(sample_scores)
        assert result == {"subtask_score": 0.5}

    def test_subtask_score_missing_returns_empty_dict(self):
        """Test subtask_score returns empty dict when not present."""
        metric_fn = subtask_score()
        sample_scores = [
            SampleScore(sample_id="1", score=Score(value=1.0, answer="", metadata={}), sample_metadata={})
        ]
        result = metric_fn(sample_scores)
        assert result == {}

    def test_subtask_score_empty_list_returns_empty_dict(self):
        """Test subtask_score returns empty dict for empty list."""
        metric_fn = subtask_score()
        result = metric_fn([])
        assert result == {}


class TestPerTaskSubmissionScores:
    """Test cases for per_task_submission_scores metric."""

    def test_per_task_submission_scores_single_task(self):
        """Test per-task submission scores for single task."""
        metric_fn = per_task_submission_scores()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBMISSION_SCORE: 0.8}),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            )
        ]
        result = metric_fn(sample_scores)
        assert result == {"task_1_submission_score": 0.8}

    def test_per_task_submission_scores_multiple_tasks(self):
        """Test per-task submission scores for multiple tasks."""
        metric_fn = per_task_submission_scores()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBMISSION_SCORE: 0.8}),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            ),
            SampleScore(
                sample_id="2",
                score=Score(value=0.5, answer="", metadata={MetadataKeys.SUBMISSION_SCORE: 0.5}),
                sample_metadata={MetadataKeys.TASK_ID: "task-2"},
            ),
        ]
        result = metric_fn(sample_scores)
        assert result == {"task_1_submission_score": 0.8, "task_2_submission_score": 0.5}

    def test_per_task_submission_scores_averages_same_task(self):
        """Test per-task submission scores averages samples from same task."""
        metric_fn = per_task_submission_scores()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBMISSION_SCORE: 1.0}),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            ),
            SampleScore(
                sample_id="2",
                score=Score(value=0.5, answer="", metadata={MetadataKeys.SUBMISSION_SCORE: 0.5}),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            ),
        ]
        result = metric_fn(sample_scores)
        assert result == {"task_1_submission_score": 0.75}

    def test_per_task_submission_scores_sanitizes_task_ids(self):
        """Test that task IDs with hyphens and spaces are sanitized."""
        metric_fn = per_task_submission_scores()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBMISSION_SCORE: 0.9}),
                sample_metadata={MetadataKeys.TASK_ID: "my-task with-spaces"},
            )
        ]
        result = metric_fn(sample_scores)
        assert result == {"my_task_with_spaces_submission_score": 0.9}


class TestPerTaskSubtaskScores:
    """Test cases for per_task_subtask_scores metric."""

    def test_per_task_subtask_scores_single_task(self):
        """Test per-task subtask scores for single task."""
        metric_fn = per_task_subtask_scores()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.5, answer="", metadata={MetadataKeys.SUBTASK_SCORE: 0.5}),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            )
        ]
        result = metric_fn(sample_scores)
        assert result == {"task_1_subtask_score": 0.5}

    def test_per_task_subtask_scores_skips_missing(self):
        """Test per-task subtask scores skips samples without subtask scores."""
        metric_fn = per_task_subtask_scores()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBMISSION_SCORE: 1.0}),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            )
        ]
        result = metric_fn(sample_scores)
        assert result == {}

    def test_per_task_subtask_scores_multiple_tasks(self):
        """Test per-task subtask scores for multiple tasks."""
        metric_fn = per_task_subtask_scores()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.5, answer="", metadata={MetadataKeys.SUBTASK_SCORE: 0.6}),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            ),
            SampleScore(
                sample_id="2",
                score=Score(value=1.3, answer="", metadata={MetadataKeys.SUBTASK_SCORE: 0.3}),
                sample_metadata={MetadataKeys.TASK_ID: "task-2"},
            ),
        ]
        result = metric_fn(sample_scores)
        assert result == {"task_1_subtask_score": 0.6, "task_2_subtask_score": 0.3}


class TestSubtaskScoreMetrics:
    """Test cases for subtask_score_metrics metric."""

    def test_subtask_score_metrics_single_subtask(self):
        """Test subtask score metrics for single subtask."""
        metric_fn = subtask_score_metrics()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBTASK_SCORES: {"subtask-1": 0.8}}),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            )
        ]
        result = metric_fn(sample_scores)
        assert result == {"task_1_subtask_1_score": 0.8}

    def test_subtask_score_metrics_multiple_subtasks(self):
        """Test subtask score metrics for multiple subtasks."""
        metric_fn = subtask_score_metrics()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=1.5,
                    answer="",
                    metadata={MetadataKeys.SUBTASK_SCORES: {"subtask-1": 0.8, "subtask-2": 0.7}},
                ),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            )
        ]
        result = metric_fn(sample_scores)
        assert result == {"task_1_subtask_1_score": 0.8, "task_1_subtask_2_score": 0.7}

    def test_subtask_score_metrics_averages_across_samples(self):
        """Test subtask score metrics averages across samples."""
        metric_fn = subtask_score_metrics()
        sample_scores = [
            SampleScore(
                sample_id="1",
                score=Score(value=1.0, answer="", metadata={MetadataKeys.SUBTASK_SCORES: {"subtask-1": 1.0}}),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            ),
            SampleScore(
                sample_id="2",
                score=Score(value=0.5, answer="", metadata={MetadataKeys.SUBTASK_SCORES: {"subtask-1": 0.5}}),
                sample_metadata={MetadataKeys.TASK_ID: "task-1"},
            ),
        ]
        result = metric_fn(sample_scores)
        assert result == {"task_1_subtask_1_score": 0.75}

    def test_subtask_score_metrics_skips_missing_metadata(self):
        """Test subtask score metrics skips samples without subtask scores."""
        metric_fn = subtask_score_metrics()
        sample_scores = [
            SampleScore(sample_id="1", score=Score(value=1.0, answer="", metadata={}), sample_metadata={})
        ]
        result = metric_fn(sample_scores)
        assert result == {}


# ============================================================================
# Test Submission Scoring
# ============================================================================


class TestScoreSubmissionStatic:
    """Test cases for score_submission_static."""

    @pytest.mark.asyncio
    async def test_static_submission_exact_match(self, task_context):
        """Test static submission scoring with exact match."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="The answer is 42",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.STATIC,
            criteria={"expected_answers": ["42"]},
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        mock_state = Mock(spec=TaskState)
        score, explanation = await score_submission_static(submission_data, criteria, None, mock_state)
        assert score == 1.0
        assert "42" in explanation

    @pytest.mark.asyncio
    async def test_static_submission_no_match(self, task_context):
        """Test static submission scoring with no match."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="The answer is 99",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.STATIC,
            criteria={"expected_answers": ["42"]},
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        mock_state = Mock(spec=TaskState)
        score, explanation = await score_submission_static(submission_data, criteria, None, mock_state)
        assert score == 0.0
        assert "Expected" in explanation

    @pytest.mark.asyncio
    async def test_static_submission_case_insensitive(self, task_context):
        """Test static submission scoring is case-insensitive."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="The answer is CORRECT",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.STATIC,
            criteria={"expected_answers": ["correct"]},
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        mock_state = Mock(spec=TaskState)
        score, explanation = await score_submission_static(submission_data, criteria, None, mock_state)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_static_submission_multiple_expected_answers(self, task_context):
        """Test static submission with multiple expected answers."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="blue",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.STATIC,
            criteria={"expected_answers": ["red", "blue", "green"]},
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        mock_state = Mock(spec=TaskState)
        score, explanation = await score_submission_static(submission_data, criteria, None, mock_state)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_static_submission_string_to_list_conversion(self, task_context):
        """Test static submission converts string expected_answers to list."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="answer",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.STATIC,
            criteria={"expected_answers": "answer"},  # String instead of list
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        mock_state = Mock(spec=TaskState)
        score, explanation = await score_submission_static(submission_data, criteria, None, mock_state)
        assert score == 1.0


class TestScoreSubmissionLLM:
    """Test cases for score_submission_llm."""

    @pytest.mark.asyncio
    async def test_llm_submission_correct_response(self, task_context):
        """Test LLM submission scoring with CORRECT response."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="Paris",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.LLM_JUDGE,
            criteria={
                "judge_system_template": "You are a judge",
                "judge_user_template": "Is {{ submission }} correct?",
                "model": "openai/gpt-4",
                "golden_answer": "Paris",
            },
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        # Mock state and model
        state = Mock(spec=TaskState)
        state.messages = []
        mock_output = Mock(spec=ModelOutput)
        mock_output.completion = "CORRECT"
        state.output = mock_output

        # Mock get_model
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.core.scoring.standard_submission.resolve_template_content", new_callable=AsyncMock, side_effect=lambda _sm, val: val):
            with patch("saber.inspect_ai.core.scoring.standard_submission.get_model", return_value=mock_model):
                score, explanation = await score_submission_llm(submission_data, criteria, Mock(), state)

        assert score == 1.0
        assert "1.0" in explanation
        assert mock_model.generate.called

    @pytest.mark.asyncio
    async def test_llm_submission_incorrect_response(self, task_context):
        """Test LLM submission scoring with INCORRECT response."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="London",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.LLM_JUDGE,
            criteria={
                "judge_system_template": "You are a judge",
                "judge_user_template": "Is {{ submission }} correct?",
                "model": "openai/gpt-4",
                "golden_answer": "Paris",
            },
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        state = Mock(spec=TaskState)
        state.messages = []
        mock_output = Mock(spec=ModelOutput)
        mock_output.completion = "INCORRECT"
        state.output = mock_output

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.core.scoring.standard_submission.resolve_template_content", new_callable=AsyncMock, side_effect=lambda _sm, val: val):
            with patch("saber.inspect_ai.core.scoring.standard_submission.get_model", return_value=mock_model):
                score, explanation = await score_submission_llm(submission_data, criteria, Mock(), state)

        assert score == 0.0
        assert "0.0" in explanation

    @pytest.mark.asyncio
    async def test_llm_submission_unclear_response(self, task_context):
        """Test LLM submission scoring with unclear response defaults to 0.0."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="Maybe",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.LLM_JUDGE,
            criteria={
                "judge_system_template": "You are a judge",
                "judge_user_template": "Evaluate {{ submission }}",
                "model": "openai/gpt-4",
            },
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        state = Mock(spec=TaskState)
        state.messages = []
        mock_output = Mock(spec=ModelOutput)
        mock_output.completion = "I'm not sure about this"
        state.output = mock_output

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.core.scoring.standard_submission.resolve_template_content", new_callable=AsyncMock, side_effect=lambda _sm, val: val):
            with patch("saber.inspect_ai.core.scoring.standard_submission.get_model", return_value=mock_model):
                score, explanation = await score_submission_llm(submission_data, criteria, Mock(), state)

        assert score == 0.0
        assert "unable to parse" in explanation

    @pytest.mark.asyncio
    async def test_llm_submission_missing_templates_raises_error(self, task_context):
        """Test LLM submission raises error when templates are missing."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="test",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.LLM_JUDGE,
            criteria={"model": "openai/gpt-4"},  # Missing templates
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        state = Mock(spec=TaskState)
        state.messages = []

        with pytest.raises(RuntimeError, match="LLM submission evaluation requires"):
            await score_submission_llm(submission_data, criteria, Mock(), state)


# ============================================================================
# Test Subtask Scoring
# ============================================================================


class TestParseLLMStepEvaluations:
    """Test cases for _parse_llm_step_evaluations."""

    def test_parse_single_completion(self):
        """Test parsing a single checkpoint completion."""
        response = """
        STEP_EVALUATIONS:
        [5: checkpoint_1] - Agent discovered the vulnerability
        """
        valid_ids = ["checkpoint_1", "checkpoint_2"]
        result = _parse_llm_step_evaluations(response, valid_ids)
        assert result["checkpoint_1"] == [5]
        assert result["checkpoint_2"] == []

    def test_parse_multiple_completions(self):
        """Test parsing multiple checkpoint completions."""
        response = """
        STEP_EVALUATIONS:
        [2: checkpoint_1] - Agent enumerated endpoints
        [5: checkpoint_2] - Agent identified vulnerability
        [8: checkpoint_3] - Agent exploited the vulnerability
        """
        valid_ids = ["checkpoint_1", "checkpoint_2", "checkpoint_3", "checkpoint_4"]
        result = _parse_llm_step_evaluations(response, valid_ids)
        assert result["checkpoint_1"] == [2]
        assert result["checkpoint_2"] == [5]
        assert result["checkpoint_3"] == [8]
        assert result["checkpoint_4"] == []

    def test_parse_no_completions(self):
        """Test parsing NO_COMPLETIONS response."""
        response = "[NO_COMPLETIONS]"
        valid_ids = ["checkpoint_1", "checkpoint_2"]
        result = _parse_llm_step_evaluations(response, valid_ids)
        assert result["checkpoint_1"] == []
        assert result["checkpoint_2"] == []

    def test_parse_ignores_invalid_checkpoint_ids(self):
        """Test that invalid checkpoint IDs are ignored."""
        response = """
        STEP_EVALUATIONS:
        [5: checkpoint_1] - Valid
        [6: invalid_checkpoint] - Should be ignored
        """
        valid_ids = ["checkpoint_1", "checkpoint_2"]
        result = _parse_llm_step_evaluations(response, valid_ids)
        assert result["checkpoint_1"] == [5]
        assert result["checkpoint_2"] == []
        assert "invalid_checkpoint" not in result

    def test_parse_deduplicates_step_numbers(self):
        """Test that duplicate step numbers for same checkpoint are deduplicated."""
        response = """
        STEP_EVALUATIONS:
        [5: checkpoint_1] - First mention
        [5: checkpoint_1] - Duplicate
        """
        valid_ids = ["checkpoint_1"]
        result = _parse_llm_step_evaluations(response, valid_ids)
        assert result["checkpoint_1"] == [5]

class TestScoreSubtasksLLMBatch:
    """Test cases for _score_subtasks_llm_batch."""

    @pytest.mark.asyncio
    async def test_llm_batch_single_subtask_completed(self, task_context):
        """Test LLM batch scoring with single subtask completed."""
        steps_data = EpisodeStepsResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            steps=[
                EpisodeStepData(
                    step_number=1,
                    tool_name="bash",
                    tool_input={},
                    tool_output="success",
                    timestamp=datetime.now(timezone.utc),
                )
            ],
            total_steps=1,
        )
        criteria = SubtaskEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            subtask_id="checkpoint_1",
            title="Test Task",
            description="Test Description",
            objective="Test Objective",
            strategy=StepEvaluationStrategy.LLM_JUDGE,
            criteria={
                "judge_system_template": "Evaluate steps",
                "judge_user_template": "Check steps",
                "model": "openai/gpt-4",
                "steps_per_message": 10,
            },
            max_score=1.0,
            weight=1.0,
            task_context=task_context,
        )

        state = Mock(spec=TaskState)
        state.messages = []
        mock_output = Mock(spec=ModelOutput)
        mock_output.completion = """
        STEP_EVALUATIONS:
        [1: checkpoint_1] - Agent completed the objective
        """
        state.output = mock_output

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.core.saber_scorer.get_model", return_value=mock_model):
            results = await _score_subtasks_llm_batch(
                steps_data, [criteria], task_context, Mock(), state
            )

        assert len(results) == 1
        score, evaluations = results[0]
        assert score == 1.0
        assert len(evaluations) == 1
        assert evaluations[0].completed

    @pytest.mark.asyncio
    async def test_llm_batch_multiple_subtasks(self, task_context):
        """Test LLM batch scoring with multiple subtasks - only one LLM call."""
        steps_data = EpisodeStepsResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            steps=[
                EpisodeStepData(
                    step_number=0,
                    tool_name="bash",
                    tool_input={},
                    tool_output="enumerated",
                    timestamp=datetime.now(timezone.utc),
                ),
                EpisodeStepData(
                    step_number=1,
                    tool_name="bash",
                    tool_input={},
                    tool_output="exploited",
                    timestamp=datetime.now(timezone.utc),
                ),
            ],
            total_steps=2,
        )
        criteria_list = [
            SubtaskEvaluationCriteriaResponse(
                session_id="session1",
                episode_id="ep1",
                task_id="task1",
                subtask_id="checkpoint_1",
                title="Discovery",
                description="Discover endpoints",
                objective="Enumerate API",
                strategy=StepEvaluationStrategy.LLM_JUDGE,
                criteria={
                    "judge_system_template": "Evaluate",
                    "judge_user_template": "Check",
                    "model": "openai/gpt-4",
                },
                max_score=1.0,
                weight=0.25,
                task_context=task_context,
            ),
            SubtaskEvaluationCriteriaResponse(
                session_id="session1",
                episode_id="ep1",
                task_id="task1",
                subtask_id="checkpoint_2",
                title="Exploitation",
                description="Exploit vulnerability",
                objective="Execute exploit",
                strategy=StepEvaluationStrategy.LLM_JUDGE,
                criteria={
                    "judge_system_template": "Evaluate",
                    "judge_user_template": "Check",
                    "model": "openai/gpt-4",
                },
                max_score=1.0,
                weight=0.75,
                task_context=task_context,
            ),
        ]

        state = Mock(spec=TaskState)
        state.messages = []
        mock_output = Mock(spec=ModelOutput)
        # LLM returns both checkpoints completed
        mock_output.completion = """
        STEP_EVALUATIONS:
        [0: checkpoint_1] - Agent enumerated endpoints
        [1: checkpoint_2] - Agent exploited vulnerability
        """
        state.output = mock_output

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.core.saber_scorer.get_model", return_value=mock_model):
            results = await _score_subtasks_llm_batch(
                steps_data, criteria_list, task_context, Mock(), state
            )

        # Should only call LLM once (batched)
        assert mock_model.generate.call_count == 1

        # Both subtasks scored
        assert len(results) == 2
        score1, evals1 = results[0]
        score2, evals2 = results[1]
        assert score1 == 1.0
        assert score2 == 1.0

    @pytest.mark.asyncio
    async def test_llm_batch_no_completions(self, task_context):
        """Test LLM batch scoring when no checkpoints completed."""
        steps_data = EpisodeStepsResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            steps=[
                EpisodeStepData(
                    step_number=1,
                    tool_name="bash",
                    tool_input={},
                    tool_output="failed",
                    timestamp=datetime.now(timezone.utc),
                )
            ],
            total_steps=1,
        )
        criteria = SubtaskEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            subtask_id="checkpoint_1",
            title="Test",
            description="Test",
            objective="Test",
            strategy=StepEvaluationStrategy.LLM_JUDGE,
            criteria={
                "judge_system_template": "Evaluate",
                "judge_user_template": "Check",
                "model": "openai/gpt-4",
            },
            max_score=1.0,
            weight=1.0,
            task_context=task_context,
        )

        state = Mock(spec=TaskState)
        state.messages = []
        mock_output = Mock(spec=ModelOutput)
        mock_output.completion = "[NO_COMPLETIONS]"
        state.output = mock_output

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.core.saber_scorer.get_model", return_value=mock_model):
            results = await _score_subtasks_llm_batch(
                steps_data, [criteria], task_context, Mock(), state
            )

        score, evaluations = results[0]
        assert score == 0.0
        assert all(not e.completed for e in evaluations)

    @pytest.mark.asyncio
    async def test_llm_batch_missing_templates_raises_error(self, task_context):
        """Test LLM batch raises error when templates are missing."""
        steps_data = EpisodeStepsResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            steps=[],
            total_steps=0,
        )
        criteria = SubtaskEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            subtask_id="checkpoint_1",
            title="Test",
            description="Test",
            objective="Test",
            strategy=StepEvaluationStrategy.LLM_JUDGE,
            criteria={"model": "openai/gpt-4"},  # Missing templates
            max_score=1.0,
            weight=1.0,
            task_context=task_context,
        )

        state = Mock(spec=TaskState)
        state.messages = []

        with pytest.raises(RuntimeError, match="Missing required template"):
            await _score_subtasks_llm_batch(steps_data, [criteria], task_context, Mock(), state)

    @pytest.mark.asyncio
    async def test_llm_batch_empty_list_returns_empty(self, task_context):
        """Test LLM batch with empty criteria list returns empty."""
        steps_data = EpisodeStepsResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            steps=[],
            total_steps=0,
        )

        state = Mock(spec=TaskState)
        results = await _score_subtasks_llm_batch(steps_data, [], task_context, Mock(), state)
        assert results == []


# ============================================================================
# Test Score Submission Dispatcher
# ============================================================================


class TestScoreSubmission:
    """Test cases for _score_submission dispatcher."""

    @pytest.mark.asyncio
    async def test_score_submission_dispatches_to_static(self, task_context):
        """Test _score_submission dispatches to static strategy."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="42",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.STATIC,
            criteria={"expected_answers": ["42"]},
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        score, explanation = await _score_submission(submission_data, criteria, Mock(), Mock())
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_submission_dispatches_to_llm(self, task_context):
        """Test _score_submission dispatches to LLM strategy."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="Paris",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy=SubmissionEvaluationStrategy.LLM_JUDGE,
            criteria={
                "judge_system_template": "Judge",
                "judge_user_template": "{{ submission }}",
                "model": "openai/gpt-4",
            },
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        state = Mock(spec=TaskState)
        state.messages = []
        mock_output = Mock(spec=ModelOutput)
        mock_output.completion = "CORRECT"
        state.output = mock_output

        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        with patch("saber.inspect_ai.core.scoring.standard_submission.get_model", return_value=mock_model):
            score, explanation = await _score_submission(submission_data, criteria, Mock(), state)

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_submission_unknown_strategy_raises_error(self, task_context):
        """Test _score_submission raises error for unknown strategy."""
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="test",
            model="test",
            tokens={},
        )
        criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            strategy="UNKNOWN_STRATEGY",  # Invalid strategy
            criteria={},
            scoring={"max_score": 1.0},
            task_context=task_context,
        )

        with pytest.raises(RuntimeError, match="Unknown submission strategy"):
            await _score_submission(submission_data, criteria, Mock(), Mock())


# ============================================================================
# Test Score All Subtasks
# ============================================================================


class TestScoreAllSubtasks:
    """Test cases for _score_all_subtasks."""

    @pytest.mark.asyncio
    async def test_score_all_subtasks_single_subtask(self, task_context):
        """Test scoring all subtasks with single subtask."""
        steps_data = EpisodeStepsResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            steps=[
                EpisodeStepData(
                    step_number=1,
                    tool_name="bash",
                    tool_input={},
                    tool_output="config.yaml",
                    timestamp=datetime.now(timezone.utc),
                )
            ],
            total_steps=1,
        )
        criteria_list = [
            SubtaskEvaluationCriteriaResponse(
                session_id="session1",
                episode_id="ep1",
                task_id="task1",
                subtask_id="subtask1",
                title="Find config",
                description="Locate config",
                objective="Find config.yaml",
                strategy=StepEvaluationStrategy.STATIC,
                criteria={"expected_outputs": ["config.yaml"]},
                max_score=1.0,
                weight=1.0,
                task_context=task_context,
            )
        ]

        # Create mock submission_data
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="test",
            model="test",
            tokens={},
        )

        total_score, individual_scores, evaluations, checkpoint_summary = await _score_all_subtasks(
            steps_data, criteria_list, Mock(description="Test", domain="test"), Mock(), Mock(), submission_data
        )

        assert total_score == 1.0
        assert individual_scores == [1.0]
        assert len(evaluations) == 1
        assert isinstance(checkpoint_summary, str)

    @pytest.mark.asyncio
    async def test_score_all_subtasks_multiple_subtasks(self, task_context):
        """Test scoring all subtasks with multiple subtasks."""
        steps_data = EpisodeStepsResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            steps=[
                EpisodeStepData(
                    step_number=1,
                    tool_name="bash",
                    tool_input={},
                    tool_output="config.yaml",
                    timestamp=datetime.now(timezone.utc),
                ),
                EpisodeStepData(
                    step_number=2,
                    tool_name="python",
                    tool_input={},
                    tool_output="success",
                    timestamp=datetime.now(timezone.utc),
                ),
            ],
            total_steps=2,
        )
        criteria_list = [
            SubtaskEvaluationCriteriaResponse(
                session_id="session1",
                episode_id="ep1",
                task_id="task1",
                subtask_id="subtask1",
                title="Config",
                description="Find config",
                objective="Find config",
                strategy=StepEvaluationStrategy.STATIC,
                criteria={"expected_outputs": ["config.yaml"]},
                max_score=1.0,
                weight=1.0,
                task_context=task_context,
            ),
            SubtaskEvaluationCriteriaResponse(
                session_id="session1",
                episode_id="ep1",
                task_id="task1",
                subtask_id="subtask2",
                title="Python",
                description="Use Python",
                objective="Call Python",
                strategy=StepEvaluationStrategy.TOOL_CALL,
                criteria={"expected_tools": ["python"]},
                max_score=1.0,
                weight=1.0,
                task_context=task_context,
            ),
        ]

        # Create mock submission_data
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="test",
            model="test",
            tokens={},
        )

        total_score, individual_scores, evaluations, checkpoint_summary = await _score_all_subtasks(
            steps_data, criteria_list, Mock(description="Test", domain="test"), Mock(), Mock(), submission_data
        )

        assert total_score == 2.0
        assert individual_scores == [1.0, 1.0]
        assert len(evaluations) == 2
        assert isinstance(checkpoint_summary, str)

    @pytest.mark.asyncio
    async def test_score_all_subtasks_skips_empty_strategy(self, task_context):
        """Test scoring all subtasks skips subtasks without strategy."""
        steps_data = EpisodeStepsResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            steps=[],
            total_steps=0,
        )
        criteria_list = [
            SubtaskEvaluationCriteriaResponse(
                session_id="session1",
                episode_id="ep1",
                task_id="task1",
                subtask_id="subtask1",
                title="Info only",
                description="Informational checkpoint",
                objective="Info",
                strategy="",  # Empty strategy
                criteria={},
                max_score=1.0,
                weight=1.0,
                task_context=task_context,
            )
        ]

        # Create mock submission_data
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="test",
            model="test",
            tokens={},
        )

        total_score, individual_scores, evaluations, checkpoint_summary = await _score_all_subtasks(
            steps_data, criteria_list, Mock(description="Test"), Mock(), Mock(), submission_data
        )

        assert total_score == 0.0
        assert individual_scores == [0.0]
        assert evaluations == [[]]
        assert isinstance(checkpoint_summary, str)

    @pytest.mark.asyncio
    async def test_score_all_subtasks_unknown_strategy_raises_error(self, task_context):
        """Test scoring all subtasks raises error for unknown strategy."""
        steps_data = EpisodeStepsResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            steps=[],
            total_steps=0,
        )
        criteria_list = [
            SubtaskEvaluationCriteriaResponse(
                session_id="session1",
                episode_id="ep1",
                task_id="task1",
                subtask_id="subtask1",
                title="Test",
                description="Test",
                objective="Test",
                strategy="UNKNOWN",  # Invalid strategy
                criteria={},
                max_score=1.0,
                weight=1.0,
                task_context=task_context,
            )
        ]

        # Create mock submission_data
        submission_data = EpisodeSubmissionResponse(
            session_id="session1",
            episode_id="ep1",
            task_id="task1",
            submission="test",
            model="test",
            tokens={},
        )

        with pytest.raises(RuntimeError, match="Unknown subtask strategy"):
            await _score_all_subtasks(
                steps_data, criteria_list, Mock(description="Test"), Mock(), Mock(), submission_data
            )
