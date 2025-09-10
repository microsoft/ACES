"""
Unit tests for StaticEvaluator.
"""

import pytest

from saber.server.benchmarks.task import Task
from saber.server.evaluation.constants import EVAL_STRATEGY_STATIC
from saber.server.evaluation.evaluators.static_evaluator import StaticEvaluator
from saber.server.evaluation.exceptions import EvaluationValidationError
from saber.server.evaluation.models import EvaluationConfig, EvaluationResult, EpisodeEvaluationData


class TestStaticEvaluator:
    """Test cases for StaticEvaluator functionality."""

    @pytest.fixture
    def evaluator(self):
        """Create StaticEvaluator instance."""
        return StaticEvaluator()

    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        return Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            prompt_template_file="test_template.j2",
            allowed_executors=["test_executor"],
            evaluation_config={
                "strategy": "static",
                "criteria": {"expected_answers": ["test_answer"]},
                "scoring": {"max_score": 1.0}
            }
        )

    @pytest.fixture
    def single_answer_config(self):
        """Create evaluation config for single answer task."""
        return EvaluationConfig(
            strategy=EVAL_STRATEGY_STATIC,
            criteria={"expected_answers": ["flag{correct_answer}"]},
            scoring={"max_score": 1.0}
        )

    @pytest.fixture
    def multi_answer_config(self):
        """Create evaluation config for multiple answer task."""
        return EvaluationConfig(
            strategy=EVAL_STRATEGY_STATIC,
            criteria={"expected_answers": ["answer1", "answer2", "answer3"]},
            scoring={"max_score": 100.0}
        )

    @pytest.fixture
    def episode_data(self):
        """Create sample episode data."""
        return EpisodeEvaluationData(
            episode_id="episode_123",
            task_id="test_task",
            submission="flag{correct_answer}",
            executed_commands=["ls", "cat file.txt"],
            completion_reason="completed",
            step_count=5
        )

    @pytest.mark.asyncio
    async def test_single_answer_exact_match_success(self, evaluator, sample_task, single_answer_config, episode_data):
        """Test exact match success for single answer task."""
        result = await evaluator.evaluate(episode_data, single_answer_config, sample_task)

        assert isinstance(result, EvaluationResult)
        assert result.episode_id == "episode_123"
        assert result.task_id == "test_task"
        assert result.strategy == EVAL_STRATEGY_STATIC
        assert result.raw_score == 1.0
        assert result.max_score == 1.0
        assert result.score == 1.0
        assert result.success is True
        assert result.details["matched_answers"] == ["flag{correct_answer}"]
        assert result.details["total_expected"] == 1
        assert result.details["submission"] == "flag{correct_answer}"

    @pytest.mark.asyncio
    async def test_single_answer_exact_match_failure(self, evaluator, sample_task, single_answer_config):
        """Test exact match failure for single answer task."""
        episode_data = EpisodeEvaluationData(
            episode_id="episode_123",
            task_id="test_task",
            submission="flag{wrong_answer}",
            step_count=5
        )

        result = await evaluator.evaluate(episode_data, single_answer_config, sample_task)

        assert result.raw_score == 0.0
        assert result.score == 0.0
        assert result.success is False
        assert result.details["matched_answers"] == []
        assert result.details["submission"] == "flag{wrong_answer}"

    @pytest.mark.asyncio
    async def test_case_sensitive_matching(self, evaluator, sample_task, single_answer_config):
        """Test that matching is case-sensitive."""
        episode_data = EpisodeEvaluationData(
            episode_id="episode_123",
            task_id="test_task",
            submission="FLAG{CORRECT_ANSWER}",  # Different case
            step_count=5
        )

        result = await evaluator.evaluate(episode_data, single_answer_config, sample_task)

        assert result.raw_score == 0.0
        assert result.success is False

    @pytest.mark.asyncio
    async def test_whitespace_trimming(self, evaluator, sample_task, single_answer_config):
        """Test that whitespace is properly trimmed."""
        episode_data = EpisodeEvaluationData(
            episode_id="episode_123",
            task_id="test_task",
            submission="  flag{correct_answer}  ",
            step_count=5
        )

        result = await evaluator.evaluate(episode_data, single_answer_config, sample_task)

        assert result.raw_score == 1.0
        assert result.success is True

    @pytest.mark.asyncio
    async def test_multi_answer_partial_match(self, evaluator, sample_task, multi_answer_config):
        """Test partial match for multiple answer task."""
        episode_data = EpisodeEvaluationData(
            episode_id="episode_123",
            task_id="test_task",
            submission="answer2",
            step_count=5
        )

        result = await evaluator.evaluate(episode_data, multi_answer_config, sample_task)

        assert result.raw_score == 1.0 / 3.0  # 1 out of 3 answers
        assert abs(result.score - 100.0 / 3.0) < 0.01  # Raw score * max_score (allow for floating point precision)
        assert result.success is True  # Any match counts as success for multi-answer
        assert result.details["matched_answers"] == ["answer2"]
        assert result.details["total_expected"] == 3

    @pytest.mark.asyncio
    async def test_multi_answer_no_match(self, evaluator, sample_task, multi_answer_config):
        """Test no match for multiple answer task."""
        episode_data = EpisodeEvaluationData(
            episode_id="episode_123",
            task_id="test_task",
            submission="wrong_answer",
            step_count=5
        )

        result = await evaluator.evaluate(episode_data, multi_answer_config, sample_task)

        assert result.raw_score == 0.0
        assert result.score == 0.0
        assert result.success is False

    @pytest.mark.asyncio
    async def test_missing_expected_answers(self, evaluator, sample_task, episode_data):
        """Test validation error when expected_answers is missing."""
        config = EvaluationConfig(
            strategy=EVAL_STRATEGY_STATIC,
            criteria={},  # Missing expected_answers
            scoring={"max_score": 1.0}
        )

        with pytest.raises(EvaluationValidationError, match="Static evaluation requires expected_answers"):
            await evaluator.evaluate(episode_data, config, sample_task)

    @pytest.mark.asyncio
    async def test_invalid_expected_answers_type(self, evaluator, sample_task, episode_data):
        """Test validation error when expected_answers is not a list."""
        config = EvaluationConfig(
            strategy=EVAL_STRATEGY_STATIC,
            criteria={"expected_answers": "not_a_list"},
            scoring={"max_score": 1.0}
        )

        with pytest.raises(EvaluationValidationError, match="expected_answers must be a list"):
            await evaluator.evaluate(episode_data, config, sample_task)

    @pytest.mark.asyncio
    async def test_empty_submission(self, evaluator, sample_task, single_answer_config):
        """Test evaluation with empty submission."""
        episode_data = EpisodeEvaluationData(
            episode_id="episode_123",
            task_id="test_task",
            submission="",
            step_count=5
        )

        result = await evaluator.evaluate(episode_data, single_answer_config, sample_task)

        assert result.raw_score == 0.0
        assert result.success is False

    @pytest.mark.asyncio
    async def test_custom_max_score(self, evaluator, sample_task, episode_data):
        """Test evaluation with custom max_score."""
        config = EvaluationConfig(
            strategy=EVAL_STRATEGY_STATIC,
            criteria={"expected_answers": ["flag{correct_answer}"]},
            scoring={"max_score": 50.0}
        )

        result = await evaluator.evaluate(episode_data, config, sample_task)

        assert result.raw_score == 1.0
        assert result.max_score == 50.0
        assert result.score == 50.0
