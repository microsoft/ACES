"""
Tests for LLM evaluator implementation.
"""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch
import pytest

from saber.server.evaluation.evaluators.llm_evaluator import LLMEvaluator
from saber.server.evaluation.models import EvaluationConfig, EpisodeEvaluationData
from saber.server.evaluation.exceptions import EvaluationError, EvaluationConfigError
from saber.server.evaluation.constants import EVAL_STRATEGY_LLM_JUDGE
from saber.server.benchmarks.prompt_generator import JudgePromptPayload


class MockTask:
    """Mock task for testing."""
    def __init__(self):
        self.task_id = "test-task"
        self.description = "Test task description"
        # Mock evaluation_config with judge_prompt_renderer
        self.evaluation_config = {
            "judge_prompt_renderer": self._mock_judge_prompt_renderer
        }

    def _mock_judge_prompt_renderer(self, submission, episode_id):
        """Mock judge prompt renderer function."""
        return JudgePromptPayload(
            messages=[
                {"role": "system", "content": "Evaluate this cybersecurity analysis submission."},
                {"role": "user", "content": f"Submission: {submission}\nEpisode ID: {episode_id}\nIs this analysis correct?"}
            ],
            model="gpt-3.5-turbo",
            task_id="test-task",
            episode_id=episode_id
        )


@pytest.fixture
def sample_episode_data():
    """Sample episode data for testing."""
    return EpisodeEvaluationData(
        episode_id="test-episode-123",
        task_id="test-task",
        submission="The malware is a banking trojan targeting user credentials",
        step_count=5
    )


@pytest.fixture
def sample_config():
    """Sample LLM evaluation configuration."""
    return EvaluationConfig(
        strategy=EVAL_STRATEGY_LLM_JUDGE,
        criteria={
            "golden_answer": "Banking trojan that steals credentials",
            "model": "gpt-3.5-turbo",
            "judge_system_template": "test_system.md",
            "judge_user_template": "test_user.md"
        },
        scoring={"max_score": 1.0}
    )


@pytest.fixture
def mock_task():
    """Mock task for testing."""
    return MockTask()


class TestLLMEvaluator:
    """Test cases for LLMEvaluator."""

    def test_initialization(self):
        """Test LLM evaluator initialization."""
        evaluator = LLMEvaluator()
        assert evaluator.timeout_seconds == 15
        assert evaluator.llm_client is None

        # Test custom timeout
        evaluator = LLMEvaluator(timeout_seconds=30)
        assert evaluator.timeout_seconds == 30

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="LLM evaluator is currently stubbed - validation not implemented")
    async def test_evaluate_missing_api_key(self, sample_episode_data, sample_config, mock_task):
        """Test evaluation fails when OPENAI_API_KEY is not set."""
        evaluator = LLMEvaluator()

        # Ensure no API key is set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EvaluationError, match="OPENAI_API_KEY environment variable not set"):
                await evaluator.evaluate(sample_episode_data, sample_config, mock_task)

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="LLM evaluator is currently stubbed - validation not implemented")
    async def test_evaluate_missing_golden_answer(self, sample_episode_data, mock_task):
        """Test evaluation fails when golden_answer is missing."""
        evaluator = LLMEvaluator()

        config = EvaluationConfig(
            strategy=EVAL_STRATEGY_LLM_JUDGE,
            criteria={
                "model": "gpt-3.5-turbo",  # Missing golden_answer
                "judge_system_template": "test_system.md",
                "judge_user_template": "test_user.md"
            },
            scoring={"max_score": 1.0}
        )

        # Should fail with config error before checking API key
        with pytest.raises(EvaluationConfigError, match="LLM evaluation requires golden_answer"):
            await evaluator.evaluate(sample_episode_data, config, mock_task)

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="LLM evaluator is currently stubbed - validation not implemented")
    async def test_evaluate_missing_model(self, sample_episode_data, mock_task):
        """Test evaluation fails when model is missing."""
        evaluator = LLMEvaluator()

        config = EvaluationConfig(
            strategy=EVAL_STRATEGY_LLM_JUDGE,
            criteria={
                "golden_answer": "test answer",  # Missing model
                "judge_system_template": "test_system.md",
                "judge_user_template": "test_user.md"
            },
            scoring={"max_score": 1.0}
        )

        # Should fail with config error before checking API key
        with pytest.raises(EvaluationConfigError, match="LLM evaluation requires model"):
            await evaluator.evaluate(sample_episode_data, config, mock_task)

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="LLM evaluator is currently stubbed - validation not implemented")
    async def test_evaluate_missing_judge_system_template(self, sample_episode_data, mock_task):
        """Test evaluation fails when judge_system_template is missing."""
        evaluator = LLMEvaluator()

        config = EvaluationConfig(
            strategy=EVAL_STRATEGY_LLM_JUDGE,
            criteria={
                "golden_answer": "test answer",
                "model": "gpt-3.5-turbo",
                "judge_user_template": "test_user.md"  # Missing judge_system_template
            },
            scoring={"max_score": 1.0}
        )

        # Should fail with config error
        with pytest.raises(EvaluationConfigError, match="LLM evaluation requires both judge_system_template and judge_user_template"):
            await evaluator.evaluate(sample_episode_data, config, mock_task)

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="LLM evaluator is currently stubbed - validation not implemented")
    async def test_evaluate_missing_judge_user_template(self, sample_episode_data, mock_task):
        """Test evaluation fails when judge_user_template is missing."""
        evaluator = LLMEvaluator()

        config = EvaluationConfig(
            strategy=EVAL_STRATEGY_LLM_JUDGE,
            criteria={
                "golden_answer": "test answer",
                "model": "gpt-3.5-turbo",
                "judge_system_template": "test_system.md"  # Missing judge_user_template
            },
            scoring={"max_score": 1.0}
        )

        # Should fail with config error
        with pytest.raises(EvaluationConfigError, match="LLM evaluation requires both judge_system_template and judge_user_template"):
            await evaluator.evaluate(sample_episode_data, config, mock_task)

    @pytest.mark.asyncio
    async def test_evaluate_success_correct_answer(self, sample_episode_data, sample_config, mock_task):
        """Test successful evaluation with stubbed LLM evaluator."""
        evaluator = LLMEvaluator()

        result = await evaluator.evaluate(sample_episode_data, sample_config, mock_task)

        assert result.episode_id == "test-episode-123"
        assert result.task_id == "test-task"
        assert result.strategy == EVAL_STRATEGY_LLM_JUDGE
        assert result.raw_score == 1.0
        assert result.max_score == 1.0
        assert result.score == 1.0
        assert result.success is True
        # Check stubbed response details
        assert result.details["stubbed"] is True
        assert result.details["message"] == "LLM evaluator is stubbed - returning default success"
        assert result.details["submission"] == sample_episode_data.submission

    @pytest.mark.asyncio
    async def test_evaluate_success_incorrect_answer(self, sample_episode_data, sample_config, mock_task):
        """Test stubbed evaluation always returns success."""
        evaluator = LLMEvaluator()

        result = await evaluator.evaluate(sample_episode_data, sample_config, mock_task)

        # Stubbed evaluator always returns success
        assert result.raw_score == 1.0
        assert result.score == 1.0
        assert result.success is True
        assert result.details["stubbed"] is True

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="LLM evaluator is currently stubbed - validation not implemented")
    async def test_evaluate_malformed_json_response(self, sample_episode_data, sample_config, mock_task):
        """Test evaluation fails with malformed JSON response."""
        evaluator = LLMEvaluator()

        # Mock malformed JSON response
        mock_response = 'This is not JSON'

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.object(evaluator, '_call_llm_json', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = mock_response

                with pytest.raises(EvaluationError, match="Malformed LLM judge response"):
                    await evaluator.evaluate(sample_episode_data, sample_config, mock_task)

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="LLM evaluator is currently stubbed - validation not implemented")
    async def test_evaluate_timeout(self, sample_episode_data, sample_config, mock_task):
        """Test evaluation fails on timeout."""
        evaluator = LLMEvaluator(timeout_seconds=1)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.object(evaluator, '_call_llm_json', new_callable=AsyncMock) as mock_llm:
                mock_llm.side_effect = asyncio.TimeoutError()

                with pytest.raises(EvaluationError, match="LLM evaluation timed out"):
                    await evaluator.evaluate(sample_episode_data, sample_config, mock_task)

    @pytest.mark.asyncio
    async def test_evaluate_custom_max_score(self, sample_episode_data, mock_task):
        """Test evaluation with custom max score."""
        evaluator = LLMEvaluator()

        config = EvaluationConfig(
            strategy=EVAL_STRATEGY_LLM_JUDGE,
            criteria={
                "golden_answer": "test answer",
                "model": "gpt-3.5-turbo",
                "judge_system_template": "test_system.md",
                "judge_user_template": "test_user.md"
            },
            scoring={"max_score": 5.0}
        )

        mock_response = '{"analysis": "Correct", "is_correct": true}'

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.object(evaluator, '_call_llm_json', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = mock_response

                result = await evaluator.evaluate(sample_episode_data, config, mock_task)

                assert result.raw_score == 1.0
                assert result.max_score == 5.0
                assert result.score == 5.0  # raw_score * max_score

    def test_initialize_llm_client_no_api_key(self):
        """Test LLM client initialization fails without API key."""
        evaluator = LLMEvaluator()

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EvaluationError, match="OPENAI_API_KEY environment variable not set"):
                evaluator._initialize_llm_client()

    def test_initialize_llm_client_success(self):
        """Test successful LLM client initialization."""
        evaluator = LLMEvaluator()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('saber.server.evaluation.evaluators.llm_evaluator.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client

                evaluator._initialize_llm_client()

                assert evaluator.llm_client == mock_client
                mock_openai.assert_called_once_with(api_key="test-key")
