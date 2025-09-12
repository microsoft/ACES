"""
Phase 3 Integration Tests: SABER Client-Side Scorer Implementation

Tests the new client-side scoring functionality that retrieves evaluation criteria
from the server and performs local evaluation using inspect_ai scorers.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

from saber.client.inspect_ai.saber_scorer import (
    saber_scorer,
    _get_evaluation_criteria,
    _evaluate_static,
    _evaluate_llm,
)
from saber.client.client_session import ClientSessionManager
from saber.models.rest.evaluation import (
    EvaluationCriteriaResponse,
    TaskEvaluationContext,
    JudgeMessages,
)
from inspect_ai.scorer import Score
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Target
from inspect_ai.model import ModelOutput
from inspect_ai.util import store


class TestSABERClientScorer:
    """Test client-side scorer functionality."""

    @pytest.fixture
    def mock_static_criteria(self):
        """Mock evaluation criteria for static evaluation."""
        return EvaluationCriteriaResponse(
            session_id="sess_123",
            episode_id="ep_123",
            task_id="task_1",
            submission="flag{test_flag} - Found in /etc/passwd",
            task_context=TaskEvaluationContext(
                task_id="task_1",
                title="Find the Flag",
                description="Locate the hidden flag in the system",
                domain="linux_ctf"
            ),
            evaluation_config={
                "strategy": "static",
                "criteria": {
                    "expected_answers": ["flag{test_flag}", "test_flag"]
                },
                "scoring": {
                    "max_score": 1.0
                }
            }
        )

    @pytest.fixture
    def mock_llm_criteria(self):
        """Mock evaluation criteria for LLM evaluation."""
        return EvaluationCriteriaResponse(
            session_id="sess_123",
            episode_id="ep_123",
            task_id="task_1",
            submission="I found the flag in /etc/passwd: flag{test_flag}",
            task_context=TaskEvaluationContext(
                task_id="task_1",
                title="Find the Flag",
                description="Locate the hidden flag in the system",
                domain="linux_ctf"
            ),
            evaluation_config={
                "strategy": "llm_judge",
                "criteria": {
                    "golden_answer": "flag{test_flag}"
                },
                "scoring": {
                    "max_score": 1.0
                }
            },
            judge_messages=JudgeMessages(
                system_message="You are an expert security evaluator...",
                user_message="Task: Locate the hidden flag in the system\nSubmission: I found the flag in /etc/passwd: flag{test_flag}\nPlease evaluate this submission. End with GRADE: C or GRADE: I",
                model="gpt-4"
            )
        )

    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager."""
        return AsyncMock(spec=ClientSessionManager)

    @pytest.fixture
    def mock_task_state(self):
        """Mock TaskState with submission."""
        state = Mock(spec=TaskState)
        state.output = ModelOutput(completion="flag{test_flag} - Found in /etc/passwd")
        return state

    @pytest.fixture
    def mock_target(self):
        """Mock Target."""
        return Mock(spec=Target)

    async def test_static_evaluation_success(self, mock_static_criteria):
        """Test successful static evaluation with substring matching."""
        submission = "flag{test_flag} - Found in /etc/passwd"

        result = await _evaluate_static(submission, mock_static_criteria)

        assert result.value == 1.0
        assert result.answer == submission
        assert "Found expected answer 'flag{test_flag}'" in result.explanation
        assert result.metadata["strategy"] == "static"
        assert result.metadata["scorer_type"] == "saber_client_side"
        assert result.metadata["is_correct"] is True
        assert result.metadata["matched_answer"] == "flag{test_flag}"

    async def test_static_evaluation_failure(self, mock_static_criteria):
        """Test static evaluation failure when no expected answers found."""
        submission = "No flag found in the system"

        result = await _evaluate_static(submission, mock_static_criteria)

        assert result.value == 0.0
        assert result.answer == submission
        assert "does not contain any expected answers" in result.explanation
        assert result.metadata["strategy"] == "static"
        assert result.metadata["scorer_type"] == "saber_client_side"
        assert result.metadata["is_correct"] is False

    async def test_static_evaluation_case_insensitive(self, mock_static_criteria):
        """Test that static evaluation is case-insensitive."""
        submission = "I found FLAG{TEST_FLAG} in the file"

        result = await _evaluate_static(submission, mock_static_criteria)

        assert result.value == 1.0
        assert result.metadata["matched_answer"] == "flag{test_flag}"

    async def test_get_evaluation_criteria_success(self, mock_session_manager, mock_static_criteria):
        """Test successful evaluation criteria retrieval."""
        mock_session_manager.get_evaluation_criteria.return_value = mock_static_criteria

        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_store.return_value = {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "sess_123",
                "saber_current_episode": Mock(episode_id="ep_123")
            }

            state = Mock(spec=TaskState)
            result = await _get_evaluation_criteria(state)

            assert result == mock_static_criteria
            mock_session_manager.get_evaluation_criteria.assert_called_once_with("sess_123", "ep_123")

    async def test_get_evaluation_criteria_missing_context(self):
        """Test that evaluation criteria retrieval fails with missing context."""
        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_store.return_value = {}

            state = Mock(spec=TaskState)

            with pytest.raises(RuntimeError, match="session manager not found"):
                await _get_evaluation_criteria(state)

    @patch('saber.client.inspect_ai.saber_scorer.model_graded_qa')
    async def test_llm_evaluation_success(self, mock_llm_scorer, mock_llm_criteria, mock_task_state):
        """Test successful LLM evaluation."""
        # Mock the inspect_ai LLM scorer result
        mock_scorer_instance = AsyncMock()
        mock_scorer_instance.return_value = Score(
            value="C",
            answer="I found the flag in /etc/passwd: flag{test_flag}",
            explanation="The submission correctly identifies the flag location and value",
            metadata={"confidence": 0.95}
        )
        mock_llm_scorer.return_value = mock_scorer_instance

        submission = "I found the flag in /etc/passwd: flag{test_flag}"

        result = await _evaluate_llm(submission, mock_llm_criteria, mock_task_state)

        assert result.value == 1.0
        assert result.answer == submission
        assert "Client-side LLM evaluation" in result.explanation
        assert result.metadata["strategy"] == "llm_judge"
        assert result.metadata["scorer_type"] == "saber_client_side"
        assert result.metadata["is_correct"] is True
        assert result.metadata["original_grade"] == "C"
        assert result.metadata["model"] == "gpt-4"

    async def test_llm_evaluation_missing_judge_messages(self, mock_llm_criteria, mock_task_state):
        """Test LLM evaluation fails when judge messages are missing."""
        mock_llm_criteria.judge_messages = None
        submission = "Test submission"

        with pytest.raises(RuntimeError, match="LLM evaluation requires judge_messages"):
            await _evaluate_llm(submission, mock_llm_criteria, mock_task_state)

    async def test_client_scorer_integration(self, mock_session_manager, mock_static_criteria, mock_task_state, mock_target):
        """Test full client scorer integration."""
        mock_session_manager.get_evaluation_criteria.return_value = mock_static_criteria

        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_store.return_value = {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "sess_123",
                "saber_current_episode": Mock(episode_id="ep_123")
            }

            scorer_func = saber_scorer()
            result = await scorer_func(mock_task_state, mock_target)

            assert isinstance(result, Score)
            assert result.value == 1.0  # Should match the submission in mock_static_criteria
            assert result.metadata["scorer_type"] == "saber_client_side"

    async def test_client_scorer_empty_submission(self, mock_session_manager, mock_static_criteria, mock_task_state, mock_target):
        """Test client scorer handles empty submission from server."""
        mock_static_criteria.submission = ""
        mock_session_manager.get_evaluation_criteria.return_value = mock_static_criteria

        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_store.return_value = {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "sess_123",
                "saber_current_episode": Mock(episode_id="ep_123")
            }

            scorer_func = saber_scorer()
            result = await scorer_func(mock_task_state, mock_target)

            assert result.value == 0.0
            assert "Empty submission from server" in result.explanation
            assert result.metadata["error"] == "empty_submission"

    async def test_client_scorer_unsupported_strategy(self, mock_session_manager, mock_static_criteria, mock_task_state, mock_target):
        """Test client scorer fails fast with unsupported strategy."""
        mock_static_criteria.evaluation_config["strategy"] = "unknown_strategy"
        mock_session_manager.get_evaluation_criteria.return_value = mock_static_criteria

        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_store.return_value = {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "sess_123",
                "saber_current_episode": Mock(episode_id="ep_123")
            }

            scorer_func = saber_scorer()

            with pytest.raises(RuntimeError, match="Unsupported evaluation strategy: unknown_strategy"):
                await scorer_func(mock_task_state, mock_target)
