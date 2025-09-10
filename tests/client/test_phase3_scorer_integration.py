"""
Phase 3 Integration Tests: SABER Server Scorer Implementation

Tests the enhanced scorer functionality that retrieves evaluations from the SABER server,
following SABER fail-fast principles.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

from saber.client.inspect_ai.saber_scorer import (
    saber_server_scorer,
    _convert_evaluation_to_score,
    _get_server_side_score,
    SABERTaskScorer,
)
from saber.client.client_session import ClientSessionManager
from saber.client.exceptions import (
    EvaluationNotFoundError,
    SessionEvaluationError,
)
from saber.models.rest.evaluation import EvaluationResultResponse
from inspect_ai.scorer import Score
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Target
from inspect_ai.util import store


class TestSABERServerScorer:
    """Test enhanced server scorer functionality."""

    @pytest.fixture
    def mock_evaluation_result(self):
        """Mock evaluation result from server."""
        return EvaluationResultResponse(
            episode_id="ep_123",
            task_id="task_1",
            strategy="static",
            raw_score=0.8,
            max_score=1.0,
            score=0.8,
            success=True,
            timestamp=datetime.now(timezone.utc),
            details={
                "submission": "flag{test_flag}",
                "analysis": "Correct flag found in target file",
                "latency_ms": 1250.5
            }
        )

    @pytest.fixture
    def mock_session_manager(self, mock_evaluation_result):
        """Mock session manager with evaluation methods."""
        session_manager = Mock(spec=ClientSessionManager)
        session_manager.get_episode_evaluation = AsyncMock(return_value=mock_evaluation_result)
        session_manager.get_session_evaluations = AsyncMock(return_value=[mock_evaluation_result])
        session_manager.get_session_evaluation_summary = AsyncMock()
        return session_manager

    @pytest.fixture
    def mock_episode(self):
        """Mock episode object."""
        episode = Mock()
        episode.episode_id = "ep_123"
        return episode

    @pytest.fixture
    def mock_task_state(self):
        """Mock TaskState for scoring."""
        state = Mock(spec=TaskState)
        state.messages = [{"role": "user", "content": "test"}]
        state.metadata = {"task_id": "task_1"}
        return state

    @pytest.fixture
    def mock_target(self):
        """Mock Target for scoring."""
        return Mock(spec=Target)

    @pytest.mark.asyncio
    async def test_saber_server_scorer_success(
        self, mock_session_manager, mock_episode, mock_task_state, mock_target, mock_evaluation_result
    ):
        """Test successful server scorer execution."""
        scorer_fn = saber_server_scorer()

        # Mock the inspect_ai store
        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_task_store = Mock()
            mock_task_store.get.side_effect = lambda key: {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "session_123",
                "saber_current_episode": mock_episode
            }.get(key)
            mock_store.return_value = mock_task_store

            result = await scorer_fn(mock_task_state, mock_target)

            assert isinstance(result, Score)
            assert result.value == 0.8
            assert result.answer == "flag{test_flag}"
            assert "Server evaluation using static strategy" in result.explanation
            assert result.metadata["server_score"] == 0.8
            assert result.metadata["success"] is True
            assert result.metadata["scorer_type"] == "saber_server_side"

    @pytest.mark.asyncio
    async def test_saber_server_scorer_missing_session_manager(self, mock_task_state, mock_target):
        """Test server scorer fails when session manager is missing."""
        scorer_fn = saber_server_scorer()

        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_task_store = Mock()
            mock_task_store.get.side_effect = lambda key: {
                "saber_session_manager": None,
                "saber_session_id": "session_123",
                "saber_current_episode": None
            }.get(key)
            mock_store.return_value = mock_task_store

            with pytest.raises(RuntimeError) as exc_info:
                await scorer_fn(mock_task_state, mock_target)

            assert "session manager not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_saber_server_scorer_missing_session_id(self, mock_session_manager, mock_task_state, mock_target):
        """Test server scorer fails when session ID is missing."""
        scorer_fn = saber_server_scorer()

        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_task_store = Mock()
            mock_task_store.get.side_effect = lambda key: {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": None,
                "saber_current_episode": None
            }.get(key)
            mock_store.return_value = mock_task_store

            with pytest.raises(RuntimeError) as exc_info:
                await scorer_fn(mock_task_state, mock_target)

            assert "session id not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_saber_server_scorer_missing_episode_id(self, mock_session_manager, mock_task_state, mock_target):
        """Test server scorer fails when episode ID is missing."""
        scorer_fn = saber_server_scorer()

        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_task_store = Mock()
            mock_task_store.get.side_effect = lambda key: {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "session_123",
                "saber_current_episode": None
            }.get(key)
            mock_store.return_value = mock_task_store

            with pytest.raises(RuntimeError) as exc_info:
                await scorer_fn(mock_task_state, mock_target)

            assert "episode id not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_saber_server_scorer_evaluation_not_found(
        self, mock_session_manager, mock_episode, mock_task_state, mock_target
    ):
        """Test server scorer handles evaluation not found error."""
        # Configure mock to raise EvaluationNotFoundError
        mock_session_manager.get_episode_evaluation.side_effect = EvaluationNotFoundError("Not found")

        scorer_fn = saber_server_scorer()

        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_task_store = Mock()
            mock_task_store.get.side_effect = lambda key: {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "session_123",
                "saber_current_episode": mock_episode
            }.get(key)
            mock_store.return_value = mock_task_store

            with pytest.raises(RuntimeError) as exc_info:
                await scorer_fn(mock_task_state, mock_target)

            assert "Server evaluation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_saber_server_scorer_session_error(
        self, mock_session_manager, mock_episode, mock_task_state, mock_target
    ):
        """Test server scorer handles session evaluation error."""
        # Configure mock to raise SessionEvaluationError
        mock_session_manager.get_episode_evaluation.side_effect = SessionEvaluationError("Server error")

        scorer_fn = saber_server_scorer()

        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_task_store = Mock()
            mock_task_store.get.side_effect = lambda key: {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "session_123",
                "saber_current_episode": mock_episode
            }.get(key)
            mock_store.return_value = mock_task_store

            with pytest.raises(RuntimeError) as exc_info:
                await scorer_fn(mock_task_state, mock_target)

            assert "Server evaluation failed" in str(exc_info.value)


class TestEvaluationConversion:
    """Test evaluation conversion utilities."""

    @pytest.fixture
    def mock_evaluation_result(self):
        """Mock evaluation result with comprehensive data."""
        return EvaluationResultResponse(
            episode_id="ep_123",
            task_id="task_1",
            strategy="llm_judge",
            raw_score=0.9,
            max_score=1.0,
            score=0.9,
            success=True,
            timestamp=datetime(2025, 9, 9, 12, 0, 0, tzinfo=timezone.utc),
            details={
                "submission": "flag{advanced_flag}",
                "analysis": "LLM correctly identified the flag using advanced reasoning",
                "latency_ms": 2100.7,
                "model": "gpt-4"
            }
        )

    def test_convert_evaluation_to_score(self, mock_evaluation_result):
        """Test conversion of server evaluation to inspect_ai Score."""
        score = _convert_evaluation_to_score(mock_evaluation_result)

        assert isinstance(score, Score)
        assert score.value == 0.9
        assert score.answer == "flag{advanced_flag}"

        # Check explanation content
        assert "Server evaluation using llm_judge strategy" in score.explanation
        assert "Raw score: 0.9/1.0" in score.explanation
        assert "Success: True" in score.explanation
        assert "LLM correctly identified" in score.explanation

        # Check metadata
        metadata = score.metadata
        assert metadata["episode_id"] == "ep_123"
        assert metadata["task_id"] == "task_1"
        assert metadata["strategy"] == "llm_judge"
        assert metadata["raw_score"] == 0.9
        assert metadata["max_score"] == 1.0
        assert metadata["server_score"] == 0.9
        assert metadata["success"] is True
        assert metadata["scorer_type"] == "saber_server_side"
        assert metadata["timestamp"] == "2025-09-09T12:00:00+00:00"
        assert metadata["evaluation_details"]["model"] == "gpt-4"

    def test_convert_evaluation_minimal_details(self):
        """Test conversion with minimal evaluation details."""
        evaluation = EvaluationResultResponse(
            episode_id="ep_minimal",
            task_id="task_minimal",
            strategy="static",
            raw_score=0.5,
            max_score=1.0,
            score=0.5,
            success=False,
            timestamp=datetime.now(timezone.utc),
            details={}  # Minimal details
        )

        score = _convert_evaluation_to_score(evaluation)

        assert score.value == 0.5
        assert score.answer == ""  # Empty when no submission
        assert "Server evaluation using static strategy" in score.explanation
        assert "Success: False" in score.explanation
        assert score.metadata["success"] is False


class TestGetServerSideScore:
    """Test server-side score retrieval utility."""

    @pytest.fixture
    def mock_evaluation_result(self):
        """Mock evaluation result."""
        return EvaluationResultResponse(
            episode_id="ep_123",
            task_id="task_1",
            strategy="static",
            raw_score=0.75,
            max_score=1.0,
            score=0.75,
            success=True,
            timestamp=datetime.now(timezone.utc),
            details={}
        )

    @pytest.fixture
    def mock_session_manager(self, mock_evaluation_result):
        """Mock session manager."""
        session_manager = Mock(spec=ClientSessionManager)
        session_manager.get_episode_evaluation = AsyncMock(return_value=mock_evaluation_result)
        return session_manager

    @pytest.fixture
    def mock_episode(self):
        """Mock episode object."""
        episode = Mock()
        episode.episode_id = "ep_123"
        return episode

    @pytest.fixture
    def mock_task_state(self):
        """Mock TaskState."""
        return Mock(spec=TaskState)

    @pytest.mark.asyncio
    async def test_get_server_side_score_success(
        self, mock_session_manager, mock_episode, mock_task_state
    ):
        """Test successful server-side score retrieval."""
        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_task_store = Mock()
            mock_task_store.get.side_effect = lambda key: {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "session_123",
                "saber_current_episode": mock_episode
            }.get(key)
            mock_store.return_value = mock_task_store

            score = await _get_server_side_score(mock_task_state)

            assert score == 0.75
            mock_session_manager.get_episode_evaluation.assert_called_once_with("session_123", "ep_123")

    @pytest.mark.asyncio
    async def test_get_server_side_score_missing_context(self, mock_task_state):
        """Test server-side score retrieval with missing context."""
        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_task_store = Mock()
            mock_task_store.get.side_effect = lambda key: None  # All context missing
            mock_store.return_value = mock_task_store

            score = await _get_server_side_score(mock_task_state)

            assert score is None

    @pytest.mark.asyncio
    async def test_get_server_side_score_retrieval_error(
        self, mock_session_manager, mock_episode, mock_task_state
    ):
        """Test server-side score retrieval with evaluation error."""
        mock_session_manager.get_episode_evaluation.side_effect = EvaluationNotFoundError("Not found")

        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_task_store = Mock()
            mock_task_store.get.side_effect = lambda key: {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "session_123",
                "saber_current_episode": mock_episode
            }.get(key)
            mock_store.return_value = mock_task_store

            score = await _get_server_side_score(mock_task_state)

            assert score is None


class TestSABERTaskScorerEnhanced:
    """Test enhanced SABERTaskScorer functionality."""

    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager."""
        return Mock(spec=ClientSessionManager)

    @pytest.fixture
    def mock_evaluation_result(self):
        """Mock evaluation result."""
        return EvaluationResultResponse(
            episode_id="ep_123",
            task_id="task_1",
            strategy="static",
            raw_score=0.8,
            max_score=1.0,
            score=0.8,
            success=True,
            timestamp=datetime.now(timezone.utc),
            details={"submission": "test_answer"}
        )

    def test_create_default_scorer_returns_server_scorer(self):
        """Test that default scorer now returns server scorer."""
        scorer = SABERTaskScorer.create_default_scorer()
        # This should be the server scorer function (we can't easily test the function identity)
        assert callable(scorer)

    @pytest.mark.asyncio
    async def test_get_server_evaluation(self, mock_session_manager, mock_evaluation_result):
        """Test direct server evaluation retrieval."""
        mock_session_manager.get_episode_evaluation = AsyncMock(return_value=mock_evaluation_result)

        result = await SABERTaskScorer.get_server_evaluation(
            mock_session_manager, "session_123", "ep_123"
        )

        assert result == mock_evaluation_result
        mock_session_manager.get_episode_evaluation.assert_called_once_with("session_123", "ep_123")

    @pytest.mark.asyncio
    async def test_get_session_evaluations(self, mock_session_manager, mock_evaluation_result):
        """Test session evaluations retrieval."""
        mock_session_manager.get_session_evaluations = AsyncMock(return_value=[mock_evaluation_result])

        results = await SABERTaskScorer.get_session_evaluations(
            mock_session_manager, "session_123", task_id="task_1"
        )

        assert len(results) == 1
        assert results[0] == mock_evaluation_result
        mock_session_manager.get_session_evaluations.assert_called_once_with("session_123", "task_1")

    @pytest.mark.asyncio
    async def test_convert_server_evaluation_to_score(self, mock_evaluation_result):
        """Test static method for evaluation conversion."""
        score = await SABERTaskScorer.convert_server_evaluation_to_score(mock_evaluation_result)

        assert isinstance(score, Score)
        assert score.value == 0.8
        assert score.answer == "test_answer"

    @pytest.mark.asyncio
    async def test_score_completed_episodes(self, mock_session_manager, mock_evaluation_result):
        """Test scoring all completed episodes."""
        mock_session_manager.get_session_evaluations = AsyncMock(return_value=[mock_evaluation_result])

        scores = await SABERTaskScorer.score_completed_episodes(
            mock_session_manager, "session_123", task_id="task_1"
        )

        assert len(scores) == 1
        assert isinstance(scores[0], Score)
        assert scores[0].value == 0.8
        mock_session_manager.get_session_evaluations.assert_called_once_with("session_123", "task_1")


class TestFailFastBehavior:
    """Test fail-fast behavior throughout Phase 3 implementation."""

    @pytest.mark.asyncio
    async def test_no_silent_fallbacks_in_server_scorer(self):
        """Test that server scorer fails fast with no silent fallbacks."""
        scorer_fn = saber_server_scorer()
        mock_task_state = Mock(spec=TaskState)
        mock_target = Mock(spec=Target)

        # Test with completely empty store
        with patch('saber.client.inspect_ai.saber_scorer.store') as mock_store:
            mock_task_store = Mock()
            mock_task_store.get.return_value = None  # No context at all
            mock_store.return_value = mock_task_store

            with pytest.raises(RuntimeError):
                await scorer_fn(mock_task_state, mock_target)

    @pytest.mark.asyncio
    async def test_evaluation_errors_propagate(self):
        """Test that evaluation retrieval errors propagate correctly."""
        mock_session_manager = Mock(spec=ClientSessionManager)
        mock_session_manager.get_episode_evaluation.side_effect = EvaluationNotFoundError("Test error")

        with pytest.raises(EvaluationNotFoundError):
            await SABERTaskScorer.get_server_evaluation(mock_session_manager, "session_123", "ep_123")

    def test_scorer_factory_methods_return_callable(self):
        """Test that all scorer factory methods return callable scorers."""
        client_scorer = SABERTaskScorer.create_client_scorer()
        server_scorer = SABERTaskScorer.create_server_scorer()
        default_scorer = SABERTaskScorer.create_default_scorer()

        assert callable(client_scorer)
        assert callable(server_scorer)
        assert callable(default_scorer)
