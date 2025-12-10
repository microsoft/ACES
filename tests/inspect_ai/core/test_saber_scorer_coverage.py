"""Additional tests for saber_scorer to improve coverage.

Focuses on missing coverage lines identified in coverage report:
- Lines 234, 240, 282, 288, 338, 344, 363-364
- Lines 468-472, 475-479, 482-486
- Lines 572-596, 603-608, 640, 673, 677, 691-704, 708
- Lines 772-787, 1031, 1116, 1194
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import SampleScore, Score
from inspect_ai.solver import TaskState

from saber.inspect_ai.core.saber_scorer import (
    per_task_submission_scores,
    per_task_subtask_scores,
    saber_scorer,
    subtask_score_metrics,
)
from saber.models.constants import MetadataKeys


@pytest.fixture
def mock_state():
    """Create mock TaskState."""
    state = MagicMock(spec=TaskState)
    state.metadata = {
        MetadataKeys.SAMPLE_ID: "sample-1",
        MetadataKeys.TASK_ID: "task-1",
    }
    state.output = MagicMock()
    state.output.completion = "test answer"
    state.output.usage = None
    state.model = "test-model"
    state.messages = []
    return state


@pytest.fixture
def mock_store():
    """Create mock store with SABER context."""
    store_dict = {
        "saber_session_manager": MagicMock(),
        "saber_session_id": "session-123",
        "saber_task_id": "task-1",
        "saber_episode_mapping": {
            "sample-1": MagicMock(episode_id="episode-456", task_id="task-1")
        },
    }
    return MagicMock(get=lambda key, default=None: store_dict.get(key, default))


class TestPerTaskMetrics:
    """Test per-task metric functions for edge cases."""

    def test_per_task_submission_scores_without_metadata(self):
        """Test per_task_submission_scores with samples without metadata."""
        metric_fn = per_task_submission_scores()

        # Sample without score.metadata
        sample1 = MagicMock()
        sample1.score = Score(value=0.5, answer="test")
        sample1.score.metadata = None  # Line 234
        sample1.sample_metadata = {MetadataKeys.TASK_ID: "task-1"}

        scores = [sample1]
        result = metric_fn(scores)

        # Should return empty dict when no metadata
        assert result == {}

    def test_per_task_submission_scores_without_task_id(self):
        """Test per_task_submission_scores with samples without task_id."""
        metric_fn = per_task_submission_scores()

        # Sample with metadata but no task_id in sample_metadata
        sample1 = MagicMock()
        sample1.score = Score(value=0.5, answer="test", metadata={MetadataKeys.SUBMISSION_SCORE: 0.8})
        sample1.sample_metadata = {}  # Line 240 - no task_id

        scores = [sample1]
        result = metric_fn(scores)

        # Should return empty dict
        assert result == {}

    def test_per_task_subtask_scores_without_metadata(self):
        """Test per_task_subtask_scores with samples without metadata."""
        metric_fn = per_task_subtask_scores()

        # Sample without score.metadata
        sample1 = MagicMock()
        sample1.score = Score(value=0.5, answer="test")
        sample1.score.metadata = None  # Line 282
        sample1.sample_metadata = {MetadataKeys.TASK_ID: "task-1"}

        scores = [sample1]
        result = metric_fn(scores)

        # Should return empty dict
        assert result == {}

    def test_per_task_subtask_scores_without_task_id(self):
        """Test per_task_subtask_scores without task_id."""
        metric_fn = per_task_subtask_scores()

        # Sample with metadata but no task_id
        sample1 = MagicMock()
        sample1.score = Score(value=0.5, answer="test", metadata={MetadataKeys.SUBTASK_SCORE: 0.7})
        sample1.sample_metadata = {}  # Line 288 - no task_id

        scores = [sample1]
        result = metric_fn(scores)

        # Should return empty dict
        assert result == {}

    def test_subtask_score_metrics_without_metadata(self):
        """Test subtask_score_metrics with samples without metadata."""
        metric_fn = subtask_score_metrics()

        # Sample without score.metadata
        sample1 = MagicMock()
        sample1.score = Score(value=0.5, answer="test")
        sample1.score.metadata = None  # Line 338
        sample1.sample_metadata = {MetadataKeys.TASK_ID: "task-1"}

        scores = [sample1]
        result = metric_fn(scores)

        # Should return empty dict
        assert result == {}

    def test_subtask_score_metrics_without_task_id(self):
        """Test subtask_score_metrics without task_id."""
        metric_fn = subtask_score_metrics()

        # Sample with metadata but no task_id
        sample1 = MagicMock()
        sample1.score = Score(value=0.5, answer="test", metadata={MetadataKeys.SUBTASK_SCORES: {"sub1": 1.0}})
        sample1.sample_metadata = {}  # Line 344 - no task_id

        scores = [sample1]
        result = metric_fn(scores)

        # Should return empty dict
        assert result == {}

    def test_subtask_score_metrics_with_valid_data(self):
        """Test subtask_score_metrics with valid data to ensure normal path works."""
        metric_fn = subtask_score_metrics()

        # Sample with valid data
        sample1 = MagicMock()
        sample1.score = Score(value=0.5, answer="test", metadata={MetadataKeys.SUBTASK_SCORES: {"subtask1": 1.0}})
        sample1.sample_metadata = {MetadataKeys.TASK_ID: "task-1"}

        scores = [sample1]
        result = metric_fn(scores)

        # Should return results with sanitized task and subtask IDs
        assert "task_1_subtask1_score" in result
        assert result["task_1_subtask1_score"] == 1.0


class TestScorerErrorPaths:
    """Test error handling paths in main scorer."""

    @pytest.mark.asyncio
    async def test_scorer_missing_session_manager(self, mock_state):
        """Test scorer when session_manager is missing (line 468-472)."""
        with patch("saber.inspect_ai.core.saber_scorer.store") as mock_store_fn:
            store = MagicMock()
            store.get = MagicMock(side_effect=lambda key, default=None: {
                "saber_session_manager": None,  # Missing!
                "saber_session_id": "session-123",
                "saber_task_id": "task-1",
            }.get(key, default))
            mock_store_fn.return_value = store

            scorer = saber_scorer()
            result = await scorer(mock_state, MagicMock())

            # Should return error score
            assert result.value == 0.0
            assert "Missing saber_session_manager" in result.explanation

    @pytest.mark.asyncio
    async def test_scorer_missing_session_id(self, mock_state):
        """Test scorer when session_id is missing (line 475-479)."""
        with patch("saber.inspect_ai.core.saber_scorer.store") as mock_store_fn:
            store = MagicMock()
            store.get = MagicMock(side_effect=lambda key, default=None: {
                "saber_session_manager": MagicMock(),
                "saber_session_id": None,  # Missing!
                "saber_task_id": "task-1",
            }.get(key, default))
            mock_store_fn.return_value = store

            scorer = saber_scorer()
            result = await scorer(mock_state, MagicMock())

            # Should return error score
            assert result.value == 0.0
            assert "Missing saber_session_id" in result.explanation

    @pytest.mark.asyncio
    async def test_scorer_missing_episode_id(self, mock_state):
        """Test scorer when episode_id is missing (line 482-486)."""
        with patch("saber.inspect_ai.core.saber_scorer.store") as mock_store_fn:
            store = MagicMock()
            store.get = MagicMock(side_effect=lambda key, default=None: {
                "saber_session_manager": MagicMock(),
                "saber_session_id": "session-123",
                "saber_task_id": "task-1",
                "saber_episode_mapping": {},  # Empty - no episode_id
            }.get(key, default))
            mock_store_fn.return_value = store

            scorer = saber_scorer()
            result = await scorer(mock_state, MagicMock())

            # Should return error score
            assert result.value == 0.0
            assert "Missing episode_id" in result.explanation

    @pytest.mark.asyncio
    async def test_scorer_orchestration_coordination_success(self, mock_state, mock_store):
        """Test scorer with orchestration coordination (lines 1031, 1116)."""
        # Add orchestration metadata
        mock_state.metadata[MetadataKeys.ORCHESTRATION_ID] = "orch-123"
        mock_state.metadata[MetadataKeys.SUB_TASK_ROLE] = "blue"

        # Mock all the required components
        mock_session_manager = MagicMock()
        mock_session_manager.post_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission.return_value = MagicMock(submission="test answer")
        mock_session_manager.get_episode_steps = AsyncMock()
        mock_session_manager.get_episode_steps.return_value = MagicMock(steps=[])
        mock_session_manager.get_submission_evaluation_criteria = AsyncMock()
        mock_session_manager.get_submission_evaluation_criteria.return_value = MagicMock(
            task_id="task-1",
            strategy="static",
            criteria={"expected_answers": ["test"]},
            scoring={"max_score": 1.0},
            task_context=MagicMock(description="Test", domain="test"),
        )
        mock_session_manager.get_subtask_evaluation_criteria = AsyncMock()
        mock_session_manager.get_subtask_evaluation_criteria.return_value = []

        mock_store.get = MagicMock(side_effect=lambda key, default=None: {
            "saber_session_manager": mock_session_manager,
            "saber_session_id": "session-123",
            "saber_task_id": "task-1",
            "saber_episode_mapping": {
                "sample-1": MagicMock(episode_id="episode-456", task_id="task-1")
            },
        }.get(key, default))

        with patch("saber.inspect_ai.core.saber_scorer.store", return_value=mock_store):
            with patch("saber.inspect_ai.core.orchestration_coordinator.OrchestrationCoordinator") as mock_coordinator_class:
                mock_coordinator = MagicMock()
                mock_coordinator.wait_for_all_scored = AsyncMock()
                mock_coordinator_class.return_value = mock_coordinator

                scorer = saber_scorer()
                result = await scorer(mock_state, MagicMock())

                # Should have called coordination
                mock_coordinator.wait_for_all_scored.assert_called_once()
                assert result.value >= 0.0

    @pytest.mark.asyncio
    async def test_scorer_orchestration_coordination_timeout(self, mock_state, mock_store):
        """Test scorer with orchestration coordination timeout (line 1116)."""
        # Add orchestration metadata
        mock_state.metadata[MetadataKeys.ORCHESTRATION_ID] = "orch-123"
        mock_state.metadata[MetadataKeys.SUB_TASK_ROLE] = "red"

        # Mock all the required components
        mock_session_manager = MagicMock()
        mock_session_manager.post_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission.return_value = MagicMock(submission="test answer")
        mock_session_manager.get_episode_steps = AsyncMock()
        mock_session_manager.get_episode_steps.return_value = MagicMock(steps=[])
        mock_session_manager.get_submission_evaluation_criteria = AsyncMock()
        mock_session_manager.get_submission_evaluation_criteria.return_value = MagicMock(
            task_id="task-1",
            strategy="static",
            criteria={"expected_answers": ["test"]},
            scoring={"max_score": 1.0},
            task_context=MagicMock(description="Test", domain="test"),
        )
        mock_session_manager.get_subtask_evaluation_criteria = AsyncMock()
        mock_session_manager.get_subtask_evaluation_criteria.return_value = []

        mock_store.get = MagicMock(side_effect=lambda key, default=None: {
            "saber_session_manager": mock_session_manager,
            "saber_session_id": "session-123",
            "saber_task_id": "task-1",
            "saber_episode_mapping": {
                "sample-1": MagicMock(episode_id="episode-456", task_id="task-1")
            },
        }.get(key, default))

        with patch("saber.inspect_ai.core.saber_scorer.store", return_value=mock_store):
            with patch("saber.inspect_ai.core.orchestration_coordinator.OrchestrationCoordinator") as mock_coordinator_class:
                mock_coordinator = MagicMock()
                # Simulate timeout
                mock_coordinator.wait_for_all_scored = AsyncMock(side_effect=asyncio.TimeoutError())
                mock_coordinator_class.return_value = mock_coordinator

                scorer = saber_scorer()
                result = await scorer(mock_state, MagicMock())

                # Should still return a score despite timeout
                assert result.value >= 0.0

    @pytest.mark.asyncio
    async def test_scorer_with_token_usage(self, mock_state, mock_store):
        """Test scorer with token usage in state.output (lines 572-596)."""
        # Add token usage
        mock_usage = MagicMock()
        mock_usage.model_dump.return_value = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "some_none_value": None,  # Test None conversion to 0
        }
        mock_state.output.usage = mock_usage

        # Mock all the required components
        mock_session_manager = MagicMock()
        mock_session_manager.post_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission.return_value = MagicMock(submission="test answer")
        mock_session_manager.get_episode_steps = AsyncMock()
        mock_session_manager.get_episode_steps.return_value = MagicMock(steps=[])
        mock_session_manager.get_submission_evaluation_criteria = AsyncMock()
        mock_session_manager.get_submission_evaluation_criteria.return_value = MagicMock(
            task_id="task-1",
            strategy="static",
            criteria={"expected_answers": ["test"]},
            scoring={"max_score": 1.0},
            task_context=MagicMock(description="Test", domain="test"),
        )
        mock_session_manager.get_subtask_evaluation_criteria = AsyncMock()
        mock_session_manager.get_subtask_evaluation_criteria.return_value = []

        mock_store.get = MagicMock(side_effect=lambda key, default=None: {
            "saber_session_manager": mock_session_manager,
            "saber_session_id": "session-123",
            "saber_task_id": "task-1",
            "saber_episode_mapping": {
                "sample-1": MagicMock(episode_id="episode-456", task_id="task-1")
            },
        }.get(key, default))

        with patch("saber.inspect_ai.core.saber_scorer.store", return_value=mock_store):
            scorer = saber_scorer()
            result = await scorer(mock_state, MagicMock())

            # Verify submission was posted with token usage
            call_args = mock_session_manager.post_episode_submission.call_args[0]
            submission = call_args[2]  # Third argument is the EvalSubmission
            assert submission.tokens["input_tokens"] == 100
            assert submission.tokens["output_tokens"] == 50
            assert submission.tokens["total_tokens"] == 150
            assert submission.tokens["some_none_value"] == 0  # None converted to 0

    @pytest.mark.asyncio
    async def test_scorer_without_output(self, mock_state, mock_store):
        """Test scorer when state.output is None (line 603-608)."""
        mock_state.output = None  # No output

        # Mock all the required components
        mock_session_manager = MagicMock()
        mock_session_manager.post_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission.return_value = MagicMock(submission="")
        mock_session_manager.get_episode_steps = AsyncMock()
        mock_session_manager.get_episode_steps.return_value = MagicMock(steps=[])
        mock_session_manager.get_submission_evaluation_criteria = AsyncMock()
        mock_session_manager.get_submission_evaluation_criteria.return_value = MagicMock(
            task_id="task-1",
            strategy="static",
            criteria={"expected_answers": ["test"]},
            scoring={"max_score": 1.0},
            task_context=MagicMock(description="Test", domain="test"),
        )
        mock_session_manager.get_subtask_evaluation_criteria = AsyncMock()
        mock_session_manager.get_subtask_evaluation_criteria.return_value = []

        mock_store.get = MagicMock(side_effect=lambda key, default=None: {
            "saber_session_manager": mock_session_manager,
            "saber_session_id": "session-123",
            "saber_task_id": "task-1",
            "saber_episode_mapping": {
                "sample-1": MagicMock(episode_id="episode-456", task_id="task-1")
            },
        }.get(key, default))

        with patch("saber.inspect_ai.core.saber_scorer.store", return_value=mock_store):
            scorer = saber_scorer()
            result = await scorer(mock_state, MagicMock())

            # Should handle gracefully
            assert result.value >= 0.0

    @pytest.mark.asyncio
    async def test_scorer_without_scorable_subtasks(self, mock_state, mock_store):
        """Test scorer when subtasks exist but have no evaluation strategy (lines 640, 677, 691-704, 708)."""
        # Mock all the required components
        mock_session_manager = MagicMock()
        mock_session_manager.post_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission.return_value = MagicMock(submission="test answer")
        mock_session_manager.get_episode_steps = AsyncMock()
        mock_session_manager.get_episode_steps.return_value = MagicMock(steps=[])
        mock_session_manager.get_submission_evaluation_criteria = AsyncMock()
        mock_session_manager.get_submission_evaluation_criteria.return_value = MagicMock(
            task_id="task-1",
            strategy="static",
            criteria={"expected_answers": ["test answer"]},  # Match to get score
            scoring={"max_score": 1.0},
            task_context=MagicMock(description="Test", domain="test"),
        )
        # Return subtasks with empty/None strategy (informational only)
        mock_session_manager.get_subtask_evaluation_criteria = AsyncMock()
        mock_session_manager.get_subtask_evaluation_criteria.return_value = [
            MagicMock(
                subtask_id="info1",
                strategy="",  # Empty strategy - informational only
                max_score=1.0,
                weight=0.5,
            ),
            MagicMock(
                subtask_id="info2",
                strategy=None,  # None strategy - informational only
                max_score=1.0,
                weight=0.5,
            ),
        ]

        mock_store.get = MagicMock(side_effect=lambda key, default=None: {
            "saber_session_manager": mock_session_manager,
            "saber_session_id": "session-123",
            "saber_task_id": "task-1",
            "saber_episode_mapping": {
                "sample-1": MagicMock(episode_id="episode-456", task_id="task-1")
            },
        }.get(key, default))

        with patch("saber.inspect_ai.core.saber_scorer.store", return_value=mock_store):
            scorer = saber_scorer()
            result = await scorer(mock_state, MagicMock())

            # Should not include subtask metadata when no scorable subtasks
            assert result.value == 1.0  # Only submission score
            assert MetadataKeys.SUBTASK_SCORE not in result.metadata
            assert MetadataKeys.WEIGHTED_SUBTASK_SCORE not in result.metadata
            assert MetadataKeys.STEP_EVALUATIONS not in result.metadata
            # Explanation should not mention subtasks
            assert "weighted_subtasks" not in result.explanation

    @pytest.mark.asyncio
    async def test_scorer_with_scorable_subtasks(self, mock_state, mock_store):
        """Test scorer when subtasks have evaluation strategies (lines 772-787)."""
        # Mock all the required components
        mock_session_manager = MagicMock()
        mock_session_manager.post_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission = AsyncMock()
        mock_session_manager.get_episode_submission.return_value = MagicMock(submission="test answer")
        mock_session_manager.get_episode_steps = AsyncMock()
        mock_session_manager.get_episode_steps.return_value = MagicMock(steps=[])
        mock_session_manager.get_submission_evaluation_criteria = AsyncMock()
        mock_session_manager.get_submission_evaluation_criteria.return_value = MagicMock(
            task_id="task-1",
            strategy="static",
            criteria={"expected_answers": ["test answer"]},
            scoring={"max_score": 1.0},
            task_context=MagicMock(description="Test", domain="test"),
        )
        # Return subtasks with valid strategies
        mock_session_manager.get_subtask_evaluation_criteria = AsyncMock()
        mock_session_manager.get_subtask_evaluation_criteria.return_value = [
            MagicMock(
                subtask_id="sub1",
                strategy="static",  # Valid strategy
                max_score=1.0,
                weight=0.5,
                description="Test subtask",
                objective="Test objective",
                title="Test title",
                task_id="task-1",
                criteria={"expected_outputs": ["expected"]},
            ),
        ]

        mock_store.get = MagicMock(side_effect=lambda key, default=None: {
            "saber_session_manager": mock_session_manager,
            "saber_session_id": "session-123",
            "saber_task_id": "task-1",
            "saber_episode_mapping": {
                "sample-1": MagicMock(episode_id="episode-456", task_id="task-1")
            },
        }.get(key, default))

        with patch("saber.inspect_ai.core.saber_scorer.store", return_value=mock_store):
            scorer = saber_scorer()
            result = await scorer(mock_state, MagicMock())

            # Should include subtask metadata when scorable subtasks exist
            assert MetadataKeys.SUBTASK_SCORE in result.metadata
            assert MetadataKeys.WEIGHTED_SUBTASK_SCORE in result.metadata
            assert MetadataKeys.STEP_EVALUATIONS in result.metadata
            # Explanation should mention subtasks
            assert "weighted_subtasks" in result.explanation

    @pytest.mark.asyncio
    async def test_scorer_general_exception(self, mock_state):
        """Test scorer with general exception (line 1194)."""
        with patch("saber.inspect_ai.core.saber_scorer.store") as mock_store_fn:
            # Make store().get() raise an exception
            mock_store_fn.return_value.get.side_effect = Exception("Unexpected error")

            scorer = saber_scorer()
            result = await scorer(mock_state, MagicMock())

            # Should return error score
            assert result.value == 0.0
            assert "Client-side evaluation failed" in result.explanation
            assert "Unexpected error" in result.explanation
