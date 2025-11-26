"""Integration tests for scorer orchestration coordination.

Tests the full flow of scorer blocking until all samples complete.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from saber.inspect_ai.core.orchestration_coordinator import OrchestrationCoordinator
from saber.inspect_ai.core.saber_scorer import saber_scorer
from saber.models.constants import MetadataKeys


class TestScorerOrchestrationIntegration:
    """Test scorer integration with orchestration coordinator."""

    @pytest.fixture
    def coordinator(self):
        """Fresh coordinator."""
        coord = OrchestrationCoordinator()
        coord.reset()
        return coord

    @pytest.fixture
    def mock_state_blue(self):
        """Mock TaskState for blue team sample."""
        state = MagicMock(spec=TaskState)
        state.metadata = {
            MetadataKeys.ORCHESTRATION_ID: "test-orch",
            MetadataKeys.SUB_TASK_ROLE: "blue",
            MetadataKeys.SAMPLE_ID: "sample-blue",
        }
        state.output = MagicMock()
        state.output.completion = "Blue team completed"
        state.model = "gpt-4"
        state.messages = []
        return state

    @pytest.fixture
    def mock_state_red(self):
        """Mock TaskState for red team sample."""
        state = MagicMock(spec=TaskState)
        state.metadata = {
            MetadataKeys.ORCHESTRATION_ID: "test-orch",
            MetadataKeys.SUB_TASK_ROLE: "red",
            MetadataKeys.SAMPLE_ID: "sample-red",
        }
        state.output = MagicMock()
        state.output.completion = "Red team completed"
        state.model = "gpt-4"
        state.messages = []
        return state

    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager with all required methods."""
        from saber.models.constants import SubmissionEvaluationStrategy

        mock = AsyncMock()
        mock.post_episode_submission = AsyncMock()
        mock.get_episode_submission = AsyncMock(
            return_value=MagicMock(submission="Test submission")
        )
        mock.get_episode_steps = AsyncMock(return_value=MagicMock(steps=[]))
        mock.get_submission_evaluation_criteria = AsyncMock(
            return_value=MagicMock(
                task_id="task-1",
                scoring={"max_score": 1.0},
                task_context={},
                strategy=SubmissionEvaluationStrategy.STATIC,  # Use valid strategy enum
                model=None,
                template_path=None,
            )
        )
        mock.get_subtask_evaluation_criteria = AsyncMock(return_value=[])
        mock.submit_evaluation_result = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_non_orchestrated_sample_no_blocking(
        self, coordinator, mock_session_manager
    ):
        """Non-orchestrated sample should not block in scorer."""
        scorer_fn = saber_scorer()

        # State without orchestration metadata
        state = MagicMock(spec=TaskState)
        state.metadata = {MetadataKeys.SAMPLE_ID: "solo-sample"}
        state.output = MagicMock()
        state.output.completion = "Done"
        state.model = "gpt-4"
        state.messages = []

        # Mock dependencies
        with patch("saber.inspect_ai.core.saber_scorer.store") as mock_store:
            mock_store.return_value = {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "test-session",
                "saber_episode_mapping": {
                    "solo-sample": MagicMock(episode_id="ep-1", task_id="task-1")
                },
            }

            start = asyncio.get_event_loop().time()
            score = await scorer_fn(state, Target(target="test"))
            elapsed = asyncio.get_event_loop().time() - start

            assert elapsed < 2.0  # Should be reasonably fast (no coordination wait)
            assert score is not None

    @pytest.mark.asyncio
    async def test_orchestrated_samples_coordinate(
        self, coordinator, mock_state_blue, mock_state_red, mock_session_manager
    ):
        """Orchestrated samples should coordinate before returning scores."""
        # Register orchestration
        coordinator.register_root_sample("test-orch", "blue", "sample-blue")
        await coordinator.register_dependent_sample(
            "test-orch", "red", "sample-red", "blue", 2
        )

        scorer_fn = saber_scorer()

        # Mock dependencies for both states
        with patch("saber.inspect_ai.core.saber_scorer.store") as mock_store:
            mock_store.return_value = {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "test-session",
                "saber_episode_mapping": {
                    "sample-blue": MagicMock(episode_id="ep-blue", task_id="task-blue"),
                    "sample-red": MagicMock(episode_id="ep-red", task_id="task-red"),
                },
            }

            # Blue scores first - should block
            blue_task = asyncio.create_task(scorer_fn(mock_state_blue, Target(target="test")))
            await asyncio.sleep(0.3)

            assert not blue_task.done(), "Blue should be waiting for red"

            # Red scores - should unblock both
            red_task = asyncio.create_task(scorer_fn(mock_state_red, Target(target="test")))

            # Both should complete
            blue_score = await asyncio.wait_for(blue_task, timeout=10.0)
            red_score = await asyncio.wait_for(red_task, timeout=10.0)

            assert blue_score is not None
            assert red_score is not None

    @pytest.mark.asyncio
    async def test_orchestration_timeout_handled_gracefully(
        self, coordinator, mock_state_blue, mock_session_manager
    ):
        """Timeout in orchestration should be handled gracefully."""
        # Register orchestration with 2 samples (but we'll only score blue)
        coordinator.register_root_sample("test-orch", "blue", "sample-blue")
        await coordinator.register_dependent_sample(
            "test-orch", "red", "sample-red", "blue", 2
        )

        scorer_fn = saber_scorer()

        with patch("saber.inspect_ai.core.saber_scorer.store") as mock_store:
            mock_store.return_value = {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "test-session",
                "saber_episode_mapping": {
                    "sample-blue": MagicMock(episode_id="ep-blue", task_id="task-blue"),
                },
            }

            # Patch wait_for_all_scored to use short timeout
            with patch.object(
                coordinator, "wait_for_all_scored", wraps=coordinator.wait_for_all_scored
            ) as mock_wait:

                # Score blue with short timeout - will timeout waiting for red
                # The scorer should catch the timeout and proceed
                try:
                    # Override timeout in the actual call
                    original_wait = coordinator.wait_for_all_scored

                    async def short_timeout_wait(*args, **kwargs):
                        kwargs["timeout"] = 0.5  # Very short timeout
                        return await original_wait(*args, **kwargs)

                    coordinator.wait_for_all_scored = short_timeout_wait

                    # This should timeout but the scorer should handle it
                    # Note: The scorer needs to catch TimeoutError
                    score = await scorer_fn(mock_state_blue, Target(target="test"))

                    # Score might be 0 or valid depending on handling
                    assert score is not None
                finally:
                    # Restore original method
                    coordinator.wait_for_all_scored = original_wait

    @pytest.mark.asyncio
    async def test_three_samples_all_coordinate(
        self, coordinator, mock_session_manager
    ):
        """Three-way orchestration should coordinate correctly."""
        # Register three samples
        coordinator.register_root_sample("test-orch-3", "attacker", "sample-1")
        await coordinator.register_dependent_sample(
            "test-orch-3", "defender", "sample-2", "attacker", 2
        )
        await coordinator.register_dependent_sample(
            "test-orch-3", "monitor", "sample-3", "defender", 3
        )

        scorer_fn = saber_scorer()

        # Create mock states
        def create_state(role, sample_id):
            state = MagicMock(spec=TaskState)
            state.metadata = {
                MetadataKeys.ORCHESTRATION_ID: "test-orch-3",
                MetadataKeys.SUB_TASK_ROLE: role,
                MetadataKeys.SAMPLE_ID: sample_id,
            }
            state.output = MagicMock()
            state.output.completion = f"{role} completed"
            state.model = "gpt-4"
            state.messages = []
            return state

        state_attacker = create_state("attacker", "sample-1")
        state_defender = create_state("defender", "sample-2")
        state_monitor = create_state("monitor", "sample-3")

        with patch("saber.inspect_ai.core.saber_scorer.store") as mock_store:
            mock_store.return_value = {
                "saber_session_manager": mock_session_manager,
                "saber_session_id": "test-session",
                "saber_episode_mapping": {
                    "sample-1": MagicMock(episode_id="ep-1", task_id="task-1"),
                    "sample-2": MagicMock(episode_id="ep-2", task_id="task-2"),
                    "sample-3": MagicMock(episode_id="ep-3", task_id="task-3"),
                },
            }

            # Start all three scorers
            attacker_task = asyncio.create_task(scorer_fn(state_attacker, Target(target="test")))
            await asyncio.sleep(0.3)

            defender_task = asyncio.create_task(scorer_fn(state_defender, Target(target="test")))
            await asyncio.sleep(0.3)

            # Both should still be waiting
            assert not attacker_task.done(), "Attacker should be waiting"
            assert not defender_task.done(), "Defender should be waiting"

            # Monitor completes last - releases all
            monitor_task = asyncio.create_task(scorer_fn(state_monitor, Target(target="test")))

            # All should complete
            attacker_score = await asyncio.wait_for(attacker_task, timeout=10.0)
            defender_score = await asyncio.wait_for(defender_task, timeout=10.0)
            monitor_score = await asyncio.wait_for(monitor_task, timeout=10.0)

            assert attacker_score is not None
            assert defender_score is not None
            assert monitor_score is not None
