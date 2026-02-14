"""Unit tests for OrchestrationCoordinator.

These tests directly test the OrchestrationCoordinator class implementation,
including singleton behavior, registration, dependency tracking, and cleanup.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saber.inspect_ai.core.orchestration_coordinator import OrchestrationCoordinator


@pytest.fixture(autouse=True)
def reset_coordinator():
    """Reset coordinator singleton before each test."""
    coordinator = OrchestrationCoordinator()
    coordinator.reset()
    yield
    coordinator.reset()


class TestSingletonBehavior:
    """Test OrchestrationCoordinator singleton pattern."""

    def test_singleton_instance(self):
        """Test that OrchestrationCoordinator returns same instance."""
        coord1 = OrchestrationCoordinator()
        coord2 = OrchestrationCoordinator()

        assert coord1 is coord2

    def test_singleton_state_shared(self):
        """Test that state is shared across instances."""
        coord1 = OrchestrationCoordinator()
        coord1.register_root_sample("orch_1", "blue", "sample_1")

        coord2 = OrchestrationCoordinator()
        # Should be able to access the same orchestration
        result = coord2.cleanup_sample("orch_1", "blue")

        assert result is True  # Last sample removed


class TestRegisterRootSample:
    """Test register_root_sample method."""

    def test_register_root_sample_success(self):
        """Test successful root sample registration."""
        coordinator = OrchestrationCoordinator()

        result = coordinator.register_root_sample(
            orchestration_id="orch_123",
            role="blue",
            sample_id="sample_blue",
        )

        assert result is True

    def test_register_root_sample_creates_orchestration_group(self):
        """Test that registering root sample creates orchestration group."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        # Verify group was created by checking we can register dependent
        result = asyncio.run(coordinator.register_dependent_sample(
            "orch_123", "red", "sample_red", "blue", 2
        ))
        assert result is True

    def test_register_root_sample_duplicate_fails(self):
        """Test that registering same role twice fails."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        result = coordinator.register_root_sample("orch_123", "blue", "sample_blue_2")

        assert result is False


class TestRegisterDependentSample:
    """Test register_dependent_sample method."""

    @pytest.mark.asyncio
    async def test_register_dependent_sample_success(self):
        """Test successful dependent sample registration."""
        coordinator = OrchestrationCoordinator()

        # Register root first
        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        # Register dependent
        result = await coordinator.register_dependent_sample(
            orchestration_id="orch_123",
            role="red",
            sample_id="sample_red",
            depends_on_role="blue",
            order=2,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_register_dependent_sample_no_orchestration(self):
        """Test dependent registration fails when orchestration doesn't exist."""
        coordinator = OrchestrationCoordinator()

        result = await coordinator.register_dependent_sample(
            "orch_999", "red", "sample_red", "blue", 2, timeout=0.5
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_register_dependent_sample_missing_dependency(self):
        """Test dependent registration fails when dependency doesn't exist."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        # Try to register dependent with wrong dependency
        result = await coordinator.register_dependent_sample(
            "orch_123", "red", "sample_red", "green", 2
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_register_dependent_sample_duplicate_fails(self):
        """Test that registering same dependent role twice fails."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)

        result = await coordinator.register_dependent_sample(
            "orch_123", "red", "sample_red_2", "blue", 3
        )

        assert result is False


class TestWaitForDependencyReady:
    """Test wait_for_dependency_ready method."""

    @pytest.mark.asyncio
    async def test_wait_for_dependency_ready_success(self):
        """Test successful dependency wait."""
        coordinator = OrchestrationCoordinator()

        # Setup orchestration
        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)

        # Set episode ID in background (simulates root completing)
        async def set_episode():
            await asyncio.sleep(0.1)
            coordinator.set_episode_id("orch_123", "blue", "episode_blue_1")

        task = asyncio.create_task(set_episode())

        # Wait for dependency
        episode_id = await coordinator.wait_for_dependency_ready(
            "orch_123", "red", None, "session_123", timeout=1.0
        )

        await task
        assert episode_id == "episode_blue_1"

    @pytest.mark.asyncio
    async def test_wait_for_dependency_ready_timeout(self):
        """Test that wait times out if dependency never becomes ready."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)

        # Don't set episode ID - should timeout
        with pytest.raises(TimeoutError, match="did not become ready"):
            await coordinator.wait_for_dependency_ready(
                "orch_123", "red", None, "session_123", timeout=0.1
            )

    @pytest.mark.asyncio
    async def test_wait_for_dependency_ready_orchestration_not_found(self):
        """Test that wait fails when orchestration doesn't exist."""
        coordinator = OrchestrationCoordinator()

        with pytest.raises(ValueError, match="Orchestration .* not found"):
            await coordinator.wait_for_dependency_ready(
                "orch_999", "red", None, "session_123"
            )

    @pytest.mark.asyncio
    async def test_wait_for_dependency_ready_no_dependency(self):
        """Test that wait fails when sample has no dependency."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        with pytest.raises(ValueError, match="has no dependency"):
            await coordinator.wait_for_dependency_ready(
                "orch_123", "blue", None, "session_123"
            )

    @pytest.mark.asyncio
    async def test_wait_for_dependency_ready_after_termination(self):
        """Test that wait raises error if orchestration was terminated."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)

        # Trigger termination
        coordinator.trigger_termination("orch_123")

        # Wait should fail even if episode ID is set
        coordinator.set_episode_id("orch_123", "blue", "episode_blue_1")

        with pytest.raises(RuntimeError, match="was terminated"):
            await coordinator.wait_for_dependency_ready(
                "orch_123", "red", None, "session_123", timeout=1.0
            )


class TestSetEpisodeId:
    """Test set_episode_id method."""

    def test_set_episode_id_success(self):
        """Test successful episode ID setting."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        coordinator.set_episode_id("orch_123", "blue", "episode_blue_1")

        # Verify it was set by checking internal state
        group = coordinator._orchestrations.get("orch_123")
        assert group is not None
        assert group.samples["blue"].episode_id == "episode_blue_1"
        assert group.samples["blue"].ready_event.is_set()

    def test_set_episode_id_orchestration_not_found(self):
        """Test setting episode ID when orchestration doesn't exist."""
        coordinator = OrchestrationCoordinator()

        # Should not raise, just log warning
        coordinator.set_episode_id("orch_999", "blue", "episode_1")

    def test_set_episode_id_sample_not_found(self):
        """Test setting episode ID when sample doesn't exist."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        # Should not raise, just log warning
        coordinator.set_episode_id("orch_123", "red", "episode_red_1")

    @pytest.mark.asyncio
    async def test_set_episode_id_signals_dependents(self):
        """Test that setting episode ID signals waiting dependents."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)

        # Start waiting in background
        async def wait_for_dep():
            return await coordinator.wait_for_dependency_ready(
                "orch_123", "red", None, "session_123"
            )

        wait_task = asyncio.create_task(wait_for_dep())

        # Give wait task time to start
        await asyncio.sleep(0.1)

        # Set episode ID - should signal waiting task
        coordinator.set_episode_id("orch_123", "blue", "episode_blue_1")

        # Wait should complete successfully
        episode_id = await wait_task
        assert episode_id == "episode_blue_1"


class TestTriggerTermination:
    """Test trigger_termination method."""

    def test_trigger_termination_success(self):
        """Test successful termination trigger."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        coordinator.set_episode_id("orch_123", "blue", "episode_blue_1")

        episodes = coordinator.trigger_termination("orch_123")

        assert episodes == [("blue", "episode_blue_1")]

    def test_trigger_termination_multiple_samples(self):
        """Test termination with multiple samples."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        coordinator.set_episode_id("orch_123", "blue", "episode_blue_1")

        asyncio.run(coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2))
        coordinator.set_episode_id("orch_123", "red", "episode_red_1")

        episodes = coordinator.trigger_termination("orch_123")

        # Should return all episodes
        assert len(episodes) == 2
        assert ("blue", "episode_blue_1") in episodes
        assert ("red", "episode_red_1") in episodes

    def test_trigger_termination_no_episodes(self):
        """Test termination when no episodes have been created."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        episodes = coordinator.trigger_termination("orch_123")

        assert episodes == []

    def test_trigger_termination_orchestration_not_found(self):
        """Test termination when orchestration doesn't exist."""
        coordinator = OrchestrationCoordinator()

        episodes = coordinator.trigger_termination("orch_999")

        assert episodes == []

    def test_trigger_termination_marks_terminated(self):
        """Test that termination marks orchestration as terminated."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        coordinator.trigger_termination("orch_123")

        # Verify terminated flag is set
        group = coordinator._orchestrations.get("orch_123")
        assert group is not None
        assert group.terminated is True

    def test_trigger_termination_signals_waiting_dependents(self):
        """Test that termination signals waiting dependents."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        # Get ready event before termination
        group = coordinator._orchestrations["orch_123"]
        blue_sample = group.samples["blue"]

        assert not blue_sample.ready_event.is_set()

        coordinator.trigger_termination("orch_123")

        # Should signal the event
        assert blue_sample.ready_event.is_set()
        assert blue_sample.termination_requested is True


class TestCleanupSample:
    """Test cleanup_sample method."""

    def test_cleanup_sample_last_sample_returns_true(self):
        """Test that cleaning up last sample returns True."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        result = coordinator.cleanup_sample("orch_123", "blue")

        assert result is True

    def test_cleanup_sample_not_last_returns_false(self):
        """Test that cleaning up non-last sample returns False."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        asyncio.run(coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2))

        result = coordinator.cleanup_sample("orch_123", "blue")

        assert result is False

    def test_cleanup_sample_removes_orchestration_when_last(self):
        """Test that orchestration is removed when last sample is cleaned up."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        coordinator.cleanup_sample("orch_123", "blue")

        # Verify orchestration was removed
        assert "orch_123" not in coordinator._orchestrations

    def test_cleanup_sample_orchestration_not_found(self):
        """Test cleanup when orchestration doesn't exist."""
        coordinator = OrchestrationCoordinator()

        result = coordinator.cleanup_sample("orch_999", "blue")

        assert result is False

    def test_cleanup_sample_role_not_found(self):
        """Test cleanup when role doesn't exist in orchestration."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        result = coordinator.cleanup_sample("orch_123", "red")

        # Should return False but not remove orchestration (blue still there)
        assert result is False
        assert "orch_123" in coordinator._orchestrations

    def test_cleanup_sample_multiple_samples_sequential(self):
        """Test cleaning up multiple samples sequentially."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        asyncio.run(coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2))

        # Clean up first sample
        result1 = coordinator.cleanup_sample("orch_123", "blue")
        assert result1 is False
        assert "orch_123" in coordinator._orchestrations

        # Clean up second (last) sample
        result2 = coordinator.cleanup_sample("orch_123", "red")
        assert result2 is True
        assert "orch_123" not in coordinator._orchestrations


class TestReset:
    """Test reset method."""

    def test_reset_clears_all_orchestrations(self):
        """Test that reset clears all orchestrations."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_1", "blue", "sample_1")
        coordinator.register_root_sample("orch_2", "red", "sample_2")

        coordinator.reset()

        assert len(coordinator._orchestrations) == 0

    def test_reset_allows_reuse(self):
        """Test that coordinator can be reused after reset."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        coordinator.reset()

        # Should be able to register again
        result = coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        assert result is True


class TestComplexScenarios:
    """Test complex multi-sample orchestration scenarios."""

    @pytest.mark.asyncio
    async def test_three_sample_chain(self):
        """Test orchestration with three samples in dependency chain."""
        coordinator = OrchestrationCoordinator()

        # Setup: blue -> red -> green
        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)
        await coordinator.register_dependent_sample("orch_123", "green", "sample_green", "red", 3)

        # Simulate execution
        coordinator.set_episode_id("orch_123", "blue", "episode_blue")
        coordinator.set_episode_id("orch_123", "red", "episode_red")
        coordinator.set_episode_id("orch_123", "green", "episode_green")

        # Verify all episodes are tracked
        episodes = coordinator.trigger_termination("orch_123")
        assert len(episodes) == 3

    @pytest.mark.asyncio
    async def test_concurrent_orchestrations(self):
        """Test multiple independent orchestrations running concurrently."""
        coordinator = OrchestrationCoordinator()

        # Setup two independent orchestrations
        coordinator.register_root_sample("orch_1", "blue", "sample_1_blue")
        coordinator.register_root_sample("orch_2", "red", "sample_2_red")

        await coordinator.register_dependent_sample("orch_1", "red", "sample_1_red", "blue", 2)
        await coordinator.register_dependent_sample("orch_2", "blue", "sample_2_blue", "red", 2)

        # Set episodes for both
        coordinator.set_episode_id("orch_1", "blue", "episode_1_blue")
        coordinator.set_episode_id("orch_2", "red", "episode_2_red")

        # Verify they're independent
        episodes_1 = coordinator.trigger_termination("orch_1")
        episodes_2 = coordinator.trigger_termination("orch_2")

        assert len(episodes_1) == 1
        assert len(episodes_2) == 1
        assert ("blue", "episode_1_blue") in episodes_1
        assert ("red", "episode_2_red") in episodes_2

    @pytest.mark.asyncio
    async def test_partial_cleanup_preserves_other_samples(self):
        """Test that cleaning up one sample doesn't affect others."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)
        await coordinator.register_dependent_sample("orch_123", "green", "sample_green", "blue", 3)

        # Clean up one dependent
        coordinator.cleanup_sample("orch_123", "red")

        # Verify orchestration still exists with remaining samples
        group = coordinator._orchestrations.get("orch_123")
        assert group is not None
        assert "blue" in group.samples
        assert "green" in group.samples
        assert "red" not in group.samples


class TestSetActiveSample:
    """Test set_active_sample method for ActiveSample reference tracking."""

    def test_set_active_sample_success(self):
        """Test successful ActiveSample registration."""
        coordinator = OrchestrationCoordinator()
        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        # Create mock ActiveSample
        mock_active_sample = MagicMock()
        mock_active_sample.id = "active_sample_id_123"

        coordinator.set_active_sample("orch_123", "blue", mock_active_sample)

        # Verify it was stored
        group = coordinator._orchestrations.get("orch_123")
        assert group is not None
        assert group.samples["blue"].active_sample is mock_active_sample

    def test_set_active_sample_orchestration_not_found(self):
        """Test set_active_sample when orchestration doesn't exist."""
        coordinator = OrchestrationCoordinator()

        mock_active_sample = MagicMock()

        # Should not raise, just return silently
        coordinator.set_active_sample("orch_999", "blue", mock_active_sample)

    def test_set_active_sample_role_not_found(self):
        """Test set_active_sample when role doesn't exist."""
        coordinator = OrchestrationCoordinator()
        coordinator.register_root_sample("orch_123", "blue", "sample_blue")

        mock_active_sample = MagicMock()

        # Should not raise, just return silently
        coordinator.set_active_sample("orch_123", "red", mock_active_sample)

        # Verify nothing was set on blue
        group = coordinator._orchestrations.get("orch_123")
        assert group.samples["blue"].active_sample is None

    @pytest.mark.asyncio
    async def test_set_active_sample_multiple_samples(self):
        """Test setting ActiveSample for multiple samples in orchestration."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)

        mock_active_blue = MagicMock()
        mock_active_blue.id = "active_blue"
        mock_active_red = MagicMock()
        mock_active_red.id = "active_red"

        coordinator.set_active_sample("orch_123", "blue", mock_active_blue)
        coordinator.set_active_sample("orch_123", "red", mock_active_red)

        group = coordinator._orchestrations.get("orch_123")
        assert group.samples["blue"].active_sample is mock_active_blue
        assert group.samples["red"].active_sample is mock_active_red


class TestTriggerTerminationWithSkipRole:
    """Test trigger_termination with skip_role parameter for sibling interruption."""

    def test_trigger_termination_skip_role_not_interrupted(self):
        """Test that skip_role sample is not interrupted."""
        coordinator = OrchestrationCoordinator()
        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        coordinator.set_episode_id("orch_123", "blue", "episode_blue_1")

        # Set mock ActiveSample
        mock_active_blue = MagicMock()
        coordinator.set_active_sample("orch_123", "blue", mock_active_blue)

        # Trigger termination with skip_role=blue
        episodes = coordinator.trigger_termination("orch_123", skip_role="blue")

        # Blue should NOT have been interrupted
        mock_active_blue.interrupt.assert_not_called()
        assert episodes == [("blue", "episode_blue_1")]

    @pytest.mark.asyncio
    async def test_trigger_termination_interrupts_sibling_samples(self):
        """Test that sibling samples are interrupted when one completes."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)

        coordinator.set_episode_id("orch_123", "blue", "episode_blue")
        coordinator.set_episode_id("orch_123", "red", "episode_red")

        # Set mock ActiveSamples
        mock_active_blue = MagicMock()
        mock_active_red = MagicMock()
        coordinator.set_active_sample("orch_123", "blue", mock_active_blue)
        coordinator.set_active_sample("orch_123", "red", mock_active_red)

        # Blue completes and triggers termination
        episodes = coordinator.trigger_termination("orch_123", skip_role="blue")

        # Blue should NOT be interrupted (it's the one completing)
        mock_active_blue.interrupt.assert_not_called()

        # Red SHOULD be interrupted with "score"
        mock_active_red.interrupt.assert_called_once_with("score")

        assert len(episodes) == 2

    @pytest.mark.asyncio
    async def test_trigger_termination_handles_interrupt_exception(self):
        """Test that interrupt exceptions are handled gracefully."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)

        coordinator.set_episode_id("orch_123", "blue", "episode_blue")
        coordinator.set_episode_id("orch_123", "red", "episode_red")

        # Set mock ActiveSamples - red will raise on interrupt
        mock_active_blue = MagicMock()
        mock_active_red = MagicMock()
        mock_active_red.interrupt.side_effect = RuntimeError("Task group not available")
        coordinator.set_active_sample("orch_123", "blue", mock_active_blue)
        coordinator.set_active_sample("orch_123", "red", mock_active_red)

        # Should not raise, should handle gracefully
        episodes = coordinator.trigger_termination("orch_123", skip_role="blue")

        # Should still return episodes
        assert len(episodes) == 2
        # Group should still be marked as terminated
        group = coordinator._orchestrations.get("orch_123")
        assert group.terminated is True

    @pytest.mark.asyncio
    async def test_trigger_termination_no_active_sample_skips_interrupt(self):
        """Test that samples without ActiveSample reference are skipped."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)

        coordinator.set_episode_id("orch_123", "blue", "episode_blue")
        coordinator.set_episode_id("orch_123", "red", "episode_red")

        # Only set ActiveSample for blue, not red
        mock_active_blue = MagicMock()
        coordinator.set_active_sample("orch_123", "blue", mock_active_blue)

        # Trigger from blue - should not raise even though red has no ActiveSample
        episodes = coordinator.trigger_termination("orch_123", skip_role="blue")

        # Blue not interrupted (skip_role)
        mock_active_blue.interrupt.assert_not_called()
        # Red has no ActiveSample so nothing to interrupt
        assert len(episodes) == 2

    @pytest.mark.asyncio
    async def test_trigger_termination_three_samples_one_skipped(self):
        """Test termination with three samples, one skipped."""
        coordinator = OrchestrationCoordinator()

        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        await coordinator.register_dependent_sample("orch_123", "red", "sample_red", "blue", 2)
        await coordinator.register_dependent_sample("orch_123", "green", "sample_green", "blue", 3)

        coordinator.set_episode_id("orch_123", "blue", "episode_blue")
        coordinator.set_episode_id("orch_123", "red", "episode_red")
        coordinator.set_episode_id("orch_123", "green", "episode_green")

        mock_active_blue = MagicMock()
        mock_active_red = MagicMock()
        mock_active_green = MagicMock()
        coordinator.set_active_sample("orch_123", "blue", mock_active_blue)
        coordinator.set_active_sample("orch_123", "red", mock_active_red)
        coordinator.set_active_sample("orch_123", "green", mock_active_green)

        # Red completes first
        episodes = coordinator.trigger_termination("orch_123", skip_role="red")

        # Red not interrupted
        mock_active_red.interrupt.assert_not_called()
        # Blue and green ARE interrupted
        mock_active_blue.interrupt.assert_called_once_with("score")
        mock_active_green.interrupt.assert_called_once_with("score")

        assert len(episodes) == 3

    def test_trigger_termination_without_skip_role_interrupts_all(self):
        """Test that without skip_role, all samples with ActiveSample are interrupted."""
        coordinator = OrchestrationCoordinator()
        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        coordinator.set_episode_id("orch_123", "blue", "episode_blue")

        mock_active_blue = MagicMock()
        coordinator.set_active_sample("orch_123", "blue", mock_active_blue)

        # Trigger without skip_role (default None)
        episodes = coordinator.trigger_termination("orch_123")

        # Should interrupt all
        mock_active_blue.interrupt.assert_called_once_with("score")

    def test_trigger_termination_backward_compatible(self):
        """Test that trigger_termination works without skip_role (backward compat)."""
        coordinator = OrchestrationCoordinator()
        coordinator.register_root_sample("orch_123", "blue", "sample_blue")
        coordinator.set_episode_id("orch_123", "blue", "episode_blue")

        # No ActiveSample set - old behavior
        episodes = coordinator.trigger_termination("orch_123")

        # Should still work and return episodes
        assert episodes == [("blue", "episode_blue")]
        group = coordinator._orchestrations.get("orch_123")
        assert group.terminated is True
