"""Unit tests for orchestration score coordination.

Tests the wait_for_all_scored() method that enables synchronized
Score return for orchestrated samples.
"""

import asyncio
import logging

import pytest

from saber.inspect_ai.orchestration_coordinator import OrchestrationCoordinator


class TestScoreCoordination:
    """Test score coordination for orchestrated samples."""

    @pytest.fixture
    def coordinator(self):
        """Fresh coordinator for each test."""
        coord = OrchestrationCoordinator()
        coord.reset()
        return coord

    @pytest.mark.asyncio
    async def test_single_sample_returns_immediately(self, coordinator):
        """Single sample (no orchestration) should not block."""
        # No registration - simulates non-orchestrated sample

        # Should return immediately without blocking
        start = asyncio.get_event_loop().time()
        await coordinator.wait_for_all_scored(
            orchestration_id="nonexistent-orch", role="solo", score=1.0, timeout=1.0
        )
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1  # Should be instant

    @pytest.mark.asyncio
    async def test_two_samples_coordinate_successfully(self, coordinator):
        """Two orchestrated samples should wait for each other."""
        orch_id = "test-orch-1"

        # Register orchestration with 2 samples
        coordinator.register_root_sample(orch_id, "blue", "sample-1")
        await coordinator.register_dependent_sample(
            orch_id, "red", "sample-2", "blue", 2
        )

        # Blue finishes first - should block
        blue_task = asyncio.create_task(
            coordinator.wait_for_all_scored(orch_id, "blue", 0.8, timeout=5.0)
        )

        await asyncio.sleep(0.1)
        assert not blue_task.done(), "Blue should still be waiting"

        # Red finishes - should unblock both
        red_task = asyncio.create_task(
            coordinator.wait_for_all_scored(orch_id, "red", 0.6, timeout=5.0)
        )

        # Both should complete quickly
        await asyncio.wait_for(blue_task, timeout=1.0)
        await asyncio.wait_for(red_task, timeout=1.0)

        assert blue_task.done()
        assert red_task.done()

    @pytest.mark.asyncio
    async def test_timeout_releases_waiting_samples(self, coordinator):
        """Timeout should release samples and raise TimeoutError."""
        orch_id = "test-orch-timeout"

        # Register orchestration with 2 samples
        coordinator.register_root_sample(orch_id, "blue", "sample-1")
        await coordinator.register_dependent_sample(
            orch_id, "red", "sample-2", "blue", 2
        )

        # Blue finishes, red never does
        with pytest.raises(asyncio.TimeoutError):
            await coordinator.wait_for_all_scored(orch_id, "blue", 0.8, timeout=0.5)

    @pytest.mark.asyncio
    async def test_last_sample_doesnt_wait(self, coordinator):
        """Last sample to score should not wait."""
        orch_id = "test-orch-last"

        coordinator.register_root_sample(orch_id, "blue", "sample-1")
        await coordinator.register_dependent_sample(
            orch_id, "red", "sample-2", "blue", 2
        )

        # Blue finishes and blocks
        blue_task = asyncio.create_task(
            coordinator.wait_for_all_scored(orch_id, "blue", 0.8, timeout=5.0)
        )
        await asyncio.sleep(0.1)

        # Red finishes last - should return immediately
        start = asyncio.get_event_loop().time()
        await coordinator.wait_for_all_scored(orch_id, "red", 0.6, timeout=5.0)
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1  # Last sample doesn't wait
        await blue_task  # Blue should now be unblocked

    @pytest.mark.asyncio
    async def test_three_samples_coordinate(self, coordinator):
        """Three-way orchestration should coordinate correctly."""
        orch_id = "test-orch-three"

        coordinator.register_root_sample(orch_id, "attacker", "sample-1")
        await coordinator.register_dependent_sample(
            orch_id, "defender", "sample-2", "attacker", 2
        )
        await coordinator.register_dependent_sample(
            orch_id, "monitor", "sample-3", "defender", 3
        )

        # All start waiting
        attacker_task = asyncio.create_task(
            coordinator.wait_for_all_scored(orch_id, "attacker", 0.9, timeout=5.0)
        )
        await asyncio.sleep(0.05)

        defender_task = asyncio.create_task(
            coordinator.wait_for_all_scored(orch_id, "defender", 0.7, timeout=5.0)
        )
        await asyncio.sleep(0.05)

        # Both should still be waiting
        assert not attacker_task.done()
        assert not defender_task.done()

        # Monitor completes last - releases all
        await coordinator.wait_for_all_scored(orch_id, "monitor", 0.5, timeout=5.0)

        await asyncio.wait_for(attacker_task, timeout=1.0)
        await asyncio.wait_for(defender_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_score_tracking_in_logs(self, coordinator, caplog):
        """Scores should be tracked for logging/debugging."""
        caplog.set_level(logging.INFO)

        orch_id = "test-orch-scores"

        coordinator.register_root_sample(orch_id, "blue", "sample-1")
        await coordinator.register_dependent_sample(
            orch_id, "red", "sample-2", "blue", 2
        )

        # Submit scores
        blue_task = asyncio.create_task(
            coordinator.wait_for_all_scored(orch_id, "blue", 0.85, timeout=5.0)
        )
        await asyncio.sleep(0.1)

        await coordinator.wait_for_all_scored(orch_id, "red", 0.62, timeout=5.0)
        await blue_task

        # Check that scores were logged
        assert "0.85" in caplog.text
        assert "0.62" in caplog.text

    @pytest.mark.asyncio
    async def test_race_condition_in_event_initialization(self, coordinator):
        """Event initialization should be thread-safe."""
        orch_id = "test-orch-race"

        coordinator.register_root_sample(orch_id, "blue", "sample-1")
        await coordinator.register_dependent_sample(
            orch_id, "red", "sample-2", "blue", 2
        )

        # Start both samples simultaneously (race to initialize event)
        tasks = [
            asyncio.create_task(
                coordinator.wait_for_all_scored(orch_id, "blue", 0.8, timeout=5.0)
            ),
            asyncio.create_task(
                coordinator.wait_for_all_scored(orch_id, "red", 0.6, timeout=5.0)
            ),
        ]

        # Both should complete successfully without race conditions
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception), f"Unexpected exception: {result}"

    @pytest.mark.asyncio
    async def test_multiple_orchestrations_independent(self, coordinator):
        """Multiple orchestrations should not interfere with each other."""
        # Register two independent orchestrations
        coordinator.register_root_sample("orch-1", "blue", "sample-1")
        await coordinator.register_dependent_sample(
            "orch-1", "red", "sample-2", "blue", 2
        )

        coordinator.register_root_sample("orch-2", "attacker", "sample-3")
        await coordinator.register_dependent_sample(
            "orch-2", "defender", "sample-4", "attacker", 2
        )

        # Complete orch-1 fully
        blue_1 = asyncio.create_task(
            coordinator.wait_for_all_scored("orch-1", "blue", 0.8, timeout=5.0)
        )
        await asyncio.sleep(0.05)
        await coordinator.wait_for_all_scored("orch-1", "red", 0.6, timeout=5.0)
        await blue_1

        # orch-2 should still be independent - attacker waiting
        attacker_2 = asyncio.create_task(
            coordinator.wait_for_all_scored("orch-2", "attacker", 0.9, timeout=5.0)
        )
        await asyncio.sleep(0.1)

        assert not attacker_2.done(), "Attacker should still be waiting"

        # Complete orch-2
        await coordinator.wait_for_all_scored("orch-2", "defender", 0.7, timeout=5.0)
        await attacker_2

    @pytest.mark.asyncio
    async def test_timeout_sets_event_for_other_waiters(self, coordinator):
        """When one sample times out, it should release other waiting samples."""
        orch_id = "test-orch-timeout-release"

        coordinator.register_root_sample(orch_id, "blue", "sample-1")
        await coordinator.register_dependent_sample(
            orch_id, "red", "sample-2", "blue", 2
        )
        await coordinator.register_dependent_sample(
            orch_id, "green", "sample-3", "red", 3
        )

        # Blue and red both start waiting
        blue_task = asyncio.create_task(
            coordinator.wait_for_all_scored(orch_id, "blue", 0.8, timeout=1.0)
        )
        await asyncio.sleep(0.05)

        red_task = asyncio.create_task(
            coordinator.wait_for_all_scored(orch_id, "red", 0.6, timeout=1.0)
        )
        await asyncio.sleep(0.05)

        # Green never completes, causing timeout
        # First sample to timeout (blue) should raise TimeoutError and set event
        # Second sample (red) should be released by the event (no timeout error)
        with pytest.raises(asyncio.TimeoutError):
            await blue_task

        # Red should complete successfully (released by blue's timeout setting the event)
        await asyncio.wait_for(red_task, timeout=0.5)
