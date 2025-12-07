"""Unit tests for TranscriptStateMachine lifecycle management."""

import asyncio
import pytest
from saber.server.episodes.transcript_state_machine import TranscriptStateMachine


@pytest.mark.asyncio
class TestLifecycleManagement:
    """Test episode lifecycle hooks."""

    async def test_episode_created_initializes_lock(self):
        """on_episode_created should initialize episode lock."""
        state_machine = TranscriptStateMachine()
        episode_id = "ep-test-1"

        await state_machine.on_episode_created(episode_id)

        # Lock should exist
        assert episode_id in state_machine._episode_locks
        assert isinstance(state_machine._episode_locks[episode_id], asyncio.Lock)

    async def test_episode_created_initializes_timestamp(self):
        """on_episode_created should initialize state timestamp."""
        state_machine = TranscriptStateMachine()
        episode_id = "ep-test-1"

        import time
        before = time.time()
        await state_machine.on_episode_created(episode_id)
        after = time.time()

        # Timestamp should exist and be recent
        assert episode_id in state_machine._state_timestamps
        timestamp = state_machine._state_timestamps[episode_id]
        assert before <= timestamp <= after

    async def test_episode_terminated_cleans_up_lock(self):
        """on_episode_terminated should remove episode lock."""
        state_machine = TranscriptStateMachine()
        episode_id = "ep-test-1"

        await state_machine.on_episode_created(episode_id)
        assert episode_id in state_machine._episode_locks

        await state_machine.on_episode_terminated(episode_id)
        assert episode_id not in state_machine._episode_locks

    async def test_episode_terminated_cleans_up_timestamp(self):
        """on_episode_terminated should remove state timestamp."""
        state_machine = TranscriptStateMachine()
        episode_id = "ep-test-1"

        await state_machine.on_episode_created(episode_id)
        assert episode_id in state_machine._state_timestamps

        await state_machine.on_episode_terminated(episode_id)
        assert episode_id not in state_machine._state_timestamps

    async def test_multiple_episodes_isolated(self):
        """Multiple episodes should have isolated locks and timestamps."""
        state_machine = TranscriptStateMachine()

        await state_machine.on_episode_created("ep-1")
        await state_machine.on_episode_created("ep-2")
        await state_machine.on_episode_created("ep-3")

        # All should have locks
        assert "ep-1" in state_machine._episode_locks
        assert "ep-2" in state_machine._episode_locks
        assert "ep-3" in state_machine._episode_locks

        # All should have timestamps
        assert "ep-1" in state_machine._state_timestamps
        assert "ep-2" in state_machine._state_timestamps
        assert "ep-3" in state_machine._state_timestamps

        # Terminate one shouldn't affect others
        await state_machine.on_episode_terminated("ep-2")

        assert "ep-1" in state_machine._episode_locks
        assert "ep-2" not in state_machine._episode_locks
        assert "ep-3" in state_machine._episode_locks

    async def test_terminate_nonexistent_episode_is_safe(self):
        """Terminating non-existent episode should not raise."""
        state_machine = TranscriptStateMachine()
        episode_id = "ep-nonexistent"

        # Should not raise
        await state_machine.on_episode_terminated(episode_id)

    async def test_double_terminate_is_safe(self):
        """Terminating same episode twice should not raise."""
        state_machine = TranscriptStateMachine()
        episode_id = "ep-test-1"

        await state_machine.on_episode_created(episode_id)
        await state_machine.on_episode_terminated(episode_id)

        # Should not raise
        await state_machine.on_episode_terminated(episode_id)


@pytest.mark.asyncio
class TestConcurrentStateAccess:
    """Test concurrent access with episode locks."""

    async def test_state_transitions_are_atomic(self):
        """Multiple concurrent state transitions should be atomic."""
        state_machine = TranscriptStateMachine()
        episode_id = "ep-test-1"

        await state_machine.on_episode_created(episode_id)

        # Simulate concurrent modifications
        results = []

        async def modify_state(value: int):
            async with state_machine._episode_locks[episode_id]:
                # Critical section - should be atomic
                await asyncio.sleep(0.01)  # Simulate work
                results.append(value)

        # Launch concurrent tasks
        await asyncio.gather(
            modify_state(1),
            modify_state(2),
            modify_state(3),
        )

        # All modifications should complete
        assert len(results) == 3
        assert set(results) == {1, 2, 3}

    async def test_lock_prevents_race_condition(self):
        """Lock should prevent race conditions in shared state."""
        state_machine = TranscriptStateMachine()
        episode_id = "ep-test-1"

        await state_machine.on_episode_created(episode_id)

        counter = {"value": 0}

        async def increment():
            async with state_machine._episode_locks[episode_id]:
                # Read-modify-write should be atomic
                current = counter["value"]
                await asyncio.sleep(0.01)  # Simulate processing
                counter["value"] = current + 1

        # Launch 10 concurrent increments
        await asyncio.gather(*[increment() for _ in range(10)])

        # All increments should succeed
        assert counter["value"] == 10
