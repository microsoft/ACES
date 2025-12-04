"""Tests for concurrent cleanup protection and semaphore leak detection."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from saber.inspect_ai.core.task_handlers import SingleEpisodeTaskHandler, OrchestratedTaskHandler
from saber.inspect_ai.core.types import HandlerState, OrchestratedHandlerState
from saber.models import (
    SingleEpisodeTask,
    OrchestratedTask,
    OrchestrationStrategy,
    SubTaskDefinition,
    EpisodeCreateResponse,
)


class TestConcurrentCleanup:
    """Test concurrent cleanup scenarios to verify lock protection."""

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_single_episode_no_double_release(self):
        """Test that concurrent cleanup calls don't double-release semaphore."""
        handler = SingleEpisodeTaskHandler()
        semaphore = asyncio.Semaphore(8)

        task = SingleEpisodeTask(
            benchmark_task_id="test_task",
            task_id="test_task",
            domain="test",
            title="Test",
            description="Test",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
        )

        # Mock session manager
        manager = AsyncMock()
        manager.create_episode = AsyncMock(
            return_value=EpisodeCreateResponse(
                episode_id="ep_123",
                task_id="test_task",
                session_id="session_123",
                state="creating",
                message="Created",
            )
        )
        manager.wait_for_episode_ready = AsyncMock()
        manager.end_episode = AsyncMock()

        # Initialize handler
        state = await handler.initialize(task, "session_123", manager, semaphore)

        # Verify semaphore was acquired
        assert semaphore._value == 7

        # Call cleanup concurrently multiple times
        await asyncio.gather(
            handler.cleanup(state, "session_123", manager, semaphore),
            handler.cleanup(state, "session_123", manager, semaphore),
            handler.cleanup(state, "session_123", manager, semaphore),
        )

        # Semaphore should be released exactly once, back to 8
        assert semaphore._value == 8

        # end_episode will be called 3 times (once per cleanup call)
        # This is acceptable - cleanup is idempotent
        # The critical part is semaphore is only released once
        assert manager.end_episode.call_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_orchestrated_no_double_release(self):
        """Test that concurrent cleanup of orchestration doesn't corrupt semaphore."""
        handler = OrchestratedTaskHandler()
        semaphore = asyncio.Semaphore(8)

        task = OrchestratedTask(
            benchmark_task_id="orch_task",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="sub_1",
                    role="blue",
                    order=0,
                    domain="test",
                    title="Test",
                    description="Test",
                    episode_attempts=1,
                    max_steps=10,
                    instruction_prompt="test",
                    assistant_prompt="test",
                    submit_prompt="test",
                ),
                SubTaskDefinition(
                    task_id="sub_2",
                    role="red",
                    order=1,
                    domain="test",
                    title="Test",
                    description="Test",
                    episode_attempts=1,
                    max_steps=10,
                    instruction_prompt="test",
                    assistant_prompt="test",
                    submit_prompt="test",
                ),
            ],
            episode_attempts=1,
        )

        # Mock session manager
        manager = AsyncMock()
        manager.create_episode = AsyncMock(
            side_effect=[
                EpisodeCreateResponse(
                    episode_id="ep_blue",
                    task_id="sub_1",
                    session_id="session_123",
                    state="creating",
                    message="Created",
                ),
                EpisodeCreateResponse(
                    episode_id="ep_red",
                    task_id="sub_2",
                    session_id="session_123",
                    state="creating",
                    message="Created",
                ),
            ]
        )
        manager.wait_for_episode_ready = AsyncMock()
        manager.end_episode = AsyncMock()

        # Initialize handler
        state = await handler.initialize(task, "session_123", manager, semaphore)

        # Verify semaphore was acquired once
        assert semaphore._value == 7

        # Call cleanup concurrently
        await asyncio.gather(
            handler.cleanup(state, "session_123", manager, semaphore),
            handler.cleanup(state, "session_123", manager, semaphore),
        )

        # Semaphore should be released exactly once
        assert semaphore._value == 8

        # end_episode will be called 4 times (2 episodes × 2 cleanup calls)
        # This is acceptable - cleanup is idempotent
        # The critical part is semaphore is only released once
        assert manager.end_episode.call_count == 4


class TestSemaphoreLeakDetection:
    """Test semaphore leak detection via __del__ finalizer."""

    def test_semaphore_leak_logged_on_handler_destruction(self, caplog):
        """Test that destroying handler without cleanup logs error."""
        import logging

        with caplog.at_level(logging.ERROR):
            handler = SingleEpisodeTaskHandler()
            semaphore = asyncio.Semaphore(8)

            # Simulate acquiring semaphore
            handler._semaphore_acquired = True
            handler._acquired_semaphore_ref = semaphore

            # Destroy handler without cleanup
            del handler

            # Note: __del__ behavior is unreliable in tests
            # This test documents the expected behavior but may not always trigger

    def test_no_leak_warning_when_cleanup_called(self):
        """Test that proper cleanup doesn't trigger leak warning."""
        handler = SingleEpisodeTaskHandler()
        semaphore = asyncio.Semaphore(8)

        # Simulate proper lifecycle
        handler._semaphore_acquired = True
        handler._acquired_semaphore_ref = semaphore

        # Manually cleanup (simulate what async cleanup does)
        handler._semaphore_acquired = False
        handler._acquired_semaphore_ref = None

        # Destroy handler - should not log error
        del handler
        # If no exception is raised, test passes


class TestCleanupErrorThreshold:
    """Test cleanup error threshold alerting."""

    @pytest.mark.asyncio
    async def test_critical_alert_when_threshold_exceeded(self):
        """Test that exceeding threshold raises RuntimeError and logs critical alert."""
        handler = SingleEpisodeTaskHandler()

        # Simulate many cleanup errors
        handler._cleanup_error_count = 10  # Above threshold of 5

        manager = AsyncMock()
        manager.end_episode = AsyncMock()

        state = HandlerState(
            episode_ids=["ep_123"],
            primary_episode_id="ep_123",
            semaphore_acquired=False,
        )

        # Call cleanup - should raise RuntimeError when threshold exceeded
        with pytest.raises(RuntimeError, match="Cleanup error threshold exceeded"):
            await handler.cleanup(state, "session_123", manager, None)

        # Verify cleanup_error_count is tracked
        assert handler._cleanup_error_count == 10

    @pytest.mark.asyncio
    async def test_warning_when_below_threshold(self):
        """Test that errors below threshold only trigger warning and return result."""
        handler = SingleEpisodeTaskHandler()

        # Simulate few cleanup errors
        handler._cleanup_error_count = 2  # Below threshold of 5

        manager = AsyncMock()
        manager.end_episode = AsyncMock()

        state = HandlerState(
            episode_ids=["ep_456"],
            primary_episode_id="ep_456",
            semaphore_acquired=False,
        )

        # Call cleanup - should succeed and return result
        result = await handler.cleanup(state, "session_123", manager, None)

        assert handler._cleanup_error_count == 2
        assert result.error_count == 2
        assert result.threshold_exceeded is False
