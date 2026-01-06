"""Tests for concurrent cleanup protection and error threshold handling."""

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
    async def test_concurrent_cleanup_single_episode_idempotent(self):
        """Test that concurrent cleanup calls are handled safely."""
        handler = SingleEpisodeTaskHandler()

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
            continue_prompt="",
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
        state = await handler.initialize(task, "session_123", manager)

        # Call cleanup concurrently multiple times
        results = await asyncio.gather(
            handler.cleanup(state, "session_123", manager),
            handler.cleanup(state, "session_123", manager),
            handler.cleanup(state, "session_123", manager),
        )

        # All results should indicate successful cleanup
        for result in results:
            assert result.success is True

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_orchestrated_idempotent(self):
        """Test that concurrent cleanup of orchestration is handled safely."""
        handler = OrchestratedTaskHandler()

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
                    continue_prompt="",
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
                    continue_prompt="",
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
        state = await handler.initialize(task, "session_123", manager)

        # Call cleanup concurrently
        results = await asyncio.gather(
            handler.cleanup(state, "session_123", manager),
            handler.cleanup(state, "session_123", manager),
        )

        # All results should indicate successful cleanup
        for result in results:
            assert result.success is True


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
        )

        # Call cleanup - should raise RuntimeError when threshold exceeded
        with pytest.raises(RuntimeError, match="Cleanup error threshold exceeded"):
            await handler.cleanup(state, "session_123", manager)

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
        )

        # Call cleanup - should succeed and return result
        result = await handler.cleanup(state, "session_123", manager)

        assert handler._cleanup_error_count == 2
        assert result.error_count == 2
        assert result.threshold_exceeded is False
