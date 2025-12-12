"""
Unit tests for BenchmarkTaskHandler polymorphic handlers.

Tests the handler pattern for single and orchestrated task execution,
including semaphore management and episode lifecycle.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from saber.inspect_ai.core.task_handlers import (
    CleanupResult,
    OrchestratedTaskHandler,
    SingleEpisodeTaskHandler,
    get_benchmark_task_handler,
)
from saber.models import (
    EpisodeCreateResponse,
    OrchestratedTask,
    OrchestrationStrategy,
    SingleEpisodeTask,
    SubTaskDefinition,
)


class TestSingleEpisodeTaskHandler:
    """Test cases for SingleEpisodeTaskHandler."""

    @pytest.fixture
    def sample_single_task(self):
        """Create a sample single episode task."""
        return SingleEpisodeTask(
            benchmark_task_id="test_task_1",
            task_id="test_task_1",
            domain="test",
            title="Test Task",
            description="A test task",
            episode_attempts=1,
            max_steps=50,
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="",
        )

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        manager = AsyncMock()
        # Return EpisodeCreateResponse object, not just episode_id string
        manager.create_episode = AsyncMock(
            return_value=EpisodeCreateResponse(
                episode_id="episode_123",
                task_id="test_task_1",
                session_id="session_123",
                state="creating",
                message="Episode creation initiated",
            )
        )
        manager.wait_for_episode_ready = AsyncMock()
        manager.end_episode = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_initialize_single_episode(self, sample_single_task, mock_session_manager):
        """Test initializing a single episode task."""
        handler = SingleEpisodeTaskHandler()
        semaphore = asyncio.Semaphore(8)

        state = await handler.initialize(
            benchmark_task=sample_single_task,
            session_id="session_123",
            session_manager=mock_session_manager,
            semaphore=semaphore,
        )

        # Verify episode was created
        mock_session_manager.create_episode.assert_called_once_with("session_123", "test_task_1")
        mock_session_manager.wait_for_episode_ready.assert_called_once_with("session_123", "episode_123")

        # Verify state structure
        assert state.episode_ids == ["episode_123"]
        assert state.primary_episode_id == "episode_123"
        assert state.semaphore_acquired is True

        # Verify semaphore was acquired
        assert semaphore._value == 7  # Started at 8, now 7

    @pytest.mark.asyncio
    async def test_initialize_without_semaphore(self, sample_single_task, mock_session_manager):
        """Test initializing without semaphore (unlimited concurrency)."""
        handler = SingleEpisodeTaskHandler()

        state = await handler.initialize(
            benchmark_task=sample_single_task,
            session_id="session_123",
            session_manager=mock_session_manager,
            semaphore=None,
        )

        # Should still work without semaphore
        assert state.episode_ids == ["episode_123"]
        assert state.semaphore_acquired is False

    @pytest.mark.asyncio
    async def test_initialize_failure_releases_semaphore(self, sample_single_task, mock_session_manager):
        """Test that semaphore is released when initialization fails."""
        handler = SingleEpisodeTaskHandler()
        semaphore = asyncio.Semaphore(8)

        # Make create_episode fail
        mock_session_manager.create_episode.side_effect = Exception("Creation failed")

        with pytest.raises(Exception, match="Creation failed"):
            await handler.initialize(
                benchmark_task=sample_single_task,
                session_id="session_123",
                session_manager=mock_session_manager,
                semaphore=semaphore,
            )

        # Semaphore should be released
        assert semaphore._value == 8  # Back to original

    @pytest.mark.asyncio
    async def test_cleanup_single_episode(self, mock_session_manager):
        """Test cleaning up a single episode task."""
        from saber.inspect_ai.core.types import HandlerState

        handler = SingleEpisodeTaskHandler()
        semaphore = asyncio.Semaphore(7)  # Simulate acquired semaphore

        # Simulate handler that has acquired semaphore during initialize()
        handler._semaphore_acquired = True
        handler._acquired_semaphore_ref = semaphore

        state = HandlerState(
            episode_ids=["episode_123"],
            primary_episode_id="episode_123",
            semaphore_acquired=True,
        )

        result = await handler.cleanup(
            state=state,
            session_id="session_123",
            session_manager=mock_session_manager,
            semaphore=semaphore,
        )

        # Verify cleanup result
        assert isinstance(result, CleanupResult)
        assert result.success is True
        assert result.error_count == 0
        assert result.semaphore_released is True
        assert result.threshold_exceeded is False

        # Verify episode was ended
        mock_session_manager.end_episode.assert_called_once_with("session_123", "episode_123")

        # Verify semaphore was released
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_cleanup_handles_end_episode_failure(self, mock_session_manager):
        """Test that cleanup continues even if end_episode fails."""
        handler = SingleEpisodeTaskHandler()
        semaphore = asyncio.Semaphore(7)

        # Simulate handler that has acquired semaphore during initialize()
        handler._semaphore_acquired = True
        handler._acquired_semaphore_ref = semaphore

        # Make end_episode fail
        mock_session_manager.end_episode.side_effect = Exception("End failed")

        from saber.inspect_ai.core.types import HandlerState

        state = HandlerState(
            episode_ids=["episode_123"],
            primary_episode_id="episode_123",
            semaphore_acquired=True,
        )

        # Should not raise exception (errors tracked in result)
        result = await handler.cleanup(
            state=state,
            session_id="session_123",
            session_manager=mock_session_manager,
            semaphore=semaphore,
        )

        # Verify cleanup result shows error but still succeeded in cleanup
        assert isinstance(result, CleanupResult)
        assert result.success is False  # Episode end failed
        assert result.error_count == 1
        assert len(result.errors) == 1
        assert "Failed to end episode" in result.errors[0]
        assert result.semaphore_released is True  # Semaphore still released
        assert result.threshold_exceeded is False

        # Semaphore should still be released
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_type_check_single_episode_handler(self, mock_session_manager):
        """Test that handler rejects non-SingleEpisodeTask types."""
        handler = SingleEpisodeTaskHandler()

        # Create an orchestrated task (wrong type)
        orchestrated_task = OrchestratedTask(
            benchmark_task_id="orch_1",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="sub_1",
                    role="test",
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
                )
            ],
            episode_attempts=1,
        )

        with pytest.raises(TypeError, match="SingleEpisodeTaskHandler requires SingleEpisodeTask"):
            await handler.initialize(
                benchmark_task=orchestrated_task,
                session_id="session_123",
                session_manager=mock_session_manager,
                semaphore=None,
            )


class TestOrchestratedTaskHandler:
    """Test cases for OrchestratedTaskHandler."""

    @pytest.fixture
    def sample_orchestrated_task(self):
        """Create a sample orchestrated task."""
        return OrchestratedTask(
            benchmark_task_id="dual_task_1",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="blue_task_1",
                    role="blue",
                    order=0,
                    domain="test",
                    title="Blue Task",
                    description="Blue team task",
                    episode_attempts=1,
                    max_steps=30,
                    instruction_prompt="test",
                    assistant_prompt="test",
                    submit_prompt="test",
            continue_prompt="",
                ),
                SubTaskDefinition(
                    task_id="red_task_1",
                    role="red",
                    order=1,
                    depends_on_role="blue",
                    domain="test",
                    title="Red Task",
                    description="Red team task",
                    episode_attempts=1,
                    max_steps=30,
                    instruction_prompt="test",
                    assistant_prompt="test",
                    submit_prompt="test",
            continue_prompt="",
                ),
            ],
            episode_attempts=1,
        )

    @pytest.fixture
    def mock_session_manager_orchestrated(self):
        """Create a mock session manager for orchestrated tasks."""
        manager = AsyncMock()
        # Return EpisodeCreateResponse objects for each create call
        manager.create_episode = AsyncMock(
            side_effect=[
                EpisodeCreateResponse(
                    episode_id="episode_blue",
                    task_id="blue_task_1",
                    session_id="session_123",
                    state="creating",
                    message="Blue episode creation initiated",
                ),
                EpisodeCreateResponse(
                    episode_id="episode_red",
                    task_id="red_task_1",
                    session_id="session_123",
                    state="creating",
                    message="Red episode creation initiated",
                    attached_to_episode_id="episode_blue",
                ),
            ]
        )
        manager.wait_for_episode_ready = AsyncMock()
        manager.end_episode = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_initialize_orchestrated_task(
        self, sample_orchestrated_task, mock_session_manager_orchestrated
    ):
        """Test initializing an orchestrated task creates all episodes."""
        handler = OrchestratedTaskHandler()
        semaphore = asyncio.Semaphore(8)

        state = await handler.initialize(
            benchmark_task=sample_orchestrated_task,
            session_id="session_123",
            session_manager=mock_session_manager_orchestrated,
            semaphore=semaphore,
        )

        # Verify both episodes were created in order
        assert mock_session_manager_orchestrated.create_episode.call_count == 2
        calls = mock_session_manager_orchestrated.create_episode.call_args_list
        assert calls[0][0] == ("session_123", "blue_task_1")
        assert calls[1][0] == ("session_123", "red_task_1")

        # Verify state structure
        assert len(state.episode_ids) == 2
        assert state.episode_ids == ["episode_blue", "episode_red"]
        assert state.primary_episode_id == "episode_blue"
        assert state.semaphore_acquired is True

        # Verify only ONE semaphore slot was acquired for entire orchestration
        assert semaphore._value == 7  # Started at 8, now 7

    @pytest.mark.asyncio
    async def test_initialize_waits_for_each_episode_ready(
        self, sample_orchestrated_task, mock_session_manager_orchestrated
    ):
        """Test that handler waits for each episode to be ready before creating next."""
        handler = OrchestratedTaskHandler()

        await handler.initialize(
            benchmark_task=sample_orchestrated_task,
            session_id="session_123",
            session_manager=mock_session_manager_orchestrated,
            semaphore=None,
        )

        # Verify wait_for_episode_ready was called for each episode
        assert mock_session_manager_orchestrated.wait_for_episode_ready.call_count == 2
        calls = mock_session_manager_orchestrated.wait_for_episode_ready.call_args_list
        assert calls[0][0] == ("session_123", "episode_blue")
        assert calls[1][0] == ("session_123", "episode_red")

    @pytest.mark.asyncio
    async def test_initialize_failure_cleans_up_partial_episodes(
        self, sample_orchestrated_task, mock_session_manager_orchestrated
    ):
        """Test that partial episodes are cleaned up when initialization fails."""
        handler = OrchestratedTaskHandler()
        semaphore = asyncio.Semaphore(8)

        # Make second episode creation fail
        mock_session_manager_orchestrated.create_episode = AsyncMock(
            side_effect=[
                EpisodeCreateResponse(
                    episode_id="episode_blue",
                    task_id="blue_task_1",
                    session_id="session_123",
                    state="creating",
                    message="Blue episode creation initiated",
                ),
                Exception("Creation failed"),
            ]
        )

        with pytest.raises(Exception, match="Creation failed"):
            await handler.initialize(
                benchmark_task=sample_orchestrated_task,
                session_id="session_123",
                session_manager=mock_session_manager_orchestrated,
                semaphore=semaphore,
            )

        # Verify first episode was cleaned up
        mock_session_manager_orchestrated.end_episode.assert_called_once_with("session_123", "episode_blue")

        # Semaphore should be released
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_cleanup_orchestrated_task(self, mock_session_manager_orchestrated):
        """Test cleaning up an orchestrated task ends all episodes."""
        handler = OrchestratedTaskHandler()
        semaphore = asyncio.Semaphore(7)  # Simulate acquired semaphore

        # Simulate handler that has acquired semaphore during initialize()
        handler._semaphore_acquired = True
        handler._acquired_semaphore_ref = semaphore

        from saber.inspect_ai.core.types import OrchestratedHandlerState

        state = OrchestratedHandlerState(
            episode_ids=["episode_blue", "episode_red"],
            episodes=[
                {"episode_id": "episode_blue", "task_id": "blue_task_1", "role": "blue"},
                {"episode_id": "episode_red", "task_id": "red_task_1", "role": "red"},
            ],
            primary_episode_id="episode_blue",
            semaphore_acquired=True,
        )

        result = await handler.cleanup(
            state=state,
            session_id="session_123",
            session_manager=mock_session_manager_orchestrated,
            semaphore=semaphore,
        )

        # Verify cleanup result
        assert isinstance(result, CleanupResult)
        assert result.success is True
        assert result.error_count == 0
        assert result.semaphore_released is True
        assert result.threshold_exceeded is False

        # Verify all episodes were ended
        assert mock_session_manager_orchestrated.end_episode.call_count == 2
        calls = mock_session_manager_orchestrated.end_episode.call_args_list
        assert calls[0][0] == ("session_123", "episode_blue")
        assert calls[1][0] == ("session_123", "episode_red")

        # Verify semaphore was released AFTER all episodes cleaned up
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_cleanup_continues_on_individual_failures(self, mock_session_manager_orchestrated):
        """Test that cleanup continues even if individual episode ends fail."""
        handler = OrchestratedTaskHandler()
        semaphore = asyncio.Semaphore(7)

        # Simulate handler that has acquired semaphore during initialize()
        handler._semaphore_acquired = True
        handler._acquired_semaphore_ref = semaphore

        # Make first episode end fail
        mock_session_manager_orchestrated.end_episode = AsyncMock(
            side_effect=[Exception("End failed"), None]
        )

        from saber.inspect_ai.core.types import OrchestratedHandlerState

        state = OrchestratedHandlerState(
            episode_ids=["episode_blue", "episode_red"],
            primary_episode_id="episode_blue",
            semaphore_acquired=True,
        )

        # Should not raise exception (errors tracked in result)
        result = await handler.cleanup(
            state=state,
            session_id="session_123",
            session_manager=mock_session_manager_orchestrated,
            semaphore=semaphore,
        )

        # Verify cleanup result shows error
        assert isinstance(result, CleanupResult)
        assert result.success is False  # First episode failed
        assert result.error_count == 1
        assert len(result.errors) == 1
        assert result.semaphore_released is True  # Semaphore still released
        assert result.threshold_exceeded is False

        # Should have tried to end both episodes
        assert mock_session_manager_orchestrated.end_episode.call_count == 2

        # Semaphore should still be released
        assert semaphore._value == 8


class TestTaskHandlerFactory:
    """Test cases for get_benchmark_task_handler factory."""

    def test_factory_returns_single_handler_for_single_task(self):
        """Test factory returns SingleEpisodeTaskHandler for SingleEpisodeTask."""
        task = SingleEpisodeTask(
            benchmark_task_id="test_1",
            task_id="test_1",
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

        handler = get_benchmark_task_handler(task)
        assert isinstance(handler, SingleEpisodeTaskHandler)

    def test_factory_returns_orchestrated_handler_for_orchestrated_task(self):
        """Test factory returns OrchestratedTaskHandler for OrchestratedTask."""
        task = OrchestratedTask(
            benchmark_task_id="orch_1",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="sub_1",
                    role="test",
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
                )
            ],
            episode_attempts=1,
        )

        handler = get_benchmark_task_handler(task)
        assert isinstance(handler, OrchestratedTaskHandler)

    def test_factory_raises_for_unknown_type(self):
        """Test factory raises ValueError for unknown task types."""
        # Create a mock object that's not a valid task type
        invalid_task = Mock()

        with pytest.raises(ValueError, match="Unknown benchmark task type"):
            get_benchmark_task_handler(invalid_task)


class TestSemaphoreLifecycle:
    """Test semaphore lifecycle management across handlers."""

    @pytest.mark.asyncio
    async def test_semaphore_released_on_init_failure_single(self):
        """Test semaphore is released when single task init fails."""
        handler = SingleEpisodeTaskHandler()
        semaphore = asyncio.Semaphore(8)
        manager = AsyncMock()
        manager.create_episode = AsyncMock(side_effect=Exception("Failed"))

        task = SingleEpisodeTask(
            benchmark_task_id="test_1",
            task_id="test_1",
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

        with pytest.raises(Exception):
            await handler.initialize(task, "session_1", manager, semaphore)

        # Semaphore must be released
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_semaphore_released_on_init_failure_orchestrated(self):
        """Test semaphore is released when orchestrated task init fails."""
        handler = OrchestratedTaskHandler()
        semaphore = asyncio.Semaphore(8)
        manager = AsyncMock()
        manager.create_episode = AsyncMock(side_effect=Exception("Failed"))
        manager.end_episode = AsyncMock()

        task = OrchestratedTask(
            benchmark_task_id="orch_1",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="sub_1",
                    role="test",
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
                )
            ],
            episode_attempts=1,
        )

        with pytest.raises(Exception):
            await handler.initialize(task, "session_1", manager, semaphore)

        # Semaphore must be released
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_orchestration_strategy_validation(self):
        """Test that unsupported orchestration strategies are rejected."""
        handler = OrchestratedTaskHandler()
        manager = AsyncMock()

        # Create task with PARALLEL strategy (not yet implemented)
        task = OrchestratedTask(
            benchmark_task_id="orch_1",
            orchestration_strategy=OrchestrationStrategy.PARALLEL,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="sub_1",
                    role="test",
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
                )
            ],
            episode_attempts=1,
        )

        # Should raise NotImplementedError for unsupported strategy
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            await handler.initialize(task, "session_1", manager, None)

        # Test with CONDITIONAL strategy too
        task.orchestration_strategy = OrchestrationStrategy.CONDITIONAL
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            await handler.initialize(task, "session_1", manager, None)

    @pytest.mark.asyncio
    async def test_cleanup_threshold_exceeded_raises_exception(self):
        """Test that cleanup raises RuntimeError when error threshold is exceeded."""
        handler = SingleEpisodeTaskHandler()
        manager = AsyncMock()
        semaphore = asyncio.Semaphore(7)

        # Simulate handler that has acquired semaphore
        handler._semaphore_acquired = True
        handler._acquired_semaphore_ref = semaphore

        # Set error count above threshold (CLEANUP_ERROR_THRESHOLD = 5)
        handler._cleanup_error_count = 6

        from saber.inspect_ai.core.types import HandlerState

        state = HandlerState(
            episode_ids=["episode_123"],
            primary_episode_id="episode_123",
            semaphore_acquired=True,
        )

        # Should raise RuntimeError when threshold exceeded
        with pytest.raises(RuntimeError, match="Cleanup error threshold exceeded"):
            await handler.cleanup(state, "session_1", manager, semaphore)

        # Semaphore should still be released even though exception raised
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_cleanup_result_dataclass_properties(self):
        """Test CleanupResult dataclass properties and methods."""
        # Test successful cleanup
        success_result = CleanupResult(
            success=True,
            error_count=0,
            errors=[],
            semaphore_released=True,
            threshold_exceeded=False,
        )
        assert success_result.has_errors is False

        # Test failed cleanup with errors
        failure_result = CleanupResult(
            success=False,
            error_count=2,
            errors=["Error 1", "Error 2"],
            semaphore_released=True,
            threshold_exceeded=False,
        )
        assert failure_result.has_errors is True
        assert len(failure_result.errors) == 2

        # Test threshold exceeded
        threshold_result = CleanupResult(
            success=False,
            error_count=6,
            errors=["Error 1", "Error 2", "Error 3", "Error 4", "Error 5", "Error 6"],
            semaphore_released=True,
            threshold_exceeded=True,
        )
        assert threshold_result.threshold_exceeded is True
        assert threshold_result.has_errors is True

    @pytest.mark.asyncio
    async def test_single_episode_creation_timeout(self):
        """Test that episode creation timeout is handled correctly for single tasks."""
        handler = SingleEpisodeTaskHandler()
        manager = AsyncMock()
        semaphore = asyncio.Semaphore(8)

        # Make create_episode timeout
        manager.create_episode = AsyncMock(side_effect=asyncio.TimeoutError())

        task = SingleEpisodeTask(
            benchmark_task_id="test_1",
            task_id="test_1",
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

        with pytest.raises(asyncio.TimeoutError):
            await handler.initialize(task, "session_1", manager, semaphore)

        # Semaphore must be released after timeout
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_orchestrated_episode_creation_timeout(self):
        """Test that episode creation timeout is handled for orchestrated tasks."""
        handler = OrchestratedTaskHandler()
        manager = AsyncMock()
        semaphore = asyncio.Semaphore(8)

        # Make second episode creation timeout
        manager.create_episode = AsyncMock(
            side_effect=[
                EpisodeCreateResponse(
                    episode_id="episode_1",
                    task_id="task_1",
                    session_id="session_1",
                    state="creating",
                    message="Created",
                ),
                asyncio.TimeoutError(),
            ]
        )
        manager.wait_for_episode_ready = AsyncMock()
        manager.end_episode = AsyncMock()

        task = OrchestratedTask(
            benchmark_task_id="orch_1",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="task_1",
                    role="role1",
                    order=0,
                    domain="test",
                    title="Task 1",
                    description="Test",
                    episode_attempts=1,
                    max_steps=10,
                    instruction_prompt="test",
                    assistant_prompt="test",
                    submit_prompt="test",
            continue_prompt="",
                ),
                SubTaskDefinition(
                    task_id="task_2",
                    role="role2",
                    order=1,
                    domain="test",
                    title="Task 2",
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

        with pytest.raises(asyncio.TimeoutError):
            await handler.initialize(task, "session_1", manager, semaphore)

        # First episode should be cleaned up
        manager.end_episode.assert_called_once_with("session_1", "episode_1")
        # Semaphore must be released
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_single_episode_cleanup_failure_during_init_rollback(self):
        """Test handling when end_episode fails during initialization rollback."""
        handler = SingleEpisodeTaskHandler()
        manager = AsyncMock()
        semaphore = asyncio.Semaphore(8)

        # Create episode succeeds but wait_for_ready fails
        manager.create_episode = AsyncMock(
            return_value=EpisodeCreateResponse(
                episode_id="episode_123",
                task_id="test_1",
                session_id="session_1",
                state="creating",
                message="Created",
            )
        )
        manager.wait_for_episode_ready = AsyncMock(side_effect=Exception("Ready failed"))
        # end_episode also fails during cleanup
        manager.end_episode = AsyncMock(side_effect=Exception("Cleanup failed"))

        task = SingleEpisodeTask(
            benchmark_task_id="test_1",
            task_id="test_1",
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

        with pytest.raises(Exception, match="Ready failed"):
            await handler.initialize(task, "session_1", manager, semaphore)

        # end_episode was called but failed
        manager.end_episode.assert_called_once_with("session_1", "episode_123")
        # Cleanup error count should be incremented
        assert handler._cleanup_error_count == 1
        # Semaphore still released despite cleanup failure
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_orchestrated_cleanup_failure_during_init_rollback(self):
        """Test handling when cleanup fails during orchestrated init rollback."""
        handler = OrchestratedTaskHandler()
        manager = AsyncMock()
        semaphore = asyncio.Semaphore(8)

        # First episode succeeds, second fails
        manager.create_episode = AsyncMock(
            side_effect=[
                EpisodeCreateResponse(
                    episode_id="episode_1",
                    task_id="task_1",
                    session_id="session_1",
                    state="creating",
                    message="Created",
                ),
                Exception("Second episode failed"),
            ]
        )
        manager.wait_for_episode_ready = AsyncMock()
        # Cleanup of first episode fails
        manager.end_episode = AsyncMock(side_effect=Exception("Cleanup failed"))

        task = OrchestratedTask(
            benchmark_task_id="orch_1",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="task_1",
                    role="role1",
                    order=0,
                    domain="test",
                    title="Task 1",
                    description="Test",
                    episode_attempts=1,
                    max_steps=10,
                    instruction_prompt="test",
                    assistant_prompt="test",
                    submit_prompt="test",
            continue_prompt="",
                ),
                SubTaskDefinition(
                    task_id="task_2",
                    role="role2",
                    order=1,
                    domain="test",
                    title="Task 2",
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

        with pytest.raises(Exception, match="Second episode failed"):
            await handler.initialize(task, "session_1", manager, semaphore)

        # Cleanup was attempted
        manager.end_episode.assert_called_once_with("session_1", "episode_1")
        # Cleanup error logged
        assert handler._cleanup_error_count == 1
        # Semaphore still released
        assert semaphore._value == 8

    @pytest.mark.asyncio
    async def test_semaphore_release_failure_single(self):
        """Test handling when semaphore release itself fails."""
        handler = SingleEpisodeTaskHandler()
        manager = AsyncMock()
        manager.end_episode = AsyncMock()

        # Create a mock semaphore that fails on release
        semaphore = AsyncMock()
        semaphore.release = Mock(side_effect=Exception("Semaphore release failed"))

        handler._semaphore_acquired = True
        handler._acquired_semaphore_ref = semaphore

        from saber.inspect_ai.core.types import HandlerState

        state = HandlerState(
            episode_ids=["episode_123"],
            primary_episode_id="episode_123",
            semaphore_acquired=True,
        )

        result = await handler.cleanup(state, "session_1", manager, semaphore)

        # Cleanup continues despite semaphore failure
        assert result.success is False
        assert len(result.errors) == 1
        assert "Failed to release semaphore" in result.errors[0]
        assert result.semaphore_released is False

    @pytest.mark.asyncio
    async def test_semaphore_release_failure_orchestrated(self):
        """Test handling when semaphore release fails for orchestrated tasks."""
        handler = OrchestratedTaskHandler()
        manager = AsyncMock()
        manager.end_episode = AsyncMock()

        # Create a mock semaphore that fails on release
        semaphore = AsyncMock()
        semaphore.release = Mock(side_effect=Exception("Semaphore release failed"))

        handler._semaphore_acquired = True
        handler._acquired_semaphore_ref = semaphore

        from saber.inspect_ai.core.types import OrchestratedHandlerState

        state = OrchestratedHandlerState(
            episode_ids=["episode_1", "episode_2"],
            episodes=[
                {"episode_id": "episode_1", "task_id": "task_1", "role": "role1"},
                {"episode_id": "episode_2", "task_id": "task_2", "role": "role2"},
            ],
            primary_episode_id="episode_1",
            semaphore_acquired=True,
        )

        result = await handler.cleanup(state, "session_1", manager, semaphore)

        # Episodes cleaned up successfully
        assert manager.end_episode.call_count == 2
        # But semaphore release failed
        assert result.success is False
        assert "Failed to release semaphore" in result.errors[0]
        assert result.semaphore_released is False

    @pytest.mark.asyncio
    async def test_orchestrated_cleanup_with_warnings(self):
        """Test that orchestrated cleanup logs warnings for non-critical errors."""
        handler = OrchestratedTaskHandler()
        manager = AsyncMock()
        semaphore = asyncio.Semaphore(7)

        # Simulate one cleanup error (below threshold)
        manager.end_episode = AsyncMock(
            side_effect=[Exception("First cleanup failed"), None]
        )

        handler._semaphore_acquired = True
        handler._acquired_semaphore_ref = semaphore
        handler._cleanup_error_count = 2  # Below threshold of 5

        from saber.inspect_ai.core.types import OrchestratedHandlerState

        state = OrchestratedHandlerState(
            episode_ids=["episode_1", "episode_2"],
            episodes=[
                {"episode_id": "episode_1", "task_id": "task_1", "role": "role1"},
                {"episode_id": "episode_2", "task_id": "task_2", "role": "role2"},
            ],
            primary_episode_id="episode_1",
            semaphore_acquired=True,
        )

        # Should not raise, just log warnings
        result = await handler.cleanup(state, "session_1", manager, semaphore)

        assert result.success is False
        assert result.error_count == 3  # 2 previous + 1 new
        assert result.threshold_exceeded is False
        assert len(result.errors) == 1  # Only current cleanup error
        assert result.semaphore_released is True

    @pytest.mark.asyncio
    async def test_orchestrated_cleanup_threshold_exceeded(self):
        """Test that orchestrated cleanup raises when threshold exceeded."""
        handler = OrchestratedTaskHandler()
        manager = AsyncMock()
        semaphore = asyncio.Semaphore(7)

        # All cleanups succeed
        manager.end_episode = AsyncMock()

        handler._semaphore_acquired = True
        handler._acquired_semaphore_ref = semaphore
        handler._cleanup_error_count = 6  # Above threshold of 5

        from saber.inspect_ai.core.types import OrchestratedHandlerState

        state = OrchestratedHandlerState(
            episode_ids=["episode_1", "episode_2"],
            episodes=[
                {"episode_id": "episode_1", "task_id": "task_1", "role": "role1"},
                {"episode_id": "episode_2", "task_id": "task_2", "role": "role2"},
            ],
            primary_episode_id="episode_1",
            semaphore_acquired=True,
        )

        # Should raise RuntimeError when threshold exceeded
        with pytest.raises(RuntimeError, match="Cleanup error threshold exceeded"):
            await handler.cleanup(state, "session_1", manager, semaphore)

        # Semaphore should still be released
        assert semaphore._value == 8
