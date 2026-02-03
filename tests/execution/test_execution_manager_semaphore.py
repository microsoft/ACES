"""
Tests for ExecutionManager semaphore queue functionality.

This module tests the asyncio.Semaphore-based concurrency control that queues
excess concurrent execution requests instead of rejecting them.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.base import Action, CommandResult
from saber.server.execution.base import ValidationResult
from saber.server.execution.execution_manager import ExecutionManager


class TestExecutionManagerSemaphore:
    """Test cases for ExecutionManager semaphore-based concurrency control."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory for testing."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        return str(config_dir)

    @pytest.fixture
    def execution_manager(self, temp_config_dir):
        """Create an ExecutionManager instance with mocked sandbox manager."""
        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager"):
            manager = ExecutionManager(temp_config_dir)

            # Mock the sandbox environment manager
            mock_sandbox_instance = MagicMock()
            mock_sandbox_instance.is_ready.return_value = True
            manager._sandbox_environment_manager = mock_sandbox_instance

            # Create executor factory with mock
            from saber.server.execution.executors.executor_factory import ExecutorFactory
            manager._executor_factory = ExecutorFactory(
                sandbox_manager=mock_sandbox_instance,
                configuration=manager._configuration,
            )

            return manager

    def test_semaphore_creation_on_first_access(self, execution_manager):
        """Test that semaphore is created lazily on first access for an episode."""
        episode_id = "test-episode-1"

        assert episode_id not in execution_manager._episode_semaphores

        semaphore = execution_manager._get_episode_semaphore(episode_id)

        assert episode_id in execution_manager._episode_semaphores
        assert isinstance(semaphore, asyncio.Semaphore)
        assert semaphore._value == execution_manager._max_concurrent_per_episode

    def test_semaphore_reuse_for_same_episode(self, execution_manager):
        """Test that the same semaphore is returned for repeated calls with same episode."""
        episode_id = "test-episode-1"

        semaphore1 = execution_manager._get_episode_semaphore(episode_id)
        semaphore2 = execution_manager._get_episode_semaphore(episode_id)

        assert semaphore1 is semaphore2

    def test_different_semaphores_for_different_episodes(self, execution_manager):
        """Test that different episodes get different semaphores."""
        episode_id_1 = "test-episode-1"
        episode_id_2 = "test-episode-2"

        semaphore1 = execution_manager._get_episode_semaphore(episode_id_1)
        semaphore2 = execution_manager._get_episode_semaphore(episode_id_2)

        assert semaphore1 is not semaphore2

    def test_cleanup_episode_semaphore(self, execution_manager):
        """Test that cleanup_episode_semaphore removes the semaphore."""
        episode_id = "test-episode-1"

        # Create semaphore
        execution_manager._get_episode_semaphore(episode_id)
        assert episode_id in execution_manager._episode_semaphores

        # Clean up
        execution_manager.cleanup_episode_semaphore(episode_id)
        assert episode_id not in execution_manager._episode_semaphores

    def test_cleanup_nonexistent_semaphore_is_safe(self, execution_manager):
        """Test that cleaning up a non-existent semaphore doesn't raise errors."""
        episode_id = "nonexistent-episode"

        # Should not raise any exceptions
        execution_manager.cleanup_episode_semaphore(episode_id)

    @pytest.mark.asyncio
    async def test_step_acquires_and_releases_semaphore(self, execution_manager):
        """Test that step properly acquires and releases the semaphore."""
        from saber.server.execution.base import BashParameters

        episode_id = "test-episode-1"
        action = Action(tool_name="bash", parameters={"command": "echo test"})
        context = {"episode_id": episode_id}

        expected_result = CommandResult.success_result(
            data={"stdout": "test\n", "stderr": "", "return_code": 0}
        )

        # Mock the executor
        mock_executor = AsyncMock()
        mock_executor.get_parameters_class = MagicMock(return_value=BashParameters)
        mock_executor.validate_parameters = MagicMock(return_value=ValidationResult.success())
        mock_executor.return_value = expected_result

        with patch.object(execution_manager._executor_factory, "get_executor", return_value=mock_executor):
            # Get semaphore before step
            semaphore = execution_manager._get_episode_semaphore(episode_id)
            initial_value = semaphore._value

            result = await execution_manager.step(action, context)

            # After step completes, semaphore should be released
            assert semaphore._value == initial_value
            assert result.success is True

    @pytest.mark.asyncio
    async def test_concurrent_executions_are_queued(self, execution_manager):
        """Test that concurrent executions beyond the limit are queued, not rejected."""
        from saber.server.execution.base import BashParameters

        episode_id = "test-episode-1"
        max_concurrent = execution_manager._max_concurrent_per_episode

        # Track execution order
        execution_order: list[int] = []
        execution_lock = asyncio.Lock()

        async def mock_execution(params, context):
            # Simulate some work
            await asyncio.sleep(0.05)
            async with execution_lock:
                execution_order.append(len(execution_order))
            return CommandResult.success_result(data={"result": "done"})

        # Mock the executor
        mock_executor = AsyncMock(side_effect=mock_execution)
        mock_executor.get_parameters_class = MagicMock(return_value=BashParameters)
        mock_executor.validate_parameters = MagicMock(return_value=ValidationResult.success())

        # Start more tasks than max_concurrent to test queuing
        num_tasks = max_concurrent + 4
        actions = [
            Action(tool_name="bash", parameters={"command": f"echo {i}"})
            for i in range(num_tasks)
        ]
        contexts = [{"episode_id": episode_id} for _ in range(num_tasks)]

        with patch.object(execution_manager._executor_factory, "get_executor", return_value=mock_executor):
            # Run all tasks concurrently
            results = await asyncio.gather(
                *[execution_manager.step(action, ctx) for action, ctx in zip(actions, contexts, strict=True)]
            )

        # All tasks should complete successfully (queued, not rejected)
        assert all(r.success for r in results)
        assert len(execution_order) == num_tasks

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_executions(self, execution_manager):
        """Test that semaphore properly limits concurrent executions."""
        from saber.server.execution.base import BashParameters

        episode_id = "test-episode-1"
        max_concurrent = execution_manager._max_concurrent_per_episode

        # Track concurrent count
        concurrent_count = 0
        max_observed_concurrent = 0
        count_lock = asyncio.Lock()

        async def mock_execution(params, context):
            nonlocal concurrent_count, max_observed_concurrent
            async with count_lock:
                concurrent_count += 1
                max_observed_concurrent = max(max_observed_concurrent, concurrent_count)

            # Simulate some work
            await asyncio.sleep(0.1)

            async with count_lock:
                concurrent_count -= 1

            return CommandResult.success_result(data={"result": "done"})

        # Mock the executor
        mock_executor = AsyncMock(side_effect=mock_execution)
        mock_executor.get_parameters_class = MagicMock(return_value=BashParameters)
        mock_executor.validate_parameters = MagicMock(return_value=ValidationResult.success())

        # Start many more tasks than max_concurrent
        num_tasks = max_concurrent * 2
        actions = [
            Action(tool_name="bash", parameters={"command": f"echo {i}"})
            for i in range(num_tasks)
        ]
        contexts = [{"episode_id": episode_id} for _ in range(num_tasks)]

        with patch.object(execution_manager._executor_factory, "get_executor", return_value=mock_executor):
            results = await asyncio.gather(
                *[execution_manager.step(action, ctx) for action, ctx in zip(actions, contexts, strict=True)]
            )

        # All tasks should complete
        assert all(r.success for r in results)

        # Max concurrent should not exceed the limit
        assert max_observed_concurrent <= max_concurrent

    @pytest.mark.asyncio
    async def test_step_requires_episode_id(self, execution_manager):
        """Test that step returns error when episode_id is missing."""
        action = Action(tool_name="bash", parameters={"command": "echo test"})
        context = {}  # No episode_id

        result = await execution_manager.step(action, context)

        assert result.success is False
        assert "episode_id is required" in result.error

    def test_get_execution_stats_with_semaphores(self, execution_manager):
        """Test that get_execution_stats works correctly with semaphore-based tracking."""
        # Create semaphores for two episodes
        ep1 = "episode-1"
        ep2 = "episode-2"

        sem1 = execution_manager._get_episode_semaphore(ep1)
        sem2 = execution_manager._get_episode_semaphore(ep2)

        # Simulate active executions by reducing semaphore values
        # Note: Directly manipulating _value for test purposes
        sem1._value = execution_manager._max_concurrent_per_episode - 3  # 3 active
        sem2._value = execution_manager._max_concurrent_per_episode - 1  # 1 active

        stats = execution_manager.get_execution_stats()

        assert stats.total_active_executions == 4
        assert stats.active_episodes == 2
        assert stats.episode_execution_counts[ep1] == 3
        assert stats.episode_execution_counts[ep2] == 1

    def test_get_execution_stats_empty(self, execution_manager):
        """Test get_execution_stats with no active episodes."""
        stats = execution_manager.get_execution_stats()

        assert stats.total_active_executions == 0
        assert stats.active_episodes == 0
        assert stats.episode_execution_counts == {}

    def test_max_concurrent_from_environment(self, temp_config_dir):
        """Test that max_concurrent_per_episode can be set via environment variable."""
        with patch.dict("os.environ", {"SABER_MAX_CONCURRENT_PER_EPISODE": "16"}):
            with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager"):
                manager = ExecutionManager(temp_config_dir)

        assert manager._max_concurrent_per_episode == 16

    @pytest.mark.asyncio
    async def test_cleanup_episode_cleans_semaphore(self, execution_manager):
        """Test that cleanup_episode removes the episode's semaphore."""
        episode_id = "test-episode-1"

        # Create semaphore
        execution_manager._get_episode_semaphore(episode_id)
        assert episode_id in execution_manager._episode_semaphores

        # Mock sandbox manager to prevent actual cleanup
        execution_manager._sandbox_environment_manager.stop_episode_environment = AsyncMock(return_value=True)
        execution_manager._executor_factory.unregister_episode_configuration = MagicMock()

        # Clean up episode
        await execution_manager.cleanup_episode(episode_id)

        # Semaphore should be removed
        assert episode_id not in execution_manager._episode_semaphores

    def test_cleanup_session_does_not_clean_semaphores_directly(self, execution_manager):
        """Test that cleanup_session leaves semaphores alone (they're cleaned via cleanup_episode).

        Episode IDs are UUIDs that are unrelated to session IDs, so cleanup_session
        cannot know which semaphores belong to which session. Instead, the session
        termination flow calls cleanup_episode for each episode, which handles
        semaphore cleanup individually.
        """
        episode_id_1 = "11111111-1111-1111-1111-111111111111"
        episode_id_2 = "22222222-2222-2222-2222-222222222222"
        session_id = "session-abc123"

        # Create semaphores for episodes (these are unrelated to session_id)
        execution_manager._get_episode_semaphore(episode_id_1)
        execution_manager._get_episode_semaphore(episode_id_2)

        assert len(execution_manager._episode_semaphores) == 2

        # cleanup_session should NOT remove semaphores since episode IDs
        # don't start with session_id (they're UUIDs)
        execution_manager.cleanup_session(session_id)

        # Semaphores should still exist - cleanup happens via cleanup_episode
        assert len(execution_manager._episode_semaphores) == 2
        assert episode_id_1 in execution_manager._episode_semaphores
        assert episode_id_2 in execution_manager._episode_semaphores


class TestExecutionManagerSemaphoreEdgeCases:
    """Edge case tests for semaphore functionality."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory for testing."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        return str(config_dir)

    @pytest.fixture
    def execution_manager(self, temp_config_dir):
        """Create an ExecutionManager instance with mocked sandbox manager."""
        with patch("saber.server.execution.execution_manager.SandboxEnvironmentManager"):
            manager = ExecutionManager(temp_config_dir)

            mock_sandbox_instance = MagicMock()
            mock_sandbox_instance.is_ready.return_value = True
            manager._sandbox_environment_manager = mock_sandbox_instance

            from saber.server.execution.executors.executor_factory import ExecutorFactory
            manager._executor_factory = ExecutorFactory(
                sandbox_manager=mock_sandbox_instance,
                configuration=manager._configuration,
            )

            return manager

    @pytest.mark.asyncio
    async def test_semaphore_released_on_exception(self, execution_manager):
        """Test that semaphore is released even when execution raises exception."""
        from saber.server.execution.base import BashParameters

        episode_id = "test-episode-1"
        action = Action(tool_name="bash", parameters={"command": "echo test"})
        context = {"episode_id": episode_id}

        # Mock executor that raises exception
        mock_executor = AsyncMock(side_effect=Exception("Execution failed"))
        mock_executor.get_parameters_class = MagicMock(return_value=BashParameters)
        mock_executor.validate_parameters = MagicMock(return_value=ValidationResult.success())

        semaphore = execution_manager._get_episode_semaphore(episode_id)
        initial_value = semaphore._value

        with patch.object(execution_manager._executor_factory, "get_executor", return_value=mock_executor):
            result = await execution_manager.step(action, context)

        # Semaphore should be released even on failure
        assert semaphore._value == initial_value
        assert result.success is False
        assert "Execution failed" in result.error

    @pytest.mark.asyncio
    async def test_semaphore_released_on_validation_failure(self, execution_manager):
        """Test that semaphore is released on parameter validation failure."""
        episode_id = "test-episode-1"
        action = Action(tool_name="bash", parameters={"invalid": "params"})
        context = {"episode_id": episode_id}

        validation_result = ValidationResult.failure(["Invalid parameter"])

        mock_executor = MagicMock()
        mock_executor.get_parameters_class = MagicMock(side_effect=ValueError("Bad params"))
        mock_executor.validate_parameters = MagicMock(return_value=validation_result)

        semaphore = execution_manager._get_episode_semaphore(episode_id)
        initial_value = semaphore._value

        with patch.object(execution_manager._executor_factory, "get_executor", return_value=mock_executor):
            result = await execution_manager.step(action, context)

        # Semaphore should be released even on validation failure
        assert semaphore._value == initial_value
        assert result.success is False

    @pytest.mark.asyncio
    async def test_multiple_episodes_independent_semaphores(self, execution_manager):
        """Test that multiple episodes have independent concurrency limits."""
        from saber.server.execution.base import BashParameters

        episode_1 = "episode-1"
        episode_2 = "episode-2"
        max_concurrent = execution_manager._max_concurrent_per_episode

        # Track concurrent count per episode
        ep1_concurrent = 0
        ep2_concurrent = 0
        max_ep1_concurrent = 0
        max_ep2_concurrent = 0
        count_lock = asyncio.Lock()

        async def mock_execution(params, context):
            nonlocal ep1_concurrent, ep2_concurrent, max_ep1_concurrent, max_ep2_concurrent
            ep_id = context.episode_id

            async with count_lock:
                if ep_id == episode_1:
                    ep1_concurrent += 1
                    max_ep1_concurrent = max(max_ep1_concurrent, ep1_concurrent)
                else:
                    ep2_concurrent += 1
                    max_ep2_concurrent = max(max_ep2_concurrent, ep2_concurrent)

            await asyncio.sleep(0.05)

            async with count_lock:
                if ep_id == episode_1:
                    ep1_concurrent -= 1
                else:
                    ep2_concurrent -= 1

            return CommandResult.success_result(data={"result": "done"})

        mock_executor = AsyncMock(side_effect=mock_execution)
        mock_executor.get_parameters_class = MagicMock(return_value=BashParameters)
        mock_executor.validate_parameters = MagicMock(return_value=ValidationResult.success())

        # Create tasks for both episodes
        num_tasks_per_episode = max_concurrent * 2
        tasks = []
        for i in range(num_tasks_per_episode):
            action = Action(tool_name="bash", parameters={"command": f"echo {i}"})
            tasks.append((action, {"episode_id": episode_1}))
            tasks.append((action, {"episode_id": episode_2}))

        with patch.object(execution_manager._executor_factory, "get_executor", return_value=mock_executor):
            results = await asyncio.gather(
                *[execution_manager.step(action, ctx) for action, ctx in tasks]
            )

        assert all(r.success for r in results)

        # Each episode should respect its own limit independently
        assert max_ep1_concurrent <= max_concurrent
        assert max_ep2_concurrent <= max_concurrent
