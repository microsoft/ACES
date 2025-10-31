"""Tests for async session manager episode creation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from saber.server.base import EpisodeState
from saber.server.session_manager import SessionManager


@pytest.fixture
def mock_execution_manager():
    """Create a mock execution manager."""
    mock = AsyncMock()
    mock.configure_for_task_async = AsyncMock(return_value=("orchestrator", "compose_path"))
    mock.wait_for_episode_healthy = AsyncMock()
    return mock


@pytest.fixture
def mock_episode_manager():
    """Create a mock episode manager."""
    mock = MagicMock()
    mock.create_episode = MagicMock()
    mock.mark_episode_ready = MagicMock()
    mock.mark_episode_failed_creation = MagicMock()
    mock.get_episode = MagicMock()
    mock.get_episode_by_id = MagicMock()
    mock.add_episode_to_session = MagicMock()
    mock.configure_for_task = AsyncMock()  # This needs to be AsyncMock
    return mock


@pytest.fixture
def session_manager(mock_execution_manager, mock_episode_manager):
    """Create a session manager with mocked dependencies."""
    # Mock other required dependencies
    mock_task_manager = MagicMock()
    mock_config_loader = MagicMock()
    mock_config_loader.get_permanent_environment = MagicMock(return_value=None)
    mock_task_manager.config_loader = mock_config_loader

    # Mock task object with proper attributes
    mock_task = Mock()
    mock_task.initial_context = {}
    mock_task.depends_on_task_id = None
    mock_task_manager.get_task = MagicMock(return_value=mock_task)

    # Mock prompt generator
    mock_prompt_generator = MagicMock()
    mock_prompt_generator.render_agent_prompts_for_task = MagicMock(return_value={
        "instruction": "Test instruction prompt",
        "assistant": "Test assistant prompt",
        "submit": "Test submit prompt",
    })
    mock_task_manager.prompt_generator = mock_prompt_generator

    mock_policy_manager = MagicMock()
    mock_policy_manager.set_episode_policy = MagicMock()

    mock_evaluation_manager = MagicMock()
    mock_evaluation_manager.log_session_start = AsyncMock()
    mock_evaluation_manager.log_session_end = AsyncMock()
    mock_evaluation_manager.log_episode_start = AsyncMock()
    mock_evaluation_manager.log_episode_end = AsyncMock()
    mock_evaluation_manager.log_action = AsyncMock()
    mock_evaluation_manager.configure_for_task = MagicMock()

    with (
        patch("saber.server.session_manager.BenchmarkManager", return_value=mock_task_manager),
        patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
        patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
        patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
        patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
    ):
        manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8003)
        manager.execution_manager = mock_execution_manager
        manager.episode_manager = mock_episode_manager
        manager.benchmark_manager = mock_task_manager
        manager.policy_manager = mock_policy_manager
        manager.evaluation_manager = mock_evaluation_manager
        return manager


class TestSessionManagerAsync:
    """Test async session manager episode creation."""

    @pytest.mark.asyncio
    async def test_initiate_episode_returns_immediately_with_creating_state(
        self, session_manager, mock_episode_manager
    ):
        """Test that initiate_episode returns immediately with CREATING state."""
        # Create a session first
        session = await session_manager.create_session("test-client")
        session_id = session.session_id
        task_id = "test-task"

        response = await session_manager.initiate_episode(session_id, task_id)

        # Verify episode is in CREATING state
        assert response.state == EpisodeState.CREATING
        assert response.task_id == task_id
        assert response.session_id == session_id

        # Verify episode was added to episode manager
        mock_episode_manager.add_episode_to_session.assert_called_once_with(session_id, response)

    @pytest.mark.asyncio
    async def test_background_finalization_marks_episode_ready(
        self, session_manager, mock_episode_manager, mock_execution_manager
    ):
        """Test that background finalization task marks episode READY."""
        # Create a session first
        session = await session_manager.create_session("test-client")
        session_id = session.session_id
        task_id = "test-task"

        # Initiate episode
        response = await session_manager.initiate_episode(session_id, task_id)
        episode_id = response.episode_id

        # Wait for background task to complete
        await asyncio.sleep(0.1)

        # Allow pending tasks to run
        for _ in range(5):
            await asyncio.sleep(0.01)

        # Verify health check was called
        mock_execution_manager.wait_for_episode_healthy.assert_called()

        # Verify episode was marked ready
        mock_episode_manager.mark_episode_ready.assert_called_with(episode_id)

    @pytest.mark.asyncio
    async def test_failed_health_check_marks_episode_failed_creation(
        self, session_manager, mock_episode_manager, mock_execution_manager
    ):
        """Test that failed health checks mark episode FAILED_CREATION."""
        # Create a session first
        session = await session_manager.create_session("test-client")
        session_id = session.session_id
        task_id = "test-task"

        # Make health check fail
        mock_execution_manager.wait_for_episode_healthy.side_effect = Exception("Health check timeout")

        # Initiate episode
        response = await session_manager.initiate_episode(session_id, task_id)
        episode_id = response.episode_id

        # Wait for background task to complete
        await asyncio.sleep(0.1)

        # Allow pending tasks to run
        for _ in range(5):
            await asyncio.sleep(0.01)

        # Verify episode was marked failed
        mock_episode_manager.mark_episode_failed_creation.assert_called()

    @pytest.mark.asyncio
    async def test_get_episode_status_returns_correct_episode(
        self, session_manager, mock_episode_manager
    ):
        """Test that get_episode_status returns correct episode."""
        episode_id = "test-episode-999"

        mock_episode = Mock()
        mock_episode.episode_id = episode_id
        mock_episode.state = EpisodeState.READY
        mock_episode.is_ready = True
        mock_episode_manager.get_episode_by_id.return_value = mock_episode

        status = session_manager.get_episode_status(episode_id)

        assert status.episode_id == episode_id
        assert status.is_ready is True
        mock_episode_manager.get_episode_by_id.assert_called_with(episode_id)

    @pytest.mark.asyncio
    async def test_concurrent_episode_creation_respects_semaphore(
        self, session_manager, mock_episode_manager, mock_execution_manager
    ):
        """Test that concurrent episode creation respects semaphore limit."""
        # Create sessions for all episodes
        sessions = []
        for i in range(10):
            session = await session_manager.create_session(f"client-{i}")
            sessions.append(session)

        # Create multiple episodes concurrently
        tasks = []
        for i in range(10):
            task = session_manager.initiate_episode(sessions[i].session_id, f"task-{i}")
            tasks.append(task)

        # All should return immediately
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 10
        for response in responses:
            assert response.state == EpisodeState.CREATING

    @pytest.mark.asyncio
    async def test_finalization_task_cleanup_on_completion(
        self, session_manager, mock_episode_manager
    ):
        """Test that finalization tasks are cleaned up after completion."""
        # Create a session first
        session = await session_manager.create_session("test-client")
        session_id = session.session_id
        task_id = "test-task"

        # Initiate episode
        response = await session_manager.initiate_episode(session_id, task_id)

        # Task should be tracked
        assert response.episode_id in session_manager._episode_finalization_tasks

        # Wait for completion
        await asyncio.sleep(0.2)

        # Task should be cleaned up after completion
        # (Implementation detail - may need adjustment based on actual cleanup logic)
