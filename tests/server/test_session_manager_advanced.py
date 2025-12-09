"""
Advanced test coverage for SessionManager to reach >90% coverage.

Tests for:
- end_episode idempotency and edge cases
- execute_action with should_terminate
- Episode creation Docker failures
- Various error paths and edge cases
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException

from saber.models import EvalSubmission
from saber.models.benchmark_task import SingleEpisodeTask
from saber.server.base import Action, CommandResult, Episode, EpisodeState
from saber.server.episodes.constants import EpisodeTerminationReason
from saber.server.episodes.episode_manager import StepResult
from saber.server.session_manager import SessionManager


class TestSessionManagerEndEpisode:
    """Test end_episode functionality including edge cases."""

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")
            manager.execution_manager.cleanup_episode = MagicMock(return_value=True)
            manager.episode_manager.end_episode = AsyncMock()
            return manager

    @pytest.mark.asyncio
    async def test_end_episode_idempotent_already_complete(self, session_manager):
        """Test ending an already completed episode (idempotent)."""
        session = await session_manager.create_session("test-client")
        episode_id = "completed-episode"

        # Create episode
        episode = MagicMock()
        episode.episode_id = episode_id
        episode.task_id = "task1"
        episode.is_complete = True
        episode.state = EpisodeState.COMPLETED
        episode.completion_reason = "completed"
        session_manager.episode_manager.get_episode_by_id.return_value = episode

        session.add_active_episode(episode_id)

        # End already completed episode
        response = await session_manager.end_episode(session.session_id, episode_id)

        # Should return success without error
        assert response.episode_ended is True
        assert response.success is True
        assert response.reason == "completed"

    @pytest.mark.asyncio
    async def test_end_episode_in_history(self, session_manager):
        """Test ending episode that's in history but not marked complete."""
        session = await session_manager.create_session("test-client")
        episode_id = "history-episode"

        # Create episode
        episode = MagicMock()
        episode.episode_id = episode_id
        episode.task_id = "task1"
        episode.is_complete = False
        session_manager.episode_manager.get_episode_by_id.return_value = episode

        # Add to history (but not active)
        session.episode_history.append(episode_id)

        # End episode
        response = await session_manager.end_episode(session.session_id, episode_id)

        # Should return success
        assert response.episode_ended is True
        assert response.success is True
        assert response.reason == "previously_completed"

    @pytest.mark.asyncio
    async def test_end_episode_not_active(self, session_manager):
        """Test ending episode that's not in active list."""
        session = await session_manager.create_session("test-client")
        episode_id = "not-active-episode"

        # Create episode
        episode = MagicMock()
        episode.episode_id = episode_id
        episode.task_id = "task1"
        episode.is_complete = False
        session_manager.episode_manager.get_episode_by_id.return_value = episode

        # Episode not in active list or history
        with pytest.raises(HTTPException) as exc_info:
            await session_manager.end_episode(session.session_id, episode_id)

        assert exc_info.value.status_code == 400
        assert "not active" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_end_episode_with_submission(self, session_manager):
        """Test ending episode with submission."""
        session = await session_manager.create_session("test-client")
        episode_id = "episode-with-submission"

        # Create episode
        episode = MagicMock()
        episode.episode_id = episode_id
        episode.task_id = "task1"
        episode.is_complete = False
        session_manager.episode_manager.get_episode_by_id.return_value = episode
        session_manager.episode_manager.end_episode.return_value = episode

        task = MagicMock()
        session_manager.benchmark_manager.get_task.return_value = task

        session.add_active_episode(episode_id)

        # Create proper submission
        submission = EvalSubmission(
            episode_id=episode_id,
            task_id="task1",
            model="test-model",
            submission="Test submission",
            time=1.0,
        )

        with patch.object(session_manager, "_cleanup_episode_network_with_retry", new=AsyncMock(return_value=True)):
            response = await session_manager.end_episode(
                session.session_id, episode_id, reason="completed", submission=submission
            )

        # Episode should have submission set
        assert episode.eval_submission == submission
        assert episode.submission == "Test submission"

    @pytest.mark.asyncio
    async def test_end_episode_without_submission(self, session_manager):
        """Test ending episode without submission."""
        session = await session_manager.create_session("test-client")
        episode_id = "episode-no-submission"

        # Create episode
        episode = MagicMock()
        episode.episode_id = episode_id
        episode.task_id = "task1"
        episode.is_complete = False
        session_manager.episode_manager.get_episode_by_id.return_value = episode
        session_manager.episode_manager.end_episode = AsyncMock(return_value=episode)

        task = MagicMock()
        session_manager.benchmark_manager.get_task.return_value = task

        session.add_active_episode(episode_id)

        with patch.object(session_manager, "_cleanup_episode_network_with_retry", new=AsyncMock(return_value=True)):
            response = await session_manager.end_episode(session.session_id, episode_id, reason="completed")

        # Episode should have default submission
        assert episode.eval_submission is None
        assert "completed without explicit submission" in episode.submission


class TestSessionManagerExecuteAction:
    """Test execute_action functionality."""

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")
            return manager

# test_execute_action_with_should_terminate removed - comprehensive coverage in test_session_manager_auto_terminate.py

    @pytest.mark.asyncio
    async def test_execute_action_episode_not_ready(self, session_manager):
        """Test execute_action when episode is not ready."""
        session = await session_manager.create_session("test-client")
        episode_id = "creating-episode"

        # Create episode in CREATING state
        episode = MagicMock()
        episode.episode_id = episode_id
        episode.state = EpisodeState.CREATING
        episode.is_ready = False
        session_manager.episode_manager.get_episode_by_id.return_value = episode

        session.add_active_episode(episode_id)

        action = Action(tool_name="bash", parameters={"arguments": "test"})

        with pytest.raises(HTTPException) as exc_info:
            await session_manager.execute_action(session.session_id, episode_id, action)

        assert exc_info.value.status_code == 409
        assert "still being created" in str(exc_info.value.detail).lower()


class TestSessionManagerDockerFailures:
    """Test Docker environment failure scenarios."""

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")
            # Setup required mocks
            manager.benchmark_manager.prompt_generator = MagicMock()
            manager.benchmark_manager.prompt_generator.render_agent_prompts_for_task.return_value = {
                "instruction": "test"
            }
            # Mock get_single_episode_task to return proper SingleEpisodeTask
            single_episode_task = SingleEpisodeTask(
                task_id="task1",
                domain="test_domain",
                title="Test Task",
                description="Test description",
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
                episode_attempts=1,
            )
            manager.benchmark_manager.get_single_episode_task = MagicMock(return_value=single_episode_task)
            manager.execution_manager.configure_for_task_async = AsyncMock()
            manager.execution_manager.wait_for_episode_healthy = AsyncMock()
            manager.execution_manager.cleanup_episode = MagicMock()
            manager.episode_manager.configure_for_task = AsyncMock()
            # Mock initialize_episode_context to return proper dict
            manager.episode_manager.initialize_episode_context = MagicMock(return_value={})
            return manager

    @pytest.mark.asyncio
    async def test_start_episode_docker_compose_failure(self, session_manager):
        """Test starting episode when docker compose fails."""
        session = await session_manager.create_session("test-client")

        # Create task
        task = MagicMock()
        task.task_id = "task1"
        task.dependency_template = None
        task.initial_context = {}
        session_manager.benchmark_manager.get_task.return_value = task

        # Make docker compose fail during configure_for_task_async
        # This is called via asyncio.to_thread so it's a regular function
        def fail_configure(*args, **kwargs):
            raise Exception("Docker compose failed")

        session_manager.execution_manager.configure_for_task_async = fail_configure

        with pytest.raises(HTTPException) as exc_info:
            await session_manager.start_episode(session.session_id, "task1")

        assert exc_info.value.status_code == 500
        assert "Failed to start Docker environment" in str(exc_info.value.detail)

        # Should attempt cleanup
        session_manager.execution_manager.cleanup_episode.assert_called()

    # test_start_episode_health_check_timeout removed - duplicate of test_session_manager_async.py::test_failed_health_check_marks_episode_failed_creation


# test_get_current_task removed - duplicate of test_session_manager_episodes.py::test_get_current_task


class TestSessionManagerGetBenchmarkInfo:
    """Test get_benchmark_info functionality."""

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")
            return manager

    def test_get_benchmark_info(self, session_manager):
        """Test getting benchmark information."""
        mock_info = MagicMock()
        session_manager.benchmark_manager.get_benchmark_info.return_value = mock_info

        result = session_manager.get_benchmark_info()

        assert result == mock_info
        session_manager.benchmark_manager.get_benchmark_info.assert_called_once()


class TestSessionManagerGetPolicy:
    """Test get_policy functionality."""

    @pytest.fixture
    def session_manager(self):
        """Create SessionManager with mocked dependencies."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")
            return manager

    @pytest.mark.asyncio
    async def test_get_policy_success(self, session_manager):
        """Test getting policy successfully."""
        # Create a session first since get_policy validates session
        session = await session_manager.create_session("test-client")
        episode_id = "test-episode"

        # Add episode to session
        session.add_active_episode(episode_id)

        # Mock policy document
        from saber.server.policy.policy_manager import PolicyDocument
        mock_policy = PolicyDocument(prompt="test prompt")
        session_manager.policy_manager.get_policy.return_value = mock_policy

        result = session_manager.get_policy(session.session_id, episode_id)

        assert result == mock_policy
        session_manager.policy_manager.get_policy.assert_called_once_with(episode_id)
