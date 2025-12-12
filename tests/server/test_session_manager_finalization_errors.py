"""
Unit tests for SessionManager episode finalization error paths.

Tests the _finalize_episode_creation error handling for various failure scenarios:
- Prompt generation failures
- Episode manager configuration failures
- File copy failures
- Evaluation configuration failures
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saber.server.base import Episode, EpisodeState
from saber.server.benchmarks.task import Task
from saber.server.session_manager import SessionManager


class TestFinalizationErrors:
    """Test error handling in episode finalization."""

    @pytest.fixture
    def mock_task(self):
        """Create a mock task."""
        return Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="Test description",
            prompts={"instruction": "test.md", "assistant": "test.md", "submit": "test.md", "continue": "test_continue.md"},
            initial_context={},
        )

    @pytest.fixture
    async def session_manager_with_episode(self):
        """Create session manager with mocked dependencies."""
        with patch('saber.server.session_manager.BenchmarkManager') as mock_bm, \
             patch('saber.server.session_manager.ExecutionManager') as mock_exec, \
             patch('saber.server.session_manager.PolicyManager') as mock_policy, \
             patch('saber.server.session_manager.EvaluationManager') as mock_eval, \
             patch('saber.server.session_manager.EpisodeManager') as mock_ep:

            # Setup basic mocks
            mock_bm.return_value.prompt_generator = MagicMock()
            mock_bm.return_value.prompt_generator.render_agent_prompts_for_task = MagicMock()

            mock_exec.return_value.wait_for_episode_healthy = MagicMock(
                return_value=AsyncMock(return_value=None)()
            )
            mock_exec.return_value.copy_initial_files_to_episode = AsyncMock()

            mock_policy.return_value.set_episode_policy = MagicMock()

            mock_eval.return_value.log_session_start = AsyncMock()
            mock_eval.return_value.configure_for_task = MagicMock()
            mock_eval.return_value.log_episode_start = AsyncMock()

            mock_ep.return_value.configure_for_task = AsyncMock()
            mock_ep.return_value.mark_episode_failed_creation = MagicMock()
            mock_ep.return_value.get_episode_by_id = MagicMock()

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8001)
            session = await manager.create_session("test-client")

            # Create episode
            episode = MagicMock(spec=Episode)
            episode.episode_id = "episode_123"
            episode.task_id = "test_task"
            episode.state = EpisodeState.CREATING

            manager.episode_manager.get_episode_by_id.return_value = episode

            yield manager, session, episode

    @pytest.mark.asyncio
    async def test_prompt_generation_failure(self, session_manager_with_episode, mock_task):
        """Test finalization when prompt generation fails."""
        manager, session, episode = session_manager_with_episode

        # Make prompt generation raise an exception
        manager.benchmark_manager.prompt_generator.render_agent_prompts_for_task.side_effect = Exception(
            "Prompt generation error"
        )

        # Mock cleanup method
        with patch.object(manager, '_cleanup_failed_episode_environment', new_callable=AsyncMock) as mock_cleanup:
            # Call finalization
            await manager._finalize_episode_creation(
                episode_id="episode_123",
                session_id=session.session_id,
                task_id="test_task",
                task=mock_task
            )

            # Wait for background processing
            await asyncio.sleep(0.1)

            # Verify episode marked as failed
            manager.episode_manager.mark_episode_failed_creation.assert_called_once()
            call_args = manager.episode_manager.mark_episode_failed_creation.call_args
            assert call_args[0][0] == "episode_123"
            assert "Prompt generation failed" in call_args[0][1]

            # Verify cleanup was called
            mock_cleanup.assert_called_once_with("episode_123")

    @pytest.mark.asyncio
    async def test_episode_manager_config_failure(self, session_manager_with_episode, mock_task):
        """Test finalization when episode manager configuration fails."""
        manager, session, episode = session_manager_with_episode

        # Make prompt generation succeed
        manager.benchmark_manager.prompt_generator.render_agent_prompts_for_task.return_value = {
            "instruction": "test instruction"
        }

        # Make episode manager config fail
        manager.episode_manager.configure_for_task.side_effect = Exception("Config error")

        with patch.object(manager, '_cleanup_failed_episode_environment', new_callable=AsyncMock) as mock_cleanup:
            await manager._finalize_episode_creation(
                episode_id="episode_123",
                session_id=session.session_id,
                task_id="test_task",
                task=mock_task
            )

            await asyncio.sleep(0.1)

            # Verify episode marked as failed with correct message
            manager.episode_manager.mark_episode_failed_creation.assert_called_once()
            call_args = manager.episode_manager.mark_episode_failed_creation.call_args
            assert "Episode configuration failed" in call_args[0][1]

            mock_cleanup.assert_called_once_with("episode_123")

    @pytest.mark.asyncio
    async def test_file_copy_failure(self, session_manager_with_episode, mock_task):
        """Test finalization when initial file copy fails."""
        manager, session, episode = session_manager_with_episode

        # Make earlier steps succeed
        manager.benchmark_manager.prompt_generator.render_agent_prompts_for_task.return_value = {
            "instruction": "test instruction"
        }
        manager.episode_manager.configure_for_task = AsyncMock()

        # Make file copy fail
        manager.execution_manager.copy_initial_files_to_episode.side_effect = Exception("File copy error")

        with patch.object(manager, '_cleanup_failed_episode_environment', new_callable=AsyncMock) as mock_cleanup:
            await manager._finalize_episode_creation(
                episode_id="episode_123",
                session_id=session.session_id,
                task_id="test_task",
                task=mock_task
            )

            await asyncio.sleep(0.1)

            # Verify episode marked as failed
            manager.episode_manager.mark_episode_failed_creation.assert_called_once()
            call_args = manager.episode_manager.mark_episode_failed_creation.call_args
            assert "File copy failed" in call_args[0][1]

            mock_cleanup.assert_called_once_with("episode_123")

    @pytest.mark.asyncio
    async def test_evaluation_config_failure(self, session_manager_with_episode, mock_task):
        """Test finalization when evaluation configuration fails."""
        manager, session, episode = session_manager_with_episode

        # Make earlier steps succeed
        manager.benchmark_manager.prompt_generator.render_agent_prompts_for_task.return_value = {
            "instruction": "test instruction"
        }
        manager.episode_manager.configure_for_task = AsyncMock()
        manager.execution_manager.copy_initial_files_to_episode = AsyncMock()

        # Make evaluation config fail
        manager.evaluation_manager.configure_for_task.side_effect = Exception("Eval config error")

        with patch.object(manager, '_cleanup_failed_episode_environment', new_callable=AsyncMock) as mock_cleanup:
            await manager._finalize_episode_creation(
                episode_id="episode_123",
                session_id=session.session_id,
                task_id="test_task",
                task=mock_task
            )

            await asyncio.sleep(0.1)

            # Verify episode marked as failed
            manager.episode_manager.mark_episode_failed_creation.assert_called_once()
            call_args = manager.episode_manager.mark_episode_failed_creation.call_args
            assert "Evaluation configuration failed" in call_args[0][1]

            mock_cleanup.assert_called_once_with("episode_123")

    @pytest.mark.asyncio
    async def test_episode_start_logging_failure_nonfatal(self, session_manager_with_episode, mock_task):
        """Test that episode start logging failures are non-fatal."""
        manager, session, episode = session_manager_with_episode

        # Make all earlier steps succeed
        manager.benchmark_manager.prompt_generator.render_agent_prompts_for_task.return_value = {
            "instruction": "test instruction"
        }
        manager.episode_manager.configure_for_task = AsyncMock()
        manager.execution_manager.copy_initial_files_to_episode = AsyncMock()
        manager.evaluation_manager.configure_for_task = MagicMock()

        # Make episode start logging fail (should be non-fatal)
        manager.evaluation_manager.log_episode_start.side_effect = Exception("Logging error")

        # Mock mark_episode_ready to verify it's still called
        manager.episode_manager.mark_episode_ready = MagicMock()

        await manager._finalize_episode_creation(
            episode_id="episode_123",
            session_id=session.session_id,
            task_id="test_task",
            task=mock_task
        )

        await asyncio.sleep(0.1)

        # Episode should NOT be marked as failed - logging failure is non-fatal
        manager.episode_manager.mark_episode_failed_creation.assert_not_called()

        # Episode should be marked as ready
        manager.episode_manager.mark_episode_ready.assert_called_once_with("episode_123")
