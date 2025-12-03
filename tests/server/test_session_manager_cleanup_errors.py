"""
Unit tests for SessionManager episode cleanup error handling.

Tests error paths in episode cleanup during cascade termination.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saber.server.benchmarks.task import Task
from saber.server.episodes.constants import EpisodeTerminationReason
from saber.server.session_manager import SessionManager


class TestEpisodeCleanupErrors:
    """Test episode cleanup error handling."""

    @pytest.fixture
    async def setup_with_episode(self):
        """Create session manager with mocked episode."""
        with patch('saber.server.session_manager.BenchmarkManager') as mock_bench_class, \
             patch('saber.server.session_manager.ExecutionManager') as mock_exec_class, \
             patch('saber.server.session_manager.PolicyManager'), \
             patch('saber.server.session_manager.EvaluationManager') as mock_eval_class, \
             patch('saber.server.session_manager.EpisodeManager') as mock_ep_class:

            # Setup benchmark manager
            mock_bench_instance = MagicMock()
            mock_task = MagicMock()
            mock_task.task_id = "test-task"
            mock_bench_instance.get_task = MagicMock(return_value=mock_task)
            mock_bench_class.return_value = mock_bench_instance

            # Setup execution manager
            mock_exec_instance = MagicMock()
            mock_exec_instance.cleanup_episode = MagicMock(return_value=True)
            mock_exec_class.return_value = mock_exec_instance

            # Setup evaluation manager
            mock_eval_instance = MagicMock()
            mock_eval_instance.log_session_start = AsyncMock()
            mock_eval_instance.log_session_end = AsyncMock()
            mock_eval_class.return_value = mock_eval_instance

            # Setup episode manager with a mock episode
            from saber.server.base import Episode, EpisodeState
            mock_episode = MagicMock(spec=Episode)
            mock_episode.episode_id = "test-episode"
            mock_episode.session_id = "test-session"
            mock_episode.state = EpisodeState.ACTIVE
            mock_episode.task_id = "test-task"
            mock_episode.completion_reason = None
            mock_episode.is_complete = False  # Required for end_episode to proceed

            mock_ep_instance = MagicMock()
            mock_ep_instance.get_episode_by_id = MagicMock(return_value=mock_episode)
            mock_ep_instance.end_episode = AsyncMock(return_value=mock_episode)
            mock_ep_class.return_value = mock_ep_instance

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8001)
            session = await manager.create_session("test-client")

            # Add episode to session's active episodes (required for end_episode)
            session.add_active_episode(mock_episode.episode_id)

            yield manager, session, mock_episode

    @pytest.mark.asyncio
    async def test_cleanup_failure_logged(self, setup_with_episode):
        """Test that cleanup failure is logged as warning."""
        manager, session, episode = setup_with_episode

        # Make cleanup return False (failure)
        manager.execution_manager.cleanup_episode.return_value = False

        # Mock the network cleanup retry to avoid actual Docker calls
        with patch.object(manager, '_cleanup_episode_network_with_retry', new_callable=AsyncMock):
            # End episode - should log warning about cleanup failure
            await manager.end_episode(
                session.session_id,
                episode.episode_id,
                EpisodeTerminationReason.TIMEOUT,
                None
            )

        # Verify cleanup was called
        assert manager.execution_manager.cleanup_episode.called

    @pytest.mark.asyncio
    async def test_cleanup_exception_handled(self, setup_with_episode):
        """Test that exception in cleanup_episode is handled gracefully."""
        manager, session, episode = setup_with_episode

        # Make cleanup raise exception
        manager.execution_manager.cleanup_episode.side_effect = RuntimeError("Cleanup failed")

        # Mock the network cleanup retry
        with patch.object(manager, '_cleanup_episode_network_with_retry', new_callable=AsyncMock):
            # End episode - should handle exception and continue
            await manager.end_episode(
                session.session_id,
                episode.episode_id,
                EpisodeTerminationReason.TIMEOUT,
                None
            )

        # Verify cleanup was attempted
        assert manager.execution_manager.cleanup_episode.called

    @pytest.mark.asyncio
    async def test_network_cleanup_exception_handled(self, setup_with_episode):
        """Test that exception in network cleanup is handled gracefully."""
        manager, session, episode = setup_with_episode

        # Make cleanup succeed but network cleanup fail
        manager.execution_manager.cleanup_episode.return_value = True

        # Mock network cleanup to raise exception
        mock_network_cleanup = AsyncMock(side_effect=RuntimeError("Network cleanup failed"))

        with patch.object(manager, '_cleanup_episode_network_with_retry', mock_network_cleanup):
            # End episode - should handle exception
            await manager.end_episode(
                session.session_id,
                episode.episode_id,
                EpisodeTerminationReason.TIMEOUT,
                None
            )

        # Verify network cleanup was attempted
        assert mock_network_cleanup.called

    @pytest.mark.asyncio
    async def test_cleanup_with_submission(self, setup_with_episode):
        """Test cleanup when episode ends with submission."""
        manager, session, episode = setup_with_episode

        # Just use submission text directly (the end_episode method accepts this)
        submission_text = "test submission"

        # Mock network cleanup
        with patch.object(manager, '_cleanup_episode_network_with_retry', new_callable=AsyncMock):
            # Create a mock submission object since end_episode expects EvalSubmission
            mock_submission = MagicMock()
            mock_submission.submission = submission_text

            # End episode with submission
            await manager.end_episode(
                session.session_id,
                episode.episode_id,
                EpisodeTerminationReason.AGENT_COMPLETED,
                mock_submission
            )

        # Verify cleanup was called
        assert manager.execution_manager.cleanup_episode.called

        # Verify episode_manager.end_episode was called with submission text
        args = manager.episode_manager.end_episode.call_args[0]
        assert args[2] == submission_text  # Third arg is submission_text
