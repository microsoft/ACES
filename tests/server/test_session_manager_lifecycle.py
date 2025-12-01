"""Tests for session shutdown, cleanup, and lifecycle management."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from saber.server.base import Episode, EpisodeState
from saber.server.episodes.constants import EpisodeTerminationReason
from saber.server.session_manager import SessionManager


class TestSessionShutdown:
    """Test session manager shutdown paths."""

    @pytest.mark.asyncio
    async def test_shutdown_with_running_cleanup_task(self):
        """Test shutdown cancels running cleanup task."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            manager = SessionManager(
                domain_name="test_domain",
                config_dir="/tmp/config",
                host="127.0.0.1",
                port=8000,
                cleanup_interval_minutes=1,
            )

            # Start cleanup loop
            manager.cleanup_task = asyncio.create_task(manager._session_cleanup_loop())
            await asyncio.sleep(0.05)  # Let it start

            # Now shutdown
            await manager.shutdown()

            # Cleanup task should be cancelled
            assert manager.cleanup_task.cancelled() or manager.cleanup_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_with_permanent_environment_running(self):
        """Test shutdown stops permanent environment."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            exec_manager = MagicMock()
            exec_manager.is_permanent_environment_running.return_value = True
            exec_manager.stop_permanent_environment = MagicMock()

            with patch("saber.server.session_manager.ExecutionManager", return_value=exec_manager):
                manager = SessionManager(
                    domain_name="test_domain",
                    config_dir="/tmp/config",
                    host="127.0.0.1",
                    port=8000,
                )

                await manager.shutdown()

                # Should stop permanent environment
                exec_manager.stop_permanent_environment.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_permanent_environment_stop_error(self):
        """Test shutdown handles permanent environment stop errors."""
        with (
            patch("saber.server.session_manager.BenchmarkManager"),
            patch("saber.server.session_manager.ExecutionManager"),
            patch("saber.server.session_manager.PolicyManager"),
            patch("saber.server.session_manager.EvaluationManager"),
            patch("saber.server.session_manager.EpisodeManager"),
        ):
            exec_manager = MagicMock()
            exec_manager.is_permanent_environment_running.return_value = True
            exec_manager.stop_permanent_environment.side_effect = Exception("Stop error")

            with patch("saber.server.session_manager.ExecutionManager", return_value=exec_manager):
                manager = SessionManager(
                    domain_name="test_domain",
                    config_dir="/tmp/config",
                    host="127.0.0.1",
                    port=8000,
                )

                # Should not raise, just log
                await manager.shutdown()


class TestInactiveSessionCleanup:
    """Test inactive session cleanup with orphaned episodes."""

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
            manager = SessionManager(
                domain_name="test_domain",
                config_dir="/tmp/config",
                host="127.0.0.1",
                port=8000,
                session_timeout_minutes=30,
            )
            return manager

    @pytest.mark.asyncio
    async def test_cleanup_skips_complete_orphaned_episodes(self, session_manager):
        """Test cleanup skips already completed orphaned episodes."""
        session = await session_manager.create_session("client1")
        old_time = datetime.utcnow() - timedelta(minutes=session_manager.session_timeout_minutes + 1)
        session.last_activity = old_time

        session.add_active_episode("ep1")

        # Episode already complete
        episode1 = MagicMock()
        episode1.episode_id = "ep1"
        episode1.is_complete = True

        session_manager.episode_manager.get_episode_by_id.return_value = episode1
        session_manager.episode_manager.end_episode = MagicMock()
        session_manager.evaluation_manager.log_session_end = AsyncMock()

        await session_manager._cleanup_inactive_sessions()

        # Should not try to end already complete episode
        session_manager.episode_manager.end_episode.assert_not_called()
