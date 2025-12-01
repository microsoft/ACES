"""
Unit tests for SessionManager orphaned episode cleanup.

Tests the orphaned episode cleanup code that runs when episodes remain in
creating/active lists after normal termination fails.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saber.server.base import Episode, EpisodeState
from saber.server.session_manager import SessionManager


class TestOrphanedCleanup:
    """Test orphaned episode cleanup during session termination."""

    @pytest.fixture
    async def session_with_orphaned_episodes(self):
        """Create session manager with orphaned episodes."""
        with patch('saber.server.session_manager.BenchmarkManager'), \
             patch('saber.server.session_manager.ExecutionManager') as mock_exec_class, \
             patch('saber.server.session_manager.PolicyManager'), \
             patch('saber.server.session_manager.EvaluationManager') as mock_eval_class, \
             patch('saber.server.session_manager.EpisodeManager') as mock_ep_class:

            # Create mock instances
            mock_exec_instance = MagicMock()
            mock_exec_instance.cleanup_episode = MagicMock(return_value=True)
            mock_exec_class.return_value = mock_exec_instance

            mock_eval_instance = MagicMock()
            mock_eval_instance.log_session_start = AsyncMock()
            mock_eval_instance.log_session_end = AsyncMock()
            mock_eval_class.return_value = mock_eval_instance

            mock_ep_instance = MagicMock()
            mock_ep_instance.get_episode_by_id = MagicMock()
            mock_ep_instance.end_episode = MagicMock()
            mock_ep_class.return_value = mock_ep_instance

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8001)
            session = await manager.create_session("test-client")

            # Create mock episodes
            episode1 = MagicMock(spec=Episode)
            episode1.episode_id = "orphaned_1"
            episode1.is_complete = False

            episode2 = MagicMock(spec=Episode)
            episode2.episode_id = "orphaned_2"
            episode2.is_complete = False

            def get_episode_side_effect(eid):
                if eid == "orphaned_1":
                    return episode1
                elif eid == "orphaned_2":
                    return episode2
                return None

            manager.episode_manager.get_episode_by_id.side_effect = get_episode_side_effect

            yield manager, session, [episode1, episode2]

    @pytest.mark.asyncio
    async def test_orphaned_cleanup_success(self, session_with_orphaned_episodes):
        """Test orphaned episode cleanup when cleanup succeeds."""
        manager, session, episodes = session_with_orphaned_episodes

        # Add episodes to active list to simulate orphaned episodes
        session.active_episode_ids.append("orphaned_1")
        session.active_episode_ids.append("orphaned_2")

        # Make cleanup return True (success)
        manager.execution_manager.cleanup_episode.return_value = True

        # Make the asyncio.gather fail to prevent list clearing
        # This simulates a critical failure in the cleanup phase
        with patch('saber.server.session_manager.asyncio.gather', side_effect=RuntimeError("Gather failed")):
            # Terminate session - orphaned cleanup should run after gather fails
            await manager.terminate_session(session.session_id)

        # Check that orphaned cleanup was called with correct parameters
        calls = manager.execution_manager.cleanup_episode.call_args_list
        orphaned_calls = [c for c in calls if len(c[0]) > 1 and isinstance(c[0][1], dict) and c[0][1].get("orphaned_check") is True]
        assert len(orphaned_calls) >= 2, f"Expected at least 2 orphaned cleanup calls, got {len(orphaned_calls)}"

    @pytest.mark.asyncio
    async def test_orphaned_cleanup_failure(self, session_with_orphaned_episodes):
        """Test orphaned episode cleanup when cleanup fails."""
        manager, session, episodes = session_with_orphaned_episodes

        # Add episode to active list
        session.active_episode_ids.append("orphaned_1")

        # Make cleanup return False (failure)
        manager.execution_manager.cleanup_episode.return_value = False

        # Make the asyncio.gather fail
        with patch('saber.server.session_manager.asyncio.gather', side_effect=RuntimeError("Gather failed")):
            await manager.terminate_session(session.session_id)

        # Verify orphaned cleanup was attempted
        calls = manager.execution_manager.cleanup_episode.call_args_list
        orphaned_calls = [c for c in calls if len(c[0]) > 1 and isinstance(c[0][1], dict) and c[0][1].get("orphaned_check") is True]
        assert len(orphaned_calls) >= 1, f"Expected at least 1 orphaned cleanup call, got {len(orphaned_calls)}"

    @pytest.mark.asyncio
    async def test_orphaned_cleanup_exception(self, session_with_orphaned_episodes):
        """Test orphaned episode cleanup when cleanup raises exception."""
        manager, session, episodes = session_with_orphaned_episodes

        # Add episode to creating list (different from active)
        session.creating_episode_ids.append("orphaned_1")

        # Make cleanup raise exception
        manager.execution_manager.cleanup_episode.side_effect = Exception("Cleanup exception")

        # Make the asyncio.gather fail
        with patch('saber.server.session_manager.asyncio.gather', side_effect=RuntimeError("Gather failed")):
            await manager.terminate_session(session.session_id)

        # Verify cleanup was attempted
        calls = manager.execution_manager.cleanup_episode.call_args_list
        orphaned_calls = [c for c in calls if len(c[0]) > 1 and isinstance(c[0][1], dict) and c[0][1].get("orphaned_check") is True]
        assert len(orphaned_calls) >= 1, f"Expected at least 1 orphaned cleanup call, got {len(orphaned_calls)}"

    @pytest.mark.asyncio
    async def test_orphaned_cleanup_mixed_results(self, session_with_orphaned_episodes):
        """Test orphaned cleanup with mixed success/failure."""
        manager, session, episodes = session_with_orphaned_episodes

        # Add multiple episodes
        session.active_episode_ids.append("orphaned_1")
        session.creating_episode_ids.append("orphaned_2")

        # Make cleanup return different results
        call_count = [0]
        def cleanup_side_effect(episode_id, metadata):
            call_count[0] += 1
            # First call succeeds, second fails
            return call_count[0] == 1

        manager.execution_manager.cleanup_episode.side_effect = cleanup_side_effect

        # Make the asyncio.gather fail
        with patch('saber.server.session_manager.asyncio.gather', side_effect=RuntimeError("Gather failed")):
            await manager.terminate_session(session.session_id)

        # Verify cleanup was attempted for both
        calls = manager.execution_manager.cleanup_episode.call_args_list
        orphaned_calls = [c for c in calls if len(c[0]) > 1 and isinstance(c[0][1], dict) and c[0][1].get("orphaned_check") is True]
        assert len(orphaned_calls) >= 2, f"Expected at least 2 orphaned cleanup calls, got {len(orphaned_calls)}"

    @pytest.mark.asyncio
    async def test_no_orphaned_cleanup_when_lists_empty(self):
        """Test that orphaned cleanup doesn't run when episode lists are empty."""
        with patch('saber.server.session_manager.BenchmarkManager'), \
             patch('saber.server.session_manager.ExecutionManager') as mock_exec, \
             patch('saber.server.session_manager.PolicyManager'), \
             patch('saber.server.session_manager.EvaluationManager') as mock_eval, \
             patch('saber.server.session_manager.EpisodeManager'):

            mock_exec.return_value.cleanup_episode = MagicMock()
            mock_eval.return_value.log_session_start = AsyncMock()
            mock_eval.return_value.log_session_end = AsyncMock()

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8001)
            session = await manager.create_session("test-client")

            # Don't add any episodes - lists are empty

            # Terminate session
            await manager.terminate_session(session.session_id)

            # Verify no orphaned cleanup was attempted
            cleanup_calls = manager.execution_manager.cleanup_episode.call_args_list
            orphaned_calls = [c for c in cleanup_calls if len(c[0]) > 1 and isinstance(c[0][1], dict) and c[0][1].get("orphaned_check") is True]
            assert len(orphaned_calls) == 0
