"""
Unit tests for SessionManager network cleanup functionality.

Tests the _cleanup_saber_episode_networks method which cleans up Docker networks
to prevent subnet pool exhaustion.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saber.server.session_manager import SessionManager


class TestNetworkCleanup:
    """Test network cleanup functionality."""

    @pytest.fixture
    async def session_manager(self):
        """Create a session manager for testing."""
        with patch('saber.server.session_manager.BenchmarkManager'), \
             patch('saber.server.session_manager.ExecutionManager'), \
             patch('saber.server.session_manager.PolicyManager'), \
             patch('saber.server.session_manager.EvaluationManager') as mock_eval, \
             patch('saber.server.session_manager.EpisodeManager'):

            mock_eval.return_value.log_session_start = AsyncMock()
            mock_eval.return_value.log_session_end = AsyncMock()

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8001)
            yield manager

    @pytest.mark.asyncio
    async def test_network_cleanup_list_failure(self, session_manager):
        """Test network cleanup when 'docker network ls' fails."""
        # Mock subprocess that fails on network list
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"error listing networks"))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            # Should return early without raising
            await session_manager._cleanup_saber_episode_networks()

        # Verify communicate was called (process was executed)
        assert mock_process.communicate.called

    @pytest.mark.asyncio
    async def test_network_cleanup_no_networks(self, session_manager):
        """Test network cleanup when no networks are found."""
        # Mock subprocess that returns empty list
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await session_manager._cleanup_saber_episode_networks()

        assert mock_process.communicate.called

    @pytest.mark.asyncio
    async def test_network_cleanup_inspect_failure(self, session_manager):
        """Test network cleanup when 'docker network inspect' fails."""
        call_count = [0]

        async def create_subprocess_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_proc = MagicMock()

            # First call: docker network ls (succeeds)
            if call_count[0] == 1:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"saber-episode-test1\n", b""))
            # Second call: docker network inspect (fails)
            elif call_count[0] == 2:
                mock_proc.returncode = 1
                mock_proc.communicate = AsyncMock(return_value=(b"", b"network not found"))
            # Third call: docker network rm
            else:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"saber-episode-test1\n", b""))

            return mock_proc

        with patch('asyncio.create_subprocess_exec', side_effect=create_subprocess_side_effect):
            await session_manager._cleanup_saber_episode_networks()

        # Should have made 3 calls: ls, inspect, rm
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_network_cleanup_with_containers(self, session_manager):
        """Test network cleanup when network has attached containers."""
        call_count = [0]

        async def create_subprocess_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_proc = MagicMock()

            # First call: docker network ls
            if call_count[0] == 1:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"saber-episode-test1\n", b""))
            # Second call: docker network inspect (returns containers)
            elif call_count[0] == 2:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"container1 container2 ", b""))
            # Third and fourth calls: docker rm -f container1, container2
            elif call_count[0] in [3, 4]:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            # Fifth call: docker network rm
            else:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"saber-episode-test1\n", b""))

            return mock_proc

        with patch('asyncio.create_subprocess_exec', side_effect=create_subprocess_side_effect):
            await session_manager._cleanup_saber_episode_networks()

        # Should have made 5 calls: ls, inspect, rm container1, rm container2, network rm
        assert call_count[0] == 5

    @pytest.mark.asyncio
    async def test_network_cleanup_container_exception(self, session_manager):
        """Test network cleanup when container cleanup raises exception."""
        call_count = [0]

        async def create_subprocess_side_effect(*args, **kwargs):
            call_count[0] += 1

            # First call: docker network ls
            if call_count[0] == 1:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"saber-episode-test1\n", b""))
                return mock_proc
            # Second call: docker network inspect - raise exception
            elif call_count[0] == 2:
                raise RuntimeError("Inspect failed")
            # Third call: docker network rm
            else:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"saber-episode-test1\n", b""))
                return mock_proc

        with patch('asyncio.create_subprocess_exec', side_effect=create_subprocess_side_effect):
            # Should not raise despite exception
            await session_manager._cleanup_saber_episode_networks()

        # Should continue to network removal despite exception
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_network_cleanup_removal_failure(self, session_manager):
        """Test network cleanup when 'docker network rm' fails."""
        call_count = [0]

        async def create_subprocess_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_proc = MagicMock()

            # First call: docker network ls
            if call_count[0] == 1:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"saber-episode-test1\n", b""))
            # Second call: docker network inspect (no containers)
            elif call_count[0] == 2:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            # Third call: docker network rm (fails)
            else:
                mock_proc.returncode = 1
                mock_proc.communicate = AsyncMock(return_value=(b"", b"network in use"))

            return mock_proc

        with patch('asyncio.create_subprocess_exec', side_effect=create_subprocess_side_effect):
            await session_manager._cleanup_saber_episode_networks()

        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_network_cleanup_removal_exception(self, session_manager):
        """Test network cleanup when network removal raises exception."""
        call_count = [0]

        async def create_subprocess_side_effect(*args, **kwargs):
            call_count[0] += 1

            # First call: docker network ls
            if call_count[0] == 1:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"saber-episode-test1\n", b""))
                return mock_proc
            # Second call: docker network inspect
            elif call_count[0] == 2:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                return mock_proc
            # Third call: docker network rm - raise exception
            else:
                raise RuntimeError("Network removal failed")

        with patch('asyncio.create_subprocess_exec', side_effect=create_subprocess_side_effect):
            # Should not raise despite exception
            await session_manager._cleanup_saber_episode_networks()

        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_network_cleanup_multiple_networks(self, session_manager):
        """Test network cleanup with multiple networks."""
        call_count = [0]

        async def create_subprocess_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_proc = MagicMock()

            # First call: docker network ls (2 networks)
            if call_count[0] == 1:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"saber-episode-test1\nsaber-episode-test2\n", b""))
            # Inspect calls for each network
            elif call_count[0] in [2, 3]:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            # Remove calls for each network
            else:
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"network removed\n", b""))

            return mock_proc

        with patch('asyncio.create_subprocess_exec', side_effect=create_subprocess_side_effect):
            await session_manager._cleanup_saber_episode_networks()

        # Should make: 1 ls + 2 inspects + 2 removes = 5 calls
        assert call_count[0] == 5
