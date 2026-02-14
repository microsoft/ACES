"""Tests for SessionManager RedisManager wiring and Redis container lifecycle.

Verifies that SessionManager creates, connects, and disconnects
RedisManager, handles connection failure gracefully, and manages
the Redis container auto-start/stop lifecycle.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.session_manager import SessionManager


def _make_session_manager() -> SessionManager:
    """Create a SessionManager with all heavy dependencies mocked."""
    with (
        patch("saber.server.session_manager.BenchmarkManager"),
        patch("saber.server.session_manager.ExecutionManager"),
        patch("saber.server.session_manager.PolicyManager"),
        patch("saber.server.session_manager.EvaluationManager"),
        patch("saber.server.session_manager.EpisodeManager"),
    ):
        return SessionManager(
            domain_name="test_domain",
            config_dir="/tmp",
            host="127.0.0.1",
            port=8001,
        )


class TestSessionManagerRedis:
    """Test SessionManager ↔ RedisManager lifecycle."""

    def test_creates_redis_manager(self) -> None:
        """SessionManager.__init__ creates a RedisManager instance."""
        sm = _make_session_manager()
        assert sm.redis_manager is not None

    def test_redis_manager_not_connected_initially(self) -> None:
        """RedisManager is NOT connected after __init__ (connect is async)."""
        sm = _make_session_manager()
        assert not sm.redis_manager.is_connected

    @pytest.mark.asyncio
    async def test_startup_connects_redis(self) -> None:
        """start_server calls redis_manager.connect()."""
        sm = _make_session_manager()
        sm.redis_manager = MagicMock()
        sm.redis_manager.connect = AsyncMock()
        sm.redis_manager.is_connected = True
        sm.redis_manager.client = MagicMock()

        # Mock the rest of start_server machinery so it doesn't actually run servers
        sm._start_redis_container = MagicMock()
        sm._start_permanent_environment_async = AsyncMock()
        sm._session_cleanup_loop = AsyncMock()
        sm.rest_api = MagicMock()
        sm.rest_api.start_server = AsyncMock()
        sm.mcp_api = MagicMock()
        sm.mcp_api.start_mcp_server = AsyncMock()

        # Run start_server in a task and cancel after wiring runs
        task = asyncio.create_task(sm.start_server())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        sm.redis_manager.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_startup_handles_connection_failure(self) -> None:
        """If redis_manager.connect() raises, startup continues (logs warning)."""
        sm = _make_session_manager()
        sm.redis_manager = MagicMock()
        sm.redis_manager.connect = AsyncMock(side_effect=Exception("connection refused"))
        sm.redis_manager.is_connected = False

        sm._start_redis_container = MagicMock()
        sm._start_permanent_environment_async = AsyncMock()
        sm._session_cleanup_loop = AsyncMock()
        sm.rest_api = MagicMock()
        sm.rest_api.start_server = AsyncMock()
        sm.mcp_api = MagicMock()
        sm.mcp_api.start_mcp_server = AsyncMock()

        task = asyncio.create_task(sm.start_server())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # connect was attempted
        sm.redis_manager.connect.assert_awaited_once()
        # episode_manager should NOT have an event_repository injected
        # (we just verify no exception was raised — startup continued)

    @pytest.mark.asyncio
    async def test_startup_wires_event_repository_on_success(self) -> None:
        """On successful connect, an EpisodeEventRepository is created and wired."""
        sm = _make_session_manager()

        mock_client = MagicMock()
        sm.redis_manager = MagicMock()
        sm.redis_manager.connect = AsyncMock()
        sm.redis_manager.is_connected = True
        sm.redis_manager.client = mock_client

        # Use a real EpisodeManager-like mock to check wiring
        sm.episode_manager = MagicMock()
        sm.episode_manager.wire_event_repository = MagicMock()

        sm._start_redis_container = MagicMock()
        sm._start_permanent_environment_async = AsyncMock()
        sm._session_cleanup_loop = AsyncMock()
        sm.rest_api = MagicMock()
        sm.rest_api.start_server = AsyncMock()
        sm.mcp_api = MagicMock()
        sm.mcp_api.start_mcp_server = AsyncMock()

        with patch("saber.server.session_manager.EpisodeEventRepository") as MockRepo:
            mock_repo_instance = MagicMock()
            MockRepo.return_value = mock_repo_instance

            task = asyncio.create_task(sm.start_server())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            MockRepo.assert_called_once_with(mock_client)
            sm.episode_manager.wire_event_repository.assert_called_once_with(mock_repo_instance)

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_redis(self) -> None:
        """shutdown() calls redis_manager.disconnect()."""
        sm = _make_session_manager()
        sm.redis_manager = MagicMock()
        sm.redis_manager.disconnect = AsyncMock()
        sm._stop_redis_container = MagicMock()

        # Stub out everything shutdown touches
        sm.shutdown_event = MagicMock()
        sm.cleanup_task = None
        sm.execution_manager = MagicMock()
        sm.execution_manager.is_permanent_environment_running = MagicMock(return_value=False)
        sm._cleanup_saber_episode_networks = AsyncMock()
        sm.mcp_api = MagicMock()
        sm.mcp_api.shutdown_mcp_server = AsyncMock()
        sm.active_sessions = {}

        await sm.shutdown()

        sm.redis_manager.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_event_repo_when_connect_fails(self) -> None:
        """When connect fails, no EpisodeEventRepository is created."""
        sm = _make_session_manager()
        sm.redis_manager = MagicMock()
        sm.redis_manager.connect = AsyncMock(side_effect=Exception("no redis"))
        sm.redis_manager.is_connected = False

        sm.episode_manager = MagicMock()
        sm.episode_manager.wire_event_repository = MagicMock()

        sm._start_redis_container = MagicMock()
        sm._start_permanent_environment_async = AsyncMock()
        sm._session_cleanup_loop = AsyncMock()
        sm.rest_api = MagicMock()
        sm.rest_api.start_server = AsyncMock()
        sm.mcp_api = MagicMock()
        sm.mcp_api.start_mcp_server = AsyncMock()

        with patch("saber.server.session_manager.EpisodeEventRepository") as MockRepo:
            task = asyncio.create_task(sm.start_server())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            MockRepo.assert_not_called()


class TestSessionManagerRedisContainer:
    """Test SessionManager Redis container auto-start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_server_starts_redis_container_when_auto_start_true(self) -> None:
        """With SABER_REDIS_AUTO_START unset (default true), subprocess.run is called
        with correct docker compose command before _connect_redis()."""
        sm = _make_session_manager()

        # Verify default auto-start is True
        assert sm._redis_auto_start is True

        sm.redis_manager = MagicMock()
        sm.redis_manager.connect = AsyncMock()
        sm.redis_manager.is_connected = False

        sm._start_permanent_environment_async = AsyncMock()
        sm._session_cleanup_loop = AsyncMock()
        sm.rest_api = MagicMock()
        sm.rest_api.start_server = AsyncMock()
        sm.mcp_api = MagicMock()
        sm.mcp_api.start_mcp_server = AsyncMock()

        with patch("saber.server.session_manager.subprocess") as mock_subprocess, \
             patch("saber.domain.resources.resolve_redis_compose_path") as mock_resolve:
            mock_path = MagicMock()
            mock_path.__str__ = lambda self: "/fake/redis-compose.yml"
            mock_resolve.return_value = mock_path
            mock_subprocess.run.return_value = MagicMock(returncode=0)
            mock_subprocess.CalledProcessError = __import__("subprocess").CalledProcessError

            task = asyncio.create_task(sm.start_server())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            mock_subprocess.run.assert_called_once()
            call_args = mock_subprocess.run.call_args[0][0]
            assert call_args[:2] == ["docker", "compose"]
            assert "-f" in call_args
            assert "up" in call_args
            assert "-d" in call_args
            assert "--wait" in call_args

    @pytest.mark.asyncio
    async def test_start_server_skips_redis_container_when_auto_start_false(self) -> None:
        """With SABER_REDIS_AUTO_START=false, subprocess.run is NOT called."""
        with patch.dict(os.environ, {"SABER_REDIS_AUTO_START": "false"}):
            sm = _make_session_manager()

        assert sm._redis_auto_start is False

        sm.redis_manager = MagicMock()
        sm.redis_manager.connect = AsyncMock()
        sm.redis_manager.is_connected = False

        sm._start_permanent_environment_async = AsyncMock()
        sm._session_cleanup_loop = AsyncMock()
        sm.rest_api = MagicMock()
        sm.rest_api.start_server = AsyncMock()
        sm.mcp_api = MagicMock()
        sm.mcp_api.start_mcp_server = AsyncMock()

        with patch("saber.server.session_manager.subprocess") as mock_subprocess:
            task = asyncio.create_task(sm.start_server())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            mock_subprocess.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_stops_redis_container_after_disconnect(self) -> None:
        """subprocess.run is called with 'down' command during shutdown."""
        sm = _make_session_manager()
        sm._redis_compose_path = MagicMock()
        sm._redis_compose_path.__str__ = lambda self: "/fake/redis-compose.yml"

        sm.redis_manager = MagicMock()
        sm.redis_manager.disconnect = AsyncMock()
        sm.shutdown_event = MagicMock()
        sm.cleanup_task = None
        sm.execution_manager = MagicMock()
        sm.execution_manager.is_permanent_environment_running = MagicMock(return_value=False)
        sm._cleanup_saber_episode_networks = AsyncMock()
        sm.mcp_api = MagicMock()
        sm.mcp_api.shutdown_mcp_server = AsyncMock()
        sm.active_sessions = {}

        with patch("saber.server.session_manager.subprocess") as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(returncode=0)
            await sm.shutdown()

            mock_subprocess.run.assert_called_once()
            call_args = mock_subprocess.run.call_args[0][0]
            assert "down" in call_args
            assert "-p" in call_args
            assert "saber-redis" in call_args

    @pytest.mark.asyncio
    async def test_shutdown_skips_redis_stop_when_not_managed(self) -> None:
        """When auto_start was false (no compose_path), no stop call on shutdown."""
        with patch.dict(os.environ, {"SABER_REDIS_AUTO_START": "false"}):
            sm = _make_session_manager()

        assert sm._redis_compose_path is None

        sm.redis_manager = MagicMock()
        sm.redis_manager.disconnect = AsyncMock()
        sm.shutdown_event = MagicMock()
        sm.cleanup_task = None
        sm.execution_manager = MagicMock()
        sm.execution_manager.is_permanent_environment_running = MagicMock(return_value=False)
        sm._cleanup_saber_episode_networks = AsyncMock()
        sm.mcp_api = MagicMock()
        sm.mcp_api.shutdown_mcp_server = AsyncMock()
        sm.active_sessions = {}

        with patch("saber.server.session_manager.subprocess") as mock_subprocess:
            await sm.shutdown()
            mock_subprocess.run.assert_not_called()

    def test_start_redis_container_handles_subprocess_failure(self) -> None:
        """CalledProcessError is caught, logged as warning, doesn't crash."""
        import subprocess as real_subprocess

        sm = _make_session_manager()
        sm._redis_auto_start = True

        with patch("saber.domain.resources.resolve_redis_compose_path") as mock_resolve, \
             patch("saber.server.session_manager.subprocess") as mock_subprocess:
            mock_path = MagicMock()
            mock_path.__str__ = lambda self: "/fake/redis-compose.yml"
            mock_resolve.return_value = mock_path
            mock_subprocess.CalledProcessError = real_subprocess.CalledProcessError
            mock_subprocess.run.side_effect = real_subprocess.CalledProcessError(
                1, "docker compose", stderr="error starting"
            )

            # Should not raise
            sm._start_redis_container()

            # compose_path should be reset to None on failure
            assert sm._redis_compose_path is None

    def test_start_redis_container_handles_docker_not_installed(self) -> None:
        """FileNotFoundError (docker not found) is caught, logged as warning."""
        sm = _make_session_manager()
        sm._redis_auto_start = True

        with patch("saber.domain.resources.resolve_redis_compose_path") as mock_resolve, \
             patch("saber.server.session_manager.subprocess") as mock_subprocess:
            mock_path = MagicMock()
            mock_resolve.return_value = mock_path
            mock_subprocess.CalledProcessError = __import__("subprocess").CalledProcessError
            mock_subprocess.run.side_effect = FileNotFoundError("docker not found")

            # Should not raise
            sm._start_redis_container()
            assert sm._redis_compose_path is None

    def test_stop_redis_container_handles_failure(self) -> None:
        """Exception during stop is caught and logged, not raised."""
        sm = _make_session_manager()
        sm._redis_compose_path = MagicMock()
        sm._redis_compose_path.__str__ = lambda self: "/fake/redis-compose.yml"

        with patch("saber.server.session_manager.subprocess") as mock_subprocess:
            mock_subprocess.run.side_effect = Exception("docker broke")

            # Should not raise
            sm._stop_redis_container()
            # compose_path is cleared in finally block
            assert sm._redis_compose_path is None

    def test_start_redis_uses_wait_timeout(self) -> None:
        """Command includes --wait-timeout 30."""
        sm = _make_session_manager()
        sm._redis_auto_start = True

        with patch("saber.domain.resources.resolve_redis_compose_path") as mock_resolve, \
             patch("saber.server.session_manager.subprocess") as mock_subprocess:
            mock_path = MagicMock()
            mock_path.__str__ = lambda self: "/fake/redis-compose.yml"
            mock_resolve.return_value = mock_path
            mock_subprocess.run.return_value = MagicMock(returncode=0)
            mock_subprocess.CalledProcessError = __import__("subprocess").CalledProcessError

            sm._start_redis_container()

            call_args = mock_subprocess.run.call_args[0][0]
            assert "--wait-timeout" in call_args
            timeout_idx = call_args.index("--wait-timeout")
            assert call_args[timeout_idx + 1] == "30"
