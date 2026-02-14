"""Unit tests for RedisManager.

Tests cover:
- Explicit config overrides defaults
- Default config from env/hardcoded
- Not connected initially
- connect() creates client and pings
- connect() is idempotent
- connect() raises on failure
- disconnect() closes client and resets
- disconnect() safe when not connected
- health_check() returns True on success
- health_check() returns False when not connected
- health_check() returns False on ping failure
- client property raises when not connected
- client property returns client when connected
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.db.redis_manager import RedisManager


@pytest.fixture
def manager() -> RedisManager:
    """Create a RedisManager with test config."""
    return RedisManager(
        host="testhost",
        port=6380,
        db=2,
        password="testpass",
    )


class TestRedisManagerInit:
    """Tests for RedisManager initialization."""

    def test_explicit_config(self, manager: RedisManager) -> None:
        """Explicit parameters override defaults."""
        assert manager.host == "testhost"
        assert manager.port == 6380
        assert manager.db == 2
        assert manager.password == "testpass"

    def test_default_config(self) -> None:
        """Defaults come from env vars or hardcoded fallbacks."""
        mgr = RedisManager()
        assert mgr.host == "localhost"
        assert mgr.port == 6379
        assert mgr.db == 0
        assert mgr.password is None

    def test_not_connected_initially(self, manager: RedisManager) -> None:
        """Manager starts disconnected."""
        assert manager.is_connected is False


@pytest.mark.asyncio
class TestRedisManagerConnect:
    """Tests for connect() behavior."""

    async def test_connect_creates_client(self, manager: RedisManager) -> None:
        """connect() creates a Redis client and pings to verify."""
        import fakeredis.aioredis

        fake_client = fakeredis.aioredis.FakeRedis()
        try:
            with patch("redis.asyncio.Redis", return_value=fake_client) as mock_cls:
                await manager.connect()

            assert manager.is_connected is True
            mock_cls.assert_called_once_with(
                host="testhost",
                port=6380,
                db=2,
                password="testpass",
            )
        finally:
            await fake_client.aclose()

    async def test_connect_idempotent(self, manager: RedisManager) -> None:
        """Second connect() is a no-op when already connected."""
        mock_client = AsyncMock()
        manager._client = mock_client
        await manager.connect()
        assert manager._client is mock_client

    async def test_connect_raises_on_failure(self, manager: RedisManager) -> None:
        """connect() propagates errors when ping fails."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("redis.asyncio.Redis", return_value=mock_client):
            with pytest.raises(ConnectionError, match="refused"):
                await manager.connect()
        assert manager.is_connected is False


@pytest.mark.asyncio
class TestRedisManagerDisconnect:
    """Tests for disconnect() behavior."""

    async def test_disconnect_closes_client(self, manager: RedisManager) -> None:
        """disconnect() closes the client and resets state."""
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        manager._client = mock_client

        await manager.disconnect()

        mock_client.aclose.assert_awaited_once()
        assert manager.is_connected is False
        assert manager._client is None

    async def test_disconnect_safe_when_not_connected(self, manager: RedisManager) -> None:
        """disconnect() is safe when not connected."""
        await manager.disconnect()
        assert manager.is_connected is False


@pytest.mark.asyncio
class TestRedisManagerHealthCheck:
    """Tests for health_check() behavior."""

    async def test_health_check_returns_true(self, manager: RedisManager) -> None:
        """health_check() returns True when Redis is reachable."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        manager._client = mock_client

        result = await manager.health_check()
        assert result is True

    async def test_health_check_false_when_not_connected(self, manager: RedisManager) -> None:
        """health_check() returns False when not connected."""
        result = await manager.health_check()
        assert result is False

    async def test_health_check_false_on_ping_failure(self, manager: RedisManager) -> None:
        """health_check() returns False when ping fails."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(side_effect=Exception("connection lost"))
        manager._client = mock_client

        result = await manager.health_check()
        assert result is False


class TestRedisManagerClient:
    """Tests for client property."""

    def test_client_raises_when_not_connected(self, manager: RedisManager) -> None:
        """client property raises RuntimeError when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            _ = manager.client

    def test_client_returns_client_when_connected(self, manager: RedisManager) -> None:
        """client property returns the client when connected."""
        mock_client = AsyncMock()
        manager._client = mock_client
        assert manager.client is mock_client
