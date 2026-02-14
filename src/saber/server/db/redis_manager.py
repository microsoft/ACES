"""Redis connection manager for SABER episode event storage."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ...logging_config import LogCategory, get_saber_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_saber_logger(LogCategory.DATABASE, __name__)


class RedisManager:
    """Manages Redis client for episode event storage.

    This is an OPTIONAL component - SABER works without Redis connection.
    When Redis is not available, the manager gracefully degrades and
    can be checked via the ``is_connected`` property.

    Environment Variables:
        SABER_REDIS_HOST: Redis host (default: localhost)
        SABER_REDIS_PORT: Redis port (default: 6379)
        SABER_REDIS_DB: Redis database number (default: 0)
        SABER_REDIS_PASSWORD: Redis password (default: None)

    Usage:
        manager = RedisManager()
        await manager.connect()

        if manager.is_connected:
            await manager.client.ping()

        await manager.disconnect()
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        password: str | None = None,
    ) -> None:
        """Initialize RedisManager with connection parameters.

        Args:
            host: Redis host (env: SABER_REDIS_HOST, default: localhost)
            port: Redis port (env: SABER_REDIS_PORT, default: 6379)
            db: Redis database number (env: SABER_REDIS_DB, default: 0)
            password: Redis password (env: SABER_REDIS_PASSWORD, default: None)
        """
        self.host = host or os.getenv("SABER_REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("SABER_REDIS_PORT", "6379"))
        self.db = db if db is not None else int(os.getenv("SABER_REDIS_DB", "0"))
        self.password = password or os.getenv("SABER_REDIS_PASSWORD")
        self._client: Redis | None = None

    @property
    def is_connected(self) -> bool:
        """Check if Redis client is established."""
        return self._client is not None

    async def connect(self) -> None:
        """Create Redis client and verify connectivity with PING.

        Raises:
            ImportError: If redis is not installed (optional dependency)
            Exception: If connection or PING fails
        """
        if self._client is not None:
            return

        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            logger.warning(
                "redis not installed, Redis features disabled. Install with: pip install saber[redis]",
                extra={"event": "redis_package_missing"},
            )
            raise ImportError("redis is required for Redis support. Install with: pip install saber[redis]") from e

        logger.info(
            "Connecting to Redis",
            extra={
                "event": "redis_connecting",
                "host": self.host,
                "port": self.port,
                "db": self.db,
            },
        )

        try:
            client = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
            )
            await client.ping()
            self._client = client
            logger.info(
                "Redis connection established",
                extra={"event": "redis_connected"},
            )
        except Exception as e:
            logger.error(
                f"Failed to connect to Redis: {e}",
                extra={
                    "event": "redis_connection_failed",
                    "host": self.host,
                    "port": self.port,
                    "error": str(e),
                },
            )
            raise

    async def disconnect(self) -> None:
        """Close Redis client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info(
                "Redis connection closed",
                extra={"event": "redis_disconnected"},
            )

    @property
    def client(self) -> Redis:
        """Get Redis client, raising if not connected.

        Returns:
            redis.asyncio.Redis client

        Raises:
            RuntimeError: If not connected (call connect() first)
        """
        if self._client is None:
            raise RuntimeError("RedisManager not connected. Call connect() first.")
        return self._client

    async def health_check(self) -> bool:
        """Check Redis connectivity via PING.

        Returns:
            True if Redis is reachable, False otherwise
        """
        if not self._client:
            return False

        try:
            await self._client.ping()
            logger.debug(
                "Redis health check passed",
                extra={"event": "redis_health_ok"},
            )
            return True
        except Exception as e:
            logger.error(
                f"Redis health check failed: {e}",
                extra={
                    "event": "redis_health_failed",
                    "error": str(e),
                },
            )
            return False
