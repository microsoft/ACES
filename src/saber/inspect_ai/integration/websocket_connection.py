"""WebSocket connection management for transcript synchronization.

This module provides WebSocket lifecycle management including:
- Connection establishment with timeout
- Automatic reconnection with exponential backoff
- Connection state tracking
- Graceful cleanup

Logging category: AGENT.
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse

if TYPE_CHECKING:
    from websockets import ClientConnection

from ...logging_config import LogCategory, get_saber_logger
from ...models.rest.websocket_config import WebSocketConfig
from ...models.rest.websocket_constants import WebSocketDefaults
from .message_serialization import is_websocket_closed

logger = get_saber_logger(LogCategory.AGENT, __name__)


class WebSocketConnectionManager:
    """Manages WebSocket connection lifecycle.

    Handles:
    - Initial connection establishment
    - Reconnection with exponential backoff
    - Connection state tracking
    - Graceful shutdown

    Thread-safety: Uses asyncio.Lock for connection state changes.
    """

    def __init__(
        self,
        episode_id: str,
        rest_url: str,
        ws_config: WebSocketConfig | None = None,
    ) -> None:
        """Initialize the connection manager.

        Args:
            episode_id: SABER episode ID for WebSocket endpoint
            rest_url: Base URL of SABER REST API
            ws_config: WebSocket configuration (uses defaults if None)
        """
        self._episode_id = episode_id
        self._rest_url = rest_url
        self._ws_config = ws_config or WebSocketConfig()

        # Connection state
        self._websocket: ClientConnection | None = None
        self._ws_lock = asyncio.Lock()
        self._listener_task: asyncio.Task[None] | None = None

        # Build WebSocket URL
        self._ws_url = self._build_websocket_url(rest_url, episode_id)

        logger.debug(
            "Created WebSocketConnectionManager",
            extra={
                "episode_id": episode_id,
                "ws_url": self._ws_url,
                "connection_timeout": self._ws_config.connection_timeout,
                "reconnect_enabled": self._ws_config.reconnect_enabled,
            },
        )

    @property
    def websocket(self) -> Optional["ClientConnection"]:
        """Get the current WebSocket connection (may be None or closed)."""
        return self._websocket

    @property
    def ws_url(self) -> str:
        """Get the WebSocket URL."""
        return self._ws_url

    @property
    def episode_id(self) -> str:
        """Get the episode ID."""
        return self._episode_id

    @property
    def config(self) -> WebSocketConfig:
        """Get the WebSocket configuration."""
        return self._ws_config

    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        return self._websocket is not None and not is_websocket_closed(self._websocket)

    def _build_websocket_url(self, rest_url: str, episode_id: str) -> str:
        """Build WebSocket URL from REST URL preserving base path.

        Handles URLs with ports, paths, and different schemes correctly.
        Preserves any base path for proxy/ingress deployments.

        Args:
            rest_url: Base REST API URL (e.g., "http://localhost:8000" or "http://proxy.com/saber")
            episode_id: Episode identifier

        Returns:
            WebSocket URL with preserved path prefix

        Examples:
            "http://localhost:8000" -> "ws://localhost:8000/api/v1/episodes/{id}/ws"
            "https://proxy.com/saber" -> "wss://proxy.com/saber/api/v1/episodes/{id}/ws"
        """
        parsed = urlparse(rest_url)

        # Map HTTP scheme to WebSocket scheme
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"

        # Preserve existing path (strip trailing slash)
        base_path = parsed.path.rstrip("/") if parsed.path else ""
        ws_path = f"{base_path}/api/v1/episodes/{episode_id}/ws"

        # Build WebSocket URL preserving host:port and base path
        ws_url = urlunparse(
            (
                ws_scheme,
                parsed.netloc,  # Preserves host:port
                ws_path,  # Preserves base path + adds WebSocket endpoint
                "",  # params
                "",  # query
                "",  # fragment
            )
        )

        return ws_url

    async def ensure_connected(
        self,
        on_connected_callback: Any | None = None,
    ) -> "ClientConnection":
        """Establish WebSocket connection with reconnection support.

        Implements exponential backoff reconnection and proper resource cleanup.

        Args:
            on_connected_callback: Optional async callback to invoke after connection
                                   is established (e.g., to start listener task).
                                   Signature: async def callback(websocket) -> asyncio.Task

        Returns:
            Connected WebSocket protocol

        Raises:
            ConnectionError: If connection fails after all retry attempts
        """
        async with self._ws_lock:
            if self._websocket is not None and not is_websocket_closed(self._websocket):
                return self._websocket  # Already connected

            # Determine number of attempts based on reconnect policy
            max_attempts = self._ws_config.max_reconnect_attempts if self._ws_config.reconnect_enabled else 1

            for attempt in range(max_attempts):
                temp_websocket = None
                temp_listener = None

                try:
                    import websockets

                    logger.info(
                        "Establishing WebSocket connection",
                        extra={
                            "episode_id": self._episode_id,
                            "url": self._ws_url,
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                        },
                    )

                    # Phase 1: Connect with timeout
                    temp_websocket = await asyncio.wait_for(
                        websockets.connect(self._ws_url), timeout=self._ws_config.connection_timeout
                    )

                    # Phase 2: Wait for connection confirmation FIRST (before starting listener)
                    # This avoids race condition where listener consumes the "connected" message
                    try:
                        timeout_seconds = self._ws_config.push.confirmation_timeout

                        async def _wait_for_connected(ws: Any = temp_websocket) -> bool:
                            assert ws is not None
                            # Use recv() to get exactly one message (the "connected" handshake)
                            message = await ws.recv()
                            data = json.loads(message)
                            if data.get("type") == "connected":
                                logger.info("WebSocket connected", extra={"episode_id": self._episode_id})
                                return True
                            else:
                                logger.warning(
                                    f"Expected 'connected' message, got '{data.get('type')}'",
                                    extra={"episode_id": self._episode_id, "message_type": data.get("type")},
                                )
                                return False

                        connected = await asyncio.wait_for(_wait_for_connected(), timeout=timeout_seconds)
                        if connected:
                            # Phase 3: Start background listener via callback AFTER connection confirmed
                            if on_connected_callback:
                                temp_listener = await on_connected_callback(temp_websocket)

                            # SUCCESS - commit state
                            self._websocket = temp_websocket
                            self._listener_task = temp_listener
                            temp_websocket = None  # Don't cleanup
                            temp_listener = None  # Don't cleanup
                            return self._websocket
                        else:
                            raise ConnectionError("WebSocket handshake failed - unexpected message type")
                    except TimeoutError:
                        raise ConnectionError(
                            f"WebSocket confirmation timeout after {self._ws_config.push.confirmation_timeout}s"
                        ) from None

                except Exception as e:
                    # Retry logic
                    is_last_attempt = attempt == max_attempts - 1

                    if not is_last_attempt:
                        # Calculate exponential backoff delay
                        delay = min(
                            self._ws_config.initial_reconnect_delay
                            * (self._ws_config.reconnect_backoff_multiplier**attempt),
                            self._ws_config.max_reconnect_delay,
                        )

                        logger.warning(
                            f"WebSocket connection failed (attempt {attempt + 1}/{max_attempts}), "
                            f"retrying in {delay:.1f}s",
                            extra={
                                "episode_id": self._episode_id,
                                "error": str(e),
                                "delay": delay,
                            },
                        )

                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "WebSocket connection failed after all attempts",
                            extra={
                                "episode_id": self._episode_id,
                                "error": str(e),
                                "attempts": max_attempts,
                            },
                        )
                        raise

                finally:
                    # Cleanup listener task with timeout
                    if temp_listener and not temp_listener.done():
                        temp_listener.cancel()
                        try:
                            await asyncio.wait_for(
                                temp_listener, timeout=WebSocketDefaults.LISTENER_TASK_CANCEL_TIMEOUT_SECONDS
                            )
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            logger.debug("Listener task cancelled during cleanup")
                        except Exception as e:
                            logger.warning(f"Error cancelling listener task: {e}")

                    # Cleanup WebSocket with timeout
                    if temp_websocket and not is_websocket_closed(temp_websocket):
                        try:
                            await asyncio.wait_for(
                                temp_websocket.close(), timeout=WebSocketDefaults.WEBSOCKET_CLOSE_TIMEOUT_SECONDS
                            )
                        except asyncio.TimeoutError:
                            logger.warning("WebSocket close timed out during cleanup")
                        except Exception as e:
                            logger.warning(f"Error closing WebSocket during cleanup: {e}")

        # Should not reach here - raise for mypy
        raise ConnectionError("WebSocket connection failed")

    async def close(self) -> None:
        """Close WebSocket connection and release resources."""
        # Cancel listener task
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        # Close WebSocket
        if self._websocket and not is_websocket_closed(self._websocket):
            await self._websocket.close()
            logger.info("WebSocket connection closed", extra={"episode_id": self._episode_id})

        self._websocket = None

    async def force_reconnect(self, on_connected_callback: Any | None = None) -> "ClientConnection":
        """Force close and reconnect.

        Useful when connection is in bad state and needs reset.

        Args:
            on_connected_callback: Callback for starting listener (see ensure_connected)

        Returns:
            New WebSocket connection
        """
        # Close existing connection
        if self._websocket and not is_websocket_closed(self._websocket):
            try:
                await self._websocket.close()
            except Exception:
                pass
        self._websocket = None

        # Establish new connection
        return await self.ensure_connected(on_connected_callback)


__all__ = ["WebSocketConnectionManager"]
