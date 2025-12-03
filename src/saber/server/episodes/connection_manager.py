"""WebSocket connection manager for episode-scoped real-time notifications.

This module provides WebSocket connection management with episode-level isolation
for the transcript coordination service.

Logging category: EPISODE
"""

import asyncio
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

from ...logging_config import LogCategory, get_saber_logger
from ...models.rest.websocket_constants import WebSocketCloseCode
from ..time_source import TimeSource, UTCTimeSource

logger = get_saber_logger(LogCategory.EPISODE, __name__)


class ConnectionManager:
    """
    Manages WebSocket connections with episode-level isolation.

    Each blue team episode establishes ONE persistent WebSocket connection.
    Events are routed only to connections matching the target episode_id.

    Typical scenario (50 concurrent evaluations):
    - 50 episodes × 1 connection each = 50 WebSocket connections
    - Each episode has 1 blue agent + 1 red agent = 2 agents
    - Only blue agent maintains WebSocket (red uses REST)

    Thread-safety: All operations are protected by an asyncio.Lock to ensure
    thread-safe concurrent access from multiple async tasks.
    """

    def __init__(self, episode_manager: Optional[Any] = None, time_source: TimeSource | None = None) -> None:
        """Initialize the ConnectionManager with empty state.

        Args:
            episode_manager: Optional reference to EpisodeManager for cleanup notifications
            time_source: Time source for getting current time (defaults to UTCTimeSource)
        """
        # Map episode_id → set of active WebSocket connections
        # Usually 1 connection per episode
        self._active_connections: Dict[str, Set[WebSocket]] = {}
        self._connection_metadata: Dict[WebSocket, dict] = {}
        self._lock = asyncio.Lock()
        self._episode_manager = episode_manager  # For cleanup notifications
        self._time_source = time_source or UTCTimeSource()

    async def connect(self, episode_id: str, websocket: WebSocket, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register new WebSocket connection for episode.

        Args:
            episode_id: Target episode identifier
            websocket: WebSocket connection object
            metadata: Optional connection metadata (role, session_id, etc.)

        Raises:
            Exception: If WebSocket.accept() fails
        """
        await websocket.accept()

        async with self._lock:
            if episode_id not in self._active_connections:
                self._active_connections[episode_id] = set()

            self._active_connections[episode_id].add(websocket)
            self._connection_metadata[websocket] = {
                "episode_id": episode_id,
                "connected_at": self._time_source.now().isoformat(),
                "metadata": metadata or {},
            }

        logger.info(
            "WebSocket connected",
            extra={
                "episode_id": episode_id,
                "connection_count": len(self._active_connections[episode_id]),
            },
        )

        # Send connection confirmation
        await websocket.send_json(
            {
                "type": "connected",
                "episode_id": episode_id,
                "timestamp": self._time_source.now().isoformat(),
            }
        )

    async def disconnect(self, episode_id: str, websocket: WebSocket) -> None:
        """
        Unregister WebSocket connection and cleanup.

        Detects unexpected disconnections (e.g., client crash) and can notify
        episode manager for orphaned episode cleanup.

        Args:
            episode_id: Episode identifier
            websocket: WebSocket connection to remove
        """
        was_last_connection = False
        conn_metadata = {}

        async with self._lock:
            # Track if this was the last connection
            if episode_id in self._active_connections:
                was_last_connection = len(self._active_connections[episode_id]) == 1
                self._active_connections[episode_id].discard(websocket)

                # Cleanup empty episode sets
                if not self._active_connections[episode_id]:
                    del self._active_connections[episode_id]

            # Remove metadata
            conn_metadata = self._connection_metadata.pop(websocket, {})

        logger.info(
            "WebSocket disconnected",
            extra={
                "episode_id": episode_id,
                "was_last_connection": was_last_connection,
            },
        )

        # Check if we need to notify about potential orphaned episode
        if was_last_connection and self._episode_manager:
            from ..base import EpisodeState

            episode = self._episode_manager.get_episode_by_id(episode_id)

            # Only warn if episode is still ACTIVE (not already being cleaned up)
            if episode and episode.state == EpisodeState.ACTIVE:
                logger.warning(
                    "Last WebSocket connection lost for active episode - episode may be orphaned",
                    extra={
                        "episode_id": episode_id,
                        "episode_state": episode.state.value,
                        "connection_metadata": conn_metadata,
                    },
                )
                # TODO: Policy decision - auto-terminate? mark as failed? grace period?
                # For now, just log warning. Episode will cleanup on timeout/completion.

    async def broadcast_to_episode(self, episode_id: str, message: dict) -> None:
        """
        Send message to all WebSocket connections for specific episode.

        Handles disconnected clients gracefully and removes stale connections.

        Args:
            episode_id: Target episode to broadcast to
            message: JSON-serializable message dictionary
        """
        if episode_id not in self._active_connections:
            logger.debug(
                "No active WebSocket connections for episode",
                extra={"episode_id": episode_id},
            )
            return

        # Track disconnected connections for cleanup
        disconnected = []

        # Send to all connections for this episode
        for connection in self._active_connections[episode_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(
                    "Failed to send WebSocket message, marking for disconnect",
                    extra={"episode_id": episode_id, "error": str(e)},
                )
                disconnected.append(connection)

        # Cleanup disconnected connections
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    self._active_connections[episode_id].discard(conn)
                    self._connection_metadata.pop(conn, None)

                if not self._active_connections[episode_id]:
                    del self._active_connections[episode_id]

    async def cleanup_episode(self, episode_id: str) -> None:
        """
        Cleanup all connections for an episode (called when episode ends).

        Closes all WebSocket connections and removes all state for the episode.
        Handles close failures gracefully to ensure state is cleaned up.

        Args:
            episode_id: Episode identifier to cleanup
        """
        async with self._lock:
            if episode_id in self._active_connections:
                connection_count = len(self._active_connections[episode_id])

                # Close all connections
                for conn in self._active_connections[episode_id]:
                    try:
                        await conn.close(code=WebSocketCloseCode.NORMAL_CLOSURE, reason="Episode ended")
                    except Exception as e:
                        logger.warning(
                            "Failed to close WebSocket during cleanup (connection may already be closed)",
                            extra={"episode_id": episode_id, "error": str(e)},
                        )
                    self._connection_metadata.pop(conn, None)

                del self._active_connections[episode_id]

                logger.info(
                    "Cleaned up episode WebSocket connections",
                    extra={"episode_id": episode_id, "connections_closed": connection_count},
                )
            else:
                logger.debug("No WebSocket connections to cleanup for episode", extra={"episode_id": episode_id})
