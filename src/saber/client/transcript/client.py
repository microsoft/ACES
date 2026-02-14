"""Transcript push client orchestrator.

This module provides the main TranscriptSyncClient class that orchestrates
WebSocket-based transcript push operations for any evaluation harness.

The client composes:
- WebSocketConnectionManager for connection lifecycle
- WebSocketEventProcessor for event handling
- GenericTranscriptSyncOperations for push operations

Logging category: AGENT.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from websockets import ClientConnection

from ...logging_config import LogCategory, get_saber_logger
from ...models.rest.websocket_config import WebSocketConfig
from .connection import WebSocketConnectionManager
from .event_processor import WebSocketEventProcessor
from .protocols import MessageSerializer
from .sync_operations import GenericTranscriptSyncOperations, ReconnectCallback

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Generic message type
M = TypeVar("M")


class TranscriptSyncClient(Generic[M]):
    """Generic, harness-agnostic transcript push client.

    This client provides transcript push over WebSocket,
    supporting any evaluation harness via the MessageSerializer protocol.

    Provides:
    - Persistent WebSocket connection (established on first use or via context manager)
    - Instant state event notifications (<100ms when server state changes)
    - Push messages to server with retry logic
    - Automatic reconnection on connection loss

    Components:
    - _connection: WebSocketConnectionManager for connection lifecycle
    - _events: WebSocketEventProcessor for event handling
    - _sync: GenericTranscriptSyncOperations for push operations

    Type Parameter:
        M: The harness-specific message type (e.g., ChatMessage for inspect_ai)

    Example usage with a custom harness:

        class MyMessageSerializer(MessageSerializer[MyMessage]):
            def serialize(self, msg: MyMessage) -> dict:
                return {"role": msg.role, "content": msg.text}
            def deserialize(self, data: dict) -> MyMessage:
                return MyMessage(role=data["role"], text=data["content"])
            # ... other methods

        async with TranscriptSyncClient(
            episode_id="ep-123",
            rest_url="http://localhost:8000",
            serializer=MyMessageSerializer(),
        ) as client:
            await client.push_message(response_msg)
    """

    def __init__(
        self,
        episode_id: str,
        rest_url: str,
        serializer: MessageSerializer[M],
        ws_config: WebSocketConfig | None = None,
    ) -> None:
        """Initialize transcript sync client.

        Args:
            episode_id: SABER episode ID
            rest_url: Base URL of SABER REST API
            serializer: MessageSerializer for harness message type
            ws_config: WebSocket configuration (uses defaults if None)
        """
        self._episode_id = episode_id
        self._rest_url = rest_url
        self._serializer = serializer
        self._ws_config = ws_config or WebSocketConfig()

        # Initialize component modules
        self._connection = WebSocketConnectionManager(
            episode_id=episode_id,
            rest_url=rest_url,
            ws_config=self._ws_config,
        )

        self._events = WebSocketEventProcessor(
            episode_id=episode_id,
            ws_config=self._ws_config,
        )

        self._sync = GenericTranscriptSyncOperations[M](
            episode_id=episode_id,
            event_processor=self._events,
            serializer=serializer,
            ws_config=self._ws_config,
        )

        logger.debug(
            "Created TranscriptSyncClient",
            extra={
                "episode_id": episode_id,
                "ws_config": {
                    "connection_timeout": self._ws_config.connection_timeout,
                    "pull_event_timeout": self._ws_config.pull.event_timeout,
                    "push_confirmation_timeout": self._ws_config.push.confirmation_timeout,
                    "reconnect_enabled": self._ws_config.reconnect_enabled,
                },
                "ws_url": self._connection.ws_url,
            },
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def episode_id(self) -> str:
        """Get the episode ID."""
        return self._episode_id

    @property
    def config(self) -> WebSocketConfig:
        """Get the WebSocket configuration."""
        return self._ws_config

    @property
    def pull_enabled(self) -> bool:
        """Check if pull (server-controlled transcript) is enabled."""
        return self._ws_config.pull.enabled

    @property
    def push_enabled(self) -> bool:
        """Check if push (client sends messages to server) is enabled."""
        return self._ws_config.push.enabled

    @property
    def local_messages(self) -> list[M]:
        """Get current local transcript messages."""
        return self._sync.local_messages

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        return self._connection.is_connected()

    # =========================================================================
    # Properties for component access (for advanced usage/testing)
    # =========================================================================

    @property
    def websocket(self) -> Any:
        """Get WebSocket (for advanced usage/testing)."""
        return self._connection.websocket

    # =========================================================================
    # Context manager
    # =========================================================================

    async def __aenter__(self) -> "TranscriptSyncClient[M]":
        """Context manager entry - establish WebSocket connection."""
        await self.ensure_connected()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit - cleanup WebSocket connection."""
        await self.cleanup()
        return False  # Don't suppress exceptions

    # =========================================================================
    # Connection lifecycle
    # =========================================================================

    async def ensure_connected(self) -> None:
        """Establish WebSocket connection if not already connected."""

        async def _start_listener(websocket: Any) -> asyncio.Task:
            """Callback to start listener after connection."""
            return await self._events.start_listener(websocket)

        await self._connection.ensure_connected(on_connected_callback=_start_listener)

    async def _get_reconnect_callback(self) -> ReconnectCallback:
        """Get a callback for reconnection."""

        async def reconnect() -> "ClientConnection":
            return await self._connection.force_reconnect(
                on_connected_callback=lambda ws: self._events.start_listener(ws)
            )

        return reconnect

    # =========================================================================
    # Push operations
    # =========================================================================

    async def push_message(
        self,
        message: M,
        context: str = "message_push",
    ) -> bool:
        """Push a message to the server with retry logic.

        Args:
            message: Message to push
            context: Context string for logging

        Returns:
            True when push succeeds
        """
        if not self._ws_config.push.enabled:
            logger.debug(
                "Push disabled, tracking locally only",
                extra={"episode_id": self._episode_id},
            )
            self._sync._local_messages.append(message)
            return True

        await self.ensure_connected()
        websocket = self._connection.websocket
        if not websocket:
            raise RuntimeError("WebSocket not connected")

        return await self._sync.push_message_with_retry(
            websocket,
            message,
            context=context,
            reconnect_callback=await self._get_reconnect_callback(),
        )

    async def push_messages_if_new(
        self,
        messages: list[M],
    ) -> bool:
        """Push any messages not in local transcript (typically tool results).

        Used for pushing tool results that the harness adds to the input list.

        Args:
            messages: Full message list from harness

        Returns:
            True if any messages were pushed
        """
        if not self._ws_config.push.enabled:
            return False

        await self.ensure_connected()
        websocket = self._connection.websocket
        if not websocket:
            return False

        return await self._sync.push_tool_results_if_needed(
            websocket,
            messages,
            reconnect_callback=await self._get_reconnect_callback(),
        )

    # =========================================================================
    # State event operations
    # =========================================================================

    async def wait_for_state_event(self) -> bool:
        """Wait for a state event from the server.

        Returns:
            True if event received, False on timeout
        """
        return await self._events.wait_for_state_event()

    async def wait_for_state_event_with_retry(self) -> bool:
        """Wait for a state event with unlimited retries.

        Returns:
            True when event is received (always succeeds or blocks forever)
        """
        return await self._events.wait_for_state_event_with_retry()

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup(self) -> None:
        """Close WebSocket connection and release resources."""
        logger.info(
            "[CLEANUP] cleanup() CALLED - starting shutdown sequence",
            extra={"episode_id": self._episode_id},
        )

        # Signal shutdown to unblock any waiting operations (e.g., wait_for_injection_event)
        logger.debug(
            "[CLEANUP] Signaling shutdown to event processor",
            extra={"episode_id": self._episode_id},
        )
        self._events.signal_shutdown()

        # Cancel listener task
        logger.debug(
            "[CLEANUP] Cancelling listener task",
            extra={"episode_id": self._episode_id},
        )
        await self._events.cancel_listener()

        # Close connection
        logger.debug(
            "[CLEANUP] Closing WebSocket connection",
            extra={"episode_id": self._episode_id},
        )
        await self._connection.close()

        # Drain event queue
        logger.debug(
            "[CLEANUP] Draining event queue",
            extra={"episode_id": self._episode_id},
        )
        self._events.drain_queue()

        # Clear sync state
        logger.debug(
            "[CLEANUP] Clearing sync state",
            extra={"episode_id": self._episode_id},
        )
        self._sync.clear_state()

        logger.info(
            "[CLEANUP] cleanup() COMPLETE",
            extra={"episode_id": self._episode_id},
        )


__all__ = ["TranscriptSyncClient"]
