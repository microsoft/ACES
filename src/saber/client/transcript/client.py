"""Transcript sync client orchestrator.

This module provides the main TranscriptSyncClient class that orchestrates
WebSocket-based transcript synchronization for any evaluation harness.

The client composes:
- WebSocketConnectionManager for connection lifecycle
- WebSocketEventProcessor for event handling
- GenericTranscriptSyncOperations for sync operations

Logging category: AGENT.
"""

import asyncio
from typing import Any, ClassVar, Generic, Optional, TypeVar

from ...logging_config import LogCategory, get_saber_logger
from ...models.rest.websocket_config import WebSocketConfig
from .connection import WebSocketConnectionManager
from .event_processor import WebSocketEventProcessor
from .models import SyncResult
from .protocols import MessageSerializer
from .sync_operations import GenericTranscriptSyncOperations

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Generic message type
M = TypeVar("M")


class TranscriptSyncClient(Generic[M]):
    """Generic, harness-agnostic transcript synchronization client.

    This client provides bidirectional transcript sync over WebSocket,
    supporting any evaluation harness via the MessageSerializer protocol.

    Provides:
    - Persistent WebSocket connection (established on first use or via context manager)
    - Instant notifications (<100ms when red team injects)
    - Differential sync (only pull deltas)
    - Local version tracking (checksum validation)
    - Automatic reconnection on connection loss

    Components:
    - _connection: WebSocketConnectionManager for connection lifecycle
    - _events: WebSocketEventProcessor for event handling
    - _sync: GenericTranscriptSyncOperations for push/pull operations

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
            messages = await client.sync_transcript()
            await client.push_message(response_msg)
    """

    # Class-level registry to track clients by episode_id for cleanup
    _clients_by_episode: ClassVar[dict[str, "TranscriptSyncClient[Any]"]] = {}

    @classmethod
    def get_client_for_episode(cls, episode_id: str) -> Optional["TranscriptSyncClient[Any]"]:
        """Get the client instance for a given episode_id.

        Used by sandbox cleanup to find and cleanup the client.

        Args:
            episode_id: The episode ID to look up

        Returns:
            The client instance if found, None otherwise
        """
        return cls._clients_by_episode.get(episode_id)

    @classmethod
    def _register_client(cls, episode_id: str, client: "TranscriptSyncClient[Any]") -> None:
        """Register a client instance for cleanup lookup."""
        cls._clients_by_episode[episode_id] = client
        logger.debug(
            "Registered transcript sync client for episode",
            extra={"episode_id": episode_id, "total_registered": len(cls._clients_by_episode)},
        )

    @classmethod
    def _unregister_client(cls, episode_id: str) -> None:
        """Unregister a client instance after cleanup."""
        if episode_id in cls._clients_by_episode:
            del cls._clients_by_episode[episode_id]
            logger.debug(
                "Unregistered transcript sync client for episode",
                extra={"episode_id": episode_id, "total_registered": len(cls._clients_by_episode)},
            )

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
        self._first_call = True

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

        # Flag to skip waiting after wait_and_sync_transcript() already synced
        self._skip_next_wait = False

        # Register this client for cleanup lookup
        self._register_client(episode_id, self)

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
    def local_version(self) -> int:
        """Get current local transcript version."""
        return self._sync.local_version

    @property
    def local_checksum(self) -> str:
        """Get current local transcript checksum."""
        return self._sync.local_checksum

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

    @property
    def event_queue(self) -> asyncio.Queue:
        """Get event queue (for advanced usage/testing)."""
        return self._events.event_queue

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

    async def _get_reconnect_callback(self) -> Any:
        """Get a callback for reconnection."""

        async def reconnect() -> Any:
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
                "Push disabled, updating local state only",
                extra={"episode_id": self._episode_id},
            )
            self._sync.update_local_state_without_push(message)
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
    # Sync operations
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

    async def sync_transcript(self) -> SyncResult[M]:
        """Request and apply transcript sync from server.

        Returns:
            SyncResult with updated messages, version, and sync mode
        """
        await self.ensure_connected()
        websocket = self._connection.websocket
        if not websocket:
            raise RuntimeError("WebSocket not connected")

        sync_data = await self._sync.request_sync(websocket)
        if not sync_data:
            raise RuntimeError("No sync response received")

        messages = self._sync.apply_sync_response(sync_data)

        return SyncResult(
            messages=messages,
            version=self._sync.local_version,
            checksum=self._sync.local_checksum,
            sync_mode=sync_data.sync_mode.value,
            modified=sync_data.modified or False,
        )

    async def wait_and_sync_transcript(self) -> list[M]:
        """Wait for server transcript modification and sync, returning updated messages.

        This method is used to get server-injected messages (e.g., continue prompts,
        red team injections) without the client adding its own continue messages.

        Returns:
            List of messages with server-provided transcript

        Raises:
            RuntimeError: If sync fails or times out
        """
        logger.debug(
            "wait_and_sync_transcript called",
            extra={
                "episode_id": self._episode_id,
                "local_version": self._sync.local_version,
            },
        )

        # Ensure WebSocket is connected
        await self.ensure_connected()

        # Wait for modification event
        event_received = await self._events.wait_for_state_event_with_retry()

        if not event_received:
            logger.warning(
                "No modification event received, returning current messages",
                extra={
                    "episode_id": self._episode_id,
                    "local_message_count": len(self._sync.local_messages),
                },
            )
            return self._sync.local_messages.copy()

        # Request sync
        websocket = self._connection.websocket
        if not websocket:
            raise RuntimeError("WebSocket not connected")

        sync_data = await self._sync.request_sync(websocket)

        if sync_data:
            self._sync.apply_sync_response(sync_data)
            # Signal to skip waiting in the next sync call
            self._skip_next_wait = True
        else:
            logger.warning(
                "No sync_data received",
                extra={"episode_id": self._episode_id},
            )

        return self._sync.local_messages.copy()

    async def wait_for_injection_and_sync(self) -> list[M]:
        """Wait for red team injection, then sync transcript.

        This method waits for red team to inject a user message. It specifically
        waits for is_waiting_on_assistant event which indicates a new user message
        was added.

        Waits indefinitely - the evaluation's task timeout or red team submission
        will terminate the session if needed.

        Returns:
            List of messages with server-provided transcript after injection
        """
        logger.debug(
            "wait_for_injection_and_sync called",
            extra={"episode_id": self._episode_id},
        )

        # Ensure WebSocket is connected
        await self.ensure_connected()

        # Wait for injection event (is_waiting_on_assistant) - no timeout
        await self._events.wait_for_injection_event()

        # Injection received - sync transcript to get the new user message
        websocket = self._connection.websocket
        if not websocket:
            raise RuntimeError("WebSocket not connected")

        sync_data = await self._sync.request_sync(websocket)

        if sync_data:
            self._sync.apply_sync_response(sync_data)
            # Signal to skip waiting in the next sync call
            self._skip_next_wait = True

        logger.debug(
            "Injection received and synced",
            extra={
                "episode_id": self._episode_id,
                "message_count": len(self._sync.local_messages),
            },
        )
        return self._sync.local_messages.copy()

    def should_skip_next_wait(self) -> bool:
        """Check if the next wait should be skipped (because we just synced).

        Used by harness wrappers to avoid duplicate waiting.

        Returns:
            True if next wait should be skipped
        """
        return self._skip_next_wait

    def clear_skip_next_wait(self) -> None:
        """Clear the skip_next_wait flag after it's been consumed."""
        self._skip_next_wait = False

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
        logger.info(
            "[CLEANUP] Signaling shutdown to event processor",
            extra={"episode_id": self._episode_id},
        )
        self._events.signal_shutdown()

        # Cancel listener task
        if self._events._listener_task:
            logger.info(
                "[CLEANUP] Cancelling listener task",
                extra={"episode_id": self._episode_id},
            )
            self._events._listener_task.cancel()
            try:
                await self._events._listener_task
            except asyncio.CancelledError:
                pass
            logger.info(
                "[CLEANUP] Listener task cancelled",
                extra={"episode_id": self._episode_id},
            )

        # Close connection
        logger.info(
            "[CLEANUP] Closing WebSocket connection",
            extra={"episode_id": self._episode_id},
        )
        await self._connection.close()

        # Drain event queue
        logger.info(
            "[CLEANUP] Draining event queue",
            extra={"episode_id": self._episode_id},
        )
        self._events.drain_queue()

        # Clear sync state
        logger.info(
            "[CLEANUP] Clearing sync state",
            extra={"episode_id": self._episode_id},
        )
        self._sync.clear_state()

        # Unregister from class-level registry
        self._unregister_client(self._episode_id)

        logger.info(
            "[CLEANUP] cleanup() COMPLETE",
            extra={"episode_id": self._episode_id},
        )


__all__ = ["TranscriptSyncClient"]
