"""Model wrapper for transcript synchronization.

This module provides a transparent wrapper around inspect_ai Model objects that
automatically synchronizes transcripts with the SABER server via WebSocket.

The wrapper intercepts model.generate() calls at the lowest level, ensuring that
transcripts are synchronized BEFORE tools execute, solving the race condition where
the server needs assistant messages for context extraction.

Key features:
- Works with ANY agent type (react, basic_agent, custom agents, etc.)
- Transparent delegation - behaves exactly like the wrapped model
- Bidirectional WebSocket sync - both push and pull over single persistent connection
- Differential sync - only transfers deltas for bandwidth efficiency
- Graceful error handling - failures don't crash agent execution
- Unified coordination - all agents use WebSocketTranscriptSyncingModelWrapper

Architecture:
- WebSocketConnectionManager: Connection lifecycle management
- WebSocketEventProcessor: Event listening and processing
- TranscriptSyncOperations: Push/pull sync with retry logic
- WebSocketTranscriptSyncingModelWrapper: Orchestrates the above components

Logging category: AGENT.
"""

import asyncio
from typing import Any, ClassVar, Optional

from inspect_ai.model import ChatMessage, Model, ModelOutput

from ...logging_config import LogCategory, get_saber_logger
from ...models.rest.websocket_config import WebSocketConfig
from .event_processor import WebSocketEventProcessor
from .sync_operations import TranscriptSyncOperations
from .websocket_connection import WebSocketConnectionManager

logger = get_saber_logger(LogCategory.AGENT, __name__)


class WebSocketTranscriptSyncingModelWrapper:
    """Model wrapper with WebSocket notifications + differential sync.

    Provides:
    - Persistent WebSocket connection (established on first generate())
    - Instant notifications (<100ms when red team injects)
    - Differential sync (only pull deltas)
    - Local version tracking (checksum validation)
    - Automatic reconnection on connection loss

    Flow:
    1. First generate() → establish WebSocket connection
    2. Server sends initial state event (is_waiting_on_assistant)
    3. Client waits for state event, then syncs transcript
    4. Background listener task receives subsequent events
    5. Events queued for generate() to consume
    6. Connection reused for entire episode

    Components:
    - _connection: WebSocketConnectionManager for connection lifecycle
    - _events: WebSocketEventProcessor for event handling
    - _sync: TranscriptSyncOperations for push/pull operations
    """

    # Class-level registry to track wrappers by episode_id for cleanup
    _wrappers_by_episode: ClassVar[dict[str, "WebSocketTranscriptSyncingModelWrapper"]] = {}

    @classmethod
    def get_wrapper_for_episode(cls, episode_id: str) -> Optional["WebSocketTranscriptSyncingModelWrapper"]:
        """Get the wrapper instance for a given episode_id.

        Used by sandbox cleanup to find and cleanup the wrapper.

        Args:
            episode_id: The episode ID to look up

        Returns:
            The wrapper instance if found, None otherwise
        """
        return cls._wrappers_by_episode.get(episode_id)

    @classmethod
    def _register_wrapper(cls, episode_id: str, wrapper: "WebSocketTranscriptSyncingModelWrapper") -> None:
        """Register a wrapper instance for cleanup lookup."""
        cls._wrappers_by_episode[episode_id] = wrapper
        logger.debug(
            "Registered wrapper for episode",
            extra={"episode_id": episode_id, "total_registered": len(cls._wrappers_by_episode)},
        )

    @classmethod
    def _unregister_wrapper(cls, episode_id: str) -> None:
        """Unregister a wrapper instance after cleanup."""
        if episode_id in cls._wrappers_by_episode:
            del cls._wrappers_by_episode[episode_id]
            logger.debug(
                "Unregistered wrapper for episode",
                extra={"episode_id": episode_id, "total_registered": len(cls._wrappers_by_episode)},
            )

    def __init__(
        self,
        base_model: Model,
        session_id: str,
        episode_id: str,
        rest_url: str,
        ws_config: WebSocketConfig | None = None,
    ):
        """Initialize WebSocket transcript syncing wrapper.

        Args:
            base_model: Underlying Model to wrap
            session_id: SABER session ID
            episode_id: SABER episode ID
            rest_url: Base URL of SABER REST API
            ws_config: WebSocket configuration (uses defaults if None)
        """
        self._base_model = base_model
        self._session_id = session_id
        self._episode_id = episode_id
        self._rest_url = rest_url
        self._first_call = True
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

        self._sync = TranscriptSyncOperations(
            episode_id=episode_id,
            event_processor=self._events,
            ws_config=self._ws_config,
        )

        # Flag to skip waiting in generate() after wait_and_sync_transcript() already synced
        self._skip_next_wait = False

        # Register this wrapper for cleanup lookup
        self._register_wrapper(episode_id, self)

        logger.debug(
            "Created WebSocketTranscriptSyncingModelWrapper",
            extra={
                "session_id": session_id,
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
    # Properties for backward compatibility
    # =========================================================================

    @property
    def _websocket(self) -> Any:
        """Get WebSocket (for backward compatibility with tests)."""
        return self._connection.websocket

    @_websocket.setter
    def _websocket(self, value: Any) -> None:
        """Set WebSocket (for backward compatibility with tests)."""
        self._connection._websocket = value

    @property
    def _event_queue(self) -> asyncio.Queue:
        """Get event queue (for backward compatibility with tests)."""
        return self._events.event_queue

    @_event_queue.setter
    def _event_queue(self, value: Any) -> None:
        """Set event queue (for backward compatibility with tests)."""
        self._events._event_queue = value

    @property
    def _local_version(self) -> int:
        """Get local version (for backward compatibility)."""
        return self._sync.local_version

    @_local_version.setter
    def _local_version(self, value: int) -> None:
        """Set local version (for backward compatibility)."""
        self._sync.local_version = value

    @property
    def _local_checksum(self) -> str:
        """Get local checksum (for backward compatibility)."""
        return self._sync.local_checksum

    @_local_checksum.setter
    def _local_checksum(self, value: str) -> None:
        """Set local checksum (for backward compatibility)."""
        self._sync.local_checksum = value

    @property
    def _local_messages(self) -> list[ChatMessage]:
        """Get local messages (for backward compatibility)."""
        return self._sync.local_messages

    @_local_messages.setter
    def _local_messages(self, value: list[ChatMessage]) -> None:
        """Set local messages (for backward compatibility)."""
        self._sync.local_messages = value

    @property
    def pull_enabled(self) -> bool:
        """Check if pull (server-controlled transcript) is enabled."""
        return self._ws_config.pull.enabled

    @property
    def _ws_url(self) -> str:
        """Get WebSocket URL (for backward compatibility with tests)."""
        return self._connection._ws_url

    @property
    def _listener_task(self) -> asyncio.Task | None:
        """Get listener task (for backward compatibility with tests)."""
        return self._events._listener_task

    @_listener_task.setter
    def _listener_task(self, value: asyncio.Task | None) -> None:
        """Set listener task (for backward compatibility with tests)."""
        self._events._listener_task = value

    # =========================================================================
    # Connection lifecycle
    # =========================================================================

    async def __aenter__(self) -> "WebSocketTranscriptSyncingModelWrapper":
        """Context manager entry - establish WebSocket connection."""
        await self._ensure_connected()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit - cleanup WebSocket connection."""
        await self.cleanup()
        return False  # Don't suppress exceptions

    async def _ensure_connected(self) -> None:
        """Establish WebSocket connection."""

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
    # Main generate method
    # =========================================================================

    async def generate(
        self,
        input: str | list[ChatMessage],
        tools: list | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Generate with WebSocket-based bidirectional sync.

        All coordination happens via WebSocket:
        - Pull: Receive state events (is_waiting_on_assistant, etc.), request sync
        - Push: Send push_message with new content

        Server initializes transcript with [system, user] messages, which triggers
        WAITING_FOR_ASSISTANT state → is_waiting_on_assistant event.
        Client treats this like any other modification event.

        Args:
            input: Messages or text input to the model
            tools: Optional list of tools available to the model
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput from the base model
        """
        # Ensure WebSocket connected on first call
        if self._first_call:
            self._first_call = False
            await self._ensure_connected()

        # Push any tool results that Inspect AI added to input
        pushed_tool_results = False
        if self._ws_config.push.enabled and isinstance(input, list):
            websocket = self._connection.websocket
            if websocket:
                pushed_tool_results = await self._sync.push_tool_results_if_needed(
                    websocket,
                    input,
                    reconnect_callback=await self._get_reconnect_callback(),
                )

        # Wait for state event, then pull (unified flow for all calls)
        if self._ws_config.pull.enabled:
            if isinstance(input, list):
                logger.info(
                    "[RESTART_DEBUG] generate() starting pull flow",
                    extra={
                        "episode_id": self._episode_id,
                        "skip_next_wait": self._skip_next_wait,
                        "pushed_tool_results": pushed_tool_results,
                        "input_message_count": len(input),
                        "local_version": self._sync.local_version,
                    },
                )
                # If wait_and_sync_transcript() just synced, skip waiting
                if self._skip_next_wait:
                    event_received = True
                    self._skip_next_wait = False
                    logger.info(
                        "[RESTART_DEBUG] Skipping wait - wait_and_sync_transcript already synced",
                        extra={"episode_id": self._episode_id},
                    )
                # If we just pushed tool results, skip waiting
                elif pushed_tool_results:
                    event_received = True
                    logger.info(
                        "[RESTART_DEBUG] Skipping wait - just pushed tool results",
                        extra={"episode_id": self._episode_id},
                    )
                else:
                    logger.info(
                        "[RESTART_DEBUG] Waiting for state event...",
                        extra={"episode_id": self._episode_id},
                    )
                    event_received = await self._events.wait_for_state_event_with_retry()
                    logger.info(
                        "[RESTART_DEBUG] State event wait completed",
                        extra={"episode_id": self._episode_id, "event_received": event_received},
                    )

                if event_received:
                    websocket = self._connection.websocket
                    if websocket:
                        # Request and apply sync
                        logger.info(
                            "[RESTART_DEBUG] Requesting sync from server",
                            extra={
                                "episode_id": self._episode_id,
                                "local_version_before": self._sync.local_version,
                            },
                        )
                        sync_data = await self._sync.request_sync(websocket)
                        if sync_data:
                            logger.info(
                                "[RESTART_DEBUG] Sync response received",
                                extra={
                                    "episode_id": self._episode_id,
                                    "sync_mode": sync_data.sync_mode.value,
                                    "server_version": sync_data.current_version.sequence,
                                    "full_transcript_len": (
                                        len(sync_data.full_transcript) if sync_data.full_transcript else 0
                                    ),
                                    "delta_len": len(sync_data.delta) if sync_data.delta else 0,
                                    "modified": sync_data.modified,
                                },
                            )
                            input = self._sync.apply_sync_response(sync_data)

                            logger.info(
                                "[RESTART_DEBUG] Applied transcript sync via WebSocket",
                                extra={
                                    "episode_id": self._episode_id,
                                    "sync_mode": sync_data.sync_mode.value,
                                    "new_version": self._sync.local_version,
                                    "new_message_count": len(input),
                                    "last_message_role": input[-1].role if input else "none",
                                },
                            )
                        else:
                            logger.warning(
                                "[RESTART_DEBUG] No sync_data received from server",
                                extra={"episode_id": self._episode_id},
                            )
                    else:
                        logger.warning(
                            "[RESTART_DEBUG] No websocket available for sync",
                            extra={"episode_id": self._episode_id},
                        )
            else:
                logger.warning(
                    "WebSocket wrapper received non-list input, cannot replace with modified transcript",
                    extra={
                        "input_type": type(input).__name__,
                        "episode_id": self._episode_id,
                    },
                )
        else:
            # Pull disabled - skip waiting for modifications (red team mode)
            logger.debug(
                "Pull disabled, skipping modification sync",
                extra={"episode_id": self._episode_id},
            )

        # Generate with (possibly modified) input
        output = await self._base_model.generate(input, tools, **kwargs)

        # Push new message via WebSocket (if push enabled)
        if self._ws_config.push.enabled:
            websocket = self._connection.websocket
            if websocket:
                success = await self._sync.push_message_with_retry(
                    websocket,
                    output.message,
                    context="output_push",
                    reconnect_callback=await self._get_reconnect_callback(),
                )
                if not success:
                    logger.error(
                        "Failed to push assistant output to server after retries",
                        extra={
                            "episode_id": self._episode_id,
                            "message_role": output.message.role,
                        },
                    )
        else:
            # Push disabled - update local state only
            logger.debug(
                "Push disabled, updating local state only",
                extra={"episode_id": self._episode_id},
            )
            self._sync.update_local_state_without_push(output.message)

        return output

    # =========================================================================
    # Transcript sync helpers
    # =========================================================================

    async def wait_and_sync_transcript(self) -> list[ChatMessage]:
        """Wait for server transcript modification and sync, returning updated messages.

        This method is used by the AgentContinue callback to get server-injected
        messages (e.g., continue prompts, red team injections) without the client
        adding its own continue messages.

        Returns:
            List of ChatMessage with server-provided transcript

        Raises:
            RuntimeError: If sync fails or times out
        """
        logger.info(
            "[RESTART_DEBUG] wait_and_sync_transcript() called (AgentContinue callback)",
            extra={
                "episode_id": self._episode_id,
                "current_local_version": self._sync.local_version,
                "current_message_count": len(self._sync.local_messages),
            },
        )

        # Ensure WebSocket is connected
        await self._ensure_connected()

        # Wait for modification event
        logger.info(
            "[RESTART_DEBUG] Waiting for state event in wait_and_sync_transcript",
            extra={"episode_id": self._episode_id},
        )
        event_received = await self._events.wait_for_state_event_with_retry()
        logger.info(
            "[RESTART_DEBUG] State event received in wait_and_sync_transcript",
            extra={"episode_id": self._episode_id, "event_received": event_received},
        )

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
            logger.info(
                "[RESTART_DEBUG] Applying sync in wait_and_sync_transcript",
                extra={
                    "episode_id": self._episode_id,
                    "sync_mode": sync_data.sync_mode.value,
                    "server_version": sync_data.current_version.sequence,
                    "full_transcript_len": len(sync_data.full_transcript) if sync_data.full_transcript else 0,
                    "delta_len": len(sync_data.delta) if sync_data.delta else 0,
                },
            )
            self._sync.apply_sync_response(sync_data)

            logger.info(
                "[RESTART_DEBUG] Transcript synced for AgentContinue callback",
                extra={
                    "episode_id": self._episode_id,
                    "sync_mode": sync_data.sync_mode.value,
                    "message_count": len(self._sync.local_messages),
                    "last_message_role": self._sync.local_messages[-1].role if self._sync.local_messages else "none",
                },
            )

            # Signal generate() to skip waiting
            self._skip_next_wait = True
            logger.info(
                "[RESTART_DEBUG] Set _skip_next_wait=True",
                extra={"episode_id": self._episode_id},
            )
        else:
            logger.warning(
                "[RESTART_DEBUG] No sync_data in wait_and_sync_transcript",
                extra={"episode_id": self._episode_id},
            )

        logger.info(
            "[RESTART_DEBUG] wait_and_sync_transcript returning",
            extra={
                "episode_id": self._episode_id,
                "returning_message_count": len(self._sync.local_messages),
            },
        )
        return self._sync.local_messages.copy()

    async def wait_for_injection_and_sync(self) -> list[ChatMessage]:
        """Wait for red team injection, then sync transcript.

        This method is used by the AgentContinue callback to wait for red team
        to inject a user message. It specifically waits for is_waiting_on_assistant
        event which indicates a new user message was added.

        Waits indefinitely - the evaluation's task timeout or red team submission
        will terminate the session if needed.

        Returns:
            List of ChatMessage with server-provided transcript after injection
        """
        logger.info(
            "[INJECTION_WAIT] wait_for_injection_and_sync() called - waiting indefinitely",
            extra={
                "episode_id": self._episode_id,
                "current_local_version": self._sync.local_version,
                "current_message_count": len(self._sync.local_messages),
            },
        )

        # Ensure WebSocket is connected
        await self._ensure_connected()

        # Wait for injection event (is_waiting_on_assistant) - no timeout
        await self._events.wait_for_injection_event()

        # Injection received - sync transcript to get the new user message
        websocket = self._connection.websocket
        if not websocket:
            raise RuntimeError("WebSocket not connected")

        sync_data = await self._sync.request_sync(websocket)

        if sync_data:
            logger.info(
                "[INJECTION_WAIT] Applying sync after injection",
                extra={
                    "episode_id": self._episode_id,
                    "sync_mode": sync_data.sync_mode.value,
                    "server_version": sync_data.current_version.sequence,
                },
            )
            self._sync.apply_sync_response(sync_data)

            # Signal generate() to skip waiting
            self._skip_next_wait = True

        logger.info(
            "[INJECTION_WAIT] wait_for_injection_and_sync returning",
            extra={
                "episode_id": self._episode_id,
                "message_count": len(self._sync.local_messages),
            },
        )
        return self._sync.local_messages.copy()

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup(self) -> None:
        """Close WebSocket connection and release resources when episode ends."""
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

        # Cancel listener task (may be set via backward-compat property)
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

        # Close connection (handles its own listener task if started via ensure_connected)
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
        self._unregister_wrapper(self._episode_id)

        logger.info(
            "[CLEANUP] cleanup() COMPLETE",
            extra={"episode_id": self._episode_id},
        )

    # =========================================================================
    # Delegation to base model
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the base model.

        This makes the wrapper transparent - it behaves exactly like the
        wrapped model for all attributes/methods except generate().

        Args:
            name: Attribute name being accessed

        Returns:
            The attribute from the base model
        """
        return getattr(self._base_model, name)

    def __repr__(self) -> str:
        """Return string representation showing wrapped model."""
        return f"WebSocketTranscriptSyncingModelWrapper({self._base_model!r})"


__all__ = [
    "WebSocketTranscriptSyncingModelWrapper",
]
