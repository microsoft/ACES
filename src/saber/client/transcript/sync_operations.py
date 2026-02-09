"""Generic transcript sync operations for WebSocket communication.

This module provides sync operations including:
- Push message with unlimited retry
- Tool result pushing
- Sync request/response handling
- Local state tracking

This is the generic, harness-agnostic version that works with any message type
via the MessageSerializer protocol.

Logging category: AGENT.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from websockets import ClientConnection

from ...logging_config import LogCategory, get_saber_logger
from ...models.rest.websocket_config import WebSocketConfig
from ...models.rest.websocket_constants import WebSocketDefaults
from ...models.rest.websocket_messages import (
    PushAckData,
    PushAckMessage,
    SyncMode,
    SyncResponseData,
    SyncResponseMessage,
    TranscriptErrorMessage,
    WebSocketMessageType,
)
from ...models.transcript import compute_checksum
from .event_processor import WebSocketEventProcessor
from .protocols import MessageSerializer
from .utils import is_websocket_closed

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Generic message type
M = TypeVar("M")


class GenericTranscriptSyncOperations(Generic[M]):
    """Handles transcript synchronization operations for any message type.

    Provides:
    - Push message with retry logic
    - Tool result pushing
    - Sync request/response handling
    - Local state (version, checksum, messages) tracking

    Works with WebSocketEventProcessor for event handling.

    Type Parameter:
        M: The harness-specific message type (e.g., ChatMessage for inspect_ai)
    """

    def __init__(
        self,
        episode_id: str,
        event_processor: WebSocketEventProcessor,
        serializer: MessageSerializer[M],
        ws_config: WebSocketConfig | None = None,
    ) -> None:
        """Initialize sync operations.

        Args:
            episode_id: SABER episode ID for logging
            event_processor: Event processor for waiting on responses
            serializer: MessageSerializer for the harness message type
            ws_config: WebSocket configuration (uses defaults if None)
        """
        self._episode_id = episode_id
        self._events = event_processor
        self._serializer = serializer
        self._ws_config = ws_config or WebSocketConfig()

        # Local version tracking
        self._local_version = 0
        self._local_checksum = compute_checksum([])
        self._local_messages: list[M] = []

        logger.debug(
            "Created GenericTranscriptSyncOperations",
            extra={
                "episode_id": episode_id,
                "push_timeout": self._ws_config.push.confirmation_timeout,
                "sync_timeout": self._ws_config.pull.sync_timeout,
            },
        )

    @property
    def local_version(self) -> int:
        """Get current local transcript version."""
        return self._local_version

    @local_version.setter
    def local_version(self, value: int) -> None:
        """Set local transcript version."""
        self._local_version = value

    @property
    def local_checksum(self) -> str:
        """Get current local transcript checksum."""
        return self._local_checksum

    @local_checksum.setter
    def local_checksum(self, value: str) -> None:
        """Set local transcript checksum."""
        self._local_checksum = value

    @property
    def local_messages(self) -> list[M]:
        """Get current local transcript messages."""
        return self._local_messages

    @local_messages.setter
    def local_messages(self, value: list[M]) -> None:
        """Set local transcript messages."""
        self._local_messages = value

    async def push_message_with_retry(
        self,
        websocket: "ClientConnection",
        msg: M,
        context: str = "message_push",
        reconnect_callback: Any | None = None,
    ) -> bool:
        """Push a message to the server with unlimited retry logic.

        Retries the push with exponential backoff (capped at 60s), reconnecting
        WebSocket if needed. Push must succeed for transcript consistency - there
        is no retry limit.

        Args:
            websocket: WebSocket connection to use
            msg: The message to push
            context: Context string for logging (e.g., "tool_result_push", "output_push")
            reconnect_callback: Optional async callback to reconnect on failure.
                               Signature: async def callback() -> ClientConnection

        Returns:
            True when push succeeds (always succeeds eventually or raises)

        Note:
            On success, updates _local_version, _local_checksum, and _local_messages.
        """
        backoff = 1.0  # Initial backoff in seconds
        max_backoff = 60.0  # Cap backoff at 60 seconds
        attempt = 0
        current_websocket = websocket

        while True:
            attempt += 1
            try:
                if is_websocket_closed(current_websocket):
                    if reconnect_callback:
                        current_websocket = await reconnect_callback()
                    else:
                        raise ConnectionError("WebSocket not connected and no reconnect callback")

                # Send push message using serializer
                await current_websocket.send(
                    json.dumps(
                        {
                            "type": WebSocketMessageType.PUSH_MESSAGE.value,
                            "data": {
                                "message": self._serializer.serialize(msg),
                                "since_version": self._local_version,
                                "client_checksum": self._local_checksum,
                            },
                            "id": str(uuid.uuid4()),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                )

                # Wait for push acknowledgment
                ack_response = await self._events.wait_for_message_type(
                    expected_type=WebSocketMessageType.PUSH_ACK,
                    timeout=self._ws_config.push.confirmation_timeout,
                    max_iterations=WebSocketDefaults.MAX_ACK_WAIT_ITERATIONS,
                    context=context,
                )

                if ack_response:
                    # Get push_ack data - either from Pydantic model or parse from dict
                    if isinstance(ack_response, PushAckMessage):
                        push_ack = ack_response.data
                    elif isinstance(ack_response, dict):
                        push_ack = PushAckData(**ack_response["data"])
                    else:
                        raise TypeError(f"Unexpected ack_response type: {type(ack_response)}")

                    # Update local state from server response
                    self._local_messages.append(msg)
                    self._local_version = push_ack.version
                    self._local_checksum = push_ack.checksum

                    if attempt > 1:
                        logger.info(
                            f"Push succeeded after {attempt} attempts",
                            extra={
                                "episode_id": self._episode_id,
                                "context": context,
                                "new_version": self._local_version,
                            },
                        )
                    return True

            except Exception as e:
                logger.debug(
                    f"Push attempt {attempt} failed: {e}",
                    extra={
                        "episode_id": self._episode_id,
                        "context": context,
                        "error_type": type(e).__name__,
                    },
                )

            # Retry with exponential backoff
            logger.debug(
                f"Retrying push in {backoff:.1f}s",
                extra={
                    "episode_id": self._episode_id,
                    "context": context,
                    "attempt": attempt,
                },
            )

            await asyncio.sleep(backoff)
            backoff = min(backoff * self._ws_config.push.retry_backoff_multiplier, max_backoff)

            # Force reconnect before retry
            if reconnect_callback:
                try:
                    current_websocket = await reconnect_callback()
                except Exception as e:
                    logger.warning(f"Reconnect failed: {e}")

    async def push_tool_results_if_needed(
        self,
        websocket: "ClientConnection",
        input_messages: list[M],
        reconnect_callback: Any | None = None,
    ) -> bool:
        """Push tool results to server if harness added them to input.

        When a harness runs tools, it adds tool result messages to the input
        list before calling generate() again. These tool results exist only
        on the client - the server doesn't know about them yet. We need to
        push them to keep the server transcript in sync.

        Args:
            websocket: WebSocket connection to use
            input_messages: The input message list from the harness
            reconnect_callback: Optional callback for reconnection

        Returns:
            True if tool results were pushed, False otherwise
        """
        # Don't push if we haven't synced with server yet
        # On the first generate() call, _local_messages is empty but input has
        # the initial messages - these come from the server via sync, not local additions
        if not self._local_messages:
            return False

        # Find messages in input that aren't in our local transcript
        # These are tool results added by the harness's tool execution loop
        local_len = len(self._local_messages)
        input_len = len(input_messages)

        if input_len <= local_len:
            # No new messages to push
            return False

        # Get the new messages (tool results)
        new_messages = input_messages[local_len:]

        # Check that these are tool messages (sanity check)
        tool_messages = [m for m in new_messages if self._serializer.get_role(m) == "tool"]
        if len(tool_messages) != len(new_messages):
            logger.warning(
                "New messages in input include non-tool messages, unexpected",
                extra={
                    "episode_id": self._episode_id,
                    "new_message_roles": [self._serializer.get_role(m) for m in new_messages],
                },
            )

        if not new_messages:
            return False

        logger.debug(
            "Pushing tool results to server",
            extra={
                "episode_id": self._episode_id,
                "tool_result_count": len(new_messages),
                "local_version": self._local_version,
            },
        )

        # Push each tool result message with retry
        all_succeeded = True
        for msg in new_messages:
            success = await self.push_message_with_retry(
                websocket, msg, context="tool_result_push", reconnect_callback=reconnect_callback
            )
            if not success:
                logger.error(
                    "Failed to push tool result to server after retries - transcript is now inconsistent",
                    extra={
                        "episode_id": self._episode_id,
                        "message_role": self._serializer.get_role(msg),
                    },
                )
                all_succeeded = False
                # Continue trying to push remaining messages to minimize damage

        return all_succeeded

    async def request_sync(
        self,
        websocket: "ClientConnection",
    ) -> SyncResponseData | None:
        """Request transcript sync from server.

        Sends sync_request and waits for sync_response.

        Args:
            websocket: WebSocket connection to use

        Returns:
            SyncResponseData if sync successful, None on error/timeout

        Raises:
            RuntimeError: If sync_response contains an error
        """
        logger.debug(
            "Sending sync_request",
            extra={"episode_id": self._episode_id, "local_version": self._local_version},
        )

        # Send sync request
        await websocket.send(
            json.dumps(
                {
                    "type": WebSocketMessageType.SYNC_REQUEST.value,
                    "data": {
                        "since_version": self._local_version,
                        "client_checksum": self._local_checksum,
                    },
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        )

        # Wait for sync response
        response = None
        response_type = None
        events_discarded = 0

        for iteration in range(WebSocketDefaults.MAX_SYNC_RESPONSE_ITERATIONS):
            response = await asyncio.wait_for(
                self._events.event_queue.get(),
                timeout=self._ws_config.pull.sync_timeout,
            )

            response_type = None
            if hasattr(response, "type"):
                response_type = response.type
            elif isinstance(response, dict):
                response_type = response.get("type")

            if response_type == WebSocketMessageType.SYNC_RESPONSE.value:
                break

            if response_type == WebSocketMessageType.TRANSCRIPT_ERROR.value:
                break

            events_discarded += 1
            logger.debug(
                "Discarding non-sync event while waiting for sync response",
                extra={
                    "episode_id": self._episode_id,
                    "event_type": response_type,
                    "iteration": iteration,
                    "events_discarded": events_discarded,
                },
            )
        else:
            logger.error(
                "Exhausted iterations waiting for sync_response",
                extra={
                    "episode_id": self._episode_id,
                    "max_iterations": WebSocketDefaults.MAX_SYNC_RESPONSE_ITERATIONS,
                    "events_discarded": events_discarded,
                    "last_event_type": response_type,
                },
            )
            raise RuntimeError(
                f"Failed to get sync_response after "
                f"{WebSocketDefaults.MAX_SYNC_RESPONSE_ITERATIONS} iterations, "
                f"last event: {response_type}"
            )

        # Handle response
        if response_type == WebSocketMessageType.SYNC_RESPONSE.value:
            if isinstance(response, SyncResponseMessage):
                return response.data
            elif isinstance(response, dict):
                return SyncResponseData(**response["data"])
            else:
                raise TypeError(f"Unexpected response type: {type(response)}")

        elif response_type == WebSocketMessageType.TRANSCRIPT_ERROR.value:
            if isinstance(response, TranscriptErrorMessage):
                error_msg = response.data.message or "Unknown error"
            elif isinstance(response, dict):
                error_msg = response.get("data", {}).get("message", "Unknown error")
            else:
                error_msg = "Unknown error"
            raise RuntimeError(f"Transcript sync error: {error_msg}")

        return None

    def apply_sync_response(self, sync_data: SyncResponseData) -> list[M]:
        """Apply sync response to local state.

        Updates local_messages, local_version, and local_checksum.

        Args:
            sync_data: Sync response data from server

        Returns:
            Updated local messages list
        """
        old_version = self._local_version
        old_count = len(self._local_messages)

        logger.debug(
            "Applying sync response",
            extra={
                "episode_id": self._episode_id,
                "sync_mode": sync_data.sync_mode.value,
                "old_version": old_version,
                "old_message_count": old_count,
                "server_version": sync_data.current_version.sequence,
            },
        )

        if sync_data.sync_mode == SyncMode.FULL:
            # Rewrite detected - replace entire transcript
            if not sync_data.full_transcript:
                logger.warning("Full sync mode but no full_transcript provided")
                raise ValueError("Missing full_transcript in FULL sync mode")

            logger.debug(
                "FULL sync - replacing transcript",
                extra={
                    "episode_id": self._episode_id,
                    "new_message_count": len(sync_data.full_transcript),
                },
            )
            self._local_messages = [self._serializer.deserialize(m) for m in sync_data.full_transcript]

        elif sync_data.sync_mode == SyncMode.DELTA:
            # Delta sync - append new messages
            if sync_data.delta:
                new_messages = [self._serializer.deserialize(m) for m in sync_data.delta]
                logger.debug(
                    "DELTA sync - appending messages",
                    extra={
                        "episode_id": self._episode_id,
                        "delta_count": len(new_messages),
                    },
                )
                self._local_messages.extend(new_messages)

        elif sync_data.sync_mode == SyncMode.NO_CHANGE:
            # No changes since requested version - transcript is already up to date
            logger.debug(
                "NO_CHANGE sync - transcript up to date",
                extra={
                    "episode_id": self._episode_id,
                    "version": self._local_version,
                },
            )

        # Update local state
        self._local_version = sync_data.current_version.sequence
        self._local_checksum = sync_data.current_version.checksum

        logger.debug(
            "Sync applied",
            extra={
                "episode_id": self._episode_id,
                "sync_mode": sync_data.sync_mode.value,
                "new_version": self._local_version,
                "message_count": len(self._local_messages),
            },
        )

        return self._local_messages.copy()

    def _format_messages_for_log(self, messages: list[M]) -> list[str]:
        """Format messages for logging output.

        Args:
            messages: List of messages to format

        Returns:
            List of formatted message strings (truncated content)
        """
        if not messages:
            return []
        result = []
        for m in messages:
            role = self._serializer.get_role(m)
            content = str(self._serializer.serialize(m).get("content", ""))[:50]
            result.append(f"{role}: {content}...")
        return result

    def _analyze_tool_calls_client(self, messages: list[M]) -> dict[str, Any]:
        """Analyze tool_calls in client-side messages to detect orphaned calls."""
        all_tool_call_ids: list[str] = []
        responded_tool_call_ids: list[str] = []

        for msg in messages:
            role = self._serializer.get_role(msg)
            if role == "assistant" and self._serializer.has_tool_calls(msg):
                all_tool_call_ids.extend(self._serializer.get_tool_call_ids(msg))
            elif role == "tool":
                tool_call_id = self._serializer.get_tool_call_id(msg)
                if tool_call_id:
                    responded_tool_call_ids.append(tool_call_id)

        orphaned = [tc_id for tc_id in all_tool_call_ids if tc_id not in responded_tool_call_ids]

        return {
            "total_tool_calls": len(all_tool_call_ids),
            "responded_tool_calls": len(responded_tool_call_ids),
            "orphaned_tool_call_ids": orphaned[:5],  # Truncate for logging
        }

    def update_local_state_without_push(self, msg: M) -> None:
        """Update local state when push is disabled.

        Used when push.enabled=False to maintain local tracking.

        Args:
            msg: Message to add to local state
        """
        self._local_messages.append(msg)
        self._local_version += 1
        self._local_checksum = compute_checksum([self._serializer.serialize(m) for m in self._local_messages])

    def clear_state(self) -> None:
        """Clear local state (for cleanup)."""
        self._local_messages.clear()
        self._local_version = 0
        self._local_checksum = compute_checksum([])


__all__ = ["GenericTranscriptSyncOperations"]
