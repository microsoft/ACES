"""Generic transcript push operations for WebSocket communication.

This module provides push operations including:
- Push message with unlimited retry
- Tool result pushing
- Local message tracking

This is the generic, harness-agnostic version that works with any message type
via the MessageSerializer protocol.

Logging category: AGENT.
"""

import asyncio
import json
import uuid
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from websockets import ClientConnection

# Callback that reconnects WebSocket and returns the new connection
ReconnectCallback = Callable[[], Coroutine[None, None, "ClientConnection"]]

from ...logging_config import LogCategory, get_saber_logger
from ...models.rest.websocket_config import WebSocketConfig
from ...models.rest.websocket_constants import WebSocketDefaults
from ...models.rest.websocket_messages import (
    PushAckData,
    PushAckMessage,
    WebSocketMessageType,
)
from .event_processor import WebSocketEventProcessor
from .protocols import MessageSerializer
from .utils import is_websocket_closed

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Generic message type
M = TypeVar("M")


class GenericTranscriptSyncOperations(Generic[M]):
    """Handles transcript push operations for any message type.

    Provides:
    - Push message with retry logic
    - Tool result pushing
    - Local message tracking

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

        # Local message tracking
        self._local_messages: list[M] = []

        logger.debug(
            "Created GenericTranscriptSyncOperations",
            extra={
                "episode_id": episode_id,
                "push_timeout": self._ws_config.push.confirmation_timeout,
            },
        )

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
        reconnect_callback: ReconnectCallback | None = None,
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
            On success, appends msg to _local_messages.
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
                            },
                            "id": str(uuid.uuid4()),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
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
                    # Validate ack response type
                    if isinstance(ack_response, PushAckMessage):
                        pass  # Valid Pydantic model
                    elif isinstance(ack_response, dict):
                        PushAckData(**ack_response["data"])  # Validate structure
                    else:
                        raise TypeError(f"Unexpected ack_response type: {type(ack_response)}")

                    # Update local state from server response
                    self._local_messages.append(msg)

                    if attempt > 1:
                        logger.info(
                            f"Push succeeded after {attempt} attempts",
                            extra={
                                "episode_id": self._episode_id,
                                "context": context,
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
        reconnect_callback: ReconnectCallback | None = None,
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
        # the initial messages - these come from the server via sync, not local additions.
        # Seed local tracking with the current input so future diffs start from
        # the correct baseline (initial messages are already on the server).
        if not self._local_messages:
            self._local_messages = list(input_messages)
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

    def clear_state(self) -> None:
        """Clear local state (for cleanup)."""
        self._local_messages.clear()


__all__ = ["GenericTranscriptSyncOperations"]
