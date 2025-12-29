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

DEPRECATED:
- TranscriptSyncingModelWrapper: Legacy REST-based push-only wrapper (use WebSocket version)
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from websockets import ClientConnection

try:
    from websockets.protocol import State as WebSocketState
except ImportError:
    WebSocketState = None


def _is_websocket_closed(websocket: Any) -> bool:
    """Check if a WebSocket connection is closed.

    Compatible with both websockets <15.0 (.closed attribute)
    and websockets >=15.0 (.state attribute with State enum).

    Args:
        websocket: WebSocket connection object

    Returns:
        True if the connection is closed or closing, False if open
    """
    if websocket is None:
        return True
    # websockets >=15.0 uses .state attribute with State enum
    if hasattr(websocket, "state") and WebSocketState is not None:
        return websocket.state in (WebSocketState.CLOSED, WebSocketState.CLOSING)
    # websockets <15.0 uses .closed attribute
    if hasattr(websocket, "closed"):
        return bool(websocket.closed)
    # Default to closed if we can't determine
    return True


from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolCall

from ...logging_config import LogCategory, get_saber_logger
from ...models.rest.websocket_config import WebSocketConfig
from ...models.rest.websocket_constants import WebSocketDefaults
from ...models.rest.websocket_messages import (
    PushAckData,
    PushAckMessage,
    StateEventMessage,
    SyncMode,
    SyncResponseData,
    SyncResponseMessage,
    TranscriptErrorMessage,
    WebSocketMessageType,
    WebSocketServerMessage,
    WebSocketServerMessageAdapter,
)
from ...models.transcript import compute_checksum

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Type alias for messages that can be either Pydantic models or dicts (for test compatibility)
WebSocketMessageOrDict = WebSocketServerMessage | dict[str, Any]


class _MessageSerializationMixin:
    """Mixin providing message serialization/deserialization utilities."""

    @staticmethod
    def _serialize_message(msg: ChatMessage) -> dict[str, Any]:
        """Convert ChatMessage to JSON-safe dict.

        Args:
            msg: Inspect AI ChatMessage object

        Returns:
            Dictionary with keys: role, content, tool_calls (optional), etc.

        Raises:
            ValueError: If message type is unknown or unsupported
        """
        result: dict[str, Any] = {}

        # Extract role
        if isinstance(msg, ChatMessageSystem):
            result["role"] = "system"
        elif isinstance(msg, ChatMessageUser):
            result["role"] = "user"
        elif isinstance(msg, ChatMessageAssistant):
            result["role"] = "assistant"
        elif isinstance(msg, ChatMessageTool):
            result["role"] = "tool"
        else:
            raise ValueError(f"Unknown message type: {type(msg)}")

        # Extract content - handle both string and list of Content objects
        if isinstance(msg.content, str):
            result["content"] = msg.content
        elif isinstance(msg.content, list):
            # Extract text content parts and concatenate
            text_parts = []
            reasoning_text = None

            for content_item in msg.content:
                if isinstance(content_item, ContentText):
                    text_parts.append(content_item.text)
                elif isinstance(content_item, ContentReasoning):
                    reasoning_text = content_item.reasoning

            result["content"] = "".join(text_parts)

            # Add reasoning if present (for assistant messages)
            if reasoning_text:
                result["reasoning"] = reasoning_text
        else:
            result["content"] = ""

        # Handle assistant-specific fields
        if isinstance(msg, ChatMessageAssistant):
            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function": tc.function,
                        "arguments": tc.arguments,
                    }
                    for tc in msg.tool_calls
                ]

        # Handle tool message-specific fields
        if isinstance(msg, ChatMessageTool):
            if msg.tool_call_id:
                result["tool_call_id"] = msg.tool_call_id
            if msg.function:
                result["name"] = msg.function

        return result

    @staticmethod
    def _deserialize_message(msg_data: dict[str, Any]) -> ChatMessage:
        """Convert message dictionary to ChatMessage object.

        Inverse of _serialize_message() - converts JSON-safe dict back to
        Inspect AI ChatMessage objects.

        Args:
            msg_data: Dictionary with keys: role, content, tool_calls (optional), etc.

        Returns:
            ChatMessage object (ChatMessageUser, ChatMessageAssistant, etc.)

        Raises:
            ValueError: If role is unknown or message structure is invalid
        """
        role = msg_data.get("role")
        content = msg_data.get("content", "")

        if role == "system":
            return ChatMessageSystem(content=content)

        elif role == "user":
            return ChatMessageUser(content=content)

        elif role == "assistant":
            # Assistant messages may have tool calls
            tool_calls_data = msg_data.get("tool_calls")
            if tool_calls_data:
                # Convert tool call dicts to ToolCall objects
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        function=tc["function"],
                        arguments=tc["arguments"],
                        type="function",
                    )
                    for tc in tool_calls_data
                ]
                return ChatMessageAssistant(content=content, tool_calls=tool_calls)
            else:
                return ChatMessageAssistant(content=content)

        elif role == "tool":
            # Tool messages need tool_call_id and function name
            tool_call_id = msg_data.get("tool_call_id")
            function_name = msg_data.get("name")
            if not tool_call_id or not function_name:
                raise ValueError(f"Tool message missing tool_call_id or name: {msg_data}")
            return ChatMessageTool(
                content=content,
                tool_call_id=tool_call_id,
                function=function_name,
            )

        else:
            raise ValueError(f"Unknown message role: {role}")


class WebSocketTranscriptSyncingModelWrapper(_MessageSerializationMixin):
    """Model wrapper with WebSocket notifications + differential sync.

    Phase 3: WebSocket-based coordination with instant notifications and efficient sync.

    Features:
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

    Attributes:
        _first_call: Tracks if this is the first generate() call
        _websocket: WebSocket connection object
        _ws_lock: Lock for WebSocket connection management
        _event_queue: Queue for WebSocket server messages
        _listener_task: Background task listening for events
        _local_version: Client's current transcript version
        _local_checksum: SHA256 checksum of local transcript
        _local_messages: Local copy of transcript messages
        _ws_url: WebSocket URL constructed from REST URL
        _ws_config: WebSocket configuration settings
    """

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

        # WebSocket connection state
        self._websocket: ClientConnection | None = None
        self._ws_lock = asyncio.Lock()
        self._event_queue: asyncio.Queue[WebSocketMessageOrDict] = asyncio.Queue(
            maxsize=self._ws_config.pull.event_queue_max_size
        )
        self._listener_task: asyncio.Task[None] | None = None

        # Local version tracking
        self._local_version = 0
        self._local_checksum = compute_checksum([])
        self._local_messages: list[ChatMessage] = []

        # Flag to skip waiting in generate() after wait_and_sync_transcript() already synced
        self._skip_next_wait = False

        # Build WebSocket URL from REST URL (with proper parsing)
        self._ws_url = self._build_websocket_url(rest_url, episode_id)

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
                "ws_url": self._ws_url,
            },
        )

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
        from urllib.parse import urlparse, urlunparse

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

    async def __aenter__(self) -> "WebSocketTranscriptSyncingModelWrapper":
        """Context manager entry - establish WebSocket connection."""
        await self._ensure_connected()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit - cleanup WebSocket connection."""
        await self.cleanup()
        return False  # Don't suppress exceptions

    async def _ensure_connected(self) -> None:
        """Establish WebSocket connection with reconnection support.

        Implements exponential backoff reconnection and proper resource cleanup.
        """
        async with self._ws_lock:
            if self._websocket is not None and not _is_websocket_closed(self._websocket):
                return  # Already connected

            # Determine number of attempts based on reconnect policy
            max_attempts = self._ws_config.max_reconnect_attempts if self._ws_config.reconnect_enabled else 1

            for attempt in range(max_attempts):
                temp_websocket = None
                temp_listener = None

                try:
                    import websockets

                    logger.info(
                        "Etablishing WebSocket connection",
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
                            assert ws is not None  # mypy: temp_websocket is set above
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
                            # Phase 3: Start background listener AFTER connection confirmed
                            temp_listener = asyncio.create_task(self._listen_for_events_impl(temp_websocket))

                            # SUCCESS - commit state
                            self._websocket = temp_websocket
                            self._listener_task = temp_listener
                            temp_websocket = None  # Don't cleanup
                            temp_listener = None  # Don't cleanup
                            return
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
                    if temp_websocket and not _is_websocket_closed(temp_websocket):
                        try:
                            await asyncio.wait_for(
                                temp_websocket.close(), timeout=WebSocketDefaults.WEBSOCKET_CLOSE_TIMEOUT_SECONDS
                            )
                        except asyncio.TimeoutError:
                            logger.warning("WebSocket close timed out during cleanup")
                        except Exception as e:
                            logger.warning(f"Error closing WebSocket during cleanup: {e}")

    async def _listen_for_events_impl(self, websocket: "ClientConnection") -> None:
        """Background task that listens for WebSocket events.

        Receives transcript_modified events and queues them for generate().

        Handles both push (server events) and pull (response to client requests).
        Server-initiated events are queued for generate() to consume.
        Client-response messages are consumed directly by awaiting code.

        Args:
            websocket: WebSocket connection to listen on (passed explicitly for cleanup safety)
        """
        import websockets
        from pydantic import ValidationError

        try:
            async for message in websocket:
                # Parse JSON directly into Pydantic model using discriminated union
                try:
                    parsed_message = WebSocketServerMessageAdapter.validate_json(message)
                except ValidationError as e:
                    logger.warning(
                        f"Failed to parse WebSocket message: {e}",
                        extra={"raw_message": message[:200]},
                    )
                    continue

                event_type = parsed_message.type

                if event_type == WebSocketMessageType.TRANSCRIPT_MODIFIED.value:
                    # Server-initiated notification - queue for generate()
                    await self._event_queue.put(parsed_message)

                    logger.debug(
                        "Received transcript modification event",
                        extra={
                            "episode_id": self._episode_id,
                            "version": (
                                parsed_message.data.version if isinstance(parsed_message, StateEventMessage) else None
                            ),
                        },
                    )

                elif event_type in (
                    WebSocketMessageType.IS_WAITING_ON_USER.value,
                    WebSocketMessageType.IS_WAITING_ON_ASSISTANT.value,
                    WebSocketMessageType.IS_WAITING_ON_TOOLS.value,
                ):
                    # State machine events - log and queue
                    await self._event_queue.put(parsed_message)

                    logger.debug(
                        "Received state machine event",
                        extra={
                            "episode_id": self._episode_id,
                            "event_type": event_type,
                            "state": (
                                parsed_message.data.state if isinstance(parsed_message, StateEventMessage) else None
                            ),
                        },
                    )

                elif event_type == WebSocketMessageType.TRANSCRIPT_ERROR.value:
                    # Error events (stuck_state, etc.) - queue for handling
                    await self._event_queue.put(parsed_message)

                    if isinstance(parsed_message, TranscriptErrorMessage):
                        error_type = parsed_message.data.error
                        if error_type.value == "stuck_state":
                            logger.error(
                                "Episode stuck in state - retry required",
                                extra={
                                    "episode_id": self._episode_id,
                                    "state": parsed_message.data.state,
                                    "duration_seconds": parsed_message.data.duration_seconds,
                                    "threshold_seconds": parsed_message.data.threshold_seconds,
                                },
                            )
                        else:
                            logger.error(
                                "Received transcript error event",
                                extra={
                                    "episode_id": self._episode_id,
                                    "error": error_type.value,
                                },
                            )

                elif event_type in (
                    WebSocketMessageType.SYNC_RESPONSE.value,
                    WebSocketMessageType.PUSH_ACK.value,
                    WebSocketMessageType.PONG.value,
                ):
                    # Response to client request - queue for awaiting code
                    await self._event_queue.put(parsed_message)

                    logger.debug(
                        "Received WebSocket response",
                        extra={
                            "episode_id": self._episode_id,
                            "type": event_type,
                        },
                    )

                elif event_type == WebSocketMessageType.CONNECTED.value:
                    # Connection handshake message - should be consumed during _ensure_connected
                    # but handle it gracefully if it somehow reaches the listener
                    logger.debug(
                        "Received 'connected' message in listener (unexpected but harmless)",
                        extra={"episode_id": self._episode_id},
                    )

                else:
                    logger.warning(
                        "Received unknown WebSocket message type",
                        extra={
                            "episode_id": self._episode_id,
                            "type": event_type,
                        },
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed", extra={"episode_id": self._episode_id})
        except Exception as e:
            logger.error("WebSocket listener error", extra={"episode_id": self._episode_id, "error": str(e)})

    async def _listen_for_events(self) -> None:
        """Background task that listens for WebSocket events (legacy wrapper)."""
        if self._websocket:
            await self._listen_for_events_impl(self._websocket)

    async def _wait_for_modification_event(self) -> bool:
        """Wait for WebSocket event indicating transcript modification.

        Only accepts state events (is_waiting_on_*, transcript_modified).
        Other events (push_ack, sync_response) are discarded as they're
        stale responses from previous operations.

        Returns:
            True if state event received, False on timeout
        """
        try:
            await self._ensure_connected()

            # Loop until we get a state event, discarding other events
            for _iteration in range(WebSocketDefaults.MAX_EVENT_DISCARD_ITERATIONS):
                # Wait for event from queue (populated by background listener)
                event_data: WebSocketMessageOrDict = await asyncio.wait_for(
                    self._event_queue.get(), timeout=self._ws_config.pull.event_timeout
                )

                # Check event type
                event_type = event_data.type if hasattr(event_data, "type") else None

                # Accept state events
                state_event_types = [
                    WebSocketMessageType.IS_WAITING_ON_USER.value,
                    WebSocketMessageType.IS_WAITING_ON_ASSISTANT.value,
                    WebSocketMessageType.IS_WAITING_ON_TOOLS.value,
                    WebSocketMessageType.TRANSCRIPT_MODIFIED.value,
                    WebSocketMessageType.TRANSCRIPT_ERROR.value,
                ]

                if event_type in state_event_types:
                    # Extract version from Pydantic model (StateEventMessage has .data.version)
                    version = None
                    state = None
                    if isinstance(event_data, StateEventMessage):
                        version = event_data.data.version
                        state = event_data.data.state

                    logger.debug(
                        "Received state event",
                        extra={
                            "episode_id": self._episode_id,
                            "version": version,
                            "event_type": event_type,
                            "state": state,
                        },
                    )
                    return True
                else:
                    # Discard non-state events (push_ack, sync_response from prior operations)
                    logger.debug(
                        f"Discarding non-state event: {event_type}",
                        extra={
                            "episode_id": self._episode_id,
                            "event_type": event_type,
                        },
                    )
                    # Continue to next iteration

            # Exhausted iterations
            logger.warning(
                "Exhausted iterations waiting for state event",
                extra={
                    "episode_id": self._episode_id,
                    "max_iterations": WebSocketDefaults.MAX_EVENT_DISCARD_ITERATIONS,
                },
            )
            return False

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout ({self._ws_config.pull.event_timeout}s) waiting for modification event",
                extra={"episode_id": self._episode_id},
            )
            return False

    async def _check_for_stuck_state(self) -> bool:
        """Check event queue for stuck_state errors without blocking.

        Scans the event queue for transcript_error events with stuck_state.
        This allows proactive detection of stuck episodes.

        Returns:
            True if stuck_state error detected, False otherwise
        """
        # Non-blocking check of event queue
        try:
            # Peek at events in queue without blocking
            events_to_requeue: list[WebSocketMessageOrDict] = []
            stuck_detected = False

            while not self._event_queue.empty():
                try:
                    event: WebSocketMessageOrDict = self._event_queue.get_nowait()
                    events_to_requeue.append(event)

                    # Check for transcript_error with stuck_state using Pydantic model attributes
                    if isinstance(event, TranscriptErrorMessage):
                        if event.data.error.value == "stuck_state":
                            stuck_detected = True
                except asyncio.QueueEmpty:
                    break

            # Re-queue all events
            for event in events_to_requeue:
                await self._event_queue.put(event)

            return stuck_detected

        except Exception as e:
            logger.warning(
                "Error checking for stuck state",
                extra={
                    "episode_id": self._episode_id,
                    "error": str(e),
                },
            )
            return False

    async def _wait_for_modification_event_with_retry(self, max_retries: int = 3) -> bool:
        """Wait for modification event with stuck_state retry logic.

        Implements exponential backoff retry when stuck_state errors are detected.

        Args:
            max_retries: Maximum number of retry attempts (default 3)

        Returns:
            True if event received, False on timeout or max retries exceeded
        """
        retry_delays = [5.0, 10.0, 20.0]  # Exponential backoff in seconds

        for attempt in range(max_retries + 1):
            # Check for stuck state before waiting
            if attempt > 0:
                stuck = await self._check_for_stuck_state()
                if stuck:
                    logger.warning(
                        "Stuck state detected, retrying after delay",
                        extra={
                            "episode_id": self._episode_id,
                            "attempt": attempt,
                            "max_retries": max_retries,
                        },
                    )
                    # Wait before retry with exponential backoff
                    delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                    await asyncio.sleep(delay)

            # Try to get modification event
            event_received = await self._wait_for_modification_event()

            if event_received:
                return True

            # On timeout, check if we should retry
            if attempt < max_retries:
                logger.info(
                    "Modification event timeout, retrying",
                    extra={
                        "episode_id": self._episode_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                    },
                )

        logger.error(
            "Max retries exceeded waiting for modification event",
            extra={
                "episode_id": self._episode_id,
                "max_retries": max_retries,
            },
        )
        return False

    async def _push_message_with_retry(
        self,
        msg: ChatMessage,
        context: str = "message_push",
    ) -> bool:
        """Push a message to the server with unlimited retry logic.

        Retries the push with exponential backoff (capped at 60s), reconnecting
        WebSocket if needed. Push must succeed for transcript consistency - there
        is no retry limit.

        Args:
            msg: The ChatMessage to push
            context: Context string for logging (e.g., "tool_result_push", "output_push")

        Returns:
            True when push succeeds (always succeeds eventually or raises)

        Note:
            On success, updates _local_version, _local_checksum, and _local_messages.
        """
        backoff = 1.0  # Initial backoff in seconds
        max_backoff = 60.0  # Cap backoff at 60 seconds
        attempt = 0

        while True:
            attempt += 1
            try:
                # Ensure WebSocket is connected (will reconnect if closed)
                await self._ensure_connected()

                if not self._websocket or _is_websocket_closed(self._websocket):
                    raise ConnectionError("WebSocket not connected after _ensure_connected")

                # Send push message
                await self._websocket.send(
                    json.dumps(
                        {
                            "type": WebSocketMessageType.PUSH_MESSAGE.value,
                            "data": {
                                "message": self._serialize_message(msg),
                                "since_version": self._local_version,
                                "client_checksum": self._local_checksum,
                            },
                            "id": str(uuid.uuid4()),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                )

                # Wait for push acknowledgment
                ack_response = await self._wait_for_message_type(
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

            # Force reconnect before retry by closing current connection
            if self._websocket and not _is_websocket_closed(self._websocket):
                try:
                    await self._websocket.close()
                except Exception:
                    pass
            self._websocket = None

    async def _push_tool_results_if_needed(self, input: list[ChatMessage]) -> bool:
        """Push tool results to server if Inspect AI added them to input.

        When Inspect AI runs tools, it adds tool result messages to the input
        list before calling generate() again. These tool results exist only
        on the client - the server doesn't know about them yet. We need to
        push them to keep the server transcript in sync.

        Args:
            input: The input message list from Inspect AI

        Returns:
            True if tool results were pushed, False otherwise
        """
        # Don't push if we haven't synced with server yet
        # On the first generate() call, _local_messages is empty but input has
        # the initial messages - these come from the server via sync, not local additions
        if not self._local_messages:
            return False

        # Find messages in input that aren't in our local transcript
        # These are tool results added by Inspect AI's tool execution loop
        local_len = len(self._local_messages)
        input_len = len(input)

        if input_len <= local_len:
            # No new messages to push
            return False

        # Get the new messages (tool results)
        new_messages = input[local_len:]

        # Check that these are tool messages (sanity check)
        tool_messages = [m for m in new_messages if m.role == "tool"]
        if len(tool_messages) != len(new_messages):
            logger.warning(
                "New messages in input include non-tool messages, unexpected",
                extra={
                    "episode_id": self._episode_id,
                    "new_message_roles": [m.role for m in new_messages],
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
        # Track if ALL pushes succeeded - if any fail, transcript is in inconsistent state
        all_succeeded = True
        for msg in new_messages:
            success = await self._push_message_with_retry(msg, context="tool_result_push")
            if not success:
                logger.error(
                    "Failed to push tool result to server after retries - transcript is now inconsistent",
                    extra={
                        "episode_id": self._episode_id,
                        "message_role": msg.role,
                    },
                )
                all_succeeded = False
                # Continue trying to push remaining messages to minimize damage

        # Return True only if ALL tool results were pushed successfully
        # If any failed, return False so caller knows not to trust sync from server
        return all_succeeded

    async def _wait_for_message_type(
        self,
        expected_type: WebSocketMessageType,
        timeout: float,
        max_iterations: int = WebSocketDefaults.MAX_EVENT_DISCARD_ITERATIONS,
        context: str = "",
    ) -> WebSocketMessageOrDict | None:
        """Wait for a specific WebSocket message type, re-queuing state events.

        Helper method to reduce code duplication when waiting for specific
        message types like push_ack or sync_response.

        IMPORTANT: State events (is_waiting_on_*, transcript_modified) are re-queued
        rather than discarded, since they may be needed by _wait_for_modification_event()
        which runs after this method returns. This prevents race conditions where
        injections arrive while we're waiting for push_ack.

        Args:
            expected_type: The WebSocketMessageType to wait for
            timeout: Timeout in seconds for each queue get
            max_iterations: Maximum iterations to discard non-matching events
            context: Context string for logging (e.g., "tool_result_push")

        Returns:
            The matching message, or None if max_iterations exceeded or timeout
        """
        # State event types that should be preserved (re-queued) instead of discarded
        state_event_types = {
            WebSocketMessageType.IS_WAITING_ON_USER.value,
            WebSocketMessageType.IS_WAITING_ON_ASSISTANT.value,
            WebSocketMessageType.IS_WAITING_ON_TOOLS.value,
            WebSocketMessageType.TRANSCRIPT_MODIFIED.value,
            WebSocketMessageType.TRANSCRIPT_ERROR.value,
        }

        for iteration in range(max_iterations):
            try:
                response = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=timeout,
                )

                # Handle both Pydantic models and dicts (for test compatibility)
                response_type = response.type if hasattr(response, "type") else response.get("type")

                if response_type == expected_type.value:
                    return response
                elif response_type in state_event_types:
                    # Re-queue state events - they may be needed by _wait_for_modification_event()
                    # This prevents race conditions where injections arrive during push_ack wait
                    try:
                        self._event_queue.put_nowait(response)
                        logger.debug(
                            f"Re-queued state event while waiting for {expected_type.value}",
                            extra={
                                "episode_id": self._episode_id,
                                "response_type": response_type,
                                "expected_type": expected_type.value,
                                "iteration": iteration,
                                "context": context,
                            },
                        )
                    except asyncio.QueueFull:
                        logger.warning(
                            "Event queue full, dropping state event",
                            extra={
                                "episode_id": self._episode_id,
                                "response_type": response_type,
                            },
                        )
                else:
                    # Non-state, non-matching event - safe to discard
                    logger.debug(
                        f"Discarding non-matching event while waiting for {expected_type.value}",
                        extra={
                            "episode_id": self._episode_id,
                            "response_type": response_type,
                            "expected_type": expected_type.value,
                            "iteration": iteration,
                            "context": context,
                        },
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout waiting for {expected_type.value}",
                    extra={
                        "episode_id": self._episode_id,
                        "timeout": timeout,
                        "context": context,
                    },
                )
                return None

        logger.warning(
            f"Max iterations exceeded waiting for {expected_type.value}",
            extra={
                "episode_id": self._episode_id,
                "max_iterations": max_iterations,
                "context": context,
            },
        )
        return None

    async def wait_and_sync_transcript(self) -> list[ChatMessage]:
        """Wait for server transcript modification and sync, returning updated messages.

        This method is used by the AgentContinue callback to get server-injected
        messages (e.g., continue prompts, red team injections) without the client
        adding its own continue messages.

        The flow:
        1. Wait for transcript_modified event from server
        2. Send sync_request to get updated transcript
        3. Apply sync response (delta or full)
        4. Return the updated local messages

        Returns:
            List of ChatMessage with server-provided transcript

        Raises:
            RuntimeError: If sync fails or times out
        """
        # Ensure WebSocket is connected
        await self._ensure_connected()

        # Wait for modification event
        event_received = await self._wait_for_modification_event_with_retry()

        if not event_received:
            logger.warning(
                "No modification event received, returning current messages",
                extra={
                    "episode_id": self._episode_id,
                    "local_message_count": len(self._local_messages),
                },
            )
            return self._local_messages.copy()

        logger.debug(
            "Event received, sending sync_request",
            extra={"episode_id": self._episode_id, "local_version": self._local_version},
        )

        # Request sync via WebSocket
        if not self._websocket:
            raise RuntimeError("WebSocket not connected")

        await self._websocket.send(
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
                self._event_queue.get(),
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
            raise RuntimeError(
                f"Failed to get sync_response after {WebSocketDefaults.MAX_SYNC_RESPONSE_ITERATIONS} iterations"
            )

        if response_type == WebSocketMessageType.SYNC_RESPONSE.value:
            if isinstance(response, SyncResponseMessage):
                sync_data = response.data
            elif isinstance(response, dict):
                sync_data = SyncResponseData(**response["data"])
            else:
                raise TypeError(f"Unexpected response type: {type(response)}")

            # Apply sync
            if sync_data.sync_mode == SyncMode.FULL:
                if not sync_data.full_transcript:
                    raise ValueError("Missing full_transcript in FULL sync mode")
                self._local_messages = [self._deserialize_message(m) for m in sync_data.full_transcript]
            elif sync_data.sync_mode == SyncMode.DELTA:
                if sync_data.delta:
                    new_messages = [self._deserialize_message(m) for m in sync_data.delta]
                    self._local_messages.extend(new_messages)
            # NO_CHANGE: messages already up to date

            self._local_version = sync_data.current_version.sequence
            self._local_checksum = sync_data.current_version.checksum

            logger.info(
                "Transcript synced for AgentContinue callback",
                extra={
                    "episode_id": self._episode_id,
                    "sync_mode": sync_data.sync_mode.value,
                    "message_count": len(self._local_messages),
                },
            )

            # Signal generate() to skip waiting - we just synced the transcript
            self._skip_next_wait = True
        elif response_type == WebSocketMessageType.TRANSCRIPT_ERROR.value:
            if isinstance(response, TranscriptErrorMessage):
                error_msg = response.data.message or "Unknown error"
            elif isinstance(response, dict):
                error_msg = response.get("data", {}).get("message", "Unknown error")
            else:
                error_msg = "Unknown error"
            raise RuntimeError(f"Transcript sync error: {error_msg}")

        return self._local_messages.copy()

    @property
    def pull_enabled(self) -> bool:
        """Check if pull (server-controlled transcript) is enabled."""
        return self._ws_config.pull.enabled

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
        # These are messages that exist in input but not yet on the server
        pushed_tool_results = False
        if self._ws_config.push.enabled and isinstance(input, list):
            pushed_tool_results = await self._push_tool_results_if_needed(input)

        # Wait for state event, then pull (unified flow for all calls)
        if self._ws_config.pull.enabled:
            # Only pull if enabled (blue team waits for modifications, red team skips)
            # Block and wait for WebSocket notification with stuck_state retry
            if isinstance(input, list):
                # If wait_and_sync_transcript() just synced, skip waiting - transcript is current
                if self._skip_next_wait:
                    event_received = True
                    self._skip_next_wait = False  # Reset flag
                    logger.debug(
                        "Skipping wait - wait_and_sync_transcript already synced",
                        extra={"episode_id": self._episode_id},
                    )
                # If we just pushed tool results, skip waiting for modification event
                # We already know the transcript was modified (by us)
                elif pushed_tool_results:
                    event_received = True
                    logger.debug(
                        "Skipping wait - just pushed tool results",
                        extra={"episode_id": self._episode_id},
                    )
                else:
                    event_received = await self._wait_for_modification_event_with_retry()

                if event_received:
                    if self._websocket:
                        await self._websocket.send(
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

                    # Wait for sync response from queue (listener task queues all messages)
                    # Loop until we get the actual sync_response, discarding other events
                    response = None
                    response_type = None
                    events_discarded = 0
                    for iteration in range(WebSocketDefaults.MAX_SYNC_RESPONSE_ITERATIONS):
                        response = await asyncio.wait_for(
                            self._event_queue.get(),
                            timeout=self._ws_config.pull.sync_timeout,
                        )

                        # Handle both Pydantic models (from listener) and dicts (from tests)
                        # Check for Pydantic model first using hasattr, then fall back to dict access
                        response_type = None
                        if hasattr(response, "type"):
                            response_type = response.type
                        elif isinstance(response, dict):
                            response_type = response.get("type")

                        # Check if this is the sync_response we're waiting for
                        if response_type == WebSocketMessageType.SYNC_RESPONSE.value:
                            break

                        # Check for error response
                        if response_type == WebSocketMessageType.TRANSCRIPT_ERROR.value:
                            break

                        # Other events (state events, transcript_modified, push_ack, etc.) - discard
                        # These are either already processed or not relevant to sync
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
                        # Don't re-queue - just discard and get next event
                    else:
                        # Exhausted iterations without finding sync_response
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

                    if response_type == WebSocketMessageType.SYNC_RESPONSE.value:
                        # Get sync data - either from Pydantic model or parse from dict
                        if isinstance(response, SyncResponseMessage):
                            sync_data = response.data
                        elif isinstance(response, dict):
                            sync_data = SyncResponseData(**response["data"])
                        else:
                            raise TypeError(f"Unexpected response type: {type(response)}")

                        # Apply delta or full transcript
                        if sync_data.sync_mode == SyncMode.FULL:
                            # Rewrite detected - replace entire transcript
                            if not sync_data.full_transcript:
                                logger.warning("Full sync mode but no full_transcript provided")
                                raise ValueError("Missing full_transcript in FULL sync mode")

                            self._local_messages = [self._deserialize_message(m) for m in sync_data.full_transcript]
                        elif sync_data.sync_mode == SyncMode.DELTA:
                            # Delta sync - append new messages
                            if sync_data.delta:
                                new_messages = [self._deserialize_message(m) for m in sync_data.delta]
                                self._local_messages.extend(new_messages)
                        elif sync_data.sync_mode == SyncMode.NO_CHANGE:
                            # No changes since requested version - transcript is already up to date
                            logger.debug(
                                "Transcript sync: no changes since requested version",
                                extra={
                                    "episode_id": self._episode_id,
                                    "version": self._local_version,
                                },
                            )

                        # Update local state
                        self._local_version = sync_data.current_version.sequence
                        self._local_checksum = sync_data.current_version.checksum
                        input = self._local_messages.copy()

                        logger.debug(
                            "Applied transcript sync via WebSocket",
                            extra={
                                "episode_id": self._episode_id,
                                "sync_mode": sync_data.sync_mode.value,
                                "new_version": self._local_version,
                            },
                        )
                    elif response_type == WebSocketMessageType.TRANSCRIPT_ERROR.value:
                        # Server reported an error - log and raise
                        if isinstance(response, TranscriptErrorMessage):
                            error_msg = response.data.message or "Unknown error"
                        elif isinstance(response, dict):
                            error_msg = response.get("data", {}).get("message", "Unknown error")
                        else:
                            error_msg = "Unknown error"
                        logger.error(
                            "Server returned transcript error",
                            extra={
                                "error": error_msg,
                                "episode_id": self._episode_id,
                            },
                        )
                        raise RuntimeError(f"Transcript sync error from server: {error_msg}")
                    else:
                        # Unexpected response type
                        logger.error(
                            "Unexpected response type from sync request",
                            extra={
                                "response_type": response_type,
                                "expected": WebSocketMessageType.SYNC_RESPONSE.value,
                                "episode_id": self._episode_id,
                            },
                        )
                        raise RuntimeError(f"Unexpected sync response type: {response_type}")
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
            success = await self._push_message_with_retry(output.message, context="output_push")
            if not success:
                # Log error but don't crash - agent can continue, but server is behind
                logger.error(
                    "Failed to push assistant output to server after retries - transcript may be out of sync",
                    extra={
                        "episode_id": self._episode_id,
                        "message_role": output.message.role,
                    },
                )
        else:
            # Push disabled - update local state only (no WebSocket communication)
            logger.debug(
                "Push disabled, updating local state only",
                extra={"episode_id": self._episode_id},
            )
            self._local_messages.append(output.message)
            self._local_version += 1
            self._local_checksum = compute_checksum([self._serialize_message(m) for m in self._local_messages])

        return output

    async def cleanup(self) -> None:
        """Close WebSocket connection and release resources when episode ends."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        # Drain event queue to release message references
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Clear local state to allow garbage collection
        self._local_messages.clear()
        self._local_version = 0
        self._local_checksum = compute_checksum([])

        if self._websocket and not _is_websocket_closed(self._websocket):
            await self._websocket.close()
            logger.info("WebSocket connection closed", extra={"episode_id": self._episode_id})

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


__all__ = ["WebSocketTranscriptSyncingModelWrapper"]
