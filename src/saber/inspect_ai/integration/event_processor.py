"""WebSocket event processing for transcript synchronization.

This module provides event listening and processing capabilities:
- Background listener task for WebSocket events
- Event queue management
- State event waiting with filtering
- Stuck state detection

Logging category: AGENT.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from websockets import ClientConnection

from pydantic import ValidationError

from ...logging_config import LogCategory, get_saber_logger
from ...models.rest.websocket_config import WebSocketConfig
from ...models.rest.websocket_constants import WebSocketDefaults
from ...models.rest.websocket_messages import (
    StateEventMessage,
    TranscriptErrorMessage,
    WebSocketMessageType,
    WebSocketServerMessage,
    WebSocketServerMessageAdapter,
)

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Type alias for messages that can be either Pydantic models or dicts (for test compatibility)
WebSocketMessageOrDict = Union[WebSocketServerMessage, Dict[str, Any]]


class WebSocketEventProcessor:
    """Processes WebSocket events from the server.

    Handles:
    - Background listener task that queues events
    - Waiting for specific event types
    - State event filtering
    - Stuck state detection

    Thread-safety: Event queue is asyncio.Queue which is thread-safe.
    """

    def __init__(
        self,
        episode_id: str,
        ws_config: Optional[WebSocketConfig] = None,
    ) -> None:
        """Initialize the event processor.

        Args:
            episode_id: SABER episode ID for logging
            ws_config: WebSocket configuration (uses defaults if None)
        """
        self._episode_id = episode_id
        self._ws_config = ws_config or WebSocketConfig()

        # Event queue - populated by listener, consumed by sync operations
        self._event_queue: asyncio.Queue[WebSocketMessageOrDict] = asyncio.Queue(
            maxsize=self._ws_config.pull.event_queue_max_size
        )

        # Listener task - tracks background WebSocket listener
        self._listener_task: Optional[asyncio.Task[None]] = None

        # Shutdown event - set during cleanup to unblock wait_for_injection_event
        self._shutdown_event: asyncio.Event = asyncio.Event()

        logger.debug(
            "Created WebSocketEventProcessor",
            extra={
                "episode_id": episode_id,
                "event_timeout": self._ws_config.pull.event_timeout,
                "queue_max_size": self._ws_config.pull.event_queue_max_size,
            },
        )

    @property
    def event_queue(self) -> asyncio.Queue[WebSocketMessageOrDict]:
        """Get the event queue for direct access (e.g., testing)."""
        return self._event_queue

    @property
    def config(self) -> WebSocketConfig:
        """Get the WebSocket configuration."""
        return self._ws_config

    async def start_listener(self, websocket: "ClientConnection") -> asyncio.Task[None]:
        """Start the background listener task.

        Args:
            websocket: WebSocket connection to listen on

        Returns:
            The background task (can be cancelled to stop listening)
        """
        task = asyncio.create_task(self._listen_for_events(websocket))
        self._listener_task = task  # Store for cleanup/backward compat
        logger.debug(
            "Started WebSocket event listener",
            extra={"episode_id": self._episode_id},
        )
        return task

    async def _listen_for_events(self, websocket: "ClientConnection") -> None:
        """Background task that listens for WebSocket events.

        Receives transcript_modified events and queues them for processing.

        Handles both push (server events) and pull (response to client requests).
        Server-initiated events are queued for generate() to consume.
        Client-response messages are consumed directly by awaiting code.

        Args:
            websocket: WebSocket connection to listen on
        """
        import websockets

        try:
            async for message in websocket:
                # Parse JSON directly into Pydantic model using discriminated union
                try:
                    parsed_message = WebSocketServerMessageAdapter.validate_json(message)
                except ValidationError as e:
                    logger.warning(
                        f"Failed to parse WebSocket message: {e}",
                        extra={"raw_message": message[:200] if isinstance(message, str) else str(message)[:200]},
                    )
                    continue

                event_type = parsed_message.type

                logger.info(
                    "[RESTART_DEBUG] WebSocket listener received message",
                    extra={
                        "episode_id": self._episode_id,
                        "event_type": event_type,
                        "queue_size_before": self._event_queue.qsize(),
                    },
                )

                if event_type == WebSocketMessageType.TRANSCRIPT_MODIFIED.value:
                    # Server-initiated notification - queue for generate()
                    await self._event_queue.put(parsed_message)

                    logger.info(
                        "[RESTART_DEBUG] Received transcript modification event - QUEUED",
                        extra={
                            "episode_id": self._episode_id,
                            "version": (
                                parsed_message.data.version if isinstance(parsed_message, StateEventMessage) else None
                            ),
                            "queue_size_after": self._event_queue.qsize(),
                        },
                    )

                elif event_type in (
                    WebSocketMessageType.IS_WAITING_ON_USER.value,
                    WebSocketMessageType.IS_WAITING_ON_ASSISTANT.value,
                    WebSocketMessageType.IS_WAITING_ON_TOOLS.value,
                ):
                    # State machine events - log and queue
                    await self._event_queue.put(parsed_message)

                    logger.info(
                        "[RESTART_DEBUG] Received state machine event - QUEUED",
                        extra={
                            "episode_id": self._episode_id,
                            "event_type": event_type,
                            "state": (
                                parsed_message.data.state if isinstance(parsed_message, StateEventMessage) else None
                            ),
                            "queue_size_after": self._event_queue.qsize(),
                        },
                    )

                elif event_type == WebSocketMessageType.TRANSCRIPT_ERROR.value:
                    # Error events (stuck_state, etc.) - queue for handling
                    await self._event_queue.put(parsed_message)

                    if isinstance(parsed_message, TranscriptErrorMessage):
                        error_type = parsed_message.data.error
                        if error_type.value == "stuck_state":
                            logger.error(
                                "[RESTART_DEBUG] Episode stuck in state - retry required",
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
                    # Connection handshake message - should be consumed during connection
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

    async def wait_for_state_event(self) -> bool:
        """Wait for WebSocket event indicating transcript modification.

        Only accepts state events (is_waiting_on_*, transcript_modified).
        Other events (push_ack, sync_response) are discarded as they're
        stale responses from previous operations.

        Returns:
            True if state event received, False on timeout
        """
        logger.info(
            "[RESTART_DEBUG] wait_for_state_event() called",
            extra={
                "episode_id": self._episode_id,
                "event_timeout": self._ws_config.pull.event_timeout,
                "queue_size": self._event_queue.qsize(),
            },
        )
        try:
            # Loop until we get a state event, discarding other events
            for iteration in range(WebSocketDefaults.MAX_EVENT_DISCARD_ITERATIONS):
                logger.info(
                    f"[RESTART_DEBUG] Waiting for event from queue (iteration {iteration + 1})",
                    extra={
                        "episode_id": self._episode_id,
                        "timeout": self._ws_config.pull.event_timeout,
                    },
                )
                # Wait for event from queue (populated by background listener)
                event_data: WebSocketMessageOrDict = await asyncio.wait_for(
                    self._event_queue.get(), timeout=self._ws_config.pull.event_timeout
                )

                # Check event type
                event_type = event_data.type if hasattr(event_data, "type") else None

                logger.info(
                    "[RESTART_DEBUG] Event received from queue",
                    extra={
                        "episode_id": self._episode_id,
                        "event_type": event_type,
                        "event_class": type(event_data).__name__,
                    },
                )

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

                    logger.info(
                        "[RESTART_DEBUG] Received state event - returning True",
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
                    logger.info(
                        f"[RESTART_DEBUG] Discarding non-state event: {event_type}",
                        extra={
                            "episode_id": self._episode_id,
                            "event_type": event_type,
                        },
                    )
                    # Continue to next iteration

            # Exhausted iterations
            logger.warning(
                "[RESTART_DEBUG] Exhausted iterations waiting for state event",
                extra={
                    "episode_id": self._episode_id,
                    "max_iterations": WebSocketDefaults.MAX_EVENT_DISCARD_ITERATIONS,
                },
            )
            return False

        except asyncio.TimeoutError:
            logger.warning(
                f"[RESTART_DEBUG] Timeout ({self._ws_config.pull.event_timeout}s) waiting for modification event",
                extra={"episode_id": self._episode_id},
            )
            return False

    async def wait_for_injection_event(self) -> None:
        """Wait for is_waiting_on_assistant event (new user message injected).

        This is used by the on_continue callback to wait for red team injection.
        It ignores is_waiting_on_user events (from our own push) and only returns
        when a user message has been injected (is_waiting_on_assistant).

        Waits until injection or shutdown signal.

        Raises:
            asyncio.CancelledError: If shutdown is signaled during wait
        """
        logger.info(
            "[INJECTION_WAIT] wait_for_injection_event() STARTING",
            extra={
                "episode_id": self._episode_id,
                "queue_size": self._event_queue.qsize(),
                "shutdown_set": self._shutdown_event.is_set(),
            },
        )
        iteration = 0
        while True:
            iteration += 1
            # Check if shutdown was signaled
            if self._shutdown_event.is_set():
                logger.info(
                    "[INJECTION_WAIT] Shutdown signaled, raising CancelledError",
                    extra={"episode_id": self._episode_id, "iteration": iteration},
                )
                raise asyncio.CancelledError("Event processor shutdown")

            logger.info(
                f"[INJECTION_WAIT] Loop iteration {iteration}: waiting for event from queue (timeout=1s)",
                extra={
                    "episode_id": self._episode_id,
                    "iteration": iteration,
                    "queue_size": self._event_queue.qsize(),
                },
            )

            # Wait for event from queue with short timeout to check shutdown periodically
            try:
                event_data: WebSocketMessageOrDict = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Check shutdown and continue waiting
                logger.info(
                    f"[INJECTION_WAIT] Loop iteration {iteration}: timeout, no event yet, continuing",
                    extra={"episode_id": self._episode_id, "iteration": iteration},
                )
                continue

            # Check event type
            event_type = event_data.type if hasattr(event_data, "type") else None

            logger.info(
                f"[INJECTION_WAIT] Loop iteration {iteration}: EVENT RECEIVED from queue",
                extra={
                    "episode_id": self._episode_id,
                    "iteration": iteration,
                    "event_type": event_type,
                    "event_class": type(event_data).__name__,
                    "event_data": str(event_data)[:200],
                },
            )

            # We're specifically waiting for is_waiting_on_assistant
            # which indicates a user message was injected
            if event_type == WebSocketMessageType.IS_WAITING_ON_ASSISTANT.value:
                logger.info(
                    f"[INJECTION_WAIT] Loop iteration {iteration}: INJECTION DETECTED! Returning.",
                    extra={"episode_id": self._episode_id, "iteration": iteration},
                )
                return  # Injection received, done waiting

            # is_waiting_on_user means we (blue) just pushed assistant message
            # - ignore this and keep waiting for injection
            if event_type == WebSocketMessageType.IS_WAITING_ON_USER.value:
                logger.info(
                    f"[INJECTION_WAIT] Loop iteration {iteration}: Ignoring is_waiting_on_user (our own push)",
                    extra={"episode_id": self._episode_id, "iteration": iteration},
                )
                continue

            # Other events - log and continue waiting
            logger.info(
                f"[INJECTION_WAIT] Loop iteration {iteration}: Ignoring event type: {event_type}",
                extra={"episode_id": self._episode_id, "iteration": iteration},
            )

    def signal_shutdown(self) -> None:
        """Signal shutdown to unblock wait_for_injection_event.

        Called during cleanup to allow agent loop to exit gracefully.
        """
        logger.info(
            "[SHUTDOWN] signal_shutdown() CALLED - setting shutdown event",
            extra={
                "episode_id": self._episode_id,
                "was_already_set": self._shutdown_event.is_set(),
            },
        )
        self._shutdown_event.set()
        logger.info(
            "[SHUTDOWN] shutdown event SET - wait_for_injection_event should exit on next iteration",
            extra={"episode_id": self._episode_id},
        )

    async def check_for_stuck_state(self) -> bool:
        """Check event queue for stuck_state errors without blocking.

        Scans the event queue for transcript_error events with stuck_state.
        This allows proactive detection of stuck episodes.

        Returns:
            True if stuck_state error detected, False otherwise
        """
        # Non-blocking check of event queue
        try:
            # Peek at events in queue without blocking
            events_to_requeue: List[WebSocketMessageOrDict] = []
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

    async def wait_for_state_event_with_retry(self) -> bool:
        """Wait for modification event with unlimited retries.

        The sample is invalid without successful sync, so we retry indefinitely
        with exponential backoff (capped at 60s). The evaluation's task timeout
        or manual cancellation are the appropriate mechanisms to stop if truly stuck.

        Returns:
            True when event is received (always succeeds or blocks forever)
        """
        logger.info(
            "[RESTART_DEBUG] wait_for_state_event_with_retry() started",
            extra={"episode_id": self._episode_id},
        )
        max_backoff = 60.0  # Cap backoff at 60 seconds
        attempt = 0

        while True:
            logger.info(
                f"[RESTART_DEBUG] wait_for_state_event_with_retry attempt {attempt}",
                extra={
                    "episode_id": self._episode_id,
                    "attempt": attempt,
                    "queue_size": self._event_queue.qsize(),
                },
            )
            # Check for stuck state before waiting (after first attempt)
            if attempt > 0:
                stuck = await self.check_for_stuck_state()
                if stuck:
                    # Exponential backoff: 5, 10, 20, 40, 60, 60, 60...
                    delay = min(5.0 * (2 ** (attempt - 1)), max_backoff)
                    logger.warning(
                        "[RESTART_DEBUG] Stuck state detected, retrying after delay",
                        extra={
                            "episode_id": self._episode_id,
                            "attempt": attempt,
                            "delay_seconds": delay,
                        },
                    )
                    await asyncio.sleep(delay)

            # Try to get modification event
            event_received = await self.wait_for_state_event()

            if event_received:
                logger.info(
                    "[RESTART_DEBUG] wait_for_state_event_with_retry returning True",
                    extra={"episode_id": self._episode_id, "attempt": attempt},
                )
                return True

            # On timeout, log and retry (sample is useless without sync)
            attempt += 1
            delay = min(5.0 * (2 ** (attempt - 1)), max_backoff)
            logger.warning(
                "[RESTART_DEBUG] Modification event timeout, retrying (sample invalid without sync)",
                extra={
                    "episode_id": self._episode_id,
                    "attempt": attempt,
                    "next_delay_seconds": delay,
                },
            )
            await asyncio.sleep(delay)

    async def wait_for_message_type(
        self,
        expected_type: WebSocketMessageType,
        timeout: float,
        max_iterations: int = WebSocketDefaults.MAX_EVENT_DISCARD_ITERATIONS,
        context: str = "",
    ) -> Optional[WebSocketMessageOrDict]:
        """Wait for a specific WebSocket message type, re-queuing state events.

        IMPORTANT: State events (is_waiting_on_*, transcript_modified) are re-queued
        rather than discarded, since they may be needed by wait_for_state_event()
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
                    # Re-queue state events - they may be needed by wait_for_state_event()
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

    def drain_queue(self) -> None:
        """Drain the event queue to release message references."""
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


__all__ = ["WebSocketEventProcessor", "WebSocketMessageOrDict"]
