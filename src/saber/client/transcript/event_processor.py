"""WebSocket event processing for transcript synchronization.

This module provides event listening and processing capabilities:
- Background listener task for WebSocket events
- Dual-queue event management (ack queue + state queue)
- State event waiting
- Stuck state detection

Logging category: AGENT.
"""

import asyncio
from typing import TYPE_CHECKING, Any

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
WebSocketMessageOrDict = WebSocketServerMessage | dict[str, Any]

# Message types routed to the ack queue (responses to client requests)
_ACK_EVENT_TYPES = {
    WebSocketMessageType.PUSH_ACK.value,
    WebSocketMessageType.PONG.value,
}

# Message types routed to the state queue (server-initiated events)
_STATE_EVENT_TYPES = {
    WebSocketMessageType.IS_WAITING_ON_USER.value,
    WebSocketMessageType.IS_WAITING_ON_ASSISTANT.value,
    WebSocketMessageType.IS_WAITING_ON_TOOLS.value,
    WebSocketMessageType.TRANSCRIPT_MODIFIED.value,
    WebSocketMessageType.TRANSCRIPT_ERROR.value,
}


class WebSocketEventProcessor:
    """Processes WebSocket events from the server.

    Uses two dedicated queues to separate concerns:
    - ``_ack_queue``: push_ack and pong responses (consumed by wait_for_message_type)
    - ``_state_queue``: state events and transcript notifications (consumed by wait_for_state_event)

    This eliminates cross-contamination between ack waits and state waits.

    Thread-safety: asyncio.Queue is safe for single-loop concurrent access.
    """

    def __init__(
        self,
        episode_id: str,
        ws_config: WebSocketConfig | None = None,
    ) -> None:
        """Initialize the event processor.

        Args:
            episode_id: SABER episode ID for logging
            ws_config: WebSocket configuration (uses defaults if None)
        """
        self._episode_id = episode_id
        self._ws_config = ws_config or WebSocketConfig()

        max_size = self._ws_config.pull.event_queue_max_size

        # Ack queue — push_ack and pong (responses to client requests)
        self._ack_queue: asyncio.Queue[WebSocketMessageOrDict] = asyncio.Queue(maxsize=max_size)

        # State queue — is_waiting_on_*, transcript_modified, transcript_error
        self._state_queue: asyncio.Queue[WebSocketMessageOrDict] = asyncio.Queue(maxsize=max_size)

        # Listener task - tracks background WebSocket listener
        self._listener_task: asyncio.Task[None] | None = None

        # Shutdown event - set during cleanup to unblock wait_for_injection_event
        self._shutdown_event: asyncio.Event = asyncio.Event()

        logger.debug(
            "Created WebSocketEventProcessor",
            extra={
                "episode_id": episode_id,
                "event_timeout": self._ws_config.pull.event_timeout,
                "queue_max_size": max_size,
            },
        )

    @property
    def ack_queue(self) -> asyncio.Queue[WebSocketMessageOrDict]:
        """Get the ack queue (push_ack, pong) for direct access (e.g., testing)."""
        return self._ack_queue

    @property
    def state_queue(self) -> asyncio.Queue[WebSocketMessageOrDict]:
        """Get the state queue (state events, transcript notifications) for direct access."""
        return self._state_queue

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

        Routes incoming messages to the appropriate queue:
        - push_ack, pong → ``_ack_queue``
        - is_waiting_on_*, transcript_modified, transcript_error → ``_state_queue``
        - connected → discarded (consumed during handshake)

        Args:
            websocket: WebSocket connection to listen on
        """
        import websockets

        try:
            async for message in websocket:
                try:
                    parsed_message = WebSocketServerMessageAdapter.validate_json(message)
                except ValidationError as e:
                    logger.warning(
                        f"Failed to parse WebSocket message: {e}",
                        extra={"raw_message": message[:200] if isinstance(message, str) else str(message)[:200]},
                    )
                    continue

                event_type = parsed_message.type

                if event_type in _ACK_EVENT_TYPES:
                    await self._ack_queue.put(parsed_message)
                    logger.debug(
                        "Queued ack event",
                        extra={"episode_id": self._episode_id, "type": event_type},
                    )

                elif event_type in _STATE_EVENT_TYPES:
                    await self._state_queue.put(parsed_message)

                    # Extra logging for stuck_state errors
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
                        logger.debug(
                            "Queued state event",
                            extra={"episode_id": self._episode_id, "type": event_type},
                        )

                elif event_type == WebSocketMessageType.CONNECTED.value:
                    logger.debug(
                        "Received 'connected' message in listener (unexpected but harmless)",
                        extra={"episode_id": self._episode_id},
                    )

                else:
                    logger.warning(
                        "Received unknown WebSocket message type",
                        extra={"episode_id": self._episode_id, "type": event_type},
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed", extra={"episode_id": self._episode_id})
        except Exception as e:
            logger.error("WebSocket listener error", extra={"episode_id": self._episode_id, "error": str(e)})

    async def wait_for_state_event(self) -> bool:
        """Wait for a state event from the server.

        Reads from ``_state_queue`` which only contains state events, so no
        discard loop is needed.

        Returns:
            True if state event received, False on timeout
        """
        try:
            event_data: WebSocketMessageOrDict = await asyncio.wait_for(
                self._state_queue.get(), timeout=self._ws_config.pull.event_timeout
            )

            event_type = event_data.type if hasattr(event_data, "type") else None
            version = None
            state = None
            if isinstance(event_data, StateEventMessage):
                version = event_data.data.version
                state = event_data.data.state

            logger.debug(
                "Received state event",
                extra={
                    "episode_id": self._episode_id,
                    "event_type": event_type,
                    "version": version,
                    "state": state,
                },
            )
            return True

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout ({self._ws_config.pull.event_timeout}s) waiting for state event",
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
        logger.debug(
            "wait_for_injection_event() STARTING",
            extra={
                "episode_id": self._episode_id,
                "queue_size": self._state_queue.qsize(),
                "shutdown_set": self._shutdown_event.is_set(),
            },
        )
        iteration = 0
        while True:
            iteration += 1
            # Check if shutdown was signaled
            if self._shutdown_event.is_set():
                logger.info(
                    "Shutdown signaled, raising CancelledError",
                    extra={"episode_id": self._episode_id, "iteration": iteration},
                )
                raise asyncio.CancelledError("Event processor shutdown")

            # Wait for event from state queue with short timeout to check shutdown periodically
            try:
                event_data: WebSocketMessageOrDict = await asyncio.wait_for(self._state_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Check shutdown and continue waiting
                logger.debug(
                    f"Loop iteration {iteration}: timeout, no event yet, continuing",
                    extra={"episode_id": self._episode_id, "iteration": iteration},
                )
                continue

            # Check event type
            event_type = event_data.type if hasattr(event_data, "type") else None

            logger.debug(
                f"Loop iteration {iteration}: EVENT RECEIVED from queue",
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
                logger.debug(
                    f"Loop iteration {iteration}: INJECTION DETECTED! Returning.",
                    extra={"episode_id": self._episode_id, "iteration": iteration},
                )
                return  # Injection received, done waiting

            # is_waiting_on_user means we (blue) just pushed assistant message
            # - ignore this and keep waiting for injection
            if event_type == WebSocketMessageType.IS_WAITING_ON_USER.value:
                logger.debug(
                    f"Loop iteration {iteration}: Ignoring is_waiting_on_user (our own push)",
                    extra={"episode_id": self._episode_id, "iteration": iteration},
                )
                continue

            # Other events - log and continue waiting
            logger.debug(
                f"Loop iteration {iteration}: Ignoring event type: {event_type}",
                extra={"episode_id": self._episode_id, "iteration": iteration},
            )

    def signal_shutdown(self) -> None:
        """Signal shutdown to unblock wait_for_injection_event.

        Called during cleanup to allow agent loop to exit gracefully.
        """
        self._shutdown_event.set()
        logger.debug(
            "shutdown event SET - wait_for_injection_event should exit on next iteration",
            extra={"episode_id": self._episode_id},
        )

    async def cancel_listener(self) -> None:
        """Cancel the background listener task if running.

        Awaits the task to ensure clean cancellation before returning.
        """
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

    async def check_for_stuck_state(self) -> bool:
        """Check state queue for stuck_state errors without blocking.

        Scans the state queue for transcript_error events with stuck_state.
        This allows proactive detection of stuck episodes.

        Returns:
            True if stuck_state error detected, False otherwise
        """
        try:
            events_to_requeue: list[WebSocketMessageOrDict] = []
            stuck_detected = False

            while not self._state_queue.empty():
                try:
                    event: WebSocketMessageOrDict = self._state_queue.get_nowait()
                    events_to_requeue.append(event)

                    if isinstance(event, TranscriptErrorMessage):
                        if event.data.error.value == "stuck_state":
                            stuck_detected = True
                except asyncio.QueueEmpty:
                    break

            # Re-queue all events
            for event in events_to_requeue:
                await self._state_queue.put(event)

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
        max_backoff = 60.0  # Cap backoff at 60 seconds
        attempt = 0

        while True:
            logger.debug(
                f"wait_for_state_event_with_retry attempt {attempt}",
                extra={
                    "episode_id": self._episode_id,
                    "attempt": attempt,
                    "queue_size": self._state_queue.qsize(),
                },
            )
            # Check for stuck state before waiting (after first attempt)
            if attempt > 0:
                stuck = await self.check_for_stuck_state()
                if stuck:
                    # Exponential backoff: 5, 10, 20, 40, 60, 60, 60...
                    delay = min(5.0 * (2 ** (attempt - 1)), max_backoff)
                    logger.warning(
                        "Stuck state detected, retrying after delay",
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
                logger.debug(
                    "wait_for_state_event_with_retry returning True",
                    extra={"episode_id": self._episode_id, "attempt": attempt},
                )
                return True

            # On timeout, log and retry (sample is useless without sync)
            attempt += 1
            delay = min(5.0 * (2 ** (attempt - 1)), max_backoff)
            logger.warning(
                "Modification event timeout, retrying (sample invalid without sync)",
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
    ) -> WebSocketMessageOrDict | None:
        """Wait for a specific message type from the ack queue.

        Reads from ``_ack_queue`` which only contains push_ack and pong messages,
        so no state-event buffering is needed. Only non-matching messages
        count against ``max_iterations``.

        Args:
            expected_type: The WebSocketMessageType to wait for
            timeout: Timeout in seconds for each queue get
            max_iterations: Maximum iterations to discard non-matching events
            context: Context string for logging (e.g., "tool_result_push")

        Returns:
            The matching message, or None if max_iterations exceeded or timeout
        """
        discarded_count = 0

        while discarded_count < max_iterations:
            try:
                response = await asyncio.wait_for(
                    self._ack_queue.get(),
                    timeout=timeout,
                )

                # Handle both Pydantic models and dicts (for test compatibility)
                response_type = response.type if hasattr(response, "type") else response.get("type")

                if response_type == expected_type.value:
                    return response

                discarded_count += 1
                logger.debug(
                    f"Discarding event while waiting for {expected_type.value}",
                    extra={
                        "episode_id": self._episode_id,
                        "response_type": response_type,
                        "expected_type": expected_type.value,
                        "discarded_count": discarded_count,
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
        """Drain both queues to release message references."""
        for queue in (self._ack_queue, self._state_queue):
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break


__all__ = ["WebSocketEventProcessor", "WebSocketMessageOrDict"]
