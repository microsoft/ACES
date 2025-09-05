"""
Event Bus for SABER Client

Async producer/consumer event bus with fail-fast semantics.
No silent failures - queue overflow and handler errors raise exceptions.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .types import EventType, SaberEvent

logger = logging.getLogger(__name__)


class EventBus:
    """
    Async event bus for decoupling event producers from consumers.

    Features:
    - Producer/consumer pattern with asyncio queues
    - Event type filtering for subscribers
    - Fail-fast semantics - queue overflow raises RuntimeError
    - Handler error propagation via ERROR events
    - Graceful shutdown with event flushing
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize event bus.

        Args:
            max_queue_size: Maximum events in queue before failing fast
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._subscribers: Dict[Optional[EventType], List[Callable]] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._max_queue_size = max_queue_size

        logger.info(f"Event bus initialized with max_queue_size={max_queue_size}")

    def subscribe(self, event_type: Optional[EventType], handler: Callable[[SaberEvent], Any]) -> None:
        """
        Subscribe to events by type.

        Args:
            event_type: Event type to subscribe to, None for all events
            handler: Event handler function (sync or async)

        Raises:
            ValueError: If handler is not callable
        """
        if not callable(handler):
            raise ValueError("handler must be callable")

        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(handler)
        event_filter = event_type.value if event_type else "ALL"
        logger.info(f"Subscribed handler to events: {event_filter}")

    def unsubscribe(self, event_type: Optional[EventType], handler: Callable[[SaberEvent], Any]) -> bool:
        """
        Unsubscribe handler from event type.

        Args:
            event_type: Event type to unsubscribe from
            handler: Handler to remove

        Returns:
            True if handler was found and removed, False otherwise
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                event_filter = event_type.value if event_type else "ALL"
                logger.info(f"Unsubscribed handler from events: {event_filter}")
                return True
            except ValueError:
                pass
        return False

    async def publish(self, event: SaberEvent) -> None:
        """
        Publish event to bus.

        Args:
            event: Event to publish

        Raises:
            ValueError: If event is invalid
            RuntimeError: If queue is full (fail-fast)
        """
        if not isinstance(event, SaberEvent):
            raise ValueError("event must be SaberEvent instance")

        try:
            self._queue.put_nowait(event)
            logger.debug(f"Published event: {event.event_type.value} (correlation_id={event.correlation_id})")
        except asyncio.QueueFull:
            # FAIL FAST: No silent dropping of events
            raise RuntimeError(
                f"Event bus queue full (max: {self._max_queue_size}). " f"Event type: {event.event_type.value}"
            )

    async def start(self) -> None:
        """
        Start event processing loop.

        Raises:
            RuntimeError: If already running
        """
        if self._running:
            raise RuntimeError("Event bus is already running")

        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")

    async def stop(self) -> None:
        """
        Stop event processing and flush remaining events.

        Processes all remaining events before stopping.
        """
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        await self._flush_remaining_events()
        logger.info("Event bus stopped")

    async def _process_events(self) -> None:
        """
        Main event processing loop.

        Runs until stopped, dispatching events to subscribers.
        """
        while self._running:
            try:
                # Wait for events with timeout to check running status
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                # Timeout is normal - just check if we should continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log unexpected errors but continue processing
                logger.error(f"Unexpected error in event processing loop: {e}")

    async def _dispatch_event(self, event: SaberEvent) -> None:
        """
        Dispatch event to all matching subscribers.

        Args:
            event: Event to dispatch
        """
        dispatched_count = 0

        # Dispatch to specific event type subscribers
        for handler in self._subscribers.get(event.event_type, []):
            try:
                await self._call_handler(handler, event)
                dispatched_count += 1
            except Exception as e:
                # FAIL FAST: Create error event for handler failures
                await self._handle_subscriber_error(e, handler, event)

        # Dispatch to wildcard subscribers (None event type)
        for handler in self._subscribers.get(None, []):
            try:
                await self._call_handler(handler, event)
                dispatched_count += 1
            except Exception as e:
                # For wildcard handlers, log error but don't create infinite loop
                logger.error(
                    f"Wildcard event handler error: {e} " f"(event: {event.event_type.value}, handler: {handler})"
                )

        if dispatched_count > 0:
            logger.debug(f"Dispatched event {event.event_type.value} to {dispatched_count} handlers")

    async def _call_handler(self, handler: Callable, event: SaberEvent) -> None:
        """
        Call event handler (sync or async).

        Args:
            handler: Handler function to call
            event: Event to pass to handler
        """
        if asyncio.iscoroutinefunction(handler):
            await handler(event)
        else:
            handler(event)

    async def _handle_subscriber_error(self, error: Exception, handler: Callable, original_event: SaberEvent) -> None:
        """
        Handle subscriber error by creating ERROR event.

        Args:
            error: Exception that occurred
            handler: Handler that failed
            original_event: Event that caused the error
        """
        try:
            error_event = SaberEvent(
                event_type=EventType.ERROR,
                session_id=original_event.session_id,
                correlation_id=original_event.correlation_id,
                payload={
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "handler": str(handler),
                    "original_event_type": original_event.event_type.value,
                    "original_payload": original_event.payload,
                },
                timestamp=datetime.now(),
            )

            # Publish error event (bypass queue to avoid infinite loops)
            await self._dispatch_event(error_event)

        except Exception as e:
            # Last resort - log error if we can't even create error event
            logger.critical(
                f"Failed to create error event for handler failure: {e} "
                f"(original error: {error}, handler: {handler})"
            )

    async def _flush_remaining_events(self) -> None:
        """Flush all remaining events in queue during shutdown."""
        processed = 0
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                await self._dispatch_event(event)
                processed += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error flushing event during shutdown: {e}")

        if processed > 0:
            logger.info(f"Flushed {processed} remaining events during shutdown")

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "max_queue_size": self._max_queue_size,
            "subscriber_types": list(self._subscribers.keys()),
            "total_subscribers": sum(len(handlers) for handlers in self._subscribers.values()),
        }
