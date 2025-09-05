"""
Tool Event Publisher for SABER Server

Publishes tool execution events to session-specific subscribers via asyncio queues.
Follows fail-fast principles - queue overflow raises RuntimeError.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ....api_models import ToolCallEventComplete, ToolCallEventStart

logger = logging.getLogger(__name__)


class ToolEventPublisher:
    """
    Server-side tool event publisher for real-time SSE distribution.

    Publishes tool call events to session-specific queues with fail-fast semantics.
    No silent failures - queue overflow raises RuntimeError immediately.
    """

    def __init__(self, max_queue_size: int = 100):
        """
        Initialize tool event publisher.

        Args:
            max_queue_size: Maximum events per session queue before failing fast
        """
        self._subscribers: Dict[str, asyncio.Queue] = {}  # session_id -> queue
        self._max_queue_size = max_queue_size
        logger.info(f"Tool event publisher initialized with max_queue_size={max_queue_size}")

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        """
        Subscribe to tool events for a session.

        Args:
            session_id: SABER session ID to subscribe to

        Returns:
            asyncio.Queue for receiving tool events

        Raises:
            ValueError: If session_id is empty
        """
        if not session_id:
            raise ValueError("session_id cannot be empty")

        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers[session_id] = queue
        logger.info(f"Subscribed to tool events for session: {session_id}")
        return queue

    async def unsubscribe(self, session_id: str) -> None:
        """
        Unsubscribe from tool events for a session.

        Args:
            session_id: SABER session ID to unsubscribe from
        """
        if session_id in self._subscribers:
            del self._subscribers[session_id]
            logger.info(f"Unsubscribed from tool events for session: {session_id}")

    async def publish_tool_started(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: str,
        task_id: Optional[str] = None,
        current_step: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Publish tool call started event.

        Args:
            session_id: SABER session ID
            tool_name: Name of the tool being executed
            arguments: Tool call arguments
            call_id: Unique identifier for this tool call
            task_id: Optional task ID for context
            current_step: Current step number (1-based)
            max_steps: Maximum number of steps allowed

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If session queue is full (fail-fast)
        """
        if not all([session_id, tool_name, call_id]):
            raise ValueError("session_id, tool_name, and call_id are required")

        event_model = ToolCallEventStart(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            session_id=session_id,
            task_id=task_id,
            current_step=current_step,
            max_steps=max_steps,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Convert to dict for JSON serialization over SSE
        event = {"type": "tool_call_started", **event_model.model_dump()}

        await self._publish_to_session(session_id, event)
        logger.info(f"Published tool_call_started: {tool_name} (call_id={call_id}, session={session_id})")

    async def publish_tool_completed(
        self,
        session_id: str,
        tool_name: str,
        call_id: str,
        success: bool,
        arguments: Optional[Dict[str, Any]] = None,
        result: Any = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        task_id: Optional[str] = None,
        current_step: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Publish tool call completed event.

        Args:
            session_id: SABER session ID
            tool_name: Name of the tool that was executed
            call_id: Unique identifier for this tool call
            success: Whether tool execution succeeded
            result: Tool execution result (if successful)
            error: Error message (if failed)
            execution_time_ms: Execution time in milliseconds
            task_id: Optional task ID for context
            current_step: Current step number (1-based)
            max_steps: Maximum number of steps allowed

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If session queue is full (fail-fast)
        """
        if not all([session_id, tool_name, call_id]):
            raise ValueError("session_id, tool_name, and call_id are required")

        event_model = ToolCallEventComplete(
            call_id=call_id,
            tool_name=tool_name,
            success=success,
            arguments=arguments,
            output=str(result) if result is not None else None,
            error=error,
            execution_time_ms=execution_time_ms,
            session_id=session_id,
            task_id=task_id,
            current_step=current_step,
            max_steps=max_steps,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Convert to dict for JSON serialization over SSE, but keep the raw result field for backward compatibility
        event = {
            "type": "tool_call_completed",
            **event_model.model_dump(),
            "result": result,  # Keep original result format for client compatibility
        }

        await self._publish_to_session(session_id, event)
        status = "succeeded" if success else "failed"
        logger.info(f"Published tool_call_completed: {tool_name} {status} (call_id={call_id}, session={session_id})")

    async def _publish_to_session(self, session_id: str, event: Dict[str, Any]) -> None:
        """
        Publish event to specific session queue.

        Args:
            session_id: Session to publish to
            event: Event data to publish

        Raises:
            RuntimeError: If session queue is full (fail-fast - no silent drops)
        """
        if session_id not in self._subscribers:
            # No subscribers for this session - this is normal, not an error
            logger.debug(f"No subscribers for session {session_id}, event dropped: {event['type']}")
            return

        queue = self._subscribers[session_id]
        try:
            queue.put_nowait(event)
            logger.debug(f"Event queued for session {session_id}: {event['type']}")
        except asyncio.QueueFull:
            # FAIL FAST: No silent dropping of events
            raise RuntimeError(
                f"Tool event queue full for session {session_id} "
                f"(max: {self._max_queue_size}). Event type: {event['type']}"
            )

    def get_subscriber_count(self) -> int:
        """Get number of active subscribers."""
        return len(self._subscribers)

    def get_session_queue_size(self, session_id: str) -> Optional[int]:
        """Get queue size for a specific session."""
        if session_id in self._subscribers:
            return self._subscribers[session_id].qsize()
        return None
