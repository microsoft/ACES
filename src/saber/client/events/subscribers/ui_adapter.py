"""
UI Adapter Subscriber for SABER Event Bus

Converts event bus events to UIManager method calls for real-time UI updates.
Decouples harness from UI through event-driven architecture.
"""

import logging
from typing import Optional

from ..types import EventType, SaberEvent

logger = logging.getLogger(__name__)


class UIAdapterSubscriber:
    """
    Event bus subscriber that adapts events to UIManager calls.

    Provides a clean interface between the event bus and UI components,
    allowing the harness to emit events without directly coupling to UI.
    """

    def __init__(self, ui_manager: Optional[object]):
        """
        Initialize UI adapter subscriber.

        Args:
            ui_manager: UIManager instance or None if UI is disabled
        """
        self.ui_manager = ui_manager

        if ui_manager:
            logger.info("UI adapter subscriber initialized with UI manager")
        else:
            logger.info("UI adapter subscriber initialized without UI manager (UI disabled)")

    async def handle_event(self, event: SaberEvent) -> None:
        """
        Handle events from the event bus and dispatch to appropriate UI methods.

        Args:
            event: Event to handle

        Raises:
            Exception: Any UI method errors are propagated for fail-fast behavior
        """
        if not self.ui_manager:
            # UI is disabled - silently ignore events
            return

        # Debug: log all events received by UI adapter
        logger.info(f"🎯 UI ADAPTER received event: {event.event_type.value} (session={event.session_id})")

        try:
            if event.event_type == EventType.TOOL_CALL_STARTED:
                await self._handle_tool_call_started(event)
            elif event.event_type == EventType.TOOL_CALL_COMPLETED:
                await self._handle_tool_call_completed(event)
            elif event.event_type == EventType.TASK_STARTED:
                await self._handle_task_started(event)
            elif event.event_type == EventType.TASK_COMPLETED:
                await self._handle_task_completed(event)
            elif event.event_type == EventType.SESSION_STARTED:
                await self._handle_session_started(event)
            elif event.event_type == EventType.SESSION_COMPLETED:
                await self._handle_session_completed(event)
            elif event.event_type == EventType.ERROR:
                await self._handle_error_event(event)
            else:
                # Handle any future event types that might be added
                logger.debug(f"Unhandled event type: {event.event_type.value}")  # type: ignore[unreachable]

        except Exception as e:
            # Let errors bubble up to event bus for proper error handling
            logger.error(f"UI adapter error handling {event.event_type.value}: {e}")
            raise

    async def _handle_tool_call_started(self, event: SaberEvent) -> None:
        """
        Handle tool call started event.

        Args:
            event: Tool call started event
        """
        payload = event.payload

        # Create proper ToolCallEventStart object from payload
        from ....api_models import ToolCallEventStart

        # Deserialize the entire payload directly - server sends complete tool call data
        tool_call_event = ToolCallEventStart(**payload)

        # Check if UI manager has the expected method
        if hasattr(self.ui_manager, "tool_call_start"):
            await self.ui_manager.tool_call_start(tool_call_event)
            logger.debug(f"Dispatched tool_call_start to UI: {payload.get('tool_name')}")
        else:
            logger.warning("UI manager does not have tool_call_start method")

    async def _handle_tool_call_completed(self, event: SaberEvent) -> None:
        """
        Handle tool call completed event.

        Args:
            event: Tool call completed event
        """
        payload = event.payload

        # Debug: log the payload type and content
        logger.info(f"🔍 tool_call_completed payload type: {type(payload)}, content: {payload}")

        try:
            # Create proper ToolCallEventComplete object from payload
            from ....api_models import ToolCallEventComplete

            # Handle the result field mapping - server sends 'result' but model expects 'output'
            if "result" in payload and "output" not in payload:
                payload["output"] = payload["result"]

            # Deserialize the entire payload directly - server sends complete tool call data
            tool_call_event = ToolCallEventComplete(**payload)

            # Check if UI manager has the expected method
            if hasattr(self.ui_manager, "tool_call_complete"):
                await self.ui_manager.tool_call_complete(tool_call_event)
                status = "success" if payload.get("success") else "failed"
                logger.debug(f"Dispatched tool_call_complete to UI: {payload.get('tool_name')} ({status})")
            else:
                logger.warning("UI manager does not have tool_call_complete method")

            # Also trigger step progress update to increment task progress
            # Each tool call completion represents one step completed
            if hasattr(self.ui_manager, "update_step_progress"):
                from ...ui.models import StepStatus, StepUpdate

                # Create step info for progress update
                step_info = StepUpdate(
                    step_id=payload.get("call_id", "unknown"),
                    name=payload.get("tool_name", "unknown"),
                    description=f"Completed {payload.get('tool_name', 'tool call')}",
                    status=StepStatus.COMPLETED if payload.get("success") else StepStatus.FAILED,
                    progress=1.0,  # Tool call is complete
                    task_id=payload.get("task_id"),  # May be None, will fall back to single active task
                    metadata={
                        "tool_name": payload.get("tool_name"),
                        "execution_time_ms": payload.get("execution_time_ms"),
                        "success": payload.get("success"),
                    },
                )

                await self.ui_manager.update_step_progress(step_info)
                logger.debug(f"Dispatched update_step_progress for tool: {payload.get('tool_name')}")
            else:
                logger.warning("UI manager does not have update_step_progress method")

        except Exception as e:
            logger.error(
                f"Error processing tool_call_completed payload: {e}, payload type: {type(payload)}, payload: {payload}"
            )
            raise

    async def _handle_task_started(self, event: SaberEvent) -> None:
        """
        Handle task started event.

        Args:
            event: Task started event
        """
        payload = event.payload

        if hasattr(self.ui_manager, "task_start"):
            await self.ui_manager.task_start(payload)
            logger.debug(f"Dispatched task_start to UI: {payload.get('task_id')}")
        else:
            logger.warning("UI manager does not have task_start method")

    async def _handle_task_completed(self, event: SaberEvent) -> None:
        """
        Handle task completed event.

        Args:
            event: Task completed event
        """
        payload = event.payload

        if hasattr(self.ui_manager, "task_complete"):
            await self.ui_manager.task_complete(payload)
            logger.debug(f"Dispatched task_complete to UI: {payload.get('task_id')}")
        else:
            logger.warning("UI manager does not have task_complete method")

    async def _handle_session_started(self, event: SaberEvent) -> None:
        """
        Handle session started event.

        Args:
            event: Session started event
        """
        payload = event.payload

        if hasattr(self.ui_manager, "session_start"):
            await self.ui_manager.session_start(payload)
            logger.debug(f"Dispatched session_start to UI: {event.session_id}")
        else:
            logger.warning("UI manager does not have session_start method")

    async def _handle_session_completed(self, event: SaberEvent) -> None:
        """
        Handle session completed event.

        Args:
            event: Session completed event
        """
        payload = event.payload

        if hasattr(self.ui_manager, "session_complete"):
            await self.ui_manager.session_complete(payload)
            logger.debug(f"Dispatched session_complete to UI: {event.session_id}")
        else:
            logger.warning("UI manager does not have session_complete method")

    async def _handle_error_event(self, event: SaberEvent) -> None:
        """
        Handle error event.

        Args:
            event: Error event
        """
        payload = event.payload

        if hasattr(self.ui_manager, "error_event"):
            await self.ui_manager.error_event(payload)
            logger.debug(f"Dispatched error_event to UI: {payload.get('error')}")
        else:
            # Log errors even if UI doesn't handle them
            logger.error(f"Event bus error: {payload.get('error')} (no UI handler)")

    def is_ui_enabled(self) -> bool:
        """Check if UI is enabled."""
        return self.ui_manager is not None
