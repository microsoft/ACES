"""UI Interfaces for Tool Call Monitoring and Progress Reporting."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ...api_models import ToolCallEventComplete, ToolCallEventStart
from .models import StepStatus, StepUpdate


class ToolCallProgressReporter(ABC):
    """Abstract interface for reporting tool call progress to UI."""

    @abstractmethod
    async def tool_call_start(self, tool_event: ToolCallEventStart) -> None:
        """Report that a tool call has started."""
        pass

    @abstractmethod
    async def tool_call_complete(self, tool_event: ToolCallEventComplete) -> None:
        """Report that a tool call has completed (success, failure, or timeout)."""
        pass


class UIProgressAdapter(ToolCallProgressReporter):
    """
    Adapter that connects tool call events to the UI manager.

    This class bridges the gap between the server tool call events
    and the harness UI system for real-time progress display.
    """

    def __init__(self, ui_manager: Optional[Any] = None):
        """
        Initialize UI progress adapter.

        Args:
            ui_manager: The harness UI manager instance
        """
        self.ui_manager = ui_manager
        self._active_tool_calls: Dict[str, ToolCallEventStart] = {}
        self._lock = asyncio.Lock()

    async def tool_call_start(self, tool_event: ToolCallEventStart) -> None:
        """Report that a tool call has started."""
        async with self._lock:
            self._active_tool_calls[tool_event.call_id] = tool_event

        # Update UI with tool call start
        if self.ui_manager:
            try:
                # Convert tool event to UI progress step
                step = StepUpdate(
                    step_id=tool_event.call_id,
                    name=f"{tool_event.tool_name}",
                    description=f"Calling {tool_event.tool_name}",
                    status=StepStatus.RUNNING,
                    progress=0.0,
                    start_time=(
                        datetime.fromisoformat(tool_event.timestamp)
                        if tool_event.timestamp
                        else datetime.now(timezone.utc)
                    ),
                    task_id=tool_event.task_id,
                    metadata={
                        "tool_name": tool_event.tool_name,
                        "input_args": tool_event.arguments,
                        "agent_id": tool_event.agent_id,
                    },
                )

                # Send step update through UI manager
                await self._send_step_update(step)
                # Also send the tool event directly if supported
                if hasattr(self.ui_manager, "tool_call_start"):
                    await self.ui_manager.tool_call_start(tool_event)

            except Exception as e:
                # Don't let UI errors break tool execution
                import logging

                logging.getLogger(__name__).warning(f"UI update failed (tool_call_start): {e}")

    async def tool_call_complete(self, tool_event: ToolCallEventComplete) -> None:
        """Report that a tool call has completed."""
        async with self._lock:
            if tool_event.call_id in self._active_tool_calls:
                # Remove from active calls since it's complete
                del self._active_tool_calls[tool_event.call_id]

        # Update UI with completion
        if self.ui_manager:
            try:
                step = StepUpdate(
                    step_id=tool_event.call_id,
                    name=f"{tool_event.tool_name}",
                    description=f"Completed {tool_event.tool_name}",
                    status=StepStatus.COMPLETED if tool_event.success else StepStatus.FAILED,
                    progress=1.0,
                    end_time=(
                        datetime.fromisoformat(tool_event.timestamp)
                        if tool_event.timestamp
                        else datetime.now(timezone.utc)
                    ),
                    task_id=tool_event.task_id,
                    metadata={
                        "tool_name": tool_event.tool_name,
                        "execution_time_ms": tool_event.execution_time_ms,
                        "output": tool_event.output,
                        "error": tool_event.error,
                    },
                )

                await self._send_step_update(step)
                # Also send the tool event directly if supported
                if hasattr(self.ui_manager, "tool_call_complete"):
                    await self.ui_manager.tool_call_complete(tool_event)

            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"UI update failed (tool_call_complete): {e}")

        # NOTE: Clean up was already done above when we removed from active_tool_calls

    async def _send_step_update(self, step_info: Any) -> None:
        """Send step update to UI manager."""
        if not self.ui_manager:
            return

        # Check if UI manager has the expected interface
        if hasattr(self.ui_manager, "update_step_progress"):
            await self.ui_manager.update_step_progress(step_info)
        elif hasattr(self.ui_manager, "notify_tool_call"):
            await self.ui_manager.notify_tool_call(step_info)
        else:
            # Fallback: try to send a generic progress update
            import logging

            name = (
                getattr(step_info, "name", "<unknown>")
                if not isinstance(step_info, dict)
                else step_info.get("step_name", "<unknown>")
            )
            logging.getLogger(__name__).debug(f"UI manager doesn't support step updates, step: {name}")

    def get_active_tool_calls(self) -> Dict[str, ToolCallEventStart]:
        """Get currently active tool calls."""
        return self._active_tool_calls.copy()

    def get_tool_call_count(self) -> int:
        """Get number of active tool calls."""
        return len(self._active_tool_calls)
