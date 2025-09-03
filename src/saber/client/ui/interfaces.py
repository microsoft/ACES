"""UI Interfaces for Tool Call Monitoring and Progress Reporting."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from .models import StepStatus, StepUpdate, ToolCallEventComplete, ToolCallEventProgress, ToolCallEventStart


class MCPToolCallStatus(Enum):
    """Status of an MCP tool call."""

    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class MCPToolCall:
    """Represents an MCP tool call with progress tracking."""

    tool_name: str
    call_id: str
    status: MCPToolCallStatus
    input_args: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_ms: Optional[float] = None
    output: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None


class ToolCallProgressReporter(ABC):
    """Abstract interface for reporting tool call progress to UI."""

    @abstractmethod
    async def tool_call_start(self, tool_call: MCPToolCall) -> None:
        """Report that a tool call has started."""
        pass

    @abstractmethod
    async def tool_call_progress(self, tool_call: MCPToolCall) -> None:
        """Report progress update for a tool call."""
        pass

    @abstractmethod
    async def tool_call_complete(self, tool_call: MCPToolCall) -> None:
        """Report that a tool call has completed (success, failure, or timeout)."""
        pass


class UIProgressAdapter(ToolCallProgressReporter):
    """
    Adapter that connects MCP tool call progress to the UI manager.

    This class bridges the gap between the MCP sidecar tool call tracking
    and the harness UI system for real-time progress display.
    """

    def __init__(self, ui_manager: Optional[Any] = None):
        """
        Initialize UI progress adapter.

        Args:
            ui_manager: The harness UI manager instance
        """
        self.ui_manager = ui_manager
        self._active_tool_calls: Dict[str, MCPToolCall] = {}
        self._lock = asyncio.Lock()

    async def tool_call_start(self, tool_call: MCPToolCall) -> None:
        """Report that a tool call has started."""
        async with self._lock:
            self._active_tool_calls[tool_call.call_id] = tool_call

        # Update UI with tool call start
        if self.ui_manager:
            try:
                # Convert tool call to UI progress step
                step = StepUpdate(
                    step_id=tool_call.call_id,
                    name=f"{tool_call.tool_name}",
                    description=f"Calling {tool_call.tool_name}",
                    status=StepStatus.RUNNING,
                    progress=0.0,
                    start_time=tool_call.start_time,
                    task_id=tool_call.task_id,
                    metadata={
                        "tool_name": tool_call.tool_name,
                        "input_args": tool_call.input_args,
                        "agent_id": tool_call.agent_id,
                    },
                )

                # Send step update through UI manager
                await self._send_step_update(step)
                # Also show an explicit tool start line if supported
                if hasattr(self.ui_manager, "tool_call_start"):
                    event = ToolCallEventStart(
                        call_id=tool_call.call_id,
                        tool_name=tool_call.tool_name,
                        arguments=tool_call.input_args,
                        agent_id=tool_call.agent_id,
                        session_id=tool_call.session_id,
                        task_id=tool_call.task_id,
                    )
                    await self.ui_manager.tool_call_start(event)

            except Exception as e:
                # Don't let UI errors break tool execution
                import logging

                logging.getLogger(__name__).warning(f"UI update failed (tool_call_start): {e}")

    async def tool_call_progress(self, tool_call: MCPToolCall) -> None:
        """Report progress update for a tool call."""
        # Capture prior state before any mutation so we can access original args
        # prior_call: Optional[MCPToolCall] = None  # Not used currently
        async with self._lock:
            if tool_call.call_id in self._active_tool_calls:
                pass  # prior_call = self._active_tool_calls[tool_call.call_id]

        # Update UI with progress
        if self.ui_manager:
            try:
                step = StepUpdate(
                    step_id=tool_call.call_id,
                    name=f"{tool_call.tool_name}",
                    description=f"Executing {tool_call.tool_name}",
                    status=StepStatus.RUNNING,
                    progress=tool_call.progress,
                    task_id=tool_call.task_id,
                    metadata={
                        "tool_name": tool_call.tool_name,
                        "execution_time_ms": tool_call.execution_time_ms,
                    },
                )

                await self._send_step_update(step)
                # Also show an explicit tool progress line if supported
                if hasattr(self.ui_manager, "tool_call_progress"):
                    event = ToolCallEventProgress(
                        call_id=tool_call.call_id,
                        tool_name=tool_call.tool_name,
                        progress_info="executing...",
                        progress=tool_call.progress,
                        agent_id=tool_call.agent_id,
                        session_id=tool_call.session_id,
                        task_id=tool_call.task_id,
                    )
                    await self.ui_manager.tool_call_progress(event)

            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"UI update failed (tool_call_progress): {e}")

    async def tool_call_complete(self, tool_call: MCPToolCall) -> None:
        """Report that a tool call has completed."""
        prior_call: Optional[MCPToolCall] = None
        async with self._lock:
            if tool_call.call_id in self._active_tool_calls:
                prior_call = self._active_tool_calls[tool_call.call_id]
                self._active_tool_calls[tool_call.call_id] = tool_call

        # Update UI with completion
        if self.ui_manager:
            try:
                # status = "completed" if tool_call.status == MCPToolCallStatus.COMPLETED else "failed"
                # Not used currently, but kept for potential future use

                step = StepUpdate(
                    step_id=tool_call.call_id,
                    name=f"{tool_call.tool_name}",
                    description=f"Completed {tool_call.tool_name}",
                    status=(
                        StepStatus.COMPLETED if tool_call.status == MCPToolCallStatus.COMPLETED else StepStatus.FAILED
                    ),
                    progress=1.0,
                    end_time=tool_call.end_time,
                    task_id=tool_call.task_id,
                    metadata={
                        "tool_name": tool_call.tool_name,
                        "execution_time_ms": tool_call.execution_time_ms,
                        "output": tool_call.output,
                        "error": tool_call.error,
                    },
                )

                await self._send_step_update(step)
                # Also show an explicit tool complete line if supported
                if hasattr(self.ui_manager, "tool_call_complete"):
                    success = tool_call.status == MCPToolCallStatus.COMPLETED and not tool_call.error
                    # Fallback to stored args if missing on this event
                    args_for_event = tool_call.input_args or (prior_call.input_args if prior_call else None)
                    event = ToolCallEventComplete(
                        call_id=tool_call.call_id,
                        tool_name=tool_call.tool_name,
                        success=success,
                        arguments=args_for_event,
                        execution_time_ms=tool_call.execution_time_ms,
                        output=tool_call.output,
                        error=tool_call.error,
                        agent_id=tool_call.agent_id,
                        session_id=tool_call.session_id,
                        task_id=tool_call.task_id,
                    )
                    await self.ui_manager.tool_call_complete(event)

            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"UI update failed (tool_call_complete): {e}")

        # Clean up completed tool call
        async with self._lock:
            self._active_tool_calls.pop(tool_call.call_id, None)

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

    def get_active_tool_calls(self) -> Dict[str, MCPToolCall]:
        """Get currently active tool calls."""
        return self._active_tool_calls.copy()

    def get_tool_call_count(self) -> int:
        """Get number of active tool calls."""
        return len(self._active_tool_calls)
