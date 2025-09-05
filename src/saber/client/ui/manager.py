"""UIManager - Central orchestrator for SABER UI functionality."""

import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ...api_models import ToolCallEventComplete, ToolCallEventStart
from .inspect_ai_backend import InspectAIBackend
from .models import (
    MessageType,
    StepStatus,
    StepUpdate,
    TaskInfo,
    TaskStatus,
    UIBackend,
    UIBackendType,
    UIConfig,
    UIMessage,
)

# Conditional import for textual session
try:
    from .textual_session import SABERTextualSession, create_textual_session

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False


logger = logging.getLogger(__name__)


class UIManager:
    """
    Central UI manager that orchestrates all UI interactions.

    This is the single point of control for UI operations in SABER.
    It handles backend initialization, message routing, and task tracking.
    """

    def __init__(self, config: Optional[UIConfig] = None):
        """Initialize the UI manager with optional configuration."""
        self.config = config
        self.backend: Optional[UIBackend] = None
        self.active_tasks: Dict[str, TaskInfo] = {}

        if config:
            self._initialize_backend()

    # --- Formatting helpers for compact, readable previews ---
    MAX_ARGS_PREVIEW = 160
    MAX_OUTPUT_PREVIEW = 200

    @staticmethod
    def _to_compact_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except Exception:
            return str(value)

    @classmethod
    def _format_preview(cls, value: Any, default: str, limit: int) -> str:
        if value is None:
            return default
        # If it's a string that looks like JSON, try to parse
        if isinstance(value, str):
            v = value.strip()
            if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
                try:
                    parsed = json.loads(v)
                    text = cls._to_compact_json(parsed)
                except Exception:
                    text = value
            else:
                text = value
        elif isinstance(value, (dict, list, tuple)):
            text = cls._to_compact_json(value)
        else:
            text = str(value)
        # Collapse whitespace to keep single-line
        text = " ".join(text.split())
        # Truncate
        if len(text) > limit:
            return text[: limit - 1] + "…"
        return text

    @classmethod
    def _format_args_preview(cls, args: Any) -> str:
        return cls._format_preview(args, default="{}", limit=cls.MAX_ARGS_PREVIEW)

    @classmethod
    def _format_output_preview(cls, output: Any) -> str:
        return cls._format_preview(output, default="<no output>", limit=cls.MAX_OUTPUT_PREVIEW)

    def _initialize_backend(self) -> None:
        """Initialize the appropriate UI backend based on configuration."""
        if not self.config:
            return

        try:
            if self.config.backend_type == UIBackendType.INSPECT_AI:
                self.backend = InspectAIBackend(self.config)
                logger.info("Initialized Inspect-AI UI backend")
            # Note: Currently only INSPECT_AI is supported, but keeping structure for future backends
        except Exception as e:
            logger.error(f"Failed to initialize UI backend: {e}")
            self.backend = None

    def is_enabled(self) -> bool:
        """Check if UI is enabled and functional."""
        return self.backend is not None

    def display_message(
        self, content: str, message_type: MessageType = MessageType.INFO, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Display a message through the UI backend."""
        if not self.backend:
            return

        message = UIMessage(content=content, message_type=message_type, timestamp=datetime.now(), metadata=metadata)

        try:
            self.backend.display_message(message)
        except Exception as e:
            logger.error(f"Failed to display message: {e}")

    def start_task(self, task_id: str, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start tracking a new task."""
        task = TaskInfo(
            task_id=task_id,
            name=name,
            status=TaskStatus.RUNNING,
            progress=0.0,
            start_time=datetime.now(),
            metadata=metadata,
        )

        self.active_tasks[task_id] = task

        if self.backend:
            try:
                self.backend.show_progress(task)
            except Exception as e:
                logger.error(f"Failed to show task progress: {e}")

    def update_task_progress(self, task_id: str, progress: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update the progress of an active task."""
        if task_id not in self.active_tasks:
            logger.warning(f"Attempting to update unknown task: {task_id}")
            return

        task = self.active_tasks[task_id]
        task.progress = max(0.0, min(1.0, progress))  # Clamp to 0-1 range

        if metadata:
            if task.metadata:
                task.metadata.update(metadata)
            else:
                task.metadata = metadata

        if self.backend:
            try:
                self.backend.update_task_status(task)
            except Exception as e:
                logger.error(f"Failed to update task status: {e}")

    async def update_step_progress(self, step_info: Any) -> None:
        """Increment task progress based on tool-call step events.

        Accepts a StepUpdate object (preferred) or a legacy dict.
        """
        # Resolve task_id from StepUpdate or dict
        if isinstance(step_info, StepUpdate):
            task_id = step_info.task_id
        else:
            task_id = step_info.get("task_id") or (step_info.get("metadata") or {}).get("task_id")
        # Fallback: if task_id is missing or unknown, and exactly one task is active, use it
        if not task_id or task_id not in self.active_tasks:
            if len(self.active_tasks) == 1:
                task_id = next(iter(self.active_tasks.keys()))
            else:
                return

        task = self.active_tasks[task_id]
        total_steps = 10
        current_steps = int(task.progress * total_steps)

        # Extract status and progress in a type-safe way
        status: Any
        if isinstance(step_info, StepUpdate):
            status = step_info.status
            progress_val = step_info.progress
        else:
            status = step_info.get("status")
            progress_val = step_info.get("progress", 0.0)

        # Determine status flags based on type
        is_running = False
        is_terminal = False

        if isinstance(status, StepStatus):
            # Handle StepStatus enum values
            if status == StepStatus.RUNNING:
                is_running = True
            else:  # Must be COMPLETED or FAILED
                is_terminal = True
        elif isinstance(status, str):
            if status == "running":
                is_running = True
            elif status in ("completed", "failed"):
                is_terminal = True

        # Update step count based on status
        if is_running and (progress_val or 0.0) == 0.0:
            current_steps += 1
        if is_terminal:
            current_steps = max(current_steps, 1)

        current_steps = max(0, min(total_steps, current_steps))
        new_progress = current_steps / total_steps
        self.update_task_progress(task_id, new_progress)

    def complete_task(self, task_id: str, success: bool = True, error_message: Optional[str] = None) -> None:
        """Complete a task and update its status."""
        if task_id not in self.active_tasks:
            logger.warning(f"Attempting to complete unknown task: {task_id}")
            return

        task = self.active_tasks[task_id]
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.progress = 1.0
        task.end_time = datetime.now()
        task.error_message = error_message

        if self.backend:
            try:
                self.backend.update_task_status(task)
            except Exception as e:
                logger.error(f"Failed to update task completion: {e}")

        # Remove from active tasks
        del self.active_tasks[task_id]

    def show_results(self, results: Dict[str, Any]) -> None:
        """Display execution results."""
        if self.backend:
            try:
                self.backend.show_results(results)
            except Exception as e:
                logger.error(f"Failed to show results: {e}")

    def cleanup(self) -> None:
        """Clean up UI resources."""
        if self.backend:
            try:
                self.backend.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup UI backend: {e}")

        self.active_tasks.clear()

    def create_textual_session(
        self, session_id: str, episode_id: Optional[str] = None
    ) -> Optional["SABERTextualSession"]:
        """
        Create a SABER textual session for full TUI experience.

        This method provides access to the dedicated textual mode that runs
        SABER evaluations within Inspect-AI's textual UI context.

        Args:
            session_id: Unique session identifier
            episode_id: Optional episode identifier for grouping

        Returns:
            SABERTextualSession instance if textual mode is available, None otherwise
        """
        if not TEXTUAL_AVAILABLE:
            logger.warning("Textual mode not available - Inspect-AI not installed or textual session failed to import")
            return None

        try:
            return create_textual_session(session_id, episode_id)
        except Exception as e:
            logger.error(f"Failed to create textual session: {e}")
            return None

    def is_textual_mode_available(self) -> bool:
        """
        Check if textual mode is available.

        Returns:
            True if textual mode can be used, False otherwise
        """
        return TEXTUAL_AVAILABLE

    def run_textual_evaluation(
        self,
        session_id: str,
        tasks: List[TaskInfo],
        evaluation_func: Callable[..., Any],
        episode_id: Optional[str] = None,
    ) -> Any:
        """
        Run a complete SABER evaluation in textual mode.

        This is a convenience method that creates a textual session, starts tasks,
        and runs the evaluation within the TUI context.

        Note: This method is synchronous to properly handle the async/sync boundary
        with Inspect-AI's textual system.

        Args:
            session_id: Unique session identifier
            tasks: List of SABER tasks to execute
            evaluation_func: Function that performs the actual evaluation
            episode_id: Optional episode identifier

        Returns:
            Result of the evaluation function

        Raises:
            RuntimeError: If textual mode is not available
        """
        if not TEXTUAL_AVAILABLE:
            raise RuntimeError(
                "Textual mode not available. Please install inspect-ai and ensure "
                "textual dependencies are available."
            )

        # Create textual session
        textual_session = self.create_textual_session(session_id, episode_id)
        if not textual_session:
            raise RuntimeError("Failed to create textual session")

        # Define the evaluation wrapper that integrates with textual session
        async def evaluation_wrapper(session: "SABERTextualSession") -> Any:
            # Start task session
            await session.start_task_session(tasks)

            # Run the actual evaluation, passing the session for progress updates
            return await evaluation_func(session)

        # Run the evaluation within textual context (sync call)
        return textual_session.run_saber_evaluation(evaluation_wrapper)

    # Session management methods for compatibility with old adapter interface
    async def session_start(self, session_info: Dict[str, Any]) -> None:
        """Start a new session."""
        self.display_message(
            f"🚀 Starting session: {session_info.get('session_id', 'unknown')}", MessageType.INFO, session_info
        )

        if self.backend and hasattr(self.backend, "session_start"):
            try:
                await self.backend.session_start(session_info)
            except Exception as e:
                logger.error(f"Failed to start session in backend: {e}")

    async def session_complete(self, session_summary: Dict[str, Any]) -> None:
        """Complete a session."""
        self.display_message(
            f"✅ Session completed: {session_summary.get('session_id', 'unknown')}",
            MessageType.SUCCESS,
            session_summary,
        )

        if self.backend and hasattr(self.backend, "session_complete"):
            try:
                await self.backend.session_complete(session_summary)
            except Exception as e:
                logger.error(f"Failed to complete session in backend: {e}")

    async def task_start(self, task_info: Dict[str, Any]) -> None:
        """Start a task (enhanced version of start_task)."""
        task_id = task_info.get("task_id", f"task_{len(self.active_tasks)}")
        name = task_info.get("name", "Unknown Task")

        # Use the existing start_task method
        self.start_task(task_id, name, task_info)

        if self.backend and hasattr(self.backend, "task_start"):
            try:
                await self.backend.task_start(task_info)
            except Exception as e:
                logger.error(f"Failed to start task in backend: {e}")

    async def task_complete(self, task_result: Dict[str, Any]) -> None:
        """Complete a task (enhanced version of complete_task)."""
        task_id = task_result.get("task_id", "")
        success = task_result.get("success", True)
        error_message = task_result.get("error_message")

        # Use the existing complete_task method
        self.complete_task(task_id, success, error_message)

        if self.backend and hasattr(self.backend, "task_complete"):
            try:
                await self.backend.task_complete(task_result)
            except Exception as e:
                logger.error(f"Failed to complete task in backend: {e}")

    async def tool_call_start(self, tool_call: ToolCallEventStart) -> None:
        """Start a tool call and update backend tracking."""
        metadata = tool_call.__dict__

        # Only update backend tracking - no separate display message
        # The integrated task display will show the tool call information

        if self.backend and hasattr(self.backend, "tool_call_start"):
            try:
                await self.backend.tool_call_start(metadata)
            except Exception as e:
                logger.error(f"Failed to start tool call in backend: {e}")

    async def tool_call_complete(self, tool_call: ToolCallEventComplete) -> None:
        """Complete a tool call and update backend tracking."""
        metadata = tool_call.__dict__

        # Only update backend tracking - no separate display message
        # The integrated task display will show the tool call completion

        if self.backend and hasattr(self.backend, "tool_call_complete"):
            try:
                await self.backend.tool_call_complete(metadata)
            except Exception as e:
                logger.error(f"Failed to complete tool call in backend: {e}")

    # Extended methods for advanced UI functionality

    def start_session(self, session_id: str, total_tasks: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start a UI session."""
        if self.backend and hasattr(self.backend, "start_session"):
            try:
                self.backend.start_session(session_id, total_tasks, metadata)
            except Exception as e:
                logger.error(f"Failed to start session: {e}")

    def complete_session(
        self,
        session_id: str,
        successful_tasks: int,
        total_tasks: int,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Complete a UI session."""
        if self.backend and hasattr(self.backend, "complete_session"):
            try:
                self.backend.complete_session(session_id, successful_tasks, total_tasks, duration_seconds, metadata)
            except Exception as e:
                logger.error(f"Failed to complete session: {e}")

    def report_tool_call_start(self, tool_name: str, call_id: str, task_id: Optional[str] = None) -> None:
        """Report tool call start."""
        if self.backend and hasattr(self.backend, "report_tool_call_start"):
            try:
                self.backend.report_tool_call_start(tool_name, call_id, task_id)
            except Exception as e:
                logger.error(f"Failed to report tool call start: {e}")

    def report_tool_call_complete(
        self,
        tool_name: str,
        call_id: str,
        success: bool,
        execution_time_ms: Optional[float] = None,
        error_message: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Report tool call completion."""
        if self.backend and hasattr(self.backend, "report_tool_call_complete"):
            try:
                self.backend.report_tool_call_complete(
                    tool_name, call_id, success, execution_time_ms, error_message, task_id
                )
            except Exception as e:
                logger.error(f"Failed to report tool call complete: {e}")

    def complete_task_with_result(
        self,
        task_id: str,
        success: bool,
        episode_id: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        tool_calls: Optional[int] = None,
        successful_tool_calls: Optional[int] = None,
        flag: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Complete task with detailed results."""
        if self.backend and hasattr(self.backend, "complete_task_with_result"):
            try:
                self.backend.complete_task_with_result(
                    task_id,
                    success,
                    episode_id,
                    duration_seconds,
                    tool_calls,
                    successful_tool_calls,
                    flag,
                    error_message,
                )
            except Exception as e:
                logger.error(f"Failed to complete task with result: {e}")

        # Also handle the basic task completion
        self.complete_task(task_id, success, error_message)
