"""
Inspect-AI Backend for SABER UI

Clean implementation using inspect-ai display backends.
Fixed async/event loop issues and parameter conflicts.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

# Import actual inspect-ai display components
from inspect_ai._display import Display, TaskProfile
from inspect_ai._display.plain.display import PlainDisplay
from inspect_ai._display.rich.display import RichDisplay
from inspect_ai._display.textual.display import TextualDisplay
from inspect_ai.log import EvalConfig
from inspect_ai.model import GenerateConfig, ModelName

from .models import TaskInfo, UIBackend, UIConfig, UIMessage

logger = logging.getLogger(__name__)


class InspectAIBackendMode(Enum):
    """Supported Inspect-AI backend modes."""

    RICH = "rich"  # Full rich terminal UI with progress bars, colors
    TEXTUAL = "textual"  # Full TUI with interactive widgets
    PLAIN = "plain"  # Simple plain text output

    @classmethod
    def from_string(cls, mode_str: str) -> "InspectAIBackendMode":
        """Create backend mode from string."""
        mode_map = {"plain": cls.PLAIN, "rich": cls.RICH, "textual": cls.TEXTUAL}
        result = mode_map.get(mode_str.lower())
        if result is None:
            raise ValueError(f"Unknown backend mode: {mode_str}. Valid: {list(mode_map.keys())}")
        return result


@dataclass
class SABERTaskProfile:
    """Adapter for SABER tasks to inspect-ai TaskProfile."""

    name: str
    model: str = "saber-agent"
    dataset: str = "saber-episode"
    scorer: str = "saber-evaluator"
    samples: int = 1
    steps: int = 10
    eval_config: Optional[Dict[str, Any]] = None
    task_args: Optional[Dict[str, Any]] = None
    generate_config: Optional[Dict[str, Any]] = None
    log_location: str = ""
    file: Optional[str] = None
    tags: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        if self.eval_config is None:
            self.eval_config = {}
        if self.task_args is None:
            self.task_args = {}
        if self.generate_config is None:
            self.generate_config = {}
        if self.tags is None:
            self.tags = []

    def to_inspect_ai_profile(self) -> TaskProfile:
        """Convert to inspect-ai TaskProfile with correct parameters."""
        return TaskProfile(
            name=self.name,
            file=self.file,
            model=ModelName(f"openai/{self.model}"),
            dataset=self.dataset,
            scorer=self.scorer,
            samples=self.samples,
            steps=self.steps,
            eval_config=EvalConfig(),
            task_args=self.task_args or {},
            generate_config=GenerateConfig(),
            tags=self.tags,
            log_location=self.log_location,
        )


class InspectAIBackend(UIBackend):
    """
    Backend for integrating with inspect-ai terminal UI library.

    Clean implementation with fail-fast error handling and proper async support.
    """

    def __init__(self, config: UIConfig):
        """Initialize Inspect-AI backend."""
        self.config = config
        self.backend_mode = self._get_backend_mode()
        self._display: Optional[Display] = None
        self._current_task_displays: Dict[str, Any] = {}
        self._current_progress_contexts: Dict[str, Any] = {}
        self._task_metrics: Dict[str, List[Any]] = {}
        self._active_tasks: Dict[str, SABERTaskProfile] = {}

        # Initialize inspect-ai display - fail fast if not available
        self._init_inspect_ai_display()

        logger.info(f"🔍 Inspect-AI adapter initialized - Mode: {self.backend_mode.value}")

    def _get_backend_mode(self) -> InspectAIBackendMode:
        """Get backend mode from config."""
        if not self.config or not self.config.config:
            return InspectAIBackendMode.PLAIN

        mode_str = self.config.config.get("mode", "plain")
        return InspectAIBackendMode.from_string(mode_str)

    def _init_inspect_ai_display(self) -> None:
        """Initialize actual inspect-ai display backend - fail fast if not available."""
        try:
            if self.backend_mode == InspectAIBackendMode.PLAIN:
                self._display = PlainDisplay()
                self._display.multiple_task_names = False
                self._display.multiple_model_names = False
                self._display.tasks = []
                logger.info("✅ Inspect-AI PlainDisplay initialized")
            elif self.backend_mode == InspectAIBackendMode.RICH:
                self._display = RichDisplay()
                logger.info("✅ Inspect-AI RichDisplay initialized")
            elif self.backend_mode == InspectAIBackendMode.TEXTUAL:
                self._display = TextualDisplay()
                logger.info("✅ Inspect-AI TextualDisplay initialized")
            else:
                raise ValueError(f"Unknown backend mode: {self.backend_mode}")

        except ImportError as e:
            if "textual" in str(e).lower() and self.backend_mode == InspectAIBackendMode.TEXTUAL:
                raise ImportError("Textual backend not available. Install with: uv add textual>=0.35.0") from e
            else:
                raise ImportError(
                    "inspect-ai display backend not available. Install with: uv add inspect-ai>=0.3.0"
                ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize inspect-ai display: {e}") from e

    def _safe_display_print(self, message: str) -> None:
        """Print to display - fail fast if display not available."""
        if self._display is None:
            raise RuntimeError(
                "Display backend not initialized. Cannot print message. "
                "This indicates a critical UI initialization failure."
            )

        # Call the actual display print method
        try:
            self._display.print(message)
        except AttributeError:
            # inspect-ai displays may not have print method, use logging as fallback
            logger.info(f"Display: {message}")
        except Exception as e:
            raise RuntimeError(f"Display backend failed to print message: {e}") from e

    # UIBackend interface implementation
    def display_message(self, message: UIMessage) -> None:
        """Display a message using inspect-ai display."""
        try:
            self._safe_display_print(f"{message.message_type.value.upper()}: {message.content}")
        except Exception as e:
            logger.error(f"Failed to display message: {e}")
            raise

    def show_progress(self, task: TaskInfo) -> None:
        """Show task progress using inspect-ai display system."""
        try:
            # For textual mode, skip progress tracking since it requires different architecture
            if self.backend_mode == InspectAIBackendMode.TEXTUAL:
                self._safe_display_print(f"📋 TASK: {task.task_id}")
                self._safe_display_print(f"   ├─ Name: {task.name}")
                self._safe_display_print(f"   ├─ Status: {task.status.value}")
                self._safe_display_print("   └─ Note: Use real TUI for progress tracking")
                logger.info(f"Task {task.task_id} registered (textual mode - use real TUI for full experience)")
                return

            # Create SABER task profile for Plain and Rich modes
            saber_profile = SABERTaskProfile(
                name=task.task_id,
                model="saber-agent",
                dataset=task.metadata.get("episode_id", "saber-episode") if task.metadata else "saber-episode",
                scorer="saber-evaluator",
                samples=1,
                steps=10,  # Estimated steps for progress tracking
                eval_config={"state": task.status.value},
                task_args={"task_id": task.task_id},
                generate_config={},
                log_location=f"saber-{task.task_id}",
                tags=["saber", "security", task.status.value],
            )

            self._active_tasks[task.task_id] = saber_profile

            # Use inspect-ai's task context manager for proper display
            if self._display is not None:
                task_display = self._display.task(saber_profile.to_inspect_ai_profile())
                task_display_context = task_display.__enter__()
                self._current_task_displays[task.task_id] = task_display_context
            else:
                task_display_context = None
                self._current_task_displays[task.task_id] = None

            # Initialize progress tracking using inspect-ai's progress context manager
            if task_display_context is not None and hasattr(task_display_context, "progress"):
                try:
                    progress_context = task_display_context.progress()
                    progress = progress_context.__enter__()
                    self._current_progress_contexts[task.task_id] = {
                        "context": progress_context,
                        "progress": progress,
                        "completed_steps": 0,
                        "total_steps": saber_profile.steps,
                        "supports_progress": True,
                    }
                    logger.debug(f"Progress tracking enabled for task {task.task_id}")
                except Exception as e:
                    logger.warning(f"Progress tracking not available for task {task.task_id}: {e}")
                    self._current_progress_contexts[task.task_id] = {
                        "context": None,
                        "progress": None,
                        "completed_steps": 0,
                        "total_steps": saber_profile.steps,
                        "supports_progress": False,
                    }

            self._safe_display_print(f"📋 TASK START: {task.task_id}")
            self._safe_display_print(f"   ├─ Name: {task.name}")
            self._safe_display_print(f"   ├─ Status: {task.status.value}")
            progress_enabled = self._current_progress_contexts.get(task.task_id, {}).get("supports_progress", False)
            self._safe_display_print(f"   ├─ Progress Tracking: {'enabled' if progress_enabled else 'disabled'}")
            self._safe_display_print("   └─ Starting...")

            logger.info(
                f"Task {task.task_id} started with {saber_profile.steps} estimated steps "
                f"(progress tracking: {'enabled' if progress_enabled else 'disabled'})"
            )

        except Exception as e:
            logger.error(f"Failed to show progress for task {task.task_id}: {e}")
            raise

    def update_task_status(self, task: TaskInfo) -> None:
        """Update task status and progress."""
        try:
            # For textual mode, provide simple status updates
            if self.backend_mode == InspectAIBackendMode.TEXTUAL:
                progress_pct = task.progress * 100
                self._safe_display_print(f"🔄 TASK UPDATE: {task.task_id}")
                self._safe_display_print(f"   ├─ Status: {task.status.value}")
                self._safe_display_print(f"   ├─ Progress: {progress_pct:.1f}%")
                self._safe_display_print("   └─ Use real TUI for detailed progress")
                return

            progress_info = self._current_progress_contexts.get(task.task_id)
            if progress_info and progress_info["supports_progress"]:
                # Calculate steps based on progress percentage
                total_steps = progress_info["total_steps"]
                current_step = int(task.progress * total_steps)

                # Update progress if we have advanced
                if current_step > progress_info["completed_steps"]:
                    steps_to_advance = current_step - progress_info["completed_steps"]
                    try:
                        for _ in range(steps_to_advance):
                            # Try different progress update methods
                            progress = progress_info["progress"]
                            if hasattr(progress, "advance"):
                                progress.advance()
                            elif hasattr(progress, "update"):
                                progress.update(current_step)
                            elif hasattr(progress, "step"):
                                progress.step()
                            else:
                                # If none of the expected methods exist, log available methods
                                available_methods = [m for m in dir(progress) if not m.startswith("_")]
                                logger.debug(f"Available progress methods: {available_methods}")
                                break
                        progress_info["completed_steps"] = current_step
                    except Exception as e:
                        logger.warning(f"Failed to update progress steps for task {task.task_id}: {e}")

            # Display status update
            current_step = progress_info["completed_steps"] if progress_info else int(task.progress * 10)
            total_steps = progress_info["total_steps"] if progress_info else 10
            progress_pct = task.progress * 100

            self._safe_display_print(f"🔄 TASK UPDATE: {task.task_id}")
            self._safe_display_print(f"   ├─ Status: {task.status.value}")
            self._safe_display_print(f"   ├─ Progress: {current_step}/{total_steps} steps ({progress_pct:.1f}%)")
            self._safe_display_print("   └─ Processing...")

        except Exception as e:
            logger.error(f"Failed to update task status for {task.task_id}: {e}")
            raise

    def show_results(self, results: Dict[str, Any]) -> None:
        """Show results."""
        try:
            self._safe_display_print("📊 RESULTS:")
            for key, value in results.items():
                self._safe_display_print(f"   • {key}: {value}")
        except Exception as e:
            logger.error(f"Failed to show results: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up all resources."""
        try:
            # Clean up progress contexts first
            for task_id, progress_info in list(self._current_progress_contexts.items()):
                try:
                    if progress_info["context"] and hasattr(progress_info["context"], "__exit__"):
                        progress_info["context"].__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error cleaning up progress context for task {task_id}: {e}")

            # Clean up task displays
            for task_id, task_display in list(self._current_task_displays.items()):
                try:
                    if hasattr(task_display, "__exit__"):
                        task_display.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error cleaning up task display for task {task_id}: {e}")

            # Clear all tracking
            self._current_task_displays.clear()
            self._current_progress_contexts.clear()
            self._task_metrics.clear()
            self._active_tasks.clear()

            if self._display:
                self._safe_display_print("🧹 Inspect-AI cleanup complete")
        except Exception as e:
            logger.error(f"Failed to cleanup inspect-ai backend: {e}")
            raise

    # Session management methods for compatibility with UIManager
    async def session_start(
        self, session_id: str, total_tasks: int = 0, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start session."""
        if self.backend_mode == InspectAIBackendMode.TEXTUAL:
            # For textual mode, we provide a detailed simulation
            await self._run_textual_session_simulation(session_id, total_tasks, metadata)
        elif self._display:
            self._safe_display_print(f"🔍 SESSION START ({self.backend_mode.value} mode)")
            self._safe_display_print(f"Session ID: {session_id}")
            self._safe_display_print(f"Total Tasks: {total_tasks}")
            self._safe_display_print("=" * 60)

    async def _run_textual_session_simulation(
        self, session_id: str, total_tasks: int, metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Run a textual UI session with guidance on using the real TUI."""

        logger.info("Textual mode requested - providing guidance for real TUI usage")

        print("🖥️  INSPECT-AI TEXTUAL MODE")
        print("=" * 60)
        print(f"📋 Session ID: {session_id}")
        print(f"📊 Total Tasks: {total_tasks}")
        if metadata:
            print(f"📁 Metadata: {metadata}")
        print()
        print("💡 REAL TUI AVAILABLE!")
        print("The inspect-ai package provides a beautiful real TUI interface.")
        print("To use it, run one of these commands:")
        print()
        print("  # Run the real TUI demo:")
        print("  python examples/real_tui_demo.py")
        print()
        print("  # Or use the textual session directly:")
        print("  from saber.client.ui.textual_session import run_saber_textual_demo")
        print("  result = run_saber_textual_demo()")
        print()
        print("🎯 The real TUI provides:")
        print("  • Interactive task dashboard")
        print("  • Real-time progress bars")
        print("  • Live console output")
        print("  • Keyboard shortcuts (F1 Help, Q Quit)")
        print("  • Mouse support")
        print("  • Proper terminal application interface")
        print()
        print("⚠️  NOTE: The real TUI must run in the main thread")
        print("   (cannot be launched from within async demo context)")
        print("=" * 60)

        await asyncio.sleep(1)
        logger.info(f"Textual guidance provided for session {session_id}")

    async def session_complete(
        self, session_id: str, successful_episodes: int = 0, total_episodes: int = 0, **kwargs: Any
    ) -> None:
        """Complete session."""
        if self._display:
            success_rate = (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0
            self._safe_display_print("=" * 60)
            self._safe_display_print(f"🎉 SESSION COMPLETE ({self.backend_mode.value} mode)")
            self._safe_display_print(f"Success Rate: {success_rate:.1f}%")
            self._safe_display_print("=" * 60)

    # Tool call progress methods for real-time MCP sidecar updates
    async def tool_call_start(self, tool_call: Dict[str, Any]) -> None:
        """Handle tool call start event."""
        try:
            call_id = tool_call.get("call_id", "unknown")
            tool_name = tool_call.get("tool_name", "unknown")
            args = tool_call.get("arguments") or tool_call.get("input_args")

            if self._display:
                if args is None:
                    args_str = "({})"
                else:
                    try:
                        args_str = f"({args})"
                    except Exception:
                        args_str = "(<unprintable-args>)"
                self._safe_display_print(f"🔧 Starting: {tool_name}{args_str} ({call_id[:8]}...)")

        except Exception as e:
            logger.error(f"Failed to handle tool call start: {e}")

    async def tool_call_progress(self, tool_call: Dict[str, Any]) -> None:
        """Handle tool call progress event."""
        try:
            # call_id = tool_call.get("call_id", "unknown")  # Not used currently
            tool_name = tool_call.get("tool_name", "unknown")
            progress_info = tool_call.get("progress_info", "executing...")

            if self._display:
                self._safe_display_print(f"🔄 {tool_name}: {progress_info}")

        except Exception as e:
            logger.error(f"Failed to handle tool call progress: {e}")

    async def tool_call_complete(self, tool_call: Dict[str, Any]) -> None:
        """Handle tool call completion event."""
        try:
            # call_id = tool_call.get("call_id", "unknown")  # Not used currently
            tool_name = tool_call.get("tool_name", "unknown")
            success = tool_call.get("result", {}).get("success", True)
            args = tool_call.get("arguments") or tool_call.get("input_args")
            output = tool_call.get("output")
            execution_time_ms = tool_call.get("execution_time_ms")

            status = "✅" if success else "❌"
            time_str = f" ({execution_time_ms}ms)" if execution_time_ms else ""

            if self._display:
                if args is None:
                    args_disp = "{}"
                else:
                    try:
                        args_disp = f"{args}"
                    except Exception:
                        args_disp = "<unprintable-args>"
                if success:
                    if output is None:
                        out_disp = "<no output>"
                    else:
                        try:
                            out_disp = str(output)
                        except Exception:
                            out_disp = "<unprintable-output>"
                    suffix = f"{tool_name}({args_disp}) -> {out_disp}"
                else:
                    suffix = f"{tool_name}({args_disp})"
                self._safe_display_print(f"{status} Completed: {suffix}{time_str}")

        except Exception as e:
            logger.error(f"Failed to handle tool call completion: {e}")
