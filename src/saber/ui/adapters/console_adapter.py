#!/usr/bin/env python3
"""
Console UI Adapter

Rich console output for SABER UI without inspect-ai dependencies.
Provides enhanced terminal UI using rich library.
"""

from typing import Any, Dict, Optional

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..interfaces import (
    MCPToolCall,
    SABERTaskState,
    SABERUISessionInfo,
    SABERUISessionSummary,
    SABERUITaskInfo,
    SABERUITaskResult,
)


class ConsoleUIAdapter:
    """Rich console UI adapter using rich library."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize console adapter."""
        self.config = config or {}
        self._fallback = None

        if not RICH_AVAILABLE:
            # Fall back to null adapter
            from .null_adapter import NullUIAdapter

            self._fallback = NullUIAdapter()
            self.console = None  # type: Optional[Console]
            self.progress = None  # type: Optional[Progress]
            self.live = None  # type: Optional[Live]
            return

        # Initialize rich components
        self.console = Console()
        self.progress = Progress(
            TextColumn("[bold blue]{task.fields[task_id]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
        )
        self.live = None
        self.active_tasks: Dict[str, Any] = {}

    async def session_start(self, session_info: SABERUISessionInfo) -> None:
        """Initialize session with rich display."""
        if self._fallback:
            return await self._fallback.session_start(session_info)

        if not self.console or not self.progress:
            return

        # Create session header
        session_panel = Panel(
            f"[bold green]SABER Benchmark Session[/bold green]\n"
            f"Session ID: [cyan]{session_info.session_id}[/cyan]\n"
            f"Total Tasks: [yellow]{session_info.total_tasks}[/yellow]\n"
            f"Parallel Episodes: [blue]{session_info.parallel_episodes}[/blue]\n"
            f"Tool Detail Level: [magenta]{session_info.ui_tool_detail_level}[/magenta]",
            title="🚀 Session Started",
            border_style="green",
        )
        self.console.print(session_panel)
        self.console.print()

        # Start live display with progress
        self.live = Live(self.progress, console=self.console, refresh_per_second=4)
        self.live.start()

    async def session_update(self, session_info: SABERUISessionInfo) -> None:
        """Update session information."""
        if self._fallback:
            return await self._fallback.session_update(session_info)

        if not self.console:
            return

        # Could update header, but for now just log
        self.console.log(f"📊 Session updated - Total tasks: {session_info.total_tasks}")

    async def task_start(self, task_info: SABERUITaskInfo) -> None:
        """Start tracking a new task."""
        if self._fallback:
            return await self._fallback.task_start(task_info)

        if not self.console or not self.progress:
            return

        # Add progress bar for this task
        task_id = self.progress.add_task(
            description=f"Starting {task_info.task_id}...", total=100, task_id=task_info.task_id, status="🎬 Starting"
        )
        self.active_tasks[task_info.task_id] = task_id

        self.console.log(f"📋 Started task: [bold]{task_info.task_id}[/bold] (attempt {task_info.attempt})")

    async def task_update(self, task_info: SABERUITaskInfo) -> None:
        """Update task progress."""
        if self._fallback:
            return await self._fallback.task_update(task_info)

        if not self.progress or task_info.task_id not in self.active_tasks:
            return

        task_id = self.active_tasks[task_info.task_id]

        # Update progress based on state
        progress_mapping = {
            SABERTaskState.STARTING: (10, "🎬 Starting"),
            SABERTaskState.RUNNING: (50, "🔄 Running"),
            SABERTaskState.COMPLETED: (100, "✅ Complete"),
            SABERTaskState.FAILED: (100, "❌ Failed"),
        }

        progress, status = progress_mapping.get(task_info.state, (25, "🔄 Processing"))

        self.progress.update(
            task_id,
            completed=progress,
            status=status,
            description=task_info.progress_message or f"Task {task_info.task_id}",
        )

    async def tool_call_start(self, tool_call: MCPToolCall) -> None:
        """Log tool call start."""
        if self._fallback:
            return await self._fallback.tool_call_start(tool_call)

        if not self.console:
            return

        self.console.log(
            f"🔧 Tool call started: [bold cyan]{tool_call.tool_name}[/bold cyan] (ID: {tool_call.call_id})"
        )

    async def tool_call_complete(self, tool_call: MCPToolCall) -> None:
        """Log tool call completion."""
        if self._fallback:
            return await self._fallback.tool_call_complete(tool_call)

        if not self.console:
            return

        status_emoji = "✅" if tool_call.status.value == "completed" else "❌"
        self.console.log(f"{status_emoji} Tool call complete: [bold cyan]{tool_call.tool_name}[/bold cyan]")

    async def task_complete(self, result: SABERUITaskResult) -> None:
        """Complete task with result summary."""
        if self._fallback:
            return await self._fallback.task_complete(result)

        if not self.console or not self.progress:
            return

        # Update progress to completed state
        if result.task_id in self.active_tasks:
            task_id = self.active_tasks[result.task_id]
            status = "✅ Success" if result.success else "❌ Failed"

            self.progress.update(task_id, completed=100, status=status, description=f"Complete: {result.task_id}")

        # Log detailed result
        status_emoji = "✅" if result.success else "❌"
        result_text = f"{status_emoji} [bold]{result.task_id}[/bold] completed"

        if result.flag:
            result_text += f" | Flag: [green]{result.flag}[/green]"
        if result.error_message:
            result_text += f" | Error: [red]{result.error_message}[/red]"

        self.console.log(result_text)

    async def session_complete(self, summary: SABERUISessionSummary) -> None:
        """Complete session with summary."""
        if self._fallback:
            return await self._fallback.session_complete(summary)

        if not self.console:
            return

        # Stop live display
        if self.live:
            self.live.stop()

        # Show final summary
        success_rate = (summary.successful_episodes / summary.total_episodes * 100) if summary.total_episodes > 0 else 0

        summary_panel = Panel(
            f"[bold blue]Session Complete[/bold blue]\n"
            f"Session ID: [cyan]{summary.session_id}[/cyan]\n"
            f"Total Episodes: [yellow]{summary.total_episodes}[/yellow]\n"
            f"Successful: [green]{summary.successful_episodes}[/green]\n"
            f"Success Rate: [{'green' if success_rate >= 50 else 'red'}]{success_rate:.1f}%[/]\n"
            f"Final Result: [{'green' if summary.final_success else 'red'}]"
            f"{'SUCCESS' if summary.final_success else 'FAILURE'}[/]",
            title="🎉 Results",
            border_style="blue",
        )
        self.console.print(summary_panel)

    async def session_cleanup(self) -> None:
        """Clean up session resources."""
        if self._fallback:
            return await self._fallback.session_cleanup()

        if not self.console:
            return

        if self.live:
            self.live.stop()

        self.console.log("🧹 Session cleanup complete")

    async def __aenter__(self) -> "ConsoleUIAdapter":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Async context manager exit."""
        if self.live:
            self.live.stop()
        return False
