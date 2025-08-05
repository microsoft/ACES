"""
SABER Progress Management

Progress indicators and spinners for long-running operations.
"""

from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator, Optional

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .console import get_console


class ProgressManager:
    """
    Manages progress indicators for SABER operations.

    Provides context managers for both sync and async operations
    with automatic fallback when rich is disabled.
    """

    def __init__(self) -> None:
        self.console = get_console()
        self._active_progress: Optional[Progress] = None

    @contextmanager
    def spinner(self, description: str) -> Generator[None, None, None]:
        """
        Context manager for spinner progress indicator.

        Args:
            description: Description text to show
        """
        if self.console.rich_enabled:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console.console,
            ) as progress:
                progress.add_task(description, total=None)
                yield
        else:
            # Fallback to simple text
            self.console.print(f"⏳ {description}...")
            yield
            self.console.print("✓ Done")

    @asynccontextmanager
    async def async_spinner(self, description: str) -> AsyncGenerator[None, None]:
        """
        Async context manager for spinner progress indicator.

        Args:
            description: Description text to show
        """
        if self.console.rich_enabled:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console.console,
            ) as progress:
                progress.add_task(description, total=None)
                yield
        else:
            # Fallback to simple text
            self.console.print(f"⏳ {description}...")
            yield
            self.console.print("✓ Done")

    @contextmanager
    def bar(self, description: str, total: int) -> Generator["TaskHandle", None, None]:
        """
        Context manager for progress bar.

        Args:
            description: Description text to show
            total: Total number of items to process
        """
        if self.console.rich_enabled:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=self.console.console,
            ) as progress:
                task_id = progress.add_task(description, total=total)
                yield TaskHandle(progress, task_id)
        else:
            # Fallback to simple counter
            yield TaskHandle(None, None, description, total)

    def simple_status(self, message: str) -> None:
        """Display a simple status message."""
        self.console.print(f"⏳ {message}...")


class TaskHandle:
    """Handle for updating progress tasks."""

    def __init__(
        self, progress: Optional[Progress], task_id: Optional[Any], description: str = "", total: int = 0
    ) -> None:
        self.progress = progress
        self.task_id = task_id
        self.description: str = description
        self.total: int = total
        self.completed: int = 0
        self.console = get_console()

    def advance(self, amount: int = 1) -> None:
        """Advance the progress by the specified amount."""
        if self.progress and self.task_id is not None:
            self.progress.advance(self.task_id, amount)
        else:
            # Fallback progress display
            self.completed += amount
            if self.total > 0:
                percent = (self.completed / self.total) * 100
                self.console.print(f"Progress: {self.completed}/{self.total} ({percent:.1f}%)")

    def update(self, completed: Optional[int] = None, description: Optional[str] = None) -> None:
        """Update progress task."""
        if self.progress and self.task_id is not None:
            # Use Rich Progress.update() with individual parameters
            if completed is not None and description is not None:
                self.progress.update(self.task_id, completed=completed, description=description)
            elif completed is not None:
                self.progress.update(self.task_id, completed=completed)
            elif description is not None:
                self.progress.update(self.task_id, description=description)
        else:
            # Fallback update
            if completed is not None:
                self.completed = completed
            if description is not None:
                self.description = description
                self.console.print(f"📋 {description}")


# Global progress manager instance
progress_manager = ProgressManager()


def get_progress_manager() -> ProgressManager:
    """Get the global progress manager instance."""
    return progress_manager
