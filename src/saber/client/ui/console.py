"""
SABER Console Interface

Core console interface for SABER client applications.
Provides rich formatting with fallback to plain text.
"""

import os
from typing import Any, Optional

from rich.console import Console
from rich.theme import Theme

# SABER color theme
SABER_THEME = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "highlight": "blue",
        "dim": "dim",
        "emphasis": "bold",
        "server": "blue",
        "task": "magenta",
        "step": "cyan",
        "finding": "blue",
        "suspicious": "red bold",
        "progress": "cyan",
    }
)


class SABERConsole:
    """
    SABER-themed console interface with rich formatting.

    Automatically falls back to plain text in non-interactive environments
    or when rich output is disabled.
    """

    def __init__(
        self, force_terminal: Optional[bool] = None, theme: Optional[Theme] = None, width: Optional[int] = None
    ):
        """
        Initialize SABER console.

        Args:
            force_terminal: Force terminal mode (None=auto-detect)
            theme: Custom theme (defaults to SABER theme)
            width: Console width override
        """
        # Check if rich output is disabled
        self.rich_enabled = self._should_enable_rich(force_terminal)

        if self.rich_enabled:
            self.console = Console(theme=theme or SABER_THEME, force_terminal=force_terminal, width=width)
        else:
            # Fallback console without rich formatting
            self.console = Console(no_color=True, force_terminal=False, legacy_windows=True)

    def _should_enable_rich(self, force_terminal: Optional[bool]) -> bool:
        """Determine if rich output should be enabled."""
        # Check environment variables
        if os.getenv("SABER_NO_COLOR", "").lower() in ("1", "true", "yes"):
            return False
        if os.getenv("NO_COLOR", ""):
            return False
        if os.getenv("CI", "").lower() in ("1", "true", "yes"):
            return False

        # Use force_terminal if specified
        if force_terminal is not None:
            return force_terminal

        # Auto-detect based on terminal capabilities
        return True

    def print(self, *objects: Any, style: Optional[str] = None, **kwargs: Any) -> None:
        """Print with optional rich styling."""
        if style and self.rich_enabled:
            # Apply style markup
            formatted_objects = []
            for obj in objects:
                if isinstance(obj, str):
                    formatted_objects.append(f"[{style}]{obj}[/{style}]")
                else:
                    formatted_objects.append(obj)
            self.console.print(*formatted_objects, **kwargs)
        else:
            # Plain output
            self.console.print(*objects, **kwargs)

    def success(self, message: str, **kwargs: Any) -> None:
        """Print success message."""
        icon = "✅" if self.rich_enabled else "[OK]"
        self.print(f"{icon} {message}", style="success", **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Print error message."""
        icon = "❌" if self.rich_enabled else "[ERROR]"
        self.print(f"{icon} {message}", style="error", **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Print warning message."""
        icon = "⚠️" if self.rich_enabled else "[WARNING]"
        self.print(f"{icon} {message}", style="warning", **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Print info message."""
        icon = "ℹ️" if self.rich_enabled else "[INFO]"
        self.print(f"{icon} {message}", style="info", **kwargs)

    def server_status(self, message: str, **kwargs: Any) -> None:
        """Print server-related status."""
        self.print(message, style="server", **kwargs)

    def task_status(self, message: str, **kwargs: Any) -> None:
        """Print task-related status."""
        self.print(message, style="task", **kwargs)

    def step_status(self, message: str, **kwargs: Any) -> None:
        """Print step execution status."""
        self.print(message, style="step", **kwargs)

    def finding(self, message: str, **kwargs: Any) -> None:
        """Print investigation findings."""
        icon = "📋" if self.rich_enabled else "[FINDING]"
        self.print(f"{icon} {message}", style="finding", **kwargs)

    def suspicious(self, message: str, **kwargs: Any) -> None:
        """Print suspicious indicators."""
        icon = "⚠️" if self.rich_enabled else "[SUSPICIOUS]"
        self.print(f"{icon} {message}", style="suspicious", **kwargs)

    def dim(self, message: str, **kwargs: Any) -> None:
        """Print dimmed text."""
        self.print(message, style="dim", **kwargs)

    def emphasis(self, message: str, **kwargs: Any) -> None:
        """Print emphasized text."""
        self.print(message, style="emphasis", **kwargs)

    def rule(self, title: Optional[str] = None, **kwargs: Any) -> None:
        """Print a horizontal rule."""
        if self.rich_enabled and hasattr(self.console, "rule"):
            self.console.rule(title or "", **kwargs)
        else:
            # Fallback to simple line
            line = "=" * 80
            if title:
                self.print(f"\n{title}")
                self.print(line)
            else:
                self.print(line)


# Global console instance
console = SABERConsole()


def get_console() -> SABERConsole:
    """Get the global SABER console instance."""
    return console


def set_console_config(
    force_terminal: Optional[bool] = None, theme: Optional[Theme] = None, width: Optional[int] = None
) -> SABERConsole:
    """
    Configure the global console instance.

    Args:
        force_terminal: Force terminal mode
        theme: Custom theme
        width: Console width

    Returns:
        Configured console instance
    """
    global console
    console = SABERConsole(force_terminal=force_terminal, theme=theme, width=width)
    return console
