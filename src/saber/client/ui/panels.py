"""
SABER Panel Formatters

Panel and container formatting for structured information display.
"""

from typing import Any, Dict, Optional

from rich.panel import Panel

from .console import get_console


class PanelFormatter:
    """
    Formats information panels for SABER client output.

    Provides consistent panel styling with fallback to plain text.
    """

    def __init__(self) -> None:
        self.console = get_console()

    def connection_panel(
        self,
        server_url: str,
        session_id: Optional[str] = None,
        client_id: Optional[str] = None,
        status: str = "Connected",
    ) -> None:
        """Display connection information panel."""
        content_lines = [f"Server: {server_url}"]

        if session_id:
            content_lines.append(f"Session ID: {session_id}")
        if client_id:
            content_lines.append(f"Client ID: {client_id}")

        content = "\n".join(content_lines)

        if self.console.rich_enabled:
            if status == "Connected":
                panel = Panel(content, title="🔗 Server Connection", border_style="green", title_align="left")
            else:
                panel = Panel(content, title="❌ Connection Failed", border_style="red", title_align="left")
            self.console.console.print(panel)
        else:
            # Fallback to simple output
            self.console.print("\n=== Server Connection ===")
            for line in content_lines:
                self.console.print(f"  {line}")
            self.console.print("")

    def task_panel(
        self, task_title: str, task_description: str, progress: Optional[str] = None, status: str = "active"
    ) -> None:
        """Display task information panel."""
        content_lines = [f"[bold green]{task_title}[/bold green]" if self.console.rich_enabled else task_title]
        content_lines.append("")
        content_lines.append(task_description)

        if progress:
            content_lines.append("")
            content_lines.append(f"Progress: {progress}")

        content = "\n".join(content_lines)

        if self.console.rich_enabled:
            if status == "active":
                border_style = "blue"
                title = "🎯 Task Started"
            elif status == "completed":
                border_style = "green"
                title = "✅ Task Completed"
            else:
                border_style = "yellow"
                title = "📋 Task Status"

            panel = Panel(content, title=title, border_style=border_style, title_align="left")
            self.console.console.print(panel)
        else:
            # Fallback to simple output
            self.console.print(f"\n=== Task: {task_title} ===")
            self.console.print(f"  Description: {task_description}")
            if progress:
                self.console.print(f"  Progress: {progress}")
            self.console.print("")

    def config_panel(self, config: Dict[str, Any], title: str = "Configuration") -> None:
        """Display configuration information panel."""
        content_lines = []
        for key, value in config.items():
            # Format key nicely
            display_key = key.replace("_", " ").title()
            content_lines.append(f"{display_key}: {value}")

        content = "\n".join(content_lines)

        if self.console.rich_enabled:
            panel = Panel(content, title=f"🤖 {title}", border_style="cyan", title_align="left")
            self.console.console.print(panel)
        else:
            # Fallback to simple output
            self.console.print(f"\n=== {title} ===")
            for line in content_lines:
                self.console.print(f"  {line}")
            self.console.print("")

    def error_panel(self, error_message: str, details: Optional[str] = None) -> None:
        """Display error information panel."""
        content_lines = [error_message]
        if details:
            content_lines.append("")
            content_lines.append(details)

        content = "\n".join(content_lines)

        if self.console.rich_enabled:
            panel = Panel(content, title="❌ Error", border_style="red", title_align="left")
            self.console.console.print(panel)
        else:
            # Fallback to simple output
            self.console.print("\n=== ERROR ===")
            for line in content_lines:
                self.console.print(f"  {line}")
            self.console.print("")

    def info_panel(self, message: str, title: str = "Information") -> None:
        """Display general information panel."""
        if self.console.rich_enabled:
            panel = Panel(message, title=f"ℹ️ {title}", border_style="blue", title_align="left")
            self.console.console.print(panel)
        else:
            # Fallback to simple output
            self.console.print(f"\n=== {title} ===")
            self.console.print(f"  {message}")
            self.console.print("")

    def results_summary(self, summary_data: Dict[str, Any], title: str = "Results Summary") -> None:
        """Display results summary panel."""
        content_lines = []

        for key, value in summary_data.items():
            display_key = key.replace("_", " ").title()
            if isinstance(value, list):
                content_lines.append(f"{display_key}: {len(value)} items")
                for item in value[:3]:  # Show first 3 items
                    content_lines.append(f"  • {item}")
                if len(value) > 3:
                    content_lines.append(f"  ... and {len(value) - 3} more")
            else:
                content_lines.append(f"{display_key}: {value}")

        content = "\n".join(content_lines)

        if self.console.rich_enabled:
            panel = Panel(content, title=f"📊 {title}", border_style="green", title_align="left")
            self.console.console.print(panel)
        else:
            # Fallback to simple output
            self.console.print(f"\n=== {title} ===")
            for line in content_lines:
                self.console.print(f"  {line}")
            self.console.print("")


# Global panel formatter instance
panel_formatter = PanelFormatter()


def get_panel_formatter() -> PanelFormatter:
    """Get the global panel formatter instance."""
    return panel_formatter
