"""
SABER Table Formatters

Table formatting for structured data display.
"""

from typing import Any, Dict, List

from rich.box import ROUNDED
from rich.table import Table

from .console import get_console


class TableFormatter:
    """
    Formats tables for SABER client output.

    Provides rich table formatting with fallback to plain text.
    """

    def __init__(self) -> None:
        self.console = get_console()

    def task_list_table(self, tasks: List[Dict[str, Any]]) -> None:
        """Display available tasks in a table."""
        if not tasks:
            self.console.info("No tasks available")
            return

        if self.console.rich_enabled:
            table = Table(title="Available Tasks", box=ROUNDED, show_header=True, header_style="bold cyan")
            table.add_column("Task ID", style="magenta", no_wrap=True)
            table.add_column("Title", style="blue")
            table.add_column("Description", style="dim")
            table.add_column("Status", justify="center")

            for task in tasks:
                status = "✅ Ready" if task.get("available", True) else "❌ Unavailable"
                table.add_row(
                    task.get("task_id", "N/A"),
                    task.get("title", "N/A"),
                    (
                        task.get("description", "")[:50] + "..."
                        if len(task.get("description", "")) > 50
                        else task.get("description", "")
                    ),
                    status,
                )

            self.console.console.print(table)
        else:
            # Fallback to simple text table
            self.console.print("\n=== Available Tasks ===")
            for i, task in enumerate(tasks, 1):
                self.console.print(f"{i}. {task.get('task_id', 'N/A')}")
                self.console.print(f"   Title: {task.get('title', 'N/A')}")
                if task.get("description"):
                    desc = task["description"][:80] + "..." if len(task["description"]) > 80 else task["description"]
                    self.console.print(f"   Description: {desc}")
                self.console.print("")

    def execution_results_table(self, results: List[Dict[str, Any]]) -> None:
        """Display command execution results in a table."""
        if not results:
            self.console.warning("No execution results to display")
            return

        if self.console.rich_enabled:
            table = Table(title="Command Execution Summary", box=ROUNDED, show_header=True, header_style="bold cyan")
            table.add_column("Step", justify="center", style="cyan", width=6)
            table.add_column("Command", style="magenta", width=12)
            table.add_column("Status", justify="center", width=10)
            table.add_column("Key Findings", style="blue")

            for i, result in enumerate(results, 1):
                # Extract status
                if result.get("success", True):
                    status = "[green]✅ Success[/green]"
                else:
                    status = "[red]❌ Failed[/red]"

                # Extract key findings
                analysis = result.get("analysis", {})
                findings = analysis.get("findings", [])
                key_findings = ", ".join(findings[:2]) if findings else "No specific findings"
                if len(findings) > 2:
                    key_findings += f" (+{len(findings) - 2} more)"

                table.add_row(str(i), result.get("command", "N/A"), status, key_findings)

            self.console.console.print(table)
        else:
            # Fallback to simple text table
            self.console.print("\n=== Command Execution Summary ===")
            for i, result in enumerate(results, 1):
                status = "SUCCESS" if result.get("success", True) else "FAILED"
                self.console.print(f"{i}. {result.get('command', 'N/A')} - {status}")

                analysis = result.get("analysis", {})
                findings = analysis.get("findings", [])
                if findings:
                    self.console.print(f"   Findings: {', '.join(findings[:2])}")
                self.console.print("")

    def subtask_progress_table(self, subtasks: List[Dict[str, Any]]) -> None:
        """Display subtask progress in a table."""
        if not subtasks:
            return

        if self.console.rich_enabled:
            table = Table(title="Subtask Progress", box=ROUNDED, show_header=True, header_style="bold cyan")
            table.add_column("Subtask", style="blue", width=20)
            table.add_column("Status", justify="center", width=12)
            table.add_column("Progress", justify="center", width=10)
            table.add_column("Description", style="dim")

            for subtask in subtasks:
                # Determine status
                if subtask.get("completed", False):
                    status = "[green]✅ Complete[/green]"
                elif subtask.get("in_progress", False):
                    status = "[yellow]🔄 In Progress[/yellow]"
                else:
                    status = "[dim]⏸️ Pending[/dim]"

                # Calculate progress
                progress = "100%" if subtask.get("completed", False) else "0%"

                table.add_row(
                    subtask.get("subtask_id", "N/A"),
                    status,
                    progress,
                    (
                        subtask.get("description", "")[:40] + "..."
                        if len(subtask.get("description", "")) > 40
                        else subtask.get("description", "")
                    ),
                )

            self.console.console.print(table)
        else:
            # Fallback to simple text
            self.console.print("\n=== Subtask Progress ===")
            for subtask in subtasks:
                status = "COMPLETE" if subtask.get("completed", False) else "PENDING"
                self.console.print(f"• {subtask.get('subtask_id', 'N/A')} - {status}")
            self.console.print("")

    def findings_table(self, findings: List[str], suspicious: List[str], recommendations: List[str]) -> None:
        """Display security findings in a structured table."""
        if not any([findings, suspicious, recommendations]):
            self.console.info("No findings to display")
            return

        if self.console.rich_enabled:
            # Findings table
            if findings:
                table = Table(title="Investigation Findings", box=ROUNDED, show_header=True, header_style="bold blue")
                table.add_column("Type", style="cyan", width=12)
                table.add_column("Finding", style="blue")

                for finding in findings:
                    table.add_row("📋 Finding", finding)

                self.console.console.print(table)

            # Suspicious indicators table
            if suspicious:
                table = Table(title="Security Concerns", box=ROUNDED, show_header=True, header_style="bold red")
                table.add_column("Type", style="red", width=12)
                table.add_column("Indicator", style="red bold")

                for indicator in suspicious:
                    table.add_row("⚠️ Suspicious", indicator)

                self.console.console.print(table)

            # Recommendations table
            if recommendations:
                table = Table(title="Recommendations", box=ROUNDED, show_header=True, header_style="bold green")
                table.add_column("Priority", style="green", width=12)
                table.add_column("Recommendation", style="green")

                for rec in recommendations:
                    table.add_row("📋 Action", rec)

                self.console.console.print(table)
        else:
            # Fallback to simple lists
            if findings:
                self.console.print("\n=== Investigation Findings ===")
                for i, finding in enumerate(findings, 1):
                    self.console.print(f"{i}. {finding}")

            if suspicious:
                self.console.print("\n=== Security Concerns ===")
                for i, indicator in enumerate(suspicious, 1):
                    self.console.print(f"{i}. {indicator}")

            if recommendations:
                self.console.print("\n=== Recommendations ===")
                for i, rec in enumerate(recommendations, 1):
                    self.console.print(f"{i}. {rec}")

            self.console.print("")


# Global table formatter instance
table_formatter = TableFormatter()


def get_table_formatter() -> TableFormatter:
    """Get the global table formatter instance."""
    return table_formatter
