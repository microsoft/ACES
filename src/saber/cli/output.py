"""Rich formatting helpers for the SABER CLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from saber.cli.discovery import ComposeProject, DiscoveredDomain
from saber.environments.images import PreflightResult

# Output goes to stderr to keep stdout clean for piping/scripting
console = Console(stderr=True)


def domain_table(domains: list[DiscoveredDomain]) -> Table:
    """Build a Rich table listing discovered domains.

    Args:
        domains: List of discovered domains.

    Returns:
        A Rich Table ready for printing.
    """
    table = Table(title="SABER Domains")
    table.add_column("Slug", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Path", style="dim")
    table.add_column("Permanent Env", style="green")

    for d in domains:
        perm = d.permanent_project or "[dim]none[/dim]"
        table.add_row(d.slug, d.name, str(d.root), perm)
    return table


def build_results_table(result: PreflightResult) -> Table:
    """Build a Rich table showing image build results.

    Args:
        result: Preflight result with image build outcomes.

    Returns:
        A Rich Table ready for printing.
    """
    table = Table(title=f"Build Results — {result.domain_slug or 'unknown'}")
    table.add_column("Image", style="cyan", no_wrap=True)
    table.add_column("Tag", style="dim")
    table.add_column("Action", no_wrap=True)

    action_styles = {
        "built": "[green]built[/green]",
        "rebuilt": "[green]rebuilt[/green]",
        "skipped": "[yellow]skipped[/yellow]",
        "failed": "[red]failed[/red]",
    }

    for r in result.results:
        action_display = action_styles.get(r.action, r.action)
        table.add_row(r.name, r.tag, action_display)
    return table


def print_success(message: str) -> None:
    """Print a green success panel."""
    console.print(Panel(message, style="green", title="Success"))


def print_error(message: str) -> None:
    """Print a red error panel."""
    console.print(Panel(message, style="red", title="Error"))


def print_warning(message: str) -> None:
    """Print a yellow warning panel."""
    console.print(Panel(message, style="yellow", title="Warning"))


def teardown_projects_table(projects: list[ComposeProject]) -> Table:
    """Build a Rich table listing SABER compose projects for teardown.

    Args:
        projects: List of compose projects to display.

    Returns:
        A Rich Table ready for printing.
    """
    table = Table(title="SABER Compose Projects")
    table.add_column("Project", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")
    for p in projects:
        table.add_row(p.name, p.status)
    return table
