"""SABER CLI — operational commands for Docker environments and images.

Entry point: ``uv run saber <command>``
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from saber.cli.discovery import ComposeProject, discover_domains, find_domains_root, resolve_domain
from saber.cli.output import (
    build_results_table,
    console,
    print_error,
    print_success,
    print_warning,
    teardown_projects_table,
)
from saber.cli.workspace import create_eval_workspace
from saber.config.loader import ConfigLoader
from saber.environments.images import RebuildMode, build_domain_images
from saber.logging import get_logger
from saber.sandbox import _start_permanent_services

logger = get_logger(__name__)

app = typer.Typer(
    name="saber",
    help="SABER — Security Agent Benchmarking and Evaluation Research CLI",
    no_args_is_help=True,
)


@app.command("new-eval-workspace")  # type: ignore[misc]
def new_eval_workspace(
    directory: Annotated[Path, typer.Argument(help="Path to a new, non-existent workspace directory")],
    demo_domain: Annotated[
        bool,
        typer.Option("--demo-domain/--no-demo-domain", help="Create the starter demo domain scaffold"),
    ] = True,
) -> None:
    """Create a fresh uv-managed SABER evaluation workspace."""
    try:
        workspace = create_eval_workspace(directory, include_demo_domain=demo_domain)
    except (RuntimeError, ValueError, OSError) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    if demo_domain:
        print_success(f"Created a new SABER evaluation workspace at {workspace}")
    else:
        print_success(f"Created a new SABER evaluation workspace at {workspace} without the starter demo domain")


@app.command()  # type: ignore[misc]
def build(
    domain: str | None = typer.Argument(None, help="Domain slug (omit to build all domains)"),
    image: str | None = typer.Option(None, "--image", "-i", help="Build only this image name"),
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="Force rebuild even if images exist"),
) -> None:
    """Build Docker images for SABER domains."""
    try:
        domains_root = find_domains_root()
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    if domain is not None:
        targets = [resolve_domain(domains_root, domain)]
    else:
        targets = discover_domains(domains_root)
        if not targets:
            print_error("No domains found")
            raise typer.Exit(code=1)

    # Determine rebuild mode
    if rebuild and image:
        mode = RebuildMode.specific(frozenset({image}))
    elif rebuild:
        mode = RebuildMode.all()
    else:
        mode = RebuildMode.none()

    _ACTION_STYLES: dict[str, str] = {
        "checking": "[dim]checking[/dim]",
        "building": "[yellow]building[/yellow]",
        "built": "[green]built[/green]",
        "rebuilt": "[green]rebuilt[/green]",
        "skipped": "[dim]skipped[/dim]",
        "failed": "[red]failed[/red]",
    }

    any_failed = False
    for target in targets:
        logger.info("Starting image build for domain '%s'", target.slug)
        console.print(f"\n[bold]Building images for [cyan]{target.slug}[/cyan]...[/bold]")

        def _on_progress(name: str, tag: str, event: str) -> None:
            styled = _ACTION_STYLES.get(event, event)
            console.print(f"  {styled}  [cyan]{name}[/cyan]  [dim]{tag}[/dim]")

        result = asyncio.run(
            build_domain_images(
                domain_root=target.root,
                rebuild=mode,
                on_progress=_on_progress,
            )
        )
        console.print(build_results_table(result))
        logger.info(
            "Image build completed for domain '%s' (all_succeeded=%s)",
            target.slug,
            result.all_succeeded,
        )
        if not result.all_succeeded:
            any_failed = True

    if any_failed:
        print_error("Some image builds failed")
        raise typer.Exit(code=1)
    else:
        print_success("All images built successfully")


async def _list_compose_projects() -> list[ComposeProject]:
    """Run ``docker compose ls -a --format json`` and parse output.

    Uses ``-a`` to include stopped/exited projects that still have
    leftover containers (e.g. orphaned Inspect AI sandboxes).
    """
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "compose",
        "ls",
        "-a",
        "--format",
        "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0 or not stdout:
        return []
    try:
        raw = json.loads(stdout.decode())
        return [ComposeProject(name=item.get("Name", ""), status=item.get("Status", "")) for item in raw]
    except json.JSONDecodeError:
        return []


async def _teardown_project(project_name: str) -> tuple[bool, str]:
    """Stop a compose project and remove volumes.

    When ``docker compose down`` fails because init containers (containers
    that ran to completion and were auto-removed) are "not found", falls
    back to force-removing all project containers via the Docker CLI, then
    cleaning up networks and volumes.

    Returns:
        Tuple of (success, stderr_text).
    """
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "compose",
        "-p",
        project_name,
        "down",
        "--volumes",
        "--remove-orphans",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    stderr_text = stderr.decode().strip()

    if proc.returncode == 0:
        return True, stderr_text

    # docker compose down fails when init containers that already exited have
    # been removed by Docker — it cannot find them to stop.  Fall back to raw
    # Docker commands that are tolerant of missing containers.
    if "not found" in stderr_text.lower():
        return await _force_teardown_project(project_name, stderr_text)

    return False, stderr_text


async def _force_teardown_project(
    project_name: str,
    original_stderr: str,
) -> tuple[bool, str]:
    """Force-remove all containers, networks and volumes for a project.

    Used as a fallback when ``docker compose down`` fails due to missing
    init containers.
    """
    label = f"com.docker.compose.project={project_name}"

    # 1. Force-remove all containers belonging to this project
    list_proc = await asyncio.create_subprocess_exec(
        "docker",
        "ps",
        "-aq",
        "--filter",
        f"label={label}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    list_stdout, _ = await list_proc.communicate()
    container_ids = list_stdout.decode().split()
    if container_ids:
        rm_proc = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            *container_ids,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await rm_proc.communicate()

    # 2. Remove project networks
    net_proc = await asyncio.create_subprocess_exec(
        "docker",
        "network",
        "ls",
        "--filter",
        f"label={label}",
        "-q",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    net_stdout, _ = await net_proc.communicate()
    network_ids = net_stdout.decode().split()
    if network_ids:
        netrm = await asyncio.create_subprocess_exec(
            "docker",
            "network",
            "rm",
            *network_ids,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await netrm.communicate()

    # 3. Remove project volumes
    vol_proc = await asyncio.create_subprocess_exec(
        "docker",
        "volume",
        "ls",
        "--filter",
        f"label={label}",
        "-q",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    vol_stdout, _ = await vol_proc.communicate()
    volume_ids = vol_stdout.decode().split()
    if volume_ids:
        volrm = await asyncio.create_subprocess_exec(
            "docker",
            "volume",
            "rm",
            "-f",
            *volume_ids,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await volrm.communicate()

    # Verify nothing remains
    verify = await asyncio.create_subprocess_exec(
        "docker",
        "ps",
        "-aq",
        "--filter",
        f"label={label}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    verify_stdout, _ = await verify.communicate()
    if verify_stdout.decode().strip():
        return False, original_stderr

    return True, original_stderr


def _is_saber_project(name: str, known_slugs: frozenset[str]) -> bool:
    """Check if a Docker Compose project name belongs to SABER.

    Matches projects that:
    - Contain "saber" in the name (e.g. ``saber-sandbox-abc123``)
    - Are ``{slug}-databases`` permanent-service projects
    - Are ``inspect-{slug}-*`` projects created by Inspect AI sandboxes
    """
    lower = name.lower()
    if "saber" in lower:
        return True
    for slug in known_slugs:
        if name == f"{slug}-databases":
            return True
        if lower.startswith(f"inspect-{slug}-"):
            return True
    return False


def _filter_by_domain(projects: list[ComposeProject], domain_slug: str) -> list[ComposeProject]:
    """Filter projects to those matching a specific domain via segment matching."""
    slug = domain_slug.lower()

    def matches(name: str) -> bool:
        lower = name.lower()
        return lower == slug or lower.startswith(f"{slug}-") or slug in lower.split("-")

    return [p for p in projects if matches(p.name)]


@app.command()  # type: ignore[misc]
def teardown(
    domain: str | None = typer.Argument(None, help="Domain slug (omit to tear down all)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Tear down Docker environments from abandoned experiments."""
    # Discover running compose projects
    all_projects = asyncio.run(_list_compose_projects())

    # Discover known SABER domains
    try:
        domains_root = find_domains_root()
    except FileNotFoundError:
        domains_root = None

    if domains_root is not None:
        known_domains = discover_domains(domains_root)
    else:
        known_domains = []
    known_slugs = frozenset(d.slug for d in known_domains)

    # Filter to SABER projects
    saber_projects = [p for p in all_projects if _is_saber_project(p.name, known_slugs)]

    # Further filter by domain if specified
    if domain is not None:
        saber_projects = _filter_by_domain(saber_projects, domain)

    # Nothing to do
    if not saber_projects:
        print_warning("No SABER compose projects found")
        return

    # Display matching projects
    console.print(teardown_projects_table(saber_projects))

    # Confirm
    if not yes:
        typer.confirm("Tear down these projects?", abort=True)

    # Tear down
    logger.info("Starting teardown of %d SABER project(s)", len(saber_projects))
    any_failed = False
    for p in saber_projects:
        name = p.name
        console.print(f"Tearing down [cyan]{name}[/cyan]...")
        success, stderr_text = asyncio.run(_teardown_project(name))
        if success:
            console.print(f"  [green]✓[/green] {name}")
            logger.info("Teardown of '%s' succeeded", name)
        else:
            console.print(f"  [red]✗[/red] {name}: {stderr_text}")
            logger.info("Teardown of '%s' failed: %s", name, stderr_text)
            any_failed = True

    if any_failed:
        print_error("Some teardowns failed")
        raise typer.Exit(code=1)
    else:
        print_success("All projects torn down successfully")


@app.command()  # type: ignore[misc]
def start(
    domain: str = typer.Argument(..., help="Domain slug"),
    task: str | None = typer.Option(None, "--task", "-t", help="Also start sandbox for this task ID"),
) -> None:
    """Start permanent environment (and optionally a task sandbox)."""
    try:
        domains_root = find_domains_root()
    except FileNotFoundError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    target = resolve_domain(domains_root, domain)

    if target.permanent_compose is None or target.permanent_project is None:
        print_error(f"Domain {target.slug!r} has no permanent environment")
        raise typer.Exit(code=1)

    # Start permanent services
    logger.info("Starting permanent environment for domain '%s'", target.slug)
    console.print(f"\n[bold]Starting permanent environment for [cyan]{target.slug}[/cyan]...[/bold]")
    try:
        asyncio.run(
            _start_permanent_services(
                target.permanent_compose,
                target.permanent_project,
                project_directory=target.root,
            )
        )
    except RuntimeError as exc:
        print_error(f"Failed to start permanent services: {exc}")
        raise typer.Exit(code=1) from exc

    print_success(f"Permanent environment started (project: {target.permanent_project})")

    # Optionally start task sandbox
    if task is not None:
        loader = ConfigLoader(target.root)
        tasks = loader.load_tasks(task_filter=task)

        if not tasks:
            print_error(f"Task {task!r} not found in domain {target.slug!r}")
            raise typer.Exit(code=1)

        task_cfg = tasks[0]
        logger.info("Starting sandbox for task '%s'", task_cfg.task_id)
        console.print(f"\n[bold]Starting sandbox for task [cyan]{task_cfg.task_id}[/cyan]...[/bold]")
        console.print(f"  Title: {task_cfg.title}")
        console.print(f"  Description: {task_cfg.description}")

        sandbox_compose = target.root / "compose" / "sandbox.compose.yml"
        if not sandbox_compose.exists():
            print_error(f"Sandbox compose not found at {sandbox_compose}")
            raise typer.Exit(code=1)

        try:
            asyncio.run(
                _start_permanent_services(
                    sandbox_compose,
                    f"saber-{task_cfg.task_id}",
                    project_directory=target.root,
                )
            )
        except RuntimeError as exc:
            print_error(f"Failed to start sandbox: {exc}")
            raise typer.Exit(code=1) from exc

        print_success(f"Sandbox started for task {task_cfg.task_id}")


if __name__ == "__main__":
    app()
