"""CLI implementation for SABER domain orchestration.

This module provides the Click-based CLI interface for domain operations
with CLI-generated environment variables - no manual .env file editing required.
"""

from pathlib import Path
from typing import Any, Dict

import click

from .exceptions import DomainError
from .orchestrator import DomainOrchestrator
from .resources import get_env_example_content, resolve_compose_file, resolve_schema_file


def _get_domains_root(domains_root: Path | None) -> Path:
    """Get domains root path with auto-detection fallback."""
    if domains_root:
        return domains_root

    # Auto-detect from common locations
    candidates = [
        Path.cwd() / "domains",
        Path.cwd() / "../domains",
        Path.cwd().parent / "domains",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if it looks like a domains directory
            if any(
                (candidate / d).is_dir() and (candidate / d / "domain.yaml").exists()
                for d in candidate.iterdir()
                if d.is_dir()
            ):
                return candidate

    # Default to current directory/domains
    return Path.cwd() / "domains"


def _create_orchestrator(domains_root: Path | None) -> DomainOrchestrator:
    """Create orchestrator instance with resource resolution."""
    resolved_domains_root = _get_domains_root(domains_root)

    with resolve_compose_file() as compose_path:
        return DomainOrchestrator(resolved_domains_root, compose_path)


@click.group()  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--domains-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to domains directory (auto-detected if not provided)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def cli(ctx: click.Context, domains_root: Path | None, verbose: bool) -> None:
    """SABER Domain Orchestration CLI - Server-Only Architecture.

    Simple domain management with single server containers.
    Each domain runs one server with full privileges and Docker socket access.
    """
    ctx.ensure_object(dict)
    ctx.obj["domains_root"] = domains_root
    ctx.obj["verbose"] = verbose


@cli.command()  # type: ignore[misc]
@click.option("--verbose", "-v", is_flag=True, help="Show detailed domain information")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def list(ctx: click.Context, verbose: bool) -> None:
    """List available domains."""
    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))
        domains = orchestrator.list_domains()

        if not domains:
            click.echo("No domains found.")
            domains_root = _get_domains_root(ctx.obj.get("domains_root"))
            click.echo(f"Searched in: {domains_root}")
            return

        click.echo(f"Available domains ({len(domains)}):")
        for domain in domains:
            if verbose or ctx.obj.get("verbose", False):
                try:
                    manifest = orchestrator.validate_domain(domain)
                    name = manifest.get("domain", {}).get("name", domain)
                    description = manifest.get("domain", {}).get("description", "")
                    click.echo(f"  {domain} - {name}")
                    if description:
                        click.echo(f"    {description}")
                except Exception as e:
                    click.echo(f"  {domain} - (validation failed: {e})", err=True)
            else:
                click.echo(f"  {domain}")

    except DomainError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


@cli.command()  # type: ignore[misc]
@click.argument("domain")  # type: ignore[misc]
@click.option("--build", is_flag=True, help="Build images before starting")  # type: ignore[misc]
@click.option("--rest-port", type=int, default=8000, help="REST API port")  # type: ignore[misc]
@click.option("--mcp-port", type=int, default=8001, help="MCP port")  # type: ignore[misc]
@click.option("--log-level", default="INFO", help="Logging level")  # type: ignore[misc]
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")  # type: ignore[misc]
@click.option("--profiles", hidden=True, help="DEPRECATED: Profiles are no longer supported")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def start(
    ctx: click.Context,
    domain: str,
    build: bool,
    rest_port: int,
    mcp_port: int,
    log_level: str,
    dry_run: bool,
    profiles: str | None,
) -> None:
    """Start domain server with automatic environment generation.

    This starts a single server container with full privileges and Docker socket access.
    No profile selection required - server handles all functionality.

    The CLI automatically:
    - Validates domain configuration
    - Generates Docker Compose environment variables
    - Builds server image if --build specified
    - Starts server with full sandbox management capabilities
    """
    # FAIL FAST: Reject deprecated --profiles option
    if profiles is not None:
        click.echo("Error: --profiles option is no longer supported.", err=True)
        click.echo("The system now uses a simplified server-only architecture.", err=True)
        click.echo(f"Use: saber-domain start {domain}", err=True)
        ctx.exit(1)

    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))

        # Start domain with server-only orchestration
        orchestrator.start_domain(
            domain=domain, rest_port=rest_port, mcp_port=mcp_port, log_level=log_level, build=build, dry_run=dry_run
        )

        if not dry_run:
            click.echo(f"✓ Domain {domain} server started successfully!")
            click.echo(f"  REST API: http://localhost:{rest_port}")
            click.echo(f"  MCP Server: http://localhost:{mcp_port}")

    except DomainError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


@cli.command()  # type: ignore[misc]
@click.argument("domain")  # type: ignore[misc]
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def stop(ctx: click.Context, domain: str, dry_run: bool) -> None:
    """Stop domain services."""
    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))
        orchestrator.stop_domain(domain, dry_run)

        if not dry_run:
            click.echo(f"✓ Domain {domain} stopped successfully!")

    except DomainError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


@cli.command()  # type: ignore[misc]
@click.argument("domain")  # type: ignore[misc]
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def build(ctx: click.Context, domain: str, dry_run: bool) -> None:
    """Build domain images.

    Builds all Docker images defined in the domain manifest.
    """
    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))
        orchestrator.build_domain(domain, dry_run)

        if not dry_run:
            click.echo(f"✓ Domain {domain} images built successfully!")

    except DomainError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


@cli.command()  # type: ignore[misc]
@click.argument("domain")  # type: ignore[misc]
@click.option("--verbose", "-v", is_flag=True, help="Verbose validation output")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def validate(ctx: click.Context, domain: str, verbose: bool) -> None:
    """Validate domain configuration.

    Validates:
    - Domain manifest syntax and schema compliance
    - Required directory structure
    - Docker image availability
    - Port availability
    """
    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))
        manifest = orchestrator.validate_domain(domain)

        click.echo(f"✓ Domain {domain} validation passed!")

        if verbose or ctx.obj.get("verbose", False):
            domain_info = manifest.get("domain", {})
            click.echo(f"  Name: {domain_info.get('name', 'N/A')}")
            click.echo(f"  Version: {domain_info.get('version', 'N/A')}")
            if domain_info.get("description"):
                click.echo(f"  Description: {domain_info['description']}")

            images = manifest.get("images", {})
            click.echo(f"  Images: {', '.join(images.keys())}")

            profiles = manifest.get("profiles", [])
            if profiles:
                click.echo(f"  Custom profiles: {', '.join(p['name'] for p in profiles)}")

    except DomainError as e:
        click.echo(f"✗ Validation failed: {e}", err=True)
        ctx.exit(1)


@cli.command()  # type: ignore[misc]
@click.argument("domain", required=False)  # type: ignore[misc]
@click.option("--watch", is_flag=True, help="Watch for status changes (not implemented yet)")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def status(ctx: click.Context, domain: str | None, watch: bool) -> None:
    """Show domain status.

    Shows running status and service information for domains.
    """
    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))

        if domain:
            # Single domain status
            status_info = orchestrator.get_domain_status(domain)
            _display_domain_status(domain, status_info)
        else:
            # All domains status
            domains = orchestrator.list_domains()
            if not domains:
                click.echo("No domains found.")
                return

            click.echo(f"Status for {len(domains)} domains:\n")
            for domain_name in domains:
                try:
                    status_info = orchestrator.get_domain_status(domain_name)
                    _display_domain_status(domain_name, status_info)
                    click.echo()  # Empty line between domains
                except Exception as e:
                    click.echo(f"{domain_name}: Error getting status - {e}")

        if watch:
            click.echo("Watch mode not implemented yet. Use --help for current features.")

    except DomainError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _display_domain_status(domain: str, status_info: Dict[str, Any]) -> None:
    """Display formatted domain status information."""
    running = status_info.get("running", False)
    services = status_info.get("services", [])

    status_symbol = "✓" if running else "✗"
    status_text = "RUNNING" if running else "STOPPED"

    click.echo(f"{status_symbol} {domain}: {status_text}")

    if services and isinstance(services, list):
        for service in services:
            if isinstance(service, dict):
                name = service.get("Name", "unknown")
                state = service.get("State", "unknown")
                click.echo(f"    {name}: {state}")
    elif running:
        click.echo("    (service details unavailable)")


@cli.command(name="test-resources")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def test_resources(ctx: click.Context) -> None:
    """Test resource resolution (development command)."""
    click.echo("Testing resource resolution...")

    try:
        with resolve_compose_file() as compose_path:
            click.echo(f"✓ docker-compose.yml: {compose_path}")

        with resolve_schema_file() as schema_path:
            click.echo(f"✓ domain-manifest.schema.json: {schema_path}")

        env_content = get_env_example_content()
        click.echo(f"✓ env.example content length: {len(env_content)} chars")

        click.echo("All resources resolved successfully!")

    except DomainError as e:
        click.echo(f"✗ Resource resolution failed: {e}", err=True)
        ctx.exit(1)


def main() -> None:
    """Main entry point for console script."""
    try:
        cli()
    except DomainError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e
    except KeyboardInterrupt:
        click.echo("\nAborted by user", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
