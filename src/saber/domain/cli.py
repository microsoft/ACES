"""CLI implementation for SABER domain orchestration.

This module provides the Click-based CLI interface for domain operations
with CLI-generated environment variables - no manual .env file editing required.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, cast

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


@cli.command(name="list")  # type: ignore[misc]
@click.option("--verbose", "-v", is_flag=True, help="Show detailed domain information")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def list_domains(ctx: click.Context, verbose: bool) -> None:
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
@click.option("--build", is_flag=True, help="Build missing images before starting")  # type: ignore[misc]
@click.option("--rebuild-all", is_flag=True, help="Remove and rebuild all images before starting")  # type: ignore[misc]
@click.option(
    "--rebuild",
    help="Remove and rebuild images with names starting with this prefix before starting (e.g., 'server', 'cookie')",
)  # type: ignore[misc]
@click.option("--rest-port", type=int, default=8000, help="REST API port")  # type: ignore[misc]
@click.option("--mcp-port", type=int, default=8001, help="MCP port")  # type: ignore[misc]
@click.option("--log-level", default="INFO", help="Logging level")  # type: ignore[misc]
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose logging (sets log level to DEBUG)"
)  # type: ignore[misc]
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")  # type: ignore[misc]
@click.option("--profiles", hidden=True, help="DEPRECATED: Profiles are no longer supported")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def start(
    ctx: click.Context,
    domain: str,
    build: bool,
    rebuild_all: bool,
    rebuild: str | None,
    rest_port: int,
    mcp_port: int,
    log_level: str,
    verbose: bool,
    dry_run: bool,
    profiles: str | None,
) -> None:
    """Start domain server with automatic environment generation.

    This starts a single server container with full privileges and Docker socket access.
    No profile selection required - server handles all functionality.

    The CLI automatically:
    - Validates domain configuration
    - Generates Docker Compose environment variables
    - Builds missing images if --build specified
    - Rebuilds images if --rebuild-all or --rebuild specified
    - Starts server with full sandbox management capabilities

    Examples:
        saber-domain start romulus
        saber-domain start romulus --build             # Build only missing images
        saber-domain start romulus --rebuild-all       # Rebuild all images
        saber-domain start romulus --rebuild server    # Only rebuild server image
        saber-domain start romulus --rebuild cookie    # Only rebuild cookie_* images
    """
    # FAIL FAST: Reject deprecated --profiles option
    if profiles is not None:
        click.echo("Error: --profiles option is no longer supported.", err=True)
        click.echo("The system now uses a simplified server-only architecture.", err=True)
        click.echo(f"Use: saber-domain start {domain}", err=True)
        ctx.exit(1)

    # FAIL FAST: Reject conflicting build options
    build_options_count = sum([build, rebuild_all, bool(rebuild)])
    if build_options_count > 1:
        click.echo("Error: Cannot specify multiple build options together.", err=True)
        click.echo("Use one of:", err=True)
        click.echo("  --build: Only builds missing images", err=True)
        click.echo("  --rebuild-all: Removes and rebuilds all images", err=True)
        click.echo("  --rebuild <prefix>: Removes and rebuilds specific images", err=True)
        ctx.exit(1)

    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))

        # Override log level if verbose flag is set
        effective_log_level = "DEBUG" if verbose else log_level

        # Convert new options to old format for orchestrator
        build_param = "" if build else None
        rebuild_param = "" if rebuild_all else rebuild

        # Start domain with server-only orchestration
        orchestrator.start_domain(
            domain=domain,
            rest_port=rest_port,
            mcp_port=mcp_port,
            log_level=effective_log_level,
            build=build_param,
            rebuild=rebuild_param,
            dry_run=dry_run,
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
@click.option("--rebuild-all", is_flag=True, help="Remove and rebuild all images")  # type: ignore[misc]
@click.option(
    "--rebuild", help="Remove and rebuild images with names starting with this prefix (e.g., 'server', 'cookie')"
)  # type: ignore[misc]
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def build(ctx: click.Context, domain: str, rebuild_all: bool, rebuild: str | None, dry_run: bool) -> None:
    """Build domain images.

    By default, builds only missing Docker images (incremental build).
    Use --rebuild-all or --rebuild to force removal and rebuild of images.

    Examples:
        saber-domain build romulus                    # Build missing images only
        saber-domain build romulus --rebuild-all     # Remove and rebuild all images
        saber-domain build romulus --rebuild server  # Remove and rebuild only server image
        saber-domain build romulus --rebuild cookie  # Remove and rebuild only cookie images
    """
    try:
        # Validate mutual exclusion
        if rebuild_all and rebuild:
            click.echo("Error: Cannot specify both --rebuild-all and --rebuild options together.", err=True)
            click.echo(
                "Use --rebuild-all to rebuild all images, or --rebuild <prefix> to rebuild specific images.", err=True
            )
            ctx.exit(1)

        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))

        if rebuild_all or rebuild:
            # Rebuild mode - remove and rebuild images
            image_filter = rebuild if rebuild else None
            orchestrator.build_domain(domain, image_filter=image_filter, dry_run=dry_run, rebuild_mode=True)

            if not dry_run:
                if rebuild:
                    click.echo(f"✓ Domain {domain} images (rebuilt: {rebuild}) completed successfully!")
                else:
                    click.echo(f"✓ Domain {domain} images rebuilt successfully!")
        else:
            # Build mode - only build missing images
            orchestrator.build_domain(domain, image_filter=None, dry_run=dry_run, rebuild_mode=False)

            if not dry_run:
                click.echo(f"✓ Domain {domain} missing images built successfully!")

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
    """Display formatted domain status information with health details."""
    running = status_info.get("running", False)
    services = status_info.get("services", [])
    health_status = status_info.get("health_status", "unknown")
    error = status_info.get("error")

    # Determine status symbol and text with health information
    if running:
        if health_status == "healthy":
            status_symbol = "✓"
            status_text = "RUNNING (healthy)"
        elif health_status == "unhealthy":
            status_symbol = "⚠"
            status_text = "RUNNING (unhealthy)"
        else:
            status_symbol = "✓"
            status_text = "RUNNING"
    else:
        status_symbol = "✗"
        status_text = "STOPPED"

    click.echo(f"{status_symbol} {domain}: {status_text}")

    # Show error if present
    if error:
        click.echo(f"    Error: {error}")

    # Show service details
    if services and isinstance(services, list) and len(services) > 0:
        for service in services:
            if isinstance(service, dict):
                # Docker compose ps --format json uses these field names
                name = service.get("Name", service.get("Names", service.get("Service", "unknown")))
                state = service.get("State", "unknown")
                status = service.get("Status", "")

                # Extract health info from status if available
                health_info = ""
                if "unhealthy" in status.lower():
                    health_info = " (unhealthy)"
                elif "healthy" in status.lower():
                    health_info = " (healthy)"

                click.echo(f"    {name}: {state}{health_info}")

                # Show additional status details if health_info is present
                if health_info:
                    click.echo(f"      Status: {status}")
    elif running:
        click.echo("    (service details unavailable)")


@cli.command()  # type: ignore[misc]
@click.argument("domain")  # type: ignore[misc]
@click.option("--saber-yaml", type=Path, help="Path to SABER config file")  # type: ignore[misc]
@click.option("--stop-after", is_flag=True, help="Stop server after test completion")  # type: ignore[misc]
@click.option("--rest-port", type=int, default=8000, help="REST API port")  # type: ignore[misc]
@click.option("--mcp-port", type=int, default=8001, help="MCP port")  # type: ignore[misc]
@click.option("--build", is_flag=True, help="Build missing images before starting")  # type: ignore[misc]
@click.option("--rebuild-all", is_flag=True, help="Remove and rebuild all images before starting")  # type: ignore[misc]
@click.option(
    "--rebuild",
    help="Remove and rebuild images with names starting with this prefix before starting (e.g., 'server', 'cookie')",
)  # type: ignore[misc]
@click.option("--log-level", default="INFO", help="Logging level")  # type: ignore[misc]
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose logging (sets log level to DEBUG)"
)  # type: ignore[misc]
@click.option("--no-ui", is_flag=True, help="Disable TUI and use rich console output instead")  # type: ignore[misc]
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")  # type: ignore[misc]
@click.pass_context  # type: ignore[misc]
def test(
    ctx: click.Context,
    domain: str,
    saber_yaml: Path | None,
    stop_after: bool,
    rest_port: int,
    mcp_port: int,
    build: bool,
    rebuild_all: bool,
    rebuild: str | None,
    log_level: str,
    verbose: bool,
    no_ui: bool,
    dry_run: bool,
) -> None:
    """Run SABER evaluation tests against domain server.

    By default, the server is kept running after test completion for subsequent runs.
    Use --stop-after to explicitly stop the server when done.

    Examples:
        saber-domain test cybench
        saber-domain test cybench --no-ui
        saber-domain test cybench --saber-yaml custom.yaml
        saber-domain test cybench --build                       # Build missing images
        saber-domain test cybench --stop-after --rebuild-all    # Rebuild all images
        saber-domain test cybench --rebuild server              # Only rebuild server
        saber-domain test romulus --rebuild cookie              # Only rebuild cookie_* images
    """
    # FAIL FAST: Validate mutually exclusive build options
    build_options_count = sum([build, rebuild_all, bool(rebuild)])
    if build_options_count > 1:
        click.echo("Error: Build options are mutually exclusive:", err=True)
        click.echo("  --build: Only builds missing images", err=True)
        click.echo("  --rebuild-all: Rebuilds all images", err=True)
        click.echo("  --rebuild <prefix>: Rebuilds images matching prefix", err=True)
        ctx.exit(1)

    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))

        # Override log level if verbose flag is set
        effective_log_level = "DEBUG" if verbose else log_level

        # Convert build options to orchestrator parameters
        build_param = "" if build else None
        rebuild_param = "" if rebuild_all else rebuild

        # Run the test command implementation
        import asyncio

        asyncio.run(
            _test_command_impl(
                orchestrator=orchestrator,
                domain=domain,
                saber_yaml=saber_yaml,
                stop_after=stop_after,
                rest_port=rest_port,
                mcp_port=mcp_port,
                build=build_param,
                rebuild=rebuild_param,
                log_level=effective_log_level,
                no_ui=no_ui,
                dry_run=dry_run,
            )
        )

    except DomainError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except KeyboardInterrupt:
        click.echo("\nAborted by user", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Test failed: {e}", err=True)
        ctx.exit(1)


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


async def _test_command_impl(
    orchestrator: DomainOrchestrator,
    domain: str,
    saber_yaml: Path | None,
    stop_after: bool,
    rest_port: int,
    mcp_port: int,
    build: str | None,
    rebuild: str | None,
    log_level: str,
    no_ui: bool,
    dry_run: bool,
) -> None:
    """Implementation of the test command."""

    # Phase 0: Load environment variables FIRST
    _load_test_environment(orchestrator.manifest_loader.domains_root)

    # Phase 1: Validate domain and discover config
    click.echo(f"🔍 Checking domain {domain}...")

    try:
        # Validate domain exists and is valid
        orchestrator.validate_domain(domain)
        click.echo("✓ Domain configuration valid")
    except Exception as e:
        raise DomainError(f"Domain validation failed: {e}")

    # Discover SABER config file
    config_path = _discover_saber_config(domain, saber_yaml, orchestrator.manifest_loader.domains_root)
    click.echo(f"📋 Found test config: {config_path}")

    if dry_run:
        click.echo("🔍 Would check server status...")
        click.echo(f"🚀 Would start {domain} server if needed (ports: REST={rest_port}, MCP={mcp_port})")
        click.echo(f"📋 Would load test config: {config_path}")
        click.echo("🧪 Would run SABER evaluation")
        if stop_after:
            click.echo("🛑 Would stop server after test")
        else:
            click.echo("ℹ️  Would keep server running")
        return

    # Phase 2: Server management
    click.echo("🔍 Checking server status...")
    we_started_server = await _ensure_server_running(
        orchestrator, domain, rest_port, mcp_port, build, rebuild, log_level, dry_run
    )

    try:
        # Phase 3: Load and hydrate SABER config
        click.echo(f"📋 Loading test config: {config_path}")
        saber_config = await _load_and_hydrate_saber_config(config_path, rest_port, mcp_port)

        # Phase 4: Run SABER evaluation
        click.echo("🧪 Running SABER evaluation...")
        click.echo(f"   • Server: http://localhost:{rest_port}")
        click.echo(f"   • MCP: http://localhost:{mcp_port}")
        click.echo(f"   • Agents: {len(saber_config.agents)}")
        click.echo(f"   • Tasks: {saber_config.task_ids or 'all available'}")
        click.echo(f"   • Config: {config_path}")

        # Extract runtime URLs for CLI override
        rest_url = f"http://localhost:{rest_port}"
        mcp_url = f"http://localhost:{mcp_port}"

        # Find the repo root .env file path
        repo_root = orchestrator.manifest_loader.domains_root.parent
        env_file_path = repo_root / ".env"

        try:
            if no_ui:
                click.echo("🎯 Starting SABER client with rich console output...")
            else:
                click.echo("🎯 Starting SABER client with TUI...")

            # Build command arguments for subprocess call
            # CRITICAL: Pass the config file path AND runtime URLs as overrides
            # This ensures ALL configuration (including endpoint settings) is preserved
            # while allowing runtime URL injection for auto mode
            cmd_args = [
                sys.executable,
                "-m",
                "saber.client",
                "run",
                "--config",
                str(config_path),
                "--rest-url",
                rest_url,
                "--mcp-url",
                mcp_url,
                "--domain",
                domain,  # Pass domain for organized logging
            ]

            # Add no-ui flag if requested
            if no_ui:
                cmd_args.append("--no-ui")

            # Add env file if it exists
            if env_file_path.exists():
                cmd_args.extend(["--env-file", str(env_file_path)])

            # Call the SABER client CLI directly - this preserves TUI
            import subprocess

            subprocess.run(cmd_args, check=True, cwd=Path.cwd())

            click.echo("✓ Evaluation completed successfully!")

        except subprocess.CalledProcessError as e:
            raise DomainError(f"SABER client execution failed: {e}") from e

    finally:
        # Phase 5: Optional cleanup
        if stop_after and we_started_server:
            click.echo("🛑 Stopping server...")
            orchestrator.stop_domain(domain)
        elif stop_after:
            click.echo("ℹ️  Server was already running, not stopping")
        else:
            click.echo("ℹ️  Server kept running (use --stop-after to cleanup)")


def _discover_saber_config(domain: str, explicit_path: Path | None, domains_root: Path) -> Path:
    """Discover SABER config with fail-fast validation."""
    if explicit_path:
        if not explicit_path.exists():
            raise DomainError(f"Explicit SABER config not found: {explicit_path}")
        return explicit_path

    # Standard location
    canonical_path = domains_root / domain / "client" / "saber.yaml"
    if canonical_path.exists():
        return canonical_path

    # No fallbacks - fail fast
    raise DomainError(f"SABER config not found: {canonical_path}")


async def _ensure_server_running(
    orchestrator: DomainOrchestrator,
    domain: str,
    rest_port: int,
    mcp_port: int,
    build: str | None,
    rebuild: str | None,
    log_level: str,
    dry_run: bool,
) -> bool:
    """Ensure server is running, return True if we started it."""

    # Handle rebuild flag regardless of server status
    if rebuild is not None:
        # Stop the server first since we're rebuilding critical images
        status = orchestrator.get_domain_status(domain)
        if status.get("running", False):
            if dry_run:
                click.echo(f"🛑 Would stop {domain} server for rebuild")
            else:
                click.echo(f"🛑 Stopping {domain} server for rebuild...")
                orchestrator.stop_domain(domain, dry_run=False)
                click.echo("✓ Server stopped")

        if dry_run:
            rebuild_msg = f"🔨 Would rebuild {domain} images"
            if rebuild:  # If not empty string
                rebuild_msg += f" (filter: {rebuild})"
            click.echo(rebuild_msg)
        else:
            rebuild_msg = f"🔨 Rebuilding {domain} images"
            if rebuild:  # If not empty string
                rebuild_msg += f" (filter: {rebuild})"
            click.echo(rebuild_msg + "...")
            orchestrator.build_domain(
                domain, image_filter=rebuild if rebuild else None, dry_run=False, rebuild_mode=True
            )
            click.echo("✓ Images rebuilt successfully")

    # Handle build flag (only build missing images)
    elif build is not None:
        if dry_run:
            build_msg = f"🔨 Would build missing {domain} images"
            if build:  # If not empty string
                build_msg += f" (filter: {build})"
            click.echo(build_msg)
        else:
            build_msg = f"🔨 Building missing {domain} images"
            if build:  # If not empty string
                build_msg += f" (filter: {build})"
            click.echo(build_msg + "...")
            orchestrator.build_domain(domain, image_filter=build if build else None, dry_run=False, rebuild_mode=False)
            click.echo("✓ Images built successfully")

    status = orchestrator.get_domain_status(domain)

    if status.get("running", False):
        # Server is running - show health status
        health_status = status.get("health_status", "unknown")
        if health_status == "unhealthy":
            click.echo(f"⚠️  Server running but unhealthy for {domain} - proceeding with test")
        else:
            click.echo(f"✓ Server already running for {domain}")
        return False  # We didn't start it

    if dry_run:
        click.echo(f"🚀 Would start server for {domain}")
        return False

    click.echo(f"🚀 Starting {domain} server...")
    orchestrator.start_domain(
        domain, rest_port, mcp_port, log_level, rebuild=None, dry_run=False
    )  # rebuild=None since we already built above

    # Wait for server readiness with detailed monitoring
    await _wait_for_server_ready(rest_port, mcp_port, domain, orchestrator)
    click.echo(f"✓ Server ready at http://localhost:{rest_port}")
    return True  # We started it


async def _wait_for_server_ready(rest_port: int, mcp_port: int, domain: str, orchestrator: DomainOrchestrator) -> None:
    """
    Wait for server to be ready with detailed health reporting and no timeout.

    Shows periodic updates every 15 seconds with:
    - Elapsed time
    - Server health status
    - Permanent environment health
    - Error messages from logs

    User can Ctrl+C to gracefully shutdown the server.
    """
    import asyncio
    import signal

    import aiohttp

    health_url = f"http://localhost:{rest_port}/api/v1/health"
    start_time = asyncio.get_event_loop().time()
    last_report_time = 0
    update_interval = 15  # Report every 15 seconds
    shutdown_requested = False

    def signal_handler(signum: int, frame: Any) -> None:
        """Handle Ctrl+C gracefully."""
        nonlocal shutdown_requested
        shutdown_requested = True

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    click.echo("⏳ Waiting for server to become ready...")
    click.echo("   Press Ctrl+C to stop and shutdown the server")
    click.echo()

    try:
        while not shutdown_requested:
            elapsed = int(asyncio.get_event_loop().time() - start_time)

            # Check if server is ready
            server_ready = False
            health_data = None
            connection_error = None

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            server_ready = True
                        else:
                            connection_error = f"HTTP {response.status}"
            except asyncio.TimeoutError:
                connection_error = "Connection timeout"
            except aiohttp.ClientConnectorError:
                connection_error = "Connection refused"
            except Exception as e:
                connection_error = f"{type(e).__name__}: {str(e)[:50]}"

            # If server is ready, we're done!
            if server_ready:
                click.echo(f"✓ Server ready after {elapsed}s")
                return

            # Show periodic status updates every 15 seconds
            if elapsed - last_report_time >= update_interval:
                last_report_time = elapsed
                click.echo(f"📊 Status Update [{elapsed}s elapsed]")
                click.echo(f"{'─' * 60}")

                # Show connection status
                if connection_error:
                    click.echo(f"  🔌 Server Connection: {click.style(connection_error, fg='yellow')}")
                else:
                    click.echo(f"  🔌 Server Connection: {click.style('Connected', fg='green')}")

                # Get and show container status
                container_status = _get_container_status(domain)
                if container_status:
                    click.echo(f"  📦 Container Status: {container_status}")

                # Show health endpoint data (server status and permanent environment)
                if health_data:
                    _show_health_details(health_data)
                else:
                    click.echo(f"  💛 Server Health: {click.style('not ready', fg='yellow')}")
                    # Even if server isn't ready, try to check permanent environment directly
                    perm_env_health = _check_permanent_environment_health_direct(domain, orchestrator)
                    if perm_env_health:
                        _show_permanent_environment_health(perm_env_health)

                # Check for errors in server logs
                errors = _scan_server_logs_for_errors(domain, since_seconds=update_interval + 5)
                if errors:
                    click.echo(f"  ❌ Errors Found in Logs ({len(errors)}):")
                    for error in errors[:5]:  # Show max 5 errors
                        # Truncate long error messages
                        error_msg = error if len(error) <= 100 else error[:97] + "..."
                        click.echo(f"     • {click.style(error_msg, fg='red')}")
                    if len(errors) > 5:
                        click.echo(f"     ... and {len(errors) - 5} more errors")
                else:
                    click.echo("  ✓ No errors in recent logs")

                click.echo()

            await asyncio.sleep(1)

        # Shutdown was requested
        click.echo()
        click.echo("🛑 Shutdown requested by user")
        click.echo("   Stopping server gracefully...")
        orchestrator.stop_domain(domain)
        click.echo("✓ Server stopped successfully")
        raise DomainError("Server startup cancelled by user")

    except DomainError:
        raise
    except Exception as e:
        click.echo()
        click.echo(f"❌ Unexpected error during server startup: {e}")
        click.echo("   Attempting to stop server...")
        try:
            orchestrator.stop_domain(domain)
        except Exception:
            pass
        raise DomainError(f"Server startup failed: {e}")
    finally:
        # Restore default signal handler
        signal.signal(signal.SIGINT, signal.SIG_DFL)


def _get_container_status(domain: str) -> str:
    """Get Docker container status for the domain server."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={domain}-saber-server", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            status = result.stdout.strip()
            # Colorize status
            if "Up" in status:
                if "unhealthy" in status.lower():
                    return cast(str, click.style(status, fg="yellow"))
                elif "starting" in status.lower():
                    return cast(str, click.style(status, fg="cyan"))
                else:
                    return cast(str, click.style(status, fg="green"))
            else:
                return cast(str, click.style(status, fg="red"))
        return cast(str, click.style("Container not found", fg="red"))
    except Exception as e:
        return cast(str, click.style(f"Error: {e}", fg="red"))


def _scan_server_logs_for_errors(domain: str, since_seconds: int = 20) -> list[str]:
    """
    Scan server logs for ERROR keywords.

    Args:
        domain: Domain name
        since_seconds: Only look at logs from the last N seconds

    Returns:
        List of error messages found
    """
    try:
        result = subprocess.run(
            ["docker", "logs", "--since", f"{since_seconds}s", f"{domain}-saber-server"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return []

        # Combine stdout and stderr
        all_logs = result.stdout + result.stderr

        # Patterns to ignore (warnings, deprecations, stack traces from warnings)
        ignore_patterns = [
            "PydanticDeprecatedSince20",
            "DeprecationWarning",
            "FutureWarning",
            "UserWarning",
            "warnings.warn",
            "PendingDeprecationWarning",
            "RuntimeWarning",
            "site-packages",  # Usually part of warning stack traces
            ".py:",  # File references in warnings (e.g., "/path/file.py:123")
        ]

        # Find lines containing ERROR (case insensitive)
        errors = []
        for line in all_logs.split("\n"):
            # Skip empty lines
            if not line.strip():
                continue

            # Check if this is an ERROR line (not just warning)
            if "ERROR" in line.upper():
                # Skip if it's just a warning/deprecation or stack trace
                if any(pattern in line for pattern in ignore_patterns):
                    continue

                # Also skip lines that look like file paths or warning context
                if line.strip().startswith("/") or line.strip().startswith("File "):
                    continue

                # Clean up the line
                line = line.strip()
                if line:
                    errors.append(line)

        return errors
    except Exception:
        return []


def _show_health_details(health_data: dict) -> None:
    """Display health endpoint details."""
    try:
        # Show basic health info
        domain = health_data.get("domain", "unknown")
        status = health_data.get("status", "unknown")

        if status == "healthy":
            click.echo(f"  💚 Server Health: {click.style('healthy', fg='green')} (domain: {domain})")
        else:
            click.echo(f"  💛 Server Health: {click.style(status, fg='yellow')} (domain: {domain})")

        # Show permanent environment health if present
        perm_env = health_data.get("permanent_environment_health")
        if perm_env:
            is_healthy = perm_env.get("healthy", False)
            if is_healthy:
                click.echo(f"  🌍 Permanent Environment: {click.style('healthy', fg='green')}")
            else:
                click.echo(f"  🌍 Permanent Environment: {click.style('unhealthy', fg='yellow')}")

                # Show details if available
                details = perm_env.get("details", {})
                if isinstance(details, dict):
                    services = details.get("services", {})
                    if services:
                        click.echo("     Services:")
                        for service_name, service_info in services.items():
                            if isinstance(service_info, dict):
                                svc_healthy = service_info.get("healthy", False)
                                reason = service_info.get("reason", "unknown")
                                if svc_healthy:
                                    click.echo(f"       ✓ {service_name}: {reason}")
                                else:
                                    click.echo(f"       ✗ {service_name}: {click.style(reason, fg='yellow')}")

                # Show error if present
                error = perm_env.get("error")
                if error:
                    click.echo(f"     Error: {click.style(error, fg='red')}")
    except Exception:
        pass  # Don't fail on health detail display errors


def _check_permanent_environment_health_direct(domain: str, orchestrator: DomainOrchestrator) -> dict | None:
    """
    Check permanent environment health directly via Docker.

    This function queries the permanent environment containers directly without
    needing the server to be fully ready.

    Args:
        domain: Domain name
        orchestrator: Domain orchestrator instance

    Returns:
        Dictionary with permanent environment health info, or None if not configured
    """
    try:
        # Get domain path - validate domain exists
        orchestrator.validate_domain(domain)
        domains_root = orchestrator.manifest_loader.domains_root
        domain_path = domains_root / domain
        config_dir = domain_path / "server" / "config"

        # Check if there's a permanent environment configured
        # Look for global.yaml to get permanent environment name
        global_config_path = config_dir / "tasks" / "global.yaml"
        if not global_config_path.exists():
            return None

        import yaml

        with open(global_config_path, "r") as f:
            global_config = yaml.safe_load(f)

        permanent_env_name = global_config.get("permanent_environment")
        if not permanent_env_name:
            return None

        # Get the compose file path
        permanent_compose_path = config_dir / "environments" / "permanent" / f"{permanent_env_name}.compose.yml"
        if not permanent_compose_path.exists():
            return {
                "healthy": False,
                "status": "compose_file_missing",
                "error": f"Compose file not found: {permanent_compose_path}",
                "environment_name": permanent_env_name,
            }

        # Use ComposeHealthChecker to check service health
        from saber.server.execution.sandbox.compose_health_checker import ComposeHealthChecker

        health_checker = ComposeHealthChecker()
        project_name = f"{domain}-permanent"

        health_summary = health_checker.get_service_health_summary(str(permanent_compose_path), project_name)

        return {
            "healthy": health_summary["overall_healthy"],
            "status": "checked",
            "environment_name": permanent_env_name,
            "project_name": project_name,
            "healthy_services": health_summary["healthy_count"],
            "total_services": health_summary["total_count"],
            "services": health_summary["services"],
            "error": health_summary.get("error"),
        }
    except Exception:
        return None


def _show_permanent_environment_health(perm_env_health: dict) -> None:
    """Display permanent environment health status."""
    try:
        env_name = perm_env_health.get("environment_name", "unknown")
        is_healthy = perm_env_health.get("healthy", False)
        services = perm_env_health.get("services", {})

        # Check if any services are still starting
        has_starting_services = False
        if services:
            for service_info in services.values():
                if isinstance(service_info, dict):
                    reason = service_info.get("reason", "")
                    if "starting" in reason.lower():
                        has_starting_services = True
                        break

        # Determine status message
        if is_healthy:
            status_msg = click.style("healthy", fg="green")
            icon = "🌍"
        elif has_starting_services:
            status_msg = click.style("starting", fg="cyan")
            icon = "🌍"
        else:
            status_msg = click.style("unhealthy", fg="yellow")
            icon = "🌍"

        click.echo(f"  {icon} Permanent Environment ({env_name}): {status_msg}")

        # Show service details
        if services:
            healthy_count = perm_env_health.get("healthy_services", 0)
            total_count = perm_env_health.get("total_services", 0)
            click.echo(f"     Services ({healthy_count}/{total_count} healthy):")
            for service_name, service_info in services.items():
                if isinstance(service_info, dict):
                    svc_healthy = service_info.get("healthy", False)
                    reason = service_info.get("reason", "unknown")

                    if svc_healthy:
                        click.echo(f"       ✓ {service_name}: {reason}")
                    elif "starting" in reason.lower():
                        click.echo(f"       ⏳ {service_name}: {click.style(reason, fg='cyan')}")
                    else:
                        click.echo(f"       ✗ {service_name}: {click.style(reason, fg='yellow')}")

        # Show error if present
        error = perm_env_health.get("error")
        if error:
            click.echo(f"     Error: {click.style(error, fg='red')}")
    except Exception:
        pass  # Don't fail on display errors


async def _load_and_hydrate_saber_config(config_path: Path, rest_port: int, mcp_port: int) -> Any:
    """Load SABER config and hydrate with runtime server URLs."""
    # Import SABER client components
    SABERConfig, SABERConfigLoader = _import_saber_client()

    # Load config inputs (allows auto/missing server URLs)
    config_inputs = SABERConfigLoader.load_config_inputs(config_path)

    # Hydrate with runtime URLs if needed
    server_config = config_inputs.get("server", {})
    server_mode = server_config.get("mode")

    if server_mode == "auto" or not server_config.get("rest_url"):
        # Inject runtime URLs before creating SABERConfig
        config_inputs["server"]["rest_url"] = f"http://localhost:{rest_port}"
        config_inputs["server"]["mcp_url"] = f"http://localhost:{mcp_port}"

    # Create final SABERConfig with hydrated URLs
    saber_config = SABERConfigLoader._convert_yaml_to_saber_config(config_inputs, config_path)

    # Verify session_config was created
    if not saber_config.session_config:
        raise DomainError(
            f"Failed to create session config - server URLs may be missing. "
            f"Config has rest_url={config_inputs.get('server', {}).get('rest_url')}, "
            f"mcp_url={config_inputs.get('server', {}).get('mcp_url')}"
        )

    return saber_config


def _load_test_environment(domains_root: Path) -> None:
    """Load environment variables for SABER test execution."""
    # Find repo root from domains_root (domains_root is typically /path/to/repo/domains)
    repo_root = domains_root.parent  # /home/ms_test/repos/oss_saber
    env_file = repo_root / ".env"

    if not env_file.exists():
        click.echo(f"⚠️  No .env file found at {env_file}")
        click.echo("   LLM evaluation may fail without proper credentials")
        click.echo(f"   Create {env_file} with your OpenAI/Azure credentials")
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)  # Don't override existing env vars
        click.echo(f"🔐 Loaded environment from {env_file}")
    except ImportError:
        raise DomainError("python-dotenv is required for .env file loading. " "Install with: uv add python-dotenv")
    except Exception as e:
        raise DomainError(f"Failed to load environment file {env_file}: {e}")


def _import_saber_client() -> tuple[Any, Any]:
    """Lazy import of SABER client components."""
    try:
        from saber.client.config_loader import SABERConfigLoader
        from saber.client.models import SABERConfig

        return SABERConfig, SABERConfigLoader
    except ImportError as e:
        raise DomainError(f"SABER client components not available: {e}")


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
