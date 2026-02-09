"""CLI implementation for SABER domain orchestration.

This module provides the Click-based CLI interface for domain operations
with CLI-generated environment variables - no manual .env file editing required.
"""

from pathlib import Path
from typing import Any

import click

from .exceptions import DomainError
from .orchestrator import DomainOrchestrator
from .resources import get_env_example_content, resolve_schema_file


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
    """Create orchestrator instance."""
    resolved_domains_root = _get_domains_root(domains_root)
    return DomainOrchestrator(resolved_domains_root)


@click.group()
@click.option(
    "--domains-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to domains directory (auto-detected if not provided)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, domains_root: Path | None, verbose: bool) -> None:
    """SABER Domain Orchestration CLI - Host Subprocess Architecture.

    Simple domain management where the server runs as a host subprocess.
    Sandbox containers are managed by the server via Docker API.

    This architecture eliminates Docker-in-Docker issues and allows
    direct host path mounts for sandbox containers.
    """
    ctx.ensure_object(dict)
    ctx.obj["domains_root"] = domains_root
    ctx.obj["verbose"] = verbose


@cli.command(name="list")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed domain information")
@click.pass_context
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


@cli.command()
@click.argument("domain")
@click.option("--build", is_flag=True, help="Build missing images before starting")
@click.option("--rebuild-all", is_flag=True, help="Remove and rebuild all images before starting")
@click.option(
    "--rebuild",
    help="Remove and rebuild images with names starting with this prefix before starting (e.g., 'server', 'cookie')",
)
@click.option("--rest-port", type=int, default=8000, help="REST API port")
@click.option("--mcp-port", type=int, default=8001, help="MCP port")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging (sets log level to DEBUG)")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.option("--profiles", hidden=True, help="DEPRECATED: Profiles are no longer supported")
@click.pass_context
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
    """Start domain server as a host subprocess.

    This starts the SABER server directly on the host machine (not in a container).
    The server manages sandbox containers via the Docker API.

    Benefits of host subprocess deployment:
    - Eliminates Docker-in-Docker issues
    - Allows direct host path mounts for sandbox containers
    - Simplifies networking between server and managed containers

    The CLI automatically:
    - Validates domain configuration
    - Generates environment variables
    - Builds missing images if --build specified
    - Rebuilds images if --rebuild-all or --rebuild specified
    - Starts server subprocess with health checking

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
    except KeyboardInterrupt:
        click.echo("\n⚠️  Startup cancelled by user. Server and containers have been cleaned up.", err=True)
        ctx.exit(130)


@cli.command()
@click.argument("domain", required=False)
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.pass_context
def stop(ctx: click.Context, domain: str | None, dry_run: bool) -> None:
    """Stop domain services.

    If DOMAIN is specified, stops that domain.
    If DOMAIN is omitted, stops all running SABER domains.
    """
    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))

        if domain:
            # Stop specific domain
            orchestrator.stop_domain(domain, dry_run)
            if not dry_run:
                click.echo(f"✓ Domain {domain} stopped successfully!")
        else:
            # Stop all running domains
            running_domains = orchestrator.get_running_domains()
            if not running_domains:
                click.echo("No running SABER domains found.")
                return

            click.echo(f"Stopping {len(running_domains)} running domain(s)...")
            for domain_name in running_domains:
                if dry_run:
                    click.echo(f"  Would stop: {domain_name}")
                else:
                    orchestrator.stop_domain(domain_name, dry_run=False)
                    click.echo(f"  ✓ Stopped {domain_name}")

            if not dry_run:
                click.echo("✓ All domains stopped successfully!")

    except DomainError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.argument("domain")
@click.option("--rebuild-all", is_flag=True, help="Remove and rebuild all images")
@click.option(
    "--rebuild", help="Remove and rebuild images with names starting with this prefix (e.g., 'server', 'cookie')"
)
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.pass_context
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


@cli.command()
@click.argument("domain")
@click.option("--verbose", "-v", is_flag=True, help="Verbose validation output")
@click.pass_context
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


@cli.command()
@click.argument("domain", required=False)
@click.option("--watch", is_flag=True, help="Watch for status changes (not implemented yet)")
@click.pass_context
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


def _display_domain_status(domain: str, status_info: dict[str, Any]) -> None:
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


@cli.command()
@click.argument("domain")
@click.option("--saber-yaml", type=Path, help="Path to SABER config file (deprecated)")
@click.option("--stop-after", is_flag=True, help="Stop server after test completion (deprecated)")
@click.option("--rest-port", type=int, default=8000, help="REST API port (deprecated)")
@click.option("--mcp-port", type=int, default=8001, help="MCP port (deprecated)")
@click.option("--build", is_flag=True, help="Build missing images before starting (deprecated)")
@click.option(
    "--rebuild-all",
    is_flag=True,
    help="Remove and rebuild all images before starting (deprecated)",
)
@click.option(
    "--rebuild",
    help="Remove and rebuild images with names starting with this prefix (deprecated)",
)
@click.option("--log-level", default="INFO", help="Logging level (deprecated)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging (deprecated)")
@click.option("--no-ui", is_flag=True, help="Disable TUI (deprecated)")
@click.option("--dry-run", is_flag=True, help="Show what would be done (deprecated)")
@click.pass_context
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
    """[DEPRECATED] Run SABER evaluation tests against domain server.

    ⚠️  This command has been DEPRECATED in favor of 'inspect eval'.

    The new workflow uses Inspect AI's native evaluation system with SABER's
    SABERSandboxEnvironment integration for better performance and compatibility.

    See below for the equivalent command for your use case.
    """
    click.echo(click.style("\n⚠️  DEPRECATION WARNING", fg="yellow", bold=True))
    click.echo(click.style("=" * 60, fg="yellow"))
    click.echo()
    click.echo("The 'saber-domain test' command has been deprecated.")
    click.echo("Please use 'inspect eval' instead for better performance and compatibility.")
    click.echo()

    # Build the equivalent inspect eval command
    inspect_cmd = f"uv run inspect eval domains/{domain}"

    # Add model (required for inspect eval)
    click.echo(click.style("📋 Migration Guide:", fg="cyan", bold=True))
    click.echo()
    click.echo("Basic usage (specify your model):")
    click.echo(click.style(f"  {inspect_cmd} --model <your-model>", fg="green"))
    click.echo()
    click.echo("Examples:")
    click.echo(f"  {inspect_cmd} --model openai/azure/gpt-4")
    click.echo(f"  {inspect_cmd} --model anthropic/claude-3-opus")
    click.echo()

    # Show parameter mappings
    task_params = []

    if build:
        task_params.append("-T build=true")
    if rebuild_all:
        task_params.append("-T rebuild_all=true")
    if rebuild:
        task_params.append(f"-T rebuild={rebuild}")
    if rest_port != 8000:
        task_params.append(f"-T rest_port={rest_port}")
    if mcp_port != 8001:
        task_params.append(f"-T mcp_port={mcp_port}")
    if stop_after:
        task_params.append("-T stop_saber_after=true")

    if task_params or saber_yaml or no_ui:
        click.echo("With your current options:")
        equivalent_cmd = inspect_cmd

        if task_params:
            equivalent_cmd += " " + " ".join(task_params)

        equivalent_cmd += " --model <your-model>"

        click.echo(click.style(f"  {equivalent_cmd}", fg="green"))
        click.echo()
        click.echo("  Replace <your-model> with your model, e.g.:")
        click.echo("    --model openai/azure/gpt-4")
        click.echo("    --model anthropic/claude-3-opus")
        click.echo()

        if saber_yaml:
            click.echo(click.style("  Note:", fg="yellow") + " The --saber-yaml option is no longer needed.")
            click.echo(
                "  Task configuration is now defined in the domain task file (e.g., domains/excytin/excytin.py)."
            )
            click.echo()

        if no_ui:
            click.echo(click.style("  Note:", fg="yellow") + " UI behavior is now controlled by Inspect AI.")
            click.echo("  Use INSPECT_LOG_LEVEL=info for detailed output.")
            click.echo()

    click.echo(click.style("📚 Additional Resources:", fg="cyan", bold=True))
    click.echo()
    click.echo("  List available tasks:")
    click.echo(click.style("    uv run inspect list tasks", fg="green"))
    click.echo()
    click.echo("  Filter specific tasks:")
    click.echo(
        click.style(
            f"    uv run inspect eval domains/{domain} -T task_filter=task_name --model <your-model>", fg="green"
        )
    )
    click.echo()
    click.echo("  Documentation:")
    click.echo("    README.md - Getting started and examples")
    click.echo("    docs/INSPECT_AI_DOMAIN_TASKS.md - Complete inspect eval guide")
    click.echo()
    click.echo(click.style("=" * 60, fg="yellow"))
    click.echo()

    ctx.exit(1)


@cli.command(name="preflight")
@click.argument("domain")
@click.option(
    "--concurrency",
    "-c",
    type=int,
    default=16,
    help="Number of environments to test in parallel",
)
@click.option(
    "--timeout",
    "-t",
    type=int,
    default=90,
    help="Health check timeout in seconds per environment",
)
@click.option("--pattern", "-p", help="Filter compose files by pattern (e.g., 'cmd', 'cookie_0')")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--fail-fast", is_flag=True, help="Stop on first failure")
@click.pass_context
def preflight(
    ctx: click.Context,
    domain: str,
    concurrency: int,
    timeout: int,
    pattern: str | None,
    verbose: bool,
    fail_fast: bool,
) -> None:
    """Run preflight checks on all challenge environments.

    This command discovers and tests all compose files in the domain's sandbox
    environments directory. Each environment is brought up, health-checked, and
    torn down to verify it can start successfully.

    Examples:
        saber-domain preflight romulus                    # Test all environments
        saber-domain preflight romulus -c 8               # Test 8 in parallel
        saber-domain preflight romulus -p cmd             # Only test cmd/* environments
        saber-domain preflight romulus -p cookie_0        # Test specific environment
        saber-domain preflight romulus --fail-fast        # Stop on first error
    """
    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))

        # Run preflight check implementation
        import asyncio

        asyncio.run(
            _preflight_check_impl(
                orchestrator=orchestrator,
                domain=domain,
                concurrency=concurrency,
                timeout=timeout,
                pattern=pattern,
                verbose=verbose or ctx.obj.get("verbose", False),
                fail_fast=fail_fast,
            )
        )

    except DomainError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except KeyboardInterrupt:
        click.echo("\n⚠️  Preflight check cancelled by user", err=True)
        ctx.exit(130)


@cli.command(name="test-resources")
@click.pass_context
def test_resources(ctx: click.Context) -> None:
    """Test resource resolution (development command)."""
    click.echo("Testing resource resolution...")

    try:
        with resolve_schema_file() as schema_path:
            click.echo(f"✓ domain-manifest.schema.json: {schema_path}")

        env_content = get_env_example_content()
        click.echo(f"✓ env.example content length: {len(env_content)} chars")

        click.echo("All resources resolved successfully!")

    except DomainError as e:
        click.echo(f"✗ Resource resolution failed: {e}", err=True)
        ctx.exit(1)


async def _preflight_check_impl(
    orchestrator: DomainOrchestrator,
    domain: str,
    concurrency: int,
    timeout: int,
    pattern: str | None,
    verbose: bool,
    fail_fast: bool,
) -> None:
    """Implementation of the preflight check command."""
    import asyncio
    import signal
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    # Track cancellation state
    cancellation_requested = False

    def signal_handler(signum: int, frame: Any) -> None:
        """Handle Ctrl+C gracefully."""
        nonlocal cancellation_requested
        if not cancellation_requested:
            cancellation_requested = True
            click.echo("\n")
            click.echo("⚠️  " + click.style("Ctrl+C detected - cancelling preflight check...", fg="yellow", bold=True))
            click.echo("   Cleaning up running containers, please wait...")
            click.echo("   " + click.style("(Please don't spam Ctrl+C, cleanup is in progress)", fg="yellow"))
        else:
            click.echo("\n⚠️  Multiple interrupts detected - please wait for cleanup to complete")

    # Register signal handler
    old_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        # Phase 1: Validate domain
        click.echo(f"🔍 Validating domain {domain}...")
        try:
            orchestrator.validate_domain(domain)
            click.echo("✓ Domain configuration valid")
        except Exception as e:
            raise DomainError(f"Domain validation failed: {e}") from e

        # Phase 2: Discover compose files
        click.echo("📋 Discovering challenge environments...")
        domains_root = orchestrator.manifest_loader.domains_root
        domain_path = domains_root / domain
        sandbox_envs_path = domain_path / "server" / "config" / "environments" / "sandbox"
        permanent_envs_path = domain_path / "server" / "config" / "environments" / "permanent"

        # Find all compose files from both sandbox and permanent environments
        compose_files = []

        # Collect sandbox environments
        if sandbox_envs_path.exists():
            sandbox_files = list(sandbox_envs_path.rglob("*.compose.yml"))
            compose_files.extend(sandbox_files)
            if sandbox_files:
                click.echo(f"  Found {len(sandbox_files)} sandbox environment(s)")

        # Collect permanent environments
        if permanent_envs_path.exists():
            permanent_files = list(permanent_envs_path.glob("*.compose.yml"))
            compose_files.extend(permanent_files)
            if permanent_files:
                click.echo(f"  Found {len(permanent_files)} permanent environment(s)")

        if not compose_files:
            click.echo(f"⚠️  No compose files found in {domain_path / 'server' / 'config' / 'environments'}")
            return

        # Apply pattern filter if specified
        if pattern:
            compose_files = [f for f in compose_files if pattern in str(f)]
            if not compose_files:
                click.echo(f"⚠️  No compose files matching pattern '{pattern}'")
                return

        click.echo(f"✓ Found {len(compose_files)} total environment(s) to test")
        if pattern:
            click.echo(f"  (filtered by pattern: {pattern})")
        click.echo()

        # Keep track of base environments path for relative naming
        envs_base_path = domain_path / "server" / "config" / "environments"

        # Phase 3: Run preflight checks in parallel
        click.echo(f"🚀 Starting preflight checks (concurrency: {concurrency}, timeout: {timeout}s)...")
        click.echo("─" * 80)

        results = []
        failed_count = 0
        success_count = 0
        start_time = asyncio.get_event_loop().time()

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def test_environment(compose_file: Path) -> dict:
            """Test a single environment."""
            nonlocal failed_count, success_count

            # Get environment name (relative path from environments dir)
            env_name = str(compose_file.relative_to(envs_base_path))

            # Check for cancellation
            if cancellation_requested:
                return {
                    "name": env_name,
                    "status": "cancelled",
                    "message": "Cancelled by user",
                }

            async with semaphore:
                if fail_fast and failed_count > 0:
                    return {
                        "name": env_name,
                        "status": "skipped",
                        "message": "Skipped due to previous failure (--fail-fast)",
                    }

                # Check again inside semaphore as value could have changed
                if cancellation_requested:
                    return {  # type: ignore[unreachable]
                        "name": env_name,
                        "status": "cancelled",
                        "message": "Cancelled by user",
                    }

                if verbose:
                    click.echo(f"⏳ Testing: {env_name}")
                test_start = asyncio.get_event_loop().time()

                # Run the test in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor, _test_single_environment, compose_file, timeout, verbose
                    )

                test_elapsed = asyncio.get_event_loop().time() - test_start

                # Update counters
                if result["status"] == "success":
                    success_count += 1
                    click.echo(f"✓ {click.style('PASS', fg='green')}: {env_name} ({int(test_elapsed)}s)")
                elif result["status"] == "failed":
                    failed_count += 1
                    click.echo(f"✗ {click.style('FAIL', fg='red')}: {env_name} ({int(test_elapsed)}s)")
                    if result.get("message"):
                        # Always show error message for failures
                        click.echo(f"  Error: {result['message']}")
                else:
                    click.echo(f"⊘ {click.style('SKIP', fg='yellow')}: {env_name}")

                return result

        # Run all tests
        tasks = [test_environment(compose_file) for compose_file in compose_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions from gather
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results[i] = {
                    "name": str(compose_files[i].relative_to(envs_base_path)),
                    "status": "failed",
                    "message": f"Unexpected error: {result}",
                }
                failed_count += 1

        elapsed = asyncio.get_event_loop().time() - start_time

        # Phase 4: Report results
        click.echo()
        click.echo("─" * 80)

        if cancellation_requested:
            click.echo("📊 Preflight Check Results " + click.style("(CANCELLED)", fg="yellow", bold=True))
        else:
            click.echo("📊 Preflight Check Results")

        click.echo("─" * 80)
        click.echo(f"Total: {len(compose_files)}")
        click.echo(f"{click.style('Passed', fg='green')}: {success_count}")
        click.echo(f"{click.style('Failed', fg='red')}: {failed_count}")
        click.echo(f"Time: {elapsed:.1f}s")
        click.echo()

        # Show failed environments if any
        if failed_count > 0:
            click.echo(f"{click.style('Failed Environments:', fg='red', bold=True)}")
            for result in results:
                if isinstance(result, dict) and result["status"] == "failed":
                    click.echo(f"  • {result['name']}")
                    if verbose:
                        click.echo(f"    {result['message']}")
            click.echo()

            if cancellation_requested:
                raise DomainError(f"Preflight check cancelled by user ({failed_count} failures before cancellation)")
            else:
                raise DomainError(f"Preflight check failed: {failed_count} environment(s) failed")
        elif cancellation_requested:
            click.echo(click.style("⚠️  Preflight check was cancelled by user", fg="yellow", bold=True))
            raise DomainError("Preflight check cancelled by user")
        else:
            click.echo(f"{click.style('✓ All environments passed!', fg='green', bold=True)}")

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, old_handler)


def _test_single_environment(compose_file: Path, timeout: int, verbose: bool) -> dict:
    """Test a single environment by bringing it up and checking health.

    This runs in a thread pool to avoid blocking the async event loop.
    """
    import os
    import re
    import subprocess
    import tempfile
    import time
    import uuid

    import yaml

    from saber.server.execution.sandbox.compose_health_checker import ComposeHealthChecker
    from saber.server.execution.sandbox.environment_config import ComposeEnvironmentConfig

    # Generate unique episode ID for this test
    episode_id = str(uuid.uuid4())
    project_name = f"preflight-{episode_id}"

    # Create environment config
    config = ComposeEnvironmentConfig(
        episode_id=episode_id,
        project_name=project_name,
        config_type="sandbox",
    )

    env_vars = config.to_env_dict()

    # Process compose file to inject network (if needed) and resolve variables
    processed_compose_path = None
    env_name = str(compose_file.name)

    try:
        # Read and process compose file
        with open(compose_file) as f:
            compose_content = f.read()

        # Manually resolve environment variables in the compose content
        # This ensures ${EPISODE_ID} gets replaced before we parse the YAML
        def replace_env_var(match: Any) -> str:
            var_expr = match.group(1)
            # Handle ${VAR:-default} syntax
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return str(env_vars.get(var_name, default))
            else:
                return str(env_vars.get(var_expr, match.group(0)))

        # Replace ${VAR} and ${VAR:-default} patterns
        resolved_content = re.sub(r"\$\{([^}]+)\}", replace_env_var, compose_content)

        # Now parse the resolved YAML
        compose_data = yaml.safe_load(resolved_content)

        # Inject network if needed (same as ComposeOrchestrator does)
        if "networks" not in compose_data:
            compose_data["networks"] = {}

        if "saber-episode-network" not in compose_data["networks"]:
            compose_data["networks"]["saber-episode-network"] = {"driver": "bridge"}

        # Create temporary processed file with resolved variables
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
            yaml.dump(compose_data, tmp)
            processed_compose_path = tmp.name

        # Prepare environment
        env = os.environ.copy()
        env.update(env_vars)

        # Start environment
        if verbose:
            click.echo(f"  → Starting containers for {env_name}...")

        start_cmd = ["docker", "compose", "-f", processed_compose_path, "-p", project_name, "up", "-d"]

        result = subprocess.run(start_cmd, env=env, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return {
                "name": env_name,
                "status": "failed",
                "message": f"Failed to start: {result.stderr[:200]}",
            }

        if verbose:
            click.echo("  → Containers started, checking health...")

        # Wait for health checks with periodic updates
        health_checker = ComposeHealthChecker()

        try:
            # Custom health check with periodic updates
            start_time = time.time()
            check_interval = 2.0
            update_interval = 10  # Show update every 10 seconds
            last_update_time = 0

            while True:
                elapsed = time.time() - start_time

                # Check if we've exceeded timeout
                if elapsed > timeout:
                    raise TimeoutError(f"Health check timed out after {timeout}s")

                # Get current health status
                health_summary = health_checker.get_service_health_summary(processed_compose_path, project_name)

                # Show periodic updates
                if verbose and (elapsed - last_update_time >= update_interval):
                    healthy = health_summary["healthy_count"]
                    total = health_summary["total_count"]
                    click.echo(f"  → [{int(elapsed)}s] Health: {healthy}/{total} services healthy")

                    # Show details of unhealthy services
                    for svc_name, svc_info in health_summary["services"].items():
                        if not svc_info.get("healthy", False):
                            reason = svc_info.get("reason", "unknown")
                            click.echo(f"     • {svc_name}: {reason}")

                    last_update_time = int(elapsed)

                # Check if all services are healthy
                if health_summary["overall_healthy"]:
                    if verbose:
                        click.echo(f"  → All services healthy after {int(elapsed)}s")
                    return {
                        "name": env_name,
                        "status": "success",
                        "message": f"All services healthy ({int(elapsed)}s)",
                    }

                # Wait before next check
                time.sleep(check_interval)

        except TimeoutError as e:
            return {
                "name": env_name,
                "status": "failed",
                "message": str(e),
            }
        except Exception as e:
            return {
                "name": env_name,
                "status": "failed",
                "message": f"Health check failed: {str(e)[:200]}",
            }

    except subprocess.TimeoutExpired:
        return {
            "name": env_name,
            "status": "failed",
            "message": "Docker compose up timed out",
        }
    except Exception as e:
        return {
            "name": env_name,
            "status": "failed",
            "message": f"Error: {str(e)[:200]}",
        }
    finally:
        # Always cleanup
        if verbose:
            click.echo(f"  → Cleaning up {env_name}...")

        try:
            # Stop and remove containers
            down_cmd = ["docker", "compose", "-p", project_name, "down", "-v", "--remove-orphans"]
            subprocess.run(down_cmd, capture_output=True, timeout=30)

            # Remove temporary file
            if processed_compose_path and Path(processed_compose_path).exists():
                Path(processed_compose_path).unlink()
        except Exception:
            pass  # Best effort cleanup


def main() -> None:
    """Main entry point for console script."""
    try:
        cli()
    except DomainError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e
    except KeyboardInterrupt:
        click.echo("\nAborted by user", err=True)
        raise click.Abort() from None


if __name__ == "__main__":
    main()
