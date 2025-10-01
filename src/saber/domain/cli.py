"""CLI implementation for SABER domain orchestration.

This module provides the Click-based CLI interface for domain operations
with CLI-generated environment variables - no manual .env file editing required.
"""

import sys
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

        # Override log level if verbose flag is set
        effective_log_level = "DEBUG" if verbose else log_level

        # Start domain with server-only orchestration
        orchestrator.start_domain(
            domain=domain,
            rest_port=rest_port,
            mcp_port=mcp_port,
            log_level=effective_log_level,
            build=build,
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
@click.option("--build", is_flag=True, help="Build images before starting")  # type: ignore[misc]
@click.option("--log-level", default="INFO", help="Logging level")  # type: ignore[misc]
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose logging (sets log level to DEBUG)"
)  # type: ignore[misc]
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
    log_level: str,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Run SABER evaluation tests against domain server.

    By default, the server is kept running after test completion for subsequent runs.
    Use --stop-after to explicitly stop the server when done.

    Examples:
        saber-domain test cybench
        saber-domain test cybench --saber-yaml custom.yaml
        saber-domain test cybench --stop-after --build
    """
    try:
        orchestrator = _create_orchestrator(ctx.obj.get("domains_root"))

        # Override log level if verbose flag is set
        effective_log_level = "DEBUG" if verbose else log_level

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
                build=build,
                log_level=effective_log_level,
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
    build: bool,
    log_level: str,
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
        orchestrator, domain, rest_port, mcp_port, build, log_level, dry_run
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

        # Extract parameters from SABERConfig for CLI call
        rest_url = saber_config.session_config.base_url
        mcp_url = saber_config.session_config.mcp_server_url
        model = saber_config.model

        # Use the first agent for CLI parameters
        if saber_config.agents:
            agent_id = saber_config.agents[0].id
            task_ids = ",".join(saber_config.agents[0].tasks)
        else:
            raise DomainError("No agents configured in SABER config")

        # Find the repo root .env file path
        repo_root = orchestrator.manifest_loader.domains_root.parent
        env_file_path = repo_root / ".env"

        try:
            click.echo("🎯 Starting SABER client with TUI...")

            # Build command arguments for subprocess call
            cmd_args = [
                sys.executable,
                "-m",
                "saber.client",
                "run",
                "--rest-url",
                rest_url,
                "--mcp-url",
                mcp_url,
                "--model",
                model,
                "--agent-id",
                agent_id,
                "--task-ids",
                task_ids,
                "--domain",
                domain,  # Pass domain for organized logging
            ]

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
    build: bool,
    log_level: str,
    dry_run: bool,
) -> bool:
    """Ensure server is running, return True if we started it."""

    # Handle build flag regardless of server status
    if build:
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
            click.echo(f"🔨 Would rebuild {domain} images")
        else:
            click.echo(f"🔨 Rebuilding {domain} images...")
            orchestrator.build_domain(domain, dry_run=False)
            click.echo("✓ Images rebuilt successfully")

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
        domain, rest_port, mcp_port, log_level, build=False, dry_run=False
    )  # build=False since we already built above

    # Wait for server readiness
    await _wait_for_server_ready(rest_port, mcp_port)
    click.echo(f"✓ Server ready at http://localhost:{rest_port}")
    return True  # We started it


async def _wait_for_server_ready(rest_port: int, mcp_port: int, timeout: int = 30) -> None:
    """Wait for server to be ready by checking REST health endpoint."""
    import asyncio

    import aiohttp

    health_url = f"http://localhost:{rest_port}/api/v1/health"
    start_time = asyncio.get_event_loop().time()

    while (asyncio.get_event_loop().time() - start_time) < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        return  # Server is ready
        except Exception:
            pass  # Server not ready yet

        await asyncio.sleep(1)

    raise DomainError(f"Server did not become ready within {timeout} seconds")


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
