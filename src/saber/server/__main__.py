"""
SABER Server CLI Entry Point

Log Category: CONFIG

Provides command-line interface for starting SABER domain servers.
Usage: python -m saber.server --start --domain <domain_name> [options]
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..logging_config import LogCategory, LoggingConfig, get_saber_logger, init_logging
from .session_manager import SessionManager

logger = get_saber_logger(LogCategory.CONFIG, __name__)


def setup_cli() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="SABER Security Agent Benchmarking Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m saber.server --start --domain cybench --domains-root ./domains
  python -m saber.server --start --domain cybench --domains-root ./domains --port 8080
  python -m saber.server --start --domain cybench --domains-root ./domains --dry-run
        """,
    )

    parser.add_argument("--start", action="store_true", help="Start the SABER server")

    parser.add_argument(
        "--domain",
        type=str,
        default=os.getenv("SABER_DOMAIN"),
        help="Security domain name (e.g., pentest_demo, malware_analysis). Can also be set via SABER_DOMAIN env var.",
    )

    parser.add_argument(
        "--domains-root",
        type=str,
        default=None,
        help="Root directory containing domain definitions (required unless SABER_DOMAINS_ROOT is set)",
    )

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address (default: 0.0.0.0)")

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SABER_PORT", "8000")),
        help="REST API port (default: 8000, or SABER_PORT env var)",
    )

    parser.add_argument(
        "--mcp-port",
        type=int,
        default=int(os.getenv("SABER_MCP_PORT", "8001")),
        help="MCP API port (default: 8001, or SABER_MCP_PORT env var)",
    )

    parser.add_argument(
        "--server-network", type=str, help="Docker network name where SABER server runs (for orchestrator connectivity)"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and manifest without starting the server",
    )

    return parser


class ServerConfigError(Exception):
    """Exception raised for server configuration errors."""

    pass


def resolve_domains_root(domains_root_arg: str | None) -> Path:
    """Resolve domains root directory with fail-fast validation."""
    domains_root = domains_root_arg or os.getenv("SABER_DOMAINS_ROOT")

    if not domains_root:
        raise ServerConfigError(
            "Domains root must be specified via --domains-root argument or SABER_DOMAINS_ROOT environment variable"
        )

    domains_root_path = Path(domains_root).resolve()

    if not domains_root_path.exists():
        raise ServerConfigError(f"Domains root directory does not exist: {domains_root_path}")

    if not domains_root_path.is_dir():
        raise ServerConfigError(f"Domains root is not a directory: {domains_root_path}")

    return domains_root_path


def load_domain_manifest(domains_root: Path, domain: str) -> dict[str, Any]:
    """Load and validate domain manifest with fail-fast semantics."""
    domain_path = domains_root / domain
    manifest_path = domain_path / "domain.yaml"

    if not domain_path.exists():
        raise ServerConfigError(f"Domain directory does not exist: {domain_path}")

    if not domain_path.is_dir():
        raise ServerConfigError(f"Domain path is not a directory: {domain_path}")

    if not manifest_path.exists():
        raise ServerConfigError(
            f"Domain manifest not found: {manifest_path}\nEvery domain must have a domain.yaml manifest file."
        )

    try:
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ServerConfigError(f"Invalid YAML in domain manifest {manifest_path}: {e}") from e

    if not isinstance(manifest, dict):
        raise ServerConfigError(f"Domain manifest must be a YAML object, got {type(manifest)}: {manifest_path}")

    # Basic manifest validation
    required_fields = ["schemaVersion", "domain"]
    missing_fields = [field for field in required_fields if field not in manifest]
    if missing_fields:
        raise ServerConfigError(f"Domain manifest missing required fields: {missing_fields}\nManifest: {manifest_path}")

    domain_info = manifest.get("domain", {})
    if not isinstance(domain_info, dict):
        raise ServerConfigError(f"Domain manifest 'domain' field must be an object: {manifest_path}")

    manifest_slug = domain_info.get("slug")
    if manifest_slug != domain:
        raise ServerConfigError(
            f"Domain manifest slug mismatch: expected '{domain}', got '{manifest_slug}'\nManifest: {manifest_path}"
        )

    return manifest


def resolve_domain_paths(domains_root: Path, domain: str, manifest: dict[str, Any]) -> dict[str, Path]:
    """Resolve and validate all domain paths with fail-fast semantics."""
    domain_path = domains_root / domain

    paths = {
        "domain_root": domain_path,
        "server_root": domain_path / "server",
        "config_dir": domain_path / "server" / "config",
        "tasks_config": domain_path / "server" / "config" / "tasks",
        "environments_config": domain_path / "server" / "config" / "environments",
        "data_dir": domain_path / "server" / "data",
        "logs_dir": domain_path / "server" / "logs",
        "manifest_path": domain_path / "domain.yaml",
    }

    # Validate required paths exist
    required_paths = ["server_root", "config_dir", "tasks_config", "environments_config"]
    missing_paths = []

    for path_name in required_paths:
        path = paths[path_name]
        if not path.exists():
            missing_paths.append(f"{path_name}: {path}")

    if missing_paths:
        raise ServerConfigError(
            f"Required domain paths missing for '{domain}':\n" + "\n".join(f"  - {path}" for path in missing_paths)
        )

    # Create data and logs directories if they don't exist
    for dir_name in ["data_dir", "logs_dir"]:
        path = paths[dir_name]
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"Created directory: {path}",
                    extra={"event": "directory_created", "path": str(path), "domain": domain},
                )
            except OSError as e:
                raise ServerConfigError(f"Failed to create {dir_name}: {path} - {e}") from e

    return paths


async def start_server(args: argparse.Namespace) -> None:
    """Start the SABER server with the given arguments."""
    session_manager: SessionManager | None = None
    shutdown_event = asyncio.Event()

    async def shutdown_handler() -> None:
        """Handle graceful shutdown."""
        logger.info(
            "Shutdown signal received; initiating graceful shutdown",
            extra={"event": "server_shutdown_initiated"},
        )
        if session_manager:
            try:
                await session_manager.shutdown()
                logger.info(
                    "Graceful shutdown completed",
                    extra={"event": "server_shutdown_completed"},
                )
            except Exception as e:
                logger.exception(
                    "Error during graceful shutdown",
                    extra={"event": "server_shutdown_failed", "error": str(e)},
                )
        shutdown_event.set()

    # Setup signal handlers for graceful shutdown using asyncio
    loop = asyncio.get_running_loop()

    def signal_handler() -> None:
        logger.info(
            "Signal received; scheduling shutdown",
            extra={"event": "server_shutdown_signal_received"},
        )
        asyncio.create_task(shutdown_handler())

    # Register signal handlers (only available on Unix systems)
    try:
        if hasattr(signal, "SIGTERM"):
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
        if hasattr(signal, "SIGINT"):
            loop.add_signal_handler(signal.SIGINT, signal_handler)
        logger.info(
            "Signal handlers registered for graceful shutdown",
            extra={"event": "server_signal_handlers_registered"},
        )
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        logger.warning(
            "Signal handlers not available on this platform",
            extra={"event": "server_signal_handlers_unavailable", "platform": sys.platform},
        )

    try:
        # Resolve domains root and load manifest
        domains_root = resolve_domains_root(args.domains_root)
        manifest = load_domain_manifest(domains_root, args.domain)
        domain_paths = resolve_domain_paths(domains_root, args.domain, manifest)

        # Extract configuration paths
        config_dir = str(domain_paths["config_dir"])
        tasks_config = str(domain_paths["tasks_config"])
        environments_config = str(domain_paths["environments_config"])

        # Enhanced logging with manifest context
        manifest_info = manifest.get("domain", {})
        logger.info(
            "Starting SABER server with manifest-driven configuration",
            extra={
                "event": "server_starting",
                "domain": args.domain,
                "domain_name": manifest_info.get("name", args.domain),
                "schema_version": manifest.get("schemaVersion"),
                "rest_host": args.host,
                "rest_port": args.port,
                "mcp_port": args.mcp_port,
                "domains_root": str(domains_root),
                "config_dir": config_dir,
                "tasks_config": tasks_config,
                "environments_config": environments_config,
                "manifest_path": str(domain_paths["manifest_path"]),
            },
        )

        # Dry-run mode: validate configuration and exit
        if args.dry_run:
            logger.info(
                "Dry-run mode: configuration validation successful",
                extra={
                    "event": "server_dry_run_success",
                    "domain": args.domain,
                    "manifest_valid": True,
                    "paths_valid": True,
                },
            )
            return

        # Initialize SessionManager with manifest information
        session_manager = SessionManager(
            domain_name=args.domain,
            config_dir=config_dir,
            host=args.host,
            port=args.port,
            mcp_host=args.host,
            mcp_port=args.mcp_port,
            manifest=manifest,
            manifest_path=str(domain_paths["manifest_path"]),
            data_dir=str(domain_paths["data_dir"]),
        )

        logger.info(
            "SessionManager initialized with manifest context",
            extra={
                "event": "session_manager_initialized",
                "domain": args.domain,
                "domain_name": manifest_info.get("name"),
                "capabilities": manifest.get("capabilities", []),
            },
        )

        # Create a task for the server so we can wait for either server completion or shutdown signal
        server_task = asyncio.create_task(session_manager.start_server())

        # Wait for either the server to complete or a shutdown signal
        done, pending = await asyncio.wait(
            [server_task, asyncio.create_task(shutdown_event.wait())], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel any remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        logger.info(
            "Keyboard interrupt received; stopping server",
            extra={"event": "server_shutdown_keyboard_interrupt"},
        )
        if session_manager:
            await session_manager.shutdown()
    except ServerConfigError as e:
        logger.error(
            "Server configuration error",
            extra={
                "event": "server_config_error",
                "error": str(e),
                "domain": args.domain,
                "domains_root": args.domains_root,
            },
        )
        print(f"\n❌ Configuration Error: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print(f"  1. Verify domains root exists: {args.domains_root or os.getenv('SABER_DOMAINS_ROOT', 'NOT SET')}")
        print(f"  2. Check domain structure: domains/{args.domain}/")
        print(f"  3. Validate manifest: uv run scripts/validate_manifest.py domains/{args.domain}/domain.yaml")
        sys.exit(1)
    except Exception as e:
        logger.exception(
            "Failed to start SABER server",
            extra={"event": "server_start_failed", "error": str(e)},
        )
        if session_manager:
            try:
                await session_manager.shutdown()
            except Exception as shutdown_error:
                logger.exception(
                    "Error during emergency shutdown",
                    extra={"event": "server_emergency_shutdown_failed", "error": str(shutdown_error)},
                )
        sys.exit(1)


def setup_server_logging(domain_name: str, config_dir: str, verbose: bool = False) -> LoggingConfig:
    """Configure logging for the SABER server with timestamped log files in server-logs directory."""

    # Generate timestamped filename: saber-server-YYYY-MM-DD_HH-MM-SS.log
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    server_log_filename = f"saber-server-{timestamp}.log"

    # Determine server logs directory
    # If config_dir is absolute (like /app/config), use parent/logs/server-logs
    # If config_dir is relative (like ./domains/excytin_demo/server/config), use relative path
    config_path = Path(config_dir).resolve()

    if str(config_path).startswith("/app/config"):
        # Container environment: /app/config -> /app/logs/server-logs
        server_logs_dir = Path("/app/logs/server-logs")
    else:
        # Local development: find server directory and use logs/server-logs
        server_dir = config_path.parent  # config -> server
        server_logs_dir = server_dir / "logs" / "server-logs"

    # Create the server-logs directory if it doesn't exist
    server_logs_dir.mkdir(parents=True, exist_ok=True)

    # Build logging configuration
    base_config = LoggingConfig.from_env()
    server_config = LoggingConfig(
        level=logging.DEBUG if verbose else base_config.level,
        console=False,  # Disable console logging - file only
        structured=base_config.structured,
        enable_file=True,  # Always enable file logging for server
        log_dir=server_logs_dir,
        file_name=server_log_filename,
        max_bytes=base_config.max_bytes,
        backup_count=base_config.backup_count,
    )

    return init_logging(server_config, force=True)


def main() -> None:
    """Main CLI entry point."""
    # Initialize basic logging first for early messages
    init_logging()

    parser = setup_cli()
    args = parser.parse_args()

    if not args.start:
        parser.error("Please specify --start to start the server")

    if not args.domain:
        parser.error("--domain is required (or set SABER_DOMAIN environment variable)")

    # Find configuration directory early to set up proper logging
    try:
        domains_root = resolve_domains_root(args.domains_root)
        # Simple early path resolution for logging setup
        domain_path = domains_root / args.domain
        config_dir = str(domain_path / "server" / "config")

        # Basic validation that domain and config exist
        if not domain_path.exists():
            raise ServerConfigError(f"Domain directory does not exist: {domain_path}")
        if not Path(config_dir).exists():
            raise ServerConfigError(f"Configuration directory does not exist: {config_dir}")

    except ServerConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup server-specific logging with timestamped files
    logging_config = setup_server_logging(args.domain, config_dir, args.verbose)

    # Recreate logger after reconfiguring logging
    global logger
    logger = get_saber_logger(LogCategory.CONFIG, __name__)

    logger.info(
        "SABER server logging initialized",
        extra={
            "event": "server_logging_initialized",
            "domain": args.domain,
            "log_file": str(logging_config.log_dir / logging_config.file_name),
            "config_dir": config_dir,
        },
    )

    # Validate required arguments
    try:
        domains_root = resolve_domains_root(args.domains_root)
        logger.info(
            "CLI validation successful",
            extra={
                "event": "cli_validation_success",
                "domain": args.domain,
                "domains_root": str(domains_root),
                "dry_run": args.dry_run,
            },
        )
    except ServerConfigError as e:
        print(f"\n❌ Configuration Error: {e}", file=sys.stderr)
        print("\nUsage:")
        print("  python -m saber.server --start --domain DOMAIN --domains-root PATH")
        print("  export SABER_DOMAINS_ROOT=/path/to/domains && python -m saber.server --start --domain DOMAIN")
        print("\nExample:")
        print("  python -m saber.server --start --domain cybench --domains-root ./domains")
        sys.exit(1)

    # Run the server
    try:
        asyncio.run(start_server(args))
    except KeyboardInterrupt:
        logger.info(
            "Server stopped",
            extra={"event": "server_stopped"},
        )
        sys.exit(0)
    except Exception as e:
        logger.exception("Unhandled server error", extra={"event": "server_unhandled_error", "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
