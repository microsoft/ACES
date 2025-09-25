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
from typing import Optional

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
  python -m saber.server --start --domain pentest_demo
  python -m saber.server --start --domain malware_analysis --port 8080
  python -m saber.server --start --domain network_investigation --config-dir ./configs
        """,
    )

    parser.add_argument("--start", action="store_true", help="Start the SABER server")

    parser.add_argument(
        "--domain", type=str, required=True, help="Security domain name (e.g., pentest_demo, malware_analysis)"
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Configuration directory path (default: auto-detect from environment)",
    )

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address (default: 0.0.0.0)")

    parser.add_argument("--port", type=int, default=8000, help="REST API port (default: 8000)")

    parser.add_argument("--mcp-port", type=int, default=8001, help="MCP API port (default: 8001)")

    parser.add_argument(
        "--server-network", type=str, help="Docker network name where SABER server runs (for orchestrator connectivity)"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return parser


def find_config_directory(domain_name: str, config_dir_override: Optional[str] = None) -> str:
    """Find configuration directory for the domain."""

    if config_dir_override:
        config_dir = config_dir_override
    else:
        # Try environment variable first
        config_dir_env = os.getenv("SABER_CONFIG_DIR")

        if config_dir_env:
            config_dir = config_dir_env
        else:
            # Try common locations
            possible_paths = [
                f"./config/{domain_name}",
                f"./configs/{domain_name}",
                f"./examples/{domain_name}/server/config",
                "./config",
                "./configs",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    config_dir = path
                    break

    if not config_dir:
        raise FileNotFoundError(
            f"Configuration directory not found for domain '{domain_name}'. "
            f"Please specify --config-dir or set SABER_CONFIG_DIR environment variable."
        )

    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Configuration directory does not exist: {config_dir}")

    return config_dir


def validate_config_files(config_dir: str) -> tuple[str, str]:
    """Validate that required config files exist."""

    tasks_config_dir = os.path.join(config_dir, "tasks")
    environments_config_dir = os.path.join(config_dir, "environments")

    if not os.path.exists(tasks_config_dir):
        raise FileNotFoundError(f"Tasks configuration directory not found: {tasks_config_dir}")

    if not os.path.exists(environments_config_dir):
        raise FileNotFoundError(f"Environments configuration directory not found: {environments_config_dir}")

    return tasks_config_dir, environments_config_dir


async def start_server(args: argparse.Namespace) -> None:
    """Start the SABER server with the given arguments."""
    session_manager: Optional[SessionManager] = None
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
        # Find and validate configuration
        config_dir = find_config_directory(args.domain, args.config_dir)
        tasks_config, environments_config = validate_config_files(config_dir)

        logger.info(
            "Starting SABER server",
            extra={
                "event": "server_starting",
                "domain": args.domain,
                "rest_host": args.host,
                "rest_port": args.port,
                "mcp_port": args.mcp_port,
                "config_dir": config_dir,
                "tasks_config": tasks_config,
                "environments_config": environments_config,
            },
        )

        # Initialize SessionManager
        session_manager = SessionManager(
            domain_name=args.domain,
            config_dir=config_dir,
            host=args.host,
            port=args.port,
            mcp_host=args.host,
            mcp_port=args.mcp_port,
        )

        logger.info(
            "SessionManager initialized",
            extra={"event": "session_manager_initialized", "domain": args.domain},
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
        console=base_config.console,
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

    # Find configuration directory early to set up proper logging
    try:
        config_dir = find_config_directory(args.domain, args.config_dir)
    except FileNotFoundError as e:
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

    # Run the server
    try:
        asyncio.run(start_server(args))
    except KeyboardInterrupt:
        logger.info(
            "Server stopped",
            extra={"event": "server_stopped"},
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
