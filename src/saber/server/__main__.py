"""
SABER Server CLI Entry Point

Provides command-line interface for starting SABER domain servers.
Usage: python -m saber.server --start --domain <domain_name> [options]
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

from .session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


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

    tasks_config = os.path.join(config_dir, "tasks.yaml")
    environments_config = os.path.join(config_dir, "environments.yaml")

    if not os.path.exists(tasks_config):
        raise FileNotFoundError(f"Tasks configuration file not found: {tasks_config}")

    if not os.path.exists(environments_config):
        raise FileNotFoundError(f"Environments configuration file not found: {environments_config}")

    return tasks_config, environments_config


async def start_server(args: argparse.Namespace) -> None:
    """Start the SABER server with the given arguments."""

    try:
        # Find and validate configuration
        config_dir = find_config_directory(args.domain, args.config_dir)
        tasks_config, environments_config = validate_config_files(config_dir)

        logger.info(f"Starting SABER server for domain: {args.domain}")
        logger.info(f"Configuration directory: {config_dir}")
        logger.info(f"Tasks config: {tasks_config}")
        logger.info(f"Environments config: {environments_config}")
        logger.info(f"REST API: {args.host}:{args.port}")
        logger.info(f"MCP API: {args.host}:{args.mcp_port}")

        # Initialize SessionManager
        session_manager = SessionManager(
            domain_name=args.domain,
            config_dir=config_dir,
            host=args.host,
            port=args.port,
            mcp_host=args.host,
            mcp_port=args.mcp_port,
        )

        logger.info("SessionManager initialized successfully")

        # Start the server
        logger.info("Starting SABER server...")
        await session_manager.start_server()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping server...")
    except Exception as e:
        logger.error(f"Failed to start SABER server: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = setup_cli()
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.start:
        parser.error("Please specify --start to start the server")

    # Run the server
    try:
        asyncio.run(start_server(args))
    except KeyboardInterrupt:
        logger.info("Server stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
