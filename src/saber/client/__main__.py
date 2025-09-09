#!/usr/bin/env python3
"""
SABER Client CLI - Simple Entry Point

Simple command-line interface for testing agents against SABER server:
    python -m saber.client --agent <path_to_agent.py>
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config_loader import SABERConfigLoader
from .models import SABERConfig


def setup_client_logging(
    verbose: bool = False, log_to_file: bool = True, log_dir: Optional[Path] = None
) -> logging.Logger:
    """Setup client logging with timestamped directory."""
    if log_to_file:
        # Create timestamped log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if log_dir:
            log_base_dir = Path(log_dir)
        else:
            log_base_dir = Path("logs")

        # Create timestamped directory
        timestamped_dir = log_base_dir / f"saber_client_{timestamp}"
        timestamped_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        log_file = timestamped_dir / "saber_client.log"

        # Configure root logger - this will catch ALL loggers including inspect_ai
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        root_logger.handlers.clear()

        # File handler - captures everything
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Console handler - only warnings and errors to keep output clean
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Configure specific logger levels for better control
        logging.getLogger("saber").setLevel(logging.DEBUG if verbose else logging.INFO)
        logging.getLogger("inspect_ai").setLevel(logging.DEBUG if verbose else logging.INFO)
        logging.getLogger("docker").setLevel(logging.WARNING)  # Docker is too chatty
        logging.getLogger("urllib3").setLevel(logging.WARNING)  # HTTP requests are too chatty

        print(f"📝 Logs will be written to: {log_file}")
        print(f"🔍 Monitor with: tail -f {log_file}")
        print()

        return logging.getLogger("saber.client")
    else:
        # Standard console logging
        logging.basicConfig(
            level=getattr(logging, "DEBUG" if verbose else "INFO"), format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Still configure specific loggers for console mode
        logging.getLogger("docker").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        return logging.getLogger("saber.client")


def main() -> None:
    """Simple CLI entry point that uses YAML configuration."""
    parser = argparse.ArgumentParser(
        description="SABER Client - Unified Agent Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with specific config file
  python -m saber.client --config /path/to/saber.yaml

  # Auto-detect config from current directory
  python -m saber.client

  # Excytin demo
  docker exec -it saber-excytin-client uv run python -m saber.client --config /app/client/saber.yaml
        """,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        help="Path to SABER configuration YAML file (default: auto-detect saber.yaml in current directory)",
    )

    # Logging options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument("--no-log-file", action="store_true", help="Disable file logging (console only)")

    args = parser.parse_args()

    # Setup logging FIRST - before any other operations
    logger = setup_client_logging(
        verbose=args.verbose, log_to_file=not args.no_log_file, log_dir=None  # Will be set from config if available
    )

    if args.config:
        # Use explicitly provided config path
        config_file_path = Path(args.config)
        if not config_file_path.exists():
            print(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)
    else:
        # Look for saber.yaml in current directory
        config_file_path = Path("saber.yaml")
        if not config_file_path.exists():
            print("❌ No configuration file found. Please provide --config or create saber.yaml in current directory")
            sys.exit(1)

    try:
        config: SABERConfig = SABERConfigLoader.load_from_file(config_file_path)

        # Update logging with config's log directory if specified
        if config.log_dir and not args.no_log_file:
            logger.info(f"Updating log directory to: {config.log_dir}")
            # Re-setup logging with config's log directory
            logger = setup_client_logging(verbose=args.verbose, log_to_file=True, log_dir=Path(config.log_dir))

        agent_path = config.agent_path
        agent_id = getattr(config, "agent_id", None)

        if not agent_path and not agent_id:
            logger.error("Either agent_path or agent_id must be specified in configuration file")
            print("❌ Either agent_path or agent_id must be specified in configuration file")
            sys.exit(1)

        logger.info(f"Loaded configuration from: {config_file_path}")
        if agent_id:
            logger.info(f"Agent ID: {agent_id}")
        if agent_path:
            logger.info(f"Agent path: {agent_path}")
        logger.info(f"SABER REST URL: {config.saber_rest_url}")
        logger.info(f"SABER MCP URL: {config.saber_mcp_url}")

    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        print(f"❌ Error loading config file: {e}")
        sys.exit(1)

    # Validate agent file exists (only if using agent_path)
    if agent_path and not Path(agent_path).exists():
        logger.error(f"Agent file not found: {agent_path}")
        print(f"❌ Agent file not found: {agent_path}")
        sys.exit(1)

    logger.info("Starting SABER eval_async execution")

    if agent_id:
        print(f"🚀 Starting SABER evaluation with agent ID: {agent_id}")
    elif agent_path:
        print(f"🚀 Starting SABER evaluation with agent: {Path(agent_path).name}")
    else:
        print("🚀 Starting SABER evaluation")

    print(f"📊 Server: {config.saber_rest_url}")
    print(f"🔗 MCP: {config.saber_mcp_url}")
    print()

    # INSPECT-AI EVAL_ASYNC PATTERN - eval_async controls everything
    async def run_task_app() -> None:
        """Run SABER via inspect_ai eval_async for full UI and dataset iteration."""
        logger.info("Starting eval_async task app")

        # Import inspect_ai modules only when needed
        from .inspect_ai import run_saber_eval_async

        # eval_async becomes the main entrypoint - handles UI, dataset iteration, everything
        await run_saber_eval_async(config)
        logger.info("eval_async task app completed")

    try:
        logger.info("Starting inspect_ai task display")

        # Import inspect_ai display module only when needed
        from inspect_ai._display.core.active import display as task_display

        task_display().run_task_app(run_task_app)
        logger.info("inspect_ai task display completed")
    except asyncio.CancelledError:
        # Normal cleanup - inspect-ai cancels tasks during shutdown
        # This is expected behavior, don't show as error
        logger.info("Task cancelled during shutdown (normal)")
        pass
    except KeyboardInterrupt:
        # User interrupted - clean exit
        logger.info("User interrupted execution")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
