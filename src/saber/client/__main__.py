#!/usr/bin/env python3
"""
SABER Client CLI - Simple Entry Point

Simple command-line interface for testing agents against SABER server:
    python -m saber.client --agent <path_to_agent.py>
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from .harness_models import SABERHarnessConfig
from .saber_harness import SABERHarness


class JSONFormatter(logging.Formatter):
    """Minimal JSON log formatter (adds level, logger, message, and extra fields)."""

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        # Include any custom attributes (simple heuristic: skip built-ins)
        for k, v in record.__dict__.items():
            if k not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            } and not k.startswith("_"):
                try:
                    json.dumps({k: v})  # ensure serializable
                    base[k] = v
                except Exception:
                    base[k] = str(v)
        return json.dumps(base, ensure_ascii=False)


def setup_client_logging(
    verbose: bool = False, log_to_file: bool = True, log_dir: Optional[Path] = None
) -> logging.Logger:
    """Setup initial logging. Will be enhanced later when harness creates its timestamp directory."""
    if log_to_file:
        # For now, create minimal console logging
        # We'll redirect to harness logging directory after harness initialization
        logging.basicConfig(
            level=logging.WARNING,  # Minimal console output
            format="%(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        print("📝 Detailed logs will be integrated with harness logging directory")
        print()

        return logging.getLogger("saber.client")
    else:
        # Standard console logging
        logging.basicConfig(
            level=getattr(logging, "DEBUG" if verbose else "INFO"), format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger("saber.client")


def setup_integrated_logging(harness_log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Setup integrated logging that uses the harness timestamp directory."""
    # Create client logs directory within harness directory
    client_log_dir = harness_log_dir / "client-logs"
    client_log_dir.mkdir(exist_ok=True)

    # Create client log file
    log_file = client_log_dir / "saber_client.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # File handler - captures everything
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Console handler - only warnings and errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    print(f"📝 Client logs integrated into: {log_file}")
    print(f"🔍 Monitor with: tail -f {log_file}")
    print()

    return logging.getLogger("saber.client")


async def run_unified_benchmark(
    config: SABERHarnessConfig,
) -> None:
    """Run unified benchmark mode using configuration from YAML file."""

    # Set environment variable for debug mode (if configured)
    debug_mode = config.agent_config.get("debug_mode", False)
    if debug_mode:
        os.environ["SABER_DEBUG_MODE"] = "true"

    # Stage 1: Initial logging setup
    logger = setup_client_logging(
        verbose=(config.log_level.upper() == "DEBUG"),
        log_to_file=True,  # Always use file logging with config-driven approach
    )

    try:
        # Initialize harness (agent will be loaded in its own container)
        logger.info("� Initializing SABER harness with configuration")

        # Create harness
        harness = SABERHarness(config)

        # Validate agent path is provided
        if not config.agent_path:
            raise ValueError("Agent path must be specified in configuration")

        # Initialize with agent from config
        await harness.initialize_with_file(config.agent_path)

        # Stage 2: Integrate logging with harness timestamp directory
        if hasattr(harness, "session_log_dir") and harness.session_log_dir:
            setup_integrated_logging(harness.session_log_dir, verbose=(config.log_level.upper() == "DEBUG"))
            logger = logging.getLogger("saber.client")  # Get updated logger

        logger.info("✅ SABER harness initialized successfully")

        # Connect to server
        logger.info(f"🔗 Connecting to SABER server: {config.server_url}")
        logger.info(f"🔗 MCP server: {config.mcp_url}")

        logger.info("🚀 Starting agent execution...")

        results = await harness.run()

        # Results display
        if results.success:
            logger.info("🎉 EXECUTION COMPLETE!")
            successful = results.successful_episodes
            total = results.total_episodes
            logger.info(f"📊 Results: {successful}/{total} episodes successful")
        else:
            logger.error("❌ EXECUTION FAILED")

        # Display episode details if available
        if results.episode_results:
            logger.info("📝 Episode Details:")
            for episode_result in results.episode_results:
                status = "✅" if episode_result.success else "❌"
                task_id = episode_result.task_id
                attempt = episode_result.attempt
                reason = episode_result.termination_reason or "unknown"
                detail = f"  {status} {task_id} (attempt {attempt}): {reason}"
                logger.info(detail)

    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if config.log_level.upper() == "DEBUG":
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        sys.exit(1)


def main() -> None:
    """Simple CLI entry point that uses YAML configuration."""
    parser = argparse.ArgumentParser(
        description="SABER Client - Unified Agent Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with specific config file
  python -m saber.client --config /path/to/harness.yaml

  # Auto-detect config from current directory
  python -m saber.client

  # Excytin demo
  docker exec -it saber-excytin-client uv run python -m saber.client --config /app/client/harness.yaml
        """,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        help="Path to harness configuration YAML file (default: auto-detect harness.yaml in current directory)",
    )

    args = parser.parse_args()

    # Load configuration file
    from .config_loader import HarnessConfigLoader

    if args.config:
        # Use explicitly provided config path
        config_file_path = Path(args.config)
        if not config_file_path.exists():
            print(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)
    else:
        # Look for harness.yaml in current directory
        config_file_path = Path("harness.yaml")
        if not config_file_path.exists():
            print("❌ No configuration file found. Please provide --config or create harness.yaml in current directory")
            sys.exit(1)

    try:
        config = HarnessConfigLoader.load_from_file(config_file_path)
        agent_path = config.agent_path
        if not agent_path:
            print("❌ Agent path must be specified in configuration file")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        sys.exit(1)

    # Validate agent file exists
    if not Path(agent_path).exists():
        print(f"❌ Agent file not found: {agent_path}")
        sys.exit(1)

    # EXACT INSPECT-AI PATTERN - define async function and let display handle everything
    async def run_task_app() -> None:
        """All SABER work happens here - just like inspect-ai's eval_async."""
        await run_unified_benchmark(config)

    # EXACT INSPECT-AI PATTERN - let task_display handle event loop
    try:
        from inspect_ai._display.core.active import display as task_display

        task_display().run_task_app(run_task_app)
    except ImportError:
        # Fallback if inspect-ai not available
        print("⚠️ inspect-ai not available, using basic async mode")
        asyncio.run(run_task_app())
    except asyncio.CancelledError:
        # Normal cleanup - inspect-ai cancels tasks during shutdown
        # This is expected behavior, don't show as error
        pass
    except KeyboardInterrupt:
        # User interrupted - clean exit
        sys.exit(0)


if __name__ == "__main__":
    main()
