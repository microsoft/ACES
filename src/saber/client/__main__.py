#!/usr/bin/env python3
"""
SABER Client CLI - Simple Entry Point

Simple command-line interface for testing agents against SABER server:
    python -m saber.client --agent <path_to_agent.py>
"""

import argparse
import asyncio
import importlib.util
import inspect
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, List, Optional

from .harness_models import SABERHarnessConfig
from .saber_harness import SABERHarness


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
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
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


def load_agent_from_path(agent_path: str, agent_class: Optional[str] = None) -> Any:
    """Load agent from file path with automatic detection."""
    # Convert to absolute path
    agent_path = str(Path(agent_path).resolve())

    # Load module from file
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load module from {agent_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find agent class or function
    if agent_class:
        if not hasattr(module, agent_class):
            raise ImportError(f"Agent class '{agent_class}' not found in {agent_path}")
        return getattr(module, agent_class)

    # Auto-detect agent - prefer classes with "agent" in name
    candidates: List[Any] = []
    for name, obj in inspect.getmembers(module):
        if not name.startswith("_"):
            if inspect.isclass(obj):
                if "agent" in name.lower():
                    return obj  # Return immediately if agent in name
                candidates.append(obj)
            elif inspect.isfunction(obj) and callable(obj):
                candidates.append(obj)

    if not candidates:
        raise ImportError(f"No suitable agent class or function found in {agent_path}")

    return candidates[0]  # Return first candidate


async def run_unified_benchmark(
    agent_path: str,
    task_ids: Optional[List[str]] = None,
    agent_class: Optional[str] = None,
    server_url: Optional[str] = None,
    mcp_url: Optional[str] = None,
    env_file: Optional[str] = None,
    log_level: str = "INFO",
    ui_backend: str = "auto",
    ui_enabled: bool = True,
    ui_tool_detail_level: str = "full",
    quiet_logs: bool = False,
) -> None:
    """Run unified benchmark mode - supports single tasks, multiple tasks, or full benchmarks."""

    # Stage 1: Initial logging setup
    logger = setup_client_logging(verbose=(log_level.upper() == "DEBUG"), log_to_file=quiet_logs)

    try:
        # Load agent
        logger.info(f"Loading agent from: {agent_path}")
        agent = load_agent_from_path(agent_path, agent_class)
        logger.info(f"✅ Agent loaded: {agent}")

        # Create harness configuration
        config = SABERHarnessConfig(
            server_url=server_url or "http://saber-excytin-server:8000",
            mcp_url=mcp_url or "http://saber-excytin-server:8001",
            client_id="saber-client-unified",
            request_timeout=60.0,
            task_ids=task_ids,
            log_level=log_level,
            ui_backend=ui_backend,
            ui_enabled=ui_enabled,
            client_log_dir=Path("./logs"),  # Base directory - harness will create timestamp
            enable_container_logging=True,
        )

        # Create harness
        logger.info("🔧 Initializing SABER harness...")
        harness = SABERHarness(config)

        # Initialize with agent
        await harness.initialize_with_file(agent_path, agent_class=agent_class)

        # Stage 2: Integrate logging with harness timestamp directory
        if quiet_logs and hasattr(harness, "session_log_dir") and harness.session_log_dir:
            setup_integrated_logging(harness.session_log_dir, verbose=(log_level.upper() == "DEBUG"))
            logger = logging.getLogger("saber.client")  # Get updated logger

        logger.info("✅ Harness initialized successfully")

        # Auto-detect environment file if not provided
        if not env_file:
            # Look for .env file in agent directory
            agent_dir = Path(agent_path).parent
            potential_env = agent_dir / ".env"
            if potential_env.exists():
                env_file = str(potential_env)
                logger.info(f"📄 Found .env file: {env_file}")
                if not quiet_logs:
                    print(f"📄 Found .env file: {env_file}")

        # Configure URLs from environment if not provided
        final_server_url = server_url or os.getenv("SABER_REST_URL", "http://localhost:8000")
        final_mcp_url = mcp_url or os.getenv("SABER_MCP_URL", "http://localhost:8001")

        logger.info(f"🔗 Connecting to SABER server: {final_server_url}")
        logger.info(f"🔗 MCP server: {final_mcp_url}")
        if not quiet_logs:
            print(f"🔗 Connecting to SABER server: {final_server_url}")
            print(f"🔗 MCP server: {final_mcp_url}")

        # If no task_ids provided, fetch benchmark info and prompt user
        final_task_ids = task_ids
        if not task_ids:
            logger.info("📋 Fetching available tasks...")
            if not quiet_logs:
                print("📋 Fetching available tasks...")
            # Create a temporary REST client to get benchmark info
            from .api.rest_client import SABERRestClient

            if not final_server_url:
                raise ValueError("Server URL not configured")

            temp_client = SABERRestClient(base_url=final_server_url, client_id="temp-client")

            try:
                benchmark_data = await temp_client.get_benchmark()
                tasks_list = benchmark_data["tasks"]

                print(f"\n📊 Found {len(tasks_list)} available tasks:")
                task_list = [task["task_id"] for task in tasks_list]

                for i, task in enumerate(tasks_list, 1):
                    print(f"  {i:2d}. {task['task_id']}")

                print("\nOptions:")
                print(f"  0. Run full benchmark (all {len(task_list)} tasks)")
                print(f"  1-{len(task_list)}. Run specific task")
                print("  Enter comma-separated numbers for multiple tasks (e.g., 1,3,5)")

                choice = input("\nSelect option: ").strip()

                if choice == "0" or choice.lower() in ["all", "full", ""]:
                    final_task_ids = None  # Run all tasks
                    print("🎯 Running full benchmark (all tasks)")
                else:
                    # Parse selection(s)
                    try:
                        selected_indices = [int(x.strip()) for x in choice.split(",")]
                        final_task_ids = []
                        for idx in selected_indices:
                            if 1 <= idx <= len(task_list):
                                final_task_ids.append(task_list[idx - 1])
                            else:
                                print(f"⚠️ Invalid selection: {idx}")

                        if not final_task_ids:
                            print("❌ No valid tasks selected")
                            return

                        if len(final_task_ids) == 1:
                            print(f"🎯 Running single task: {final_task_ids[0]}")
                        else:
                            print(f"🎯 Running {len(final_task_ids)} tasks: {', '.join(final_task_ids)}")
                    except ValueError:
                        print("❌ Invalid input format")
                        return

            except Exception as e:
                logger.error(f"❌ Failed to fetch benchmark data: {e}")
                if not quiet_logs:
                    print(f"❌ Failed to fetch benchmark data: {e}")
                    print("Using provided task_ids or exiting...")
                if not task_ids:
                    return

        # Update the harness config with final task selection
        harness.config.task_ids = final_task_ids
        if final_server_url:
            harness.config.server_url = final_server_url
        if final_mcp_url:
            harness.config.mcp_url = final_mcp_url

        logger.info("🚀 Starting agent execution...")
        if not quiet_logs:
            print("🚀 Starting agent execution...")

        results = await harness.run()

        # Results display
        if not quiet_logs:
            print("\n" + "=" * 60)
        if results.success:
            logger.info("🎉 EXECUTION COMPLETE!")
            if not quiet_logs:
                print("🎉 EXECUTION COMPLETE!")
            successful = results.successful_episodes
            total = results.total_episodes
            logger.info(f"📊 Results: {successful}/{total} episodes successful")
            if not quiet_logs:
                print(f"📊 Results: {successful}/{total} episodes successful")
        else:
            logger.error("❌ EXECUTION FAILED")
            if not quiet_logs:
                print("❌ EXECUTION FAILED")

        # Display episode details if available
        if results.episode_results:
            logger.info("📝 Episode Details:")
            if not quiet_logs:
                print("\n📝 Episode Details:")
            for episode_result in results.episode_results:
                status = "✅" if episode_result.success else "❌"
                task_id = episode_result.task_id
                attempt = episode_result.attempt
                reason = episode_result.termination_reason or "unknown"
                detail = f"  {status} {task_id} (attempt {attempt}): {reason}"
                logger.info(detail)
                if not quiet_logs:
                    print(detail)

        if not quiet_logs:
            print("=" * 60)

    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
        if not quiet_logs:
            print("\n⏹️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if log_level.upper() == "DEBUG":
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        if not quiet_logs:
            print(f"❌ Error: {e}")
        sys.exit(1)


def main() -> None:
    """Simple CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SABER Client - Unified Agent Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with textual UI (clean logs to file)
  python -m saber.client --agent ./my_agent.py --ui textual --quiet-logs

  # Single task execution with rich UI
  python -m saber.client --agent ./my_agent.py --tasks xss_flag_capture --ui rich

  # Multiple specific tasks with plain UI for automation
  python -m saber.client --agent ./my_agent.py --tasks xss_flag_capture,sql_injection --ui plain

  # Full benchmark (all available tasks)
  python -m saber.client --agent ./my_agent.py --tasks all --quiet-logs

  # Excytin demo with textual UI
  python -m saber.client --agent /app/client/demo_agent.py --tasks excytin_demo --ui textual --quiet-logs
        """,
    )

    # Required arguments
    parser.add_argument("--agent", required=True, help="Path to agent file (.py)")

    # Benchmark configuration (all optional)
    parser.add_argument(
        "--tasks", help="Comma-separated task IDs to run, or 'all' for full benchmark (default: interactive prompt)"
    )

    # Optional arguments with smart defaults
    parser.add_argument("--agent-class", help="Specific agent class name (auto-detected if not provided)")
    parser.add_argument(
        "--server-url", help="SABER server URL (default: http://localhost:8000 or SABER_SERVER_URL env var)"
    )
    parser.add_argument(
        "--mcp-url", help="SABER MCP server URL (default: http://localhost:8001 or SABER_MCP_URL env var)"
    )
    parser.add_argument("--env-file", help="Path to .env file (auto-detected in agent directory if not provided)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    # UI Configuration
    parser.add_argument(
        "--ui",
        choices=["auto", "textual", "rich", "plain", "none"],
        default="auto",
        help="UI backend to use (default: auto)",
    )
    parser.add_argument("--no-ui", action="store_true", help="Disable UI (equivalent to --ui none)")
    parser.add_argument(
        "--ui-tool-detail",
        choices=["none", "basic", "full"],
        default="full",
        help="Level of MCP tool call detail to capture (default: full)",
    )
    parser.add_argument(
        "--quiet-logs",
        action="store_true",
        help="Save detailed logs to file and show minimal console output for cleaner UI",
    )

    args = parser.parse_args()

    # Validate agent file exists
    if not Path(args.agent).exists():
        print(f"❌ Agent file not found: {args.agent}")
        sys.exit(1)

    # Parse task IDs if provided
    task_ids = None
    if args.tasks:
        if args.tasks.lower() == "all":
            task_ids = None  # Run all tasks
        else:
            task_ids = [task.strip() for task in args.tasks.split(",")]

    # Handle UI arguments
    ui_backend = "none" if args.no_ui else args.ui
    ui_enabled = not args.no_ui

    # Run unified benchmark mode
    asyncio.run(
        run_unified_benchmark(
            agent_path=args.agent,
            task_ids=task_ids,
            agent_class=args.agent_class,
            server_url=args.server_url,
            mcp_url=args.mcp_url,
            env_file=args.env_file,
            log_level=args.log_level,
            ui_backend=ui_backend,
            ui_enabled=ui_enabled,
            ui_tool_detail_level=args.ui_tool_detail,
            quiet_logs=args.quiet_logs,
        )
    )


if __name__ == "__main__":
    main()
