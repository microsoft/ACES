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
from pathlib import Path
from typing import Any, List, Optional

from .harness_models import SABERHarnessConfig
from .saber_harness import SABERHarness


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
) -> None:
    """Run unified benchmark mode - supports single tasks, multiple tasks, or full benchmarks."""

    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level.upper()), format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Load agent
        print(f"🤖 Loading agent from {agent_path}")
        agent = load_agent_from_path(agent_path, agent_class)
        print(f"✅ Agent loaded: {getattr(agent, '__name__', type(agent).__name__)}")

        # Auto-detect environment file if not provided
        if not env_file:
            # Look for .env file in agent directory
            agent_dir = Path(agent_path).parent
            potential_env = agent_dir / ".env"
            if potential_env.exists():
                env_file = str(potential_env)
                print(f"📄 Found .env file: {env_file}")

        # Configure URLs from environment if not provided
        final_server_url: str = server_url or os.getenv("SABER_SERVER_URL") or "http://localhost:8000"
        final_mcp_url: str = mcp_url or os.getenv("SABER_MCP_URL") or "http://localhost:8001"

        print(f"🔗 Connecting to SABER server: {final_server_url}")
        print(f"🔗 MCP server: {final_mcp_url}")

        # If no task_ids provided, fetch benchmark info and prompt user
        final_task_ids = task_ids
        if not task_ids:
            print("📋 Fetching available tasks...")
            # Create a temporary REST client to get benchmark info
            from .api.rest_client import SABERRestClient

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
                print(f"❌ Failed to fetch benchmark data: {e}")
                print("Using provided task_ids or exiting...")
                if not task_ids:
                    return

            # Create unified configuration
            config = SABERHarnessConfig(
                server_url=final_server_url,
                mcp_url=final_mcp_url,
                task_ids=final_task_ids,
                log_level=log_level,
            )

            # Run unified test
            harness = SABERHarness(config)
            env_path = Path(env_file) if env_file else None
            await harness.initialize(agent, env_file=env_path)

            print("🚀 Starting agent execution...")
            results = await harness.run()

            # Results display
            print("\n" + "=" * 60)
            if results.success:
                print("🎉 EXECUTION COMPLETE!")
                successful = results.successful_episodes
                total = results.total_episodes
                print(f"📊 Results: {successful}/{total} episodes successful")
            else:
                print("❌ EXECUTION FAILED")
                # Note: HarnessRunResult doesn't have an 'error' field, so removing this check

            # Display episode details if available
            if results.episode_results:
                print("\n📝 Episode Details:")
                for episode_result in results.episode_results:
                    status = "✅" if episode_result.success else "❌"
                    task_id = episode_result.task_id
                    attempt = episode_result.attempt
                    reason = episode_result.termination_reason or "unknown"
                    print(f"  {status} {task_id} (attempt {attempt}): {reason}")

            print("=" * 60)

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main() -> None:
    """Simple CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SABER Client - Unified Agent Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - prompt to select tasks
  python -m saber.client --agent ./my_agent.py

  # Single task execution
  python -m saber.client --agent ./my_agent.py --tasks xss_flag_capture

  # Multiple specific tasks
  python -m saber.client --agent ./my_agent.py --tasks xss_flag_capture,sql_injection

  # Full benchmark (all available tasks)
  python -m saber.client --agent ./my_agent.py --tasks all
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
        )
    )


if __name__ == "__main__":
    main()
