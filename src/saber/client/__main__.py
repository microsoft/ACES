#!/usr/bin/env python3
"""
SABER Client CLI - Simple Entry Point

Simple command-line interface for testing agents against SABER server:
    python -m saber.client --agent <path_to_agent> --task <task_id>
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

from .saber_harness import SABERHarness, SABERHarnessConfig


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


async def run_agent_test(
    agent_path: str,
    task_id: str = "default_task",
    agent_class: Optional[str] = None,
    server_url: Optional[str] = None,
    mcp_url: Optional[str] = None,
    env_file: Optional[str] = None,
    log_level: str = "INFO",
) -> None:
    """Run agent test with minimal configuration."""

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

        # Create simple configuration
        config = SABERHarnessConfig(
            server_url=final_server_url,
            mcp_url=final_mcp_url,
            task_id=task_id,
            log_level=log_level,
        )

        print(f"🔗 Connecting to SABER server: {final_server_url}")
        print(f"🔗 MCP server: {final_mcp_url}")
        print(f"🎯 Task: {task_id}")

        # Run test
        harness = SABERHarness(config)
        env_path = Path(env_file) if env_file else None
        await harness.initialize(agent, env_file=env_path)

        print("🚀 Starting agent execution...")
        results = await harness.run_test()

        # Simple results display
        print("\n" + "=" * 50)
        if results["success"]:
            print("🎉 SUCCESS!")
            if results.get("flag"):
                print(f"� Flag: {results['flag']}")
        else:
            print("❌ FAILED")
            if results.get("error"):
                print(f"Error: {results['error']}")

        if results.get("iterations"):
            print(f"📊 Iterations: {results['iterations']}")

        print("=" * 50)

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main() -> None:
    """Simple CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SABER Client - Simple Agent Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m saber.client --agent ./my_agent.py --task xss_flag_capture
  python -m saber.client --agent ./agents/red_agent.py --task web_pentest
  python -m saber.client --agent ./agent.py --task default_task --env-file .env
        """,
    )

    # Required arguments
    parser.add_argument("--agent", required=True, help="Path to agent file (.py)")
    parser.add_argument("--task", required=True, help="Task ID to execute")

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

    # Run the test
    asyncio.run(
        run_agent_test(
            agent_path=args.agent,
            task_id=args.task,
            agent_class=args.agent_class,
            server_url=args.server_url,
            mcp_url=args.mcp_url,
            env_file=args.env_file,
            log_level=args.log_level,
        )
    )


if __name__ == "__main__":
    main()
