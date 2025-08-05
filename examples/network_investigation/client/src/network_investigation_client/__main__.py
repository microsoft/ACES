"""
Network Investigation Client - SABER Framework Example
Demonstrates how to create a simple agent for network investigation tasks.
This agent can be run through the SABER client framework.
"""

import argparse
import asyncio
import logging
from pathlib import Path

from saber.client import TestHarness, TestHarnessConfig, AgentWrapper
from .agent import NetworkInvestigationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "0.1.0"


async def main():
    """Main entry point - uses SABER TestHarness to run the agent."""
    parser = argparse.ArgumentParser(description="SABER Network Investigation Client")
    parser.add_argument("--server-url", default="http://localhost:8000", help="SABER server URL")
    parser.add_argument("--task", default="network_investigation", help="Task to execute")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--log-file", help="Optional log file path for detailed logging")
    parser.add_argument(
        "--log-structured",
        action="store_true",
        help="Enable structured JSON logging to file"
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Create and wrap agent
    agent_instance = NetworkInvestigationAgent()
    agent = AgentWrapper(agent_instance)

    # Configure test harness
    config = TestHarnessConfig(
        server_url=args.server_url,
        log_level=args.log_level,
        log_file=Path(args.log_file) if args.log_file else None,
        log_structured=args.log_structured,
    )

    print(f"🔍 Starting SABER Network Investigation Agent")
    print(f"📡 Server: {args.server_url}")
    print(f"🎯 Task: {args.task}")
    print(f"🔁 Episodes: {args.episodes}")
    if args.log_file:
        print(f"📝 Log file: {args.log_file}")
        if args.log_structured:
            print(f"📋 Structured logging: enabled")

    try:
        # Use SABER TestHarness to run the agent
        async with TestHarness(config) as harness:
            # Initialize with our agent
            await harness.initialize(agent)

            # Run the specified number of episodes
            total_steps = 0
            completed_episodes = 0

            for episode in range(args.episodes):
                if args.episodes > 1:
                    print(f"\n🎮 Episode {episode + 1}/{args.episodes}")

                # Reset agent for new episode
                await agent.reset()

                # Use enhanced TestHarness with task_id parameter
                results = await harness.run_test(task_id=args.task)

                # Track results
                episode_steps = results.get('steps_executed', 0)
                episode_completed = results.get('completed', False)

                total_steps += episode_steps
                if episode_completed:
                    completed_episodes += 1

                # Show episode results
                print(f"  Steps: {episode_steps}")
                print(f"  Completed: {'✅' if episode_completed else '❌'}")

            # Show final summary
            print(f"\n📊 Final Results:")
            print(f"  Episodes completed: {completed_episodes}/{args.episodes}")
            print(f"  Total steps: {total_steps}")
            print(f"  Success rate: {completed_episodes/args.episodes:.1%}")

    except KeyboardInterrupt:
        print("\n⚠️ Received interrupt signal")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
