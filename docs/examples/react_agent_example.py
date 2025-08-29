#!/usr/bin/env python3
"""
Example ReAct Agent Usage

Demonstrates how to use the ReAct agent with the SABER harness.
This file can be used as a reference or run directly for testing.
"""

import asyncio
import logging
from pathlib import Path

from saber.client import SABERHarness, SABERHarnessConfig, ReActAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_react_usage():
    """Example of using ReAct agent with SABER harness."""

    # Configure harness
    config = SABERHarnessConfig(
        server_url="http://localhost:8000",
        mcp_url="http://localhost:8001",
        log_level="INFO",
        max_steps_client_safety=30,  # Limit for demo
        # task_ids=["xss_flag_capture"],  # Uncomment to run specific task
    )

    # Initialize harness
    harness = SABERHarness(config)

    # The agent will be instantiated by the harness using the AgentWrapper
    # The harness will automatically provide mcp_client and llm_client
    await harness.initialize(ReActAgent)

    print("🚀 Running ReAct agent with SABER harness...")
    print("📋 This will:")
    print("   1. Discover available tools dynamically")
    print("   2. Use policy-driven initial prompts")
    print("   3. Follow ReAct pattern (Think → Act → Observe)")
    print("   4. Look for FLAG patterns in responses")
    print("   5. Handle errors gracefully")
    print()

    try:
        # Run the test
        results = await harness.run()

        # Display results
        print("=" * 60)
        print("🎯 RESULTS:")
        print(f"   Success: {results.success}")
        print(f"   Episodes: {results.successful_episodes}/{results.total_episodes}")

        for episode in results.episode_results:
            status = "✅" if episode.success else "❌"
            print(f"   {status} {episode.task_id}: {episode.termination_reason}")
            if episode.flag:
                print(f"      🎉 Flag captured: {episode.flag}")

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    print("ReAct Agent Example for SABER Framework")
    print("=" * 50)
    print()
    print("This example demonstrates:")
    print("• Universal agent adapter (AgentWrapper)")
    print("• Dynamic tool discovery")
    print("• Policy-driven prompting")
    print("• ReAct reasoning pattern")
    print("• Flag capture methodology")
    print()
    print("Requirements:")
    print("• SABER server running on localhost:8000")
    print("• MCP server running on localhost:8001")
    print("• LLM client configured (via env vars)")
    print()

    # Run example
    asyncio.run(example_react_usage())


if __name__ == "__main__":
    main()
