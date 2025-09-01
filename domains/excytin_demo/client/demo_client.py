#!/usr/bin/env uv run python
"""
Excytin Demo - Container Architecture

This script demonstrates SABER's container-based execution capabilities using
the new container-based client architecture with MCP sidecar.

The demo:
1. Uses SABERHarness for container orchestration
2. Executes a demo agent in isolated containers
3. Tests MySQL connectivity from sandbox to permanent database
4. Validates container networking and MCP communication
5. Demonstrates Excytin functionality

Usage:
    uv run demo_client.py [--verbose] [--task-id TASK_ID]
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Import SABER harness and models
sys.path.append("/app/src")
from saber.client import SABERHarness, SABERHarnessConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("excytin_demo")


async def run_demo(verbose: bool = False) -> bool:
    """Run the Excytin demo using the new container-based architecture."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("EXCYTIN DEMO - CONTAINER-BASED ARCHITECTURE")
    logger.info("=" * 70)

    # Auto-detect server URLs
    if os.getenv("DOCKER_SABER_SERVER"):
        rest_base_url = os.getenv("SABER_REST_URL", "http://saber-excytin-server:8000")
        mcp_base_url = os.getenv("SABER_MCP_URL", "http://saber-excytin-server:8001")
    else:
        rest_base_url = "http://localhost:8000"
        mcp_base_url = "http://localhost:8001"

    # Create harness configuration
    config = SABERHarnessConfig(
        rest_base_url=rest_base_url, mcp_base_url=mcp_base_url, client_id="excytin_demo", request_timeout=30.0
    )

    # Create SABER harness
    harness = SABERHarness(config)

    try:
        logger.info("🚀 Initializing SABER harness...")

        # Initialize the harness (starts MCP sidecar)
        await harness.initialize()
        logger.info("✅ SABER harness initialized successfully")

        # Get the demo agent code path
        demo_agent_path = Path(__file__).parent / "demo_agent.py"
        if not demo_agent_path.exists():
            raise FileNotFoundError(f"Demo agent not found at {demo_agent_path}")

        logger.info("🤖 Executing Excytin demo agent...")

        # Execute the agent using the harness
        result = await harness.execute_agent(
            agent_code_path=str(demo_agent_path),
            task_id="excytin_demo",
            initial_prompt="Test MySQL connectivity and demonstrate Excytin container capabilities",
        )

        # Check results
        if result and result.get("success", False):
            logger.info("✅ Agent execution completed successfully")
            logger.info(f"📊 Agent result: {result.get('message', 'No message')}")
            return True
        else:
            logger.error("❌ Agent execution failed")
            logger.error(f"📊 Agent result: {result}")
            return False

    except Exception as e:
        logger.error(f"❌ Demo failed with exception: {e}")
        return False
    finally:
        # Cleanup harness
        try:
            logger.info("🧹 Cleaning up harness resources...")
            await harness.cleanup()
            logger.info("✅ Harness cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Error during harness cleanup: {e}")

    logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Excytin Demo with MySQL Connectivity Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    try:
        success = asyncio.run(run_demo(verbose=args.verbose))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
