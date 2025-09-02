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
        server_url = os.getenv("SABER_REST_URL", "http://saber-excytin-server:8000")
        mcp_url = os.getenv("SABER_MCP_URL", "http://saber-excytin-server:8001")
    else:
        server_url = "http://localhost:8000"
        mcp_url = "http://localhost:8001"

    # Create harness configuration
    config = SABERHarnessConfig(
        server_url=server_url,
        mcp_url=mcp_url,
        client_id="excytin_demo",
        request_timeout=30.0,
        task_ids=["excytin_demo"],  # Specify the task we want to run
        log_level="DEBUG" if verbose else "INFO",
        # Enable container logging for debugging
        enable_container_logging=True,
        client_log_dir=Path("/app/logs"),  # Inside container path
        log_retention_days=7,  # Keep logs for a week
        max_log_size_mb=10,  # Smaller files for demo
    )

    # Create SABER harness
    harness = SABERHarness(config)

    try:
        logger.info("🚀 Loading demo agent...")

        # Get the demo agent file path directly
        demo_agent_path = Path(__file__).parent / "demo_agent.py"
        if not demo_agent_path.exists():
            raise FileNotFoundError(f"Demo agent not found at {demo_agent_path}")

        logger.info(f"✅ Agent file located: {demo_agent_path}")

        # Initialize the harness with the agent file path directly
        await harness.initialize_with_file(str(demo_agent_path), agent_class="ExcytinDemoAgent")
        logger.info("✅ SABER harness initialized successfully")

        logger.info("🤖 Running Excytin demo benchmark...")

        # Run the harness (this executes the agent)
        results = await harness.run()

        # Check results
        if results and results.success:
            logger.info("✅ Demo completed successfully")
            logger.info(f"📊 Results: {results.successful_episodes}/{results.total_episodes} episodes successful")
            return True
        else:
            logger.error("❌ Demo failed")
            logger.error(f"📊 Results: {results}")
            return False

    except Exception as e:
        logger.error(f"❌ Demo failed with exception: {e}")
        return False


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
