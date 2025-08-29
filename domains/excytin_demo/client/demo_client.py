#!/usr/bin/env uv run python
"""
Enhanced Container Logging Demo

This script demonstrates the enhanced container logging capabilities by directly
using SABER's REST client to:
1. Create a session and start benchmarks that create sandbox containers
2. Trigger container creation and logging via episode execution

Usage:
    uv run demo_client.py [--verbose]
"""

import argparse
import asyncio
import logging
import os
import sys

# Import SABER REST client
sys.path.append("/app/src")
from saber.client.api.rest_client import SABERRestClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enhanced_logging_demo")


async def run_demo(verbose: bool = False) -> bool:
    """Run the enhanced logging demo."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("ENHANCED CONTAINER LOGGING DEMO")
    logger.info("=" * 60)

    # Auto-detect server URL
    if os.getenv("DOCKER_SABER_SERVER"):
        base_url = os.getenv("SABER_REST_URL", "http://saber-excytin-server:8000")
    else:
        base_url = "http://localhost:8000"

    # Create REST client with proper timeout for container operations
    client = SABERRestClient(
        base_url=base_url,
        client_id="enhanced_logging_demo",
        request_timeout=30.0,  # 30 second timeout for container operations
    )

    session_id = None
    success = True

    try:
        # Test server connection
        logger.info("Testing server connection...")
        health_info = await client.health_check()
        logger.info(f"✅ Server is running: {health_info}")

        # Create session
        logger.info("Creating session...")
        session_id = await client.create_session()
        logger.info(f"Created session: {session_id}")

        # Start benchmark (this will automatically start first episode and create containers)
        logger.info("Starting benchmark - this will create sandbox containers...")
        await client.start_benchmark(session_id=session_id, task_ids=["basic_logging_demo"], episode_attempts=1)
        logger.info("Started benchmark - this creates sandbox containers and triggers logging")

        # Wait for containers to be created and logging to happen
        logger.info("Waiting for containers to be created and logs to be collected...")
        await asyncio.sleep(10)

        # Cleanup session (this will trigger final log collection)
        logger.info("Cleaning up session - this will trigger final log collection...")
        success = await client.terminate_session(session_id)

        if success:
            logger.info("✅ Demo completed - enhanced logging should have captured container info")
        else:
            logger.warning("⚠️ Session cleanup had issues but demo may have succeeded")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        success = False
    finally:
        # Ensure cleanup
        if session_id:
            try:
                await client.terminate_session(session_id)
            except Exception as e:
                logger.warning(f"Failed final cleanup: {e}")

    logger.info("=" * 60)
    if success:
        logger.info("🎉 DEMO COMPLETED SUCCESSFULLY")
        logger.info("Check ./server/logs/ for:")
        logger.info("  - compose-configs/ (docker-compose files)")
        logger.info("  - container-logs/ (container stdout/stderr)")
        logger.info("  - container-events-*.jsonl (lifecycle events)")
    else:
        logger.info("❌ DEMO FAILED")
    logger.info("=" * 60)

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Container Logging Demo")
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
