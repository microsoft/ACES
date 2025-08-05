"""
Network Investigation Server - SABER Framework Example
Demonstrates how to use the SABER SessionManager for a network investigation domain.
"""

import argparse
import asyncio
import logging
from pathlib import Path

from saber.server.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SABER Network Investigation Domain Server")
    parser.add_argument("--domain", default="network_investigation", help="Domain name")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Get paths for config files
    config_dir = Path(__file__).parent.parent.parent / "config"
    tasks_config_path = str(config_dir / "tasks.yaml")
    execution_config_path = str(config_dir / "execution.yaml")

    logger.info(f"Using tasks config: {tasks_config_path}")
    logger.info(f"Using execution config: {execution_config_path}")

    # Create SABER SessionManager
    session_manager = SessionManager(
        domain_name=args.domain,
        tasks_config_path=tasks_config_path,
        execution_config_path=execution_config_path,
        host=args.host,
        port=args.port
    )

    logger.info(f"Starting SABER {args.domain} server on {args.host}:{args.port}")

    try:
        # Start the server
        await session_manager.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        await session_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
