"""
SABER Server Main Entry Point

This module provides the main entry point for starting a SABER domain server.
"""

import asyncio
import logging
import os
import sys

from saber.server.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


async def main() -> None:
    """Main function to start the SABER server."""

    # Get configuration from environment variables
    domain_name = os.getenv("SABER_DOMAIN", "pentest_demo")
    config_dir = os.getenv("SABER_CONFIG_DIR", "/app/config")
    host = os.getenv("SABER_HOST", "0.0.0.0")
    port = int(os.getenv("SABER_PORT", "8000"))
    mcp_port = int(os.getenv("SABER_MCP_PORT", "8001"))

    # Construct config file paths
    tasks_config_path = os.path.join(config_dir, "tasks.yaml")
    environments_config_path = os.path.join(config_dir, "environments.yaml")

    logger.info(f"Starting SABER server for domain: {domain_name}")
    logger.info(f"Config directory: {config_dir}")
    logger.info(f"Tasks config: {tasks_config_path}")
    logger.info(f"Environments config: {environments_config_path}")
    logger.info(f"REST API: {host}:{port}")
    logger.info(f"MCP API: {host}:{mcp_port}")

    # Verify config files exist
    if not os.path.exists(tasks_config_path):
        logger.error(f"Tasks config file not found: {tasks_config_path}")
        sys.exit(1)

    if not os.path.exists(environments_config_path):
        logger.error(f"Environments config file not found: {environments_config_path}")
        sys.exit(1)

    try:
        # Initialize and start the SessionManager
        session_manager = SessionManager(
            domain_name=domain_name,
            tasks_config_path=tasks_config_path,
            execution_config_path=environments_config_path,  # Pass environments config to execution manager
            host=host,
            port=port,
            mcp_host=host,
            mcp_port=mcp_port,
        )

        logger.info("SessionManager initialized successfully")

        # Start the server
        await session_manager.start_server()

    except Exception as e:
        logger.error(f"Failed to start SABER server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
