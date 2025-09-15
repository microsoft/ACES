"""
SABER Server Main Entry Point

This module provides the main entry point for starting a SABER domain server.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from saber.server.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def _load_env_file() -> None:
    """Load environment variables from .env file in server directory."""
    try:
        from dotenv import load_dotenv

        # Look for .env file in the server directory
        server_dir = Path(__file__).parent
        env_file = server_dir / ".env"

        if env_file.exists():
            load_dotenv(env_file)
            logger.info("Loaded environment variables from %s", env_file)
        else:
            logger.debug("No .env file found at %s", env_file)
    except ImportError:
        logger.debug("python-dotenv not available, skipping .env file loading")


def _check_llm_environment() -> None:
    """Check if LLM evaluation environment is properly configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY environment variable not set. "
            "LLM evaluation will fail if attempted. "
            "Set OPENAI_API_KEY in .env file or environment for LLM evaluation support."
        )
    else:
        logger.info("OPENAI_API_KEY found - LLM evaluation available")


async def main() -> None:
    """Main function to start the SABER server."""

    # Load environment variables from .env file
    _load_env_file()

    # Check LLM evaluation environment
    _check_llm_environment()

    # Get configuration from environment variables
    domain_name = os.getenv("SABER_DOMAIN", "pentest_demo")
    config_dir = os.getenv("SABER_CONFIG_DIR", "/app/config")
    host = os.getenv("SABER_HOST", "0.0.0.0")
    port = int(os.getenv("SABER_PORT", "8000"))
    mcp_port = int(os.getenv("SABER_MCP_PORT", "8001"))

    # Construct config file paths
    tasks_config_path = os.path.join(config_dir, "tasks")
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
            config_dir=config_dir,
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
