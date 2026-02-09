"""SABER Server Main Entry Point.

This module provides the main entry point for starting a SABER domain server.
Logging category: ``LogCategory.SESSION_MANAGER``.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from saber.logging_config import (
    LoggingConfig,
    get_session_manager_logger,
    init_logging,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from saber.server.session_manager import SessionManager

logger = get_session_manager_logger(__name__)


def _load_env_file() -> None:
    """Load environment variables from a colocated ``.env`` file when present."""

    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - fail fast on missing optional dependency
        raise RuntimeError("python-dotenv is required to load SABER server environment files.") from exc

    load_dotenv(env_file)
    logger.info(
        "Loaded environment variables",
        extra={"event": "env_loaded", "path": str(env_file)},
    )


def _check_llm_environment() -> None:
    """Check if LLM evaluation environment is properly configured."""

    if os.getenv("OPENAI_API_KEY"):
        logger.info(
            "LLM evaluation available",
            extra={"event": "llm_env_validated"},
        )
        return

    logger.warning(
        "OPENAI_API_KEY missing; LLM evaluation will fail",
        extra={"event": "llm_env_missing"},
    )


def _setup_server_logging(domain_name: str, config_dir: Path) -> LoggingConfig:
    """Set up server logging with timestamped files in server-logs directory.

    Works with both host subprocess deployment and container deployment.
    """

    # Generate timestamped filename: saber-server-YYYY-MM-DD_HH-MM-SS.log
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    server_log_filename = f"saber-server-{timestamp}.log"

    # Determine server logs directory from config_dir
    # config_dir is typically: <domains_root>/<domain>/server/config
    config_dir_resolved = config_dir.resolve()

    # Navigate from config -> server -> logs/server-logs
    server_dir = config_dir_resolved.parent  # config -> server
    server_logs_dir = server_dir / "logs" / "server-logs"

    # Create the server-logs directory if it doesn't exist
    server_logs_dir.mkdir(parents=True, exist_ok=True)

    # Build logging configuration
    base_config = LoggingConfig.from_env()
    server_config = LoggingConfig(
        level=base_config.level,
        console=False,  # Disable console logging - file only
        structured=base_config.structured,
        enable_file=True,  # Always enable file logging for server
        log_dir=server_logs_dir,
        file_name=server_log_filename,
        max_bytes=base_config.max_bytes,
        backup_count=base_config.backup_count,
    )

    return init_logging(server_config, force=True)


async def main() -> None:
    """Main function to start the SABER server.

    The server runs as a host subprocess, not in a container.
    Required environment variables (set by DomainOrchestrator):
    - SABER_DOMAIN: Domain name (e.g., "excytin")
    - SABER_CONFIG_DIR: Path to domain config directory
    """

    # Initialize basic logging first
    init_logging()

    _load_env_file()
    _check_llm_environment()

    domain_name = os.getenv("SABER_DOMAIN", "").strip()
    if not domain_name:
        raise ValueError(
            "SABER_DOMAIN environment variable must be set.\n"
            "This is normally set by the DomainOrchestrator when starting the server.\n"
            "Use: saber-domain start <domain> or inspect eval with SABER sandbox."
        )

    config_dir_value = os.getenv("SABER_CONFIG_DIR", "").strip()
    if not config_dir_value:
        raise ValueError(
            "SABER_CONFIG_DIR environment variable must be set.\n"
            "This should point to the domain's server/config directory.\n"
            "This is normally set by the DomainOrchestrator when starting the server."
        )
    config_dir = Path(config_dir_value).expanduser()

    # Set up server-specific logging with timestamped files
    logging_config = _setup_server_logging(domain_name, config_dir)

    # Recreate logger after reconfiguring logging
    global logger
    logger = get_session_manager_logger(__name__)
    host = os.getenv("SABER_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = int(os.getenv("SABER_PORT", "8000"))
    mcp_port = int(os.getenv("SABER_MCP_PORT", "8001"))

    tasks_config_path = config_dir / "tasks"
    environments_config_path = config_dir / "environments"

    logger.info(
        "SABER server logging initialized",
        extra={
            "event": "server_logging_initialized",
            "domain": domain_name,
            "log_file": str(logging_config.log_dir / logging_config.file_name),
            "config_dir": str(config_dir),
        },
    )

    logger.info(
        "Resolved SABER server configuration",
        extra={
            "event": "server_configuration_resolved",
            "domain": domain_name,
            "config_dir": str(config_dir),
            "rest_host": host,
            "rest_port": port,
            "mcp_host": host,
            "mcp_port": mcp_port,
        },
    )

    if not tasks_config_path.exists():
        logger.error(
            "Tasks configuration directory missing",
            extra={
                "event": "config_missing",
                "path": str(tasks_config_path),
                "domain": domain_name,
            },
        )
        raise FileNotFoundError(f"Tasks configuration directory not found: {tasks_config_path}")

    if not environments_config_path.exists():
        logger.error(
            "Environments configuration directory missing",
            extra={
                "event": "config_missing",
                "path": str(environments_config_path),
                "domain": domain_name,
            },
        )
        raise FileNotFoundError(f"Environments configuration directory not found: {environments_config_path}")

    logger.info(
        "Initializing SessionManager",
        extra={
            "event": "session_manager_initializing",
            "domain": domain_name,
        },
    )

    session_manager = SessionManager(
        domain_name=domain_name,
        config_dir=str(config_dir),
        host=host,
        port=port,
        mcp_host=host,
        mcp_port=mcp_port,
    )

    logger.info(
        "SessionManager initialized",
        extra={"event": "session_manager_initialized", "domain": domain_name},
    )

    operation_name = "session_manager.start_server"
    log_operation_start(logger, operation_name, domain=domain_name)
    try:
        await session_manager.start_server()
    except Exception as exc:
        log_operation_failure(logger, operation_name, exc, domain=domain_name)
        raise
    else:
        log_operation_success(logger, operation_name, domain=domain_name)


if __name__ == "__main__":
    asyncio.run(main())
