"""
Permanent environment manager for Docker execution environments.

This module manages Docker permanent environments that persist across all sessions,
providing lifecycle management for long-running services and networks.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from ....logging_config import get_execution_logger
from ..exceptions import SandboxExecutionError
from .compose_orchestrator import ComposeOrchestrator
from .environment_config import ComposeEnvironmentConfig

logger = get_execution_logger(__name__)


class PermanentEnvironmentManager:
    """
    Manager for permanent Docker environments that persist across sessions.

    Handles creation, tracking, and cleanup of Docker permanent environments
    that provide persistent services for all benchmark sessions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PermanentEnvironmentManager.

        Args:
            config: Configuration dictionary for permanent environment settings

        Raises:
            SandboxExecutionError: If permanent environment configuration is invalid
        """
        self.config = config

        # Extract domain with fail-fast validation
        domain = config.get("domain")
        if not domain:
            raise SandboxExecutionError("Domain must be specified in permanent environment configuration")

        self.domain = domain
        self.compose_project_name = f"{domain}_permanent_environment"
        self._is_running = False
        self._compose_file_path: Optional[Path] = None  # Store for parameter-less stop

        # Configuration and metadata storage
        # Prefer explicit logs_dir; otherwise, infer from config_dir
        if "logs_dir" in config:
            logs_dir = Path(config["logs_dir"])
        elif "config_dir" in config:
            logs_dir = Path(config["config_dir"]) / "logs"
        else:
            logs_dir = Path.cwd() / "logs"

        logs_dir.mkdir(parents=True, exist_ok=True)

        # Prepare config for container logging manager (to be passed to orchestrator)
        logging_config = {
            "logs_directory": str(logs_dir),
            "domain": config.get("domain", "unknown"),
            "enable_logging": config.get("enable_logging", True),
        }

        # Initialize orchestrator with logging
        self.orchestrator = ComposeOrchestrator(logging_config=logging_config)

        logger.info("PermanentEnvironmentManager initializing...")
        logger.info(f"Project name: {self.compose_project_name}")
        logger.info(f"Container logging enabled - logs directory: {logs_dir}")

    def start_permanent_environment_from_file(self, compose_file_path: Path) -> None:
        """
        Start the permanent environment from a static compose file.

        Args:
            compose_file_path: Path to the Docker Compose file for permanent services

        Raises:
            SandboxExecutionError: If permanent environment cannot be started
        """
        if self._is_running:
            logger.warning("Permanent environment is already running")
            return

        try:
            logger.info(f"Starting permanent environment from: {compose_file_path}")

            # Store compose file path for parameter-less stop
            self._compose_file_path = compose_file_path

            # Use orchestrator to start the environment with configuration
            config = ComposeEnvironmentConfig(project_name=self.compose_project_name, config_type="permanent")
            self.orchestrator.start_environment(str(compose_file_path), config)

            self._is_running = True
            logger.info(f"Permanent environment started successfully from: {compose_file_path}")

        except Exception as e:
            # Clear stored path on failure
            self._compose_file_path = None
            logger.error(f"Failed to start permanent environment from {compose_file_path}: {e}")
            raise SandboxExecutionError(f"Failed to start permanent environment: {e}")

    def stop_permanent_environment(self) -> None:
        """
        Stop the permanent environment using stored compose file path.

        Raises:
            SandboxExecutionError: If permanent environment cannot be stopped or no file path stored
        """
        if not self._is_running:
            logger.warning("Permanent environment is not running")
            return

        if not self._compose_file_path:
            raise SandboxExecutionError("No compose file path stored - cannot stop permanent environment")

        try:
            logger.info(f"Stopping permanent environment from: {self._compose_file_path}")

            # Use orchestrator to stop the environment with explicit project name
            self.orchestrator.stop_environment(self._compose_file_path, project_name=self.compose_project_name)

            self._is_running = False
            self._compose_file_path = None  # Clear stored path

            logger.info("Permanent environment stopped successfully")

        except Exception as e:
            logger.error(f"Failed to stop permanent environment: {e}")
            raise SandboxExecutionError(f"Failed to stop permanent environment: {e}")

    def is_running(self) -> bool:
        """
        Check if the permanent environment is currently running.

        Returns:
            True if the permanent environment is running, False otherwise
        """
        return self._is_running
