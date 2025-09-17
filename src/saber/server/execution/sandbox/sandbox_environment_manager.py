"""
Sandbox environment manager for Docker execution environments.

This module manages Docker sandbox environments using static compose files,
providing lifecycle management for episode-specific environments.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

from ....logging_config import get_execution_logger
from ..exceptions import SandboxExecutionError
from .compose_orchestrator import ComposeOrchestrator
from .environment_config import ComposeEnvironmentConfig

logger = get_execution_logger(__name__)


class SandboxEnvironmentManager:
    """
    Manager for Docker-based sandbox execution environments using static compose files.

    Uses ComposeOrchestrator for direct Docker Compose operations. Handles episode-specific
    environments using static compose files with environment variable substitution.
    """

    def __init__(self, sandbox_config: Dict[str, Any]):
        """
        Initialize sandbox environment manager.

        Args:
            sandbox_config: Configuration dictionary for sandbox environments

        Raises:
            SandboxExecutionError: If configuration is invalid
        """
        self.sandbox_config = sandbox_config
        self._is_ready = False

        # Track active orchestrators by episode_id with their compose files
        self.active_orchestrators: Dict[str, ComposeOrchestrator] = {}
        self.episode_compose_files: Dict[str, Path] = {}

        # Extract domain and environments path from config
        self.domain = sandbox_config.get("domain", "excytin_demo")

        # Build base path for sandbox environments (validation happens lazily during episode creation)
        # Use config_dir if provided, otherwise fall back to relative path
        config_dir = sandbox_config.get("config_dir")
        if config_dir:
            # Inside container: /app/config/environments/sandbox
            self.environments_base_path = Path(config_dir) / "environments" / "sandbox"
        else:
            # Relative path for development: domains/{domain}/server/config/environments/sandbox
            self.environments_base_path = Path(f"domains/{self.domain}/server/config/environments/sandbox")

        # Setup logging directory for container logging
        # Prefer explicit logs_dir; otherwise, infer from config_dir
        if "logs_dir" in sandbox_config:
            logs_dir = Path(sandbox_config["logs_dir"])
        elif "config_dir" in sandbox_config:
            logs_dir = Path(sandbox_config["config_dir"]) / "logs"
        else:
            logs_dir = Path.cwd() / "logs"

        logs_dir.mkdir(parents=True, exist_ok=True)

        # Store logging configuration for orchestrators
        self.logging_config = {
            "logs_directory": str(logs_dir),
            "domain": sandbox_config.get("domain", "unknown"),
            "enable_logging": sandbox_config.get("enable_logging", True),
        }

        logger.info(
            f"Initialized SandboxEnvironmentManager for domain '{self.domain}' "
            f"with environments path: {self.environments_base_path}"
        )
        logger.info(f"Container logging enabled - logs directory: {logs_dir}")
        self._is_ready = True

    def is_ready(self) -> bool:
        """
        Check if sandbox manager is ready for operations.

        Returns:
            True if ready, False otherwise
        """
        return self._is_ready

    def wait_for_ready(self, timeout: int = 30) -> bool:
        """
        Wait for sandbox manager to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if ready within timeout, False otherwise
        """
        start_time = time.time()
        while not self._is_ready and (time.time() - start_time) < timeout:
            time.sleep(0.5)
        return self._is_ready

    def _get_compose_file_path(self, sandbox_environment: str) -> Path:
        """
        Get the compose file path for a given sandbox environment.

        Args:
            sandbox_environment: Name of the sandbox environment (e.g., "excytin_sandbox")

        Returns:
            Path to the compose file

        Raises:
            SandboxExecutionError: If compose file doesn't exist
        """
        # Check if environments directory exists (lazy validation)
        if not self.environments_base_path.exists():
            raise SandboxExecutionError(f"Sandbox environments directory not found: {self.environments_base_path}")

        compose_file_path = self.environments_base_path / f"{sandbox_environment}.compose.yml"

        if not compose_file_path.exists():
            raise SandboxExecutionError(
                f"Compose file not found for environment '{sandbox_environment}': {compose_file_path}"
            )
        if not compose_file_path.is_file():
            raise SandboxExecutionError(f"Compose file path is not a file: {compose_file_path}")

        return compose_file_path

    def create_episode_environment(self, episode_id: str, sandbox_environment: str) -> bool:
        """
        Create episode-specific sandbox environment using static compose file.

        Args:
            episode_id: Episode identifier for unique container naming
            sandbox_environment: Name of sandbox environment (e.g., "excytin_sandbox")

        Returns:
            True if environment created successfully

        Raises:
            SandboxExecutionError: If environment for episode already exists or cannot be created
        """
        if not self._is_ready:
            raise SandboxExecutionError("SandboxManager is not ready yet. Please wait for initialization to complete.")

        if episode_id in self.active_orchestrators:
            raise SandboxExecutionError(f"Environment for episode {episode_id} already exists")

        try:
            # Get compose file path for this environment
            compose_file_path = self._get_compose_file_path(sandbox_environment)

            # Create new orchestrator for this episode with logging configuration
            orchestrator = ComposeOrchestrator(logging_config=self.logging_config)

            # Create environment configuration with permanent network prefix
            # This allows sandbox environments to reference permanent environment networks
            permanent_network_prefix = f"{self.domain}_permanent_environment_"
            config = ComposeEnvironmentConfig(episode_id=episode_id, permanent_network_prefix=permanent_network_prefix)

            # Start environment with configuration
            orchestrator.start_environment(str(compose_file_path), config)

            # Track the orchestrator and its compose file
            self.active_orchestrators[episode_id] = orchestrator
            self.episode_compose_files[episode_id] = compose_file_path

            logger.info(f"Created sandbox environment for episode {episode_id} using {sandbox_environment}")
            return True

        except Exception as e:
            logger.error(f"Failed to create sandbox environment for episode {episode_id}: {e}")
            raise SandboxExecutionError(f"Failed to create sandbox environment for episode {episode_id}: {e}")

    def get_episode_environment(self, episode_id: str) -> Optional[ComposeOrchestrator]:
        """
        Retrieve orchestrator for the given episode.

        Args:
            episode_id: Episode identifier

        Returns:
            ComposeOrchestrator if episode exists, None otherwise
        """
        return self.active_orchestrators.get(episode_id)

    def stop_episode_environment(self, episode_id: str) -> bool:
        """
        Stop and clean up episode-specific sandbox environment.

        Args:
            episode_id: Episode identifier

        Returns:
            True if environment stopped successfully

        Raises:
            SandboxExecutionError: If episode environment cannot be stopped
        """
        if episode_id not in self.active_orchestrators:
            logger.warning(f"No active environment found for episode {episode_id}")
            return False

        try:
            orchestrator = self.active_orchestrators[episode_id]
            compose_file_path = self.episode_compose_files[episode_id]

            orchestrator.stop_environment(compose_file_path, episode_id)

            # Remove from active tracking
            del self.active_orchestrators[episode_id]
            del self.episode_compose_files[episode_id]

            logger.info(f"Stopped sandbox environment for episode {episode_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop sandbox environment for episode {episode_id}: {e}")
            raise SandboxExecutionError(f"Failed to stop sandbox environment for episode {episode_id}: {e}")

    def cleanup_all_episodes(self) -> None:
        """
        Clean up all active episode environments.
        """
        episode_ids = list(self.active_orchestrators.keys())

        # Stop all episodes
        for episode_id in episode_ids:
            try:
                self.stop_episode_environment(episode_id)
            except Exception as e:
                logger.error(f"Failed to cleanup episode {episode_id}: {e}")

        logger.info("Cleaned up all sandbox environments")

    def get_active_episodes(self) -> list[str]:
        """
        Get list of active episode IDs.

        Returns:
            List of active episode identifiers
        """
        return list(self.active_orchestrators.keys())

    def is_episode_active(self, episode_id: str) -> bool:
        """
        Check if episode environment is active.

        Args:
            episode_id: Episode identifier

        Returns:
            True if episode is active, False otherwise
        """
        return episode_id in self.active_orchestrators

    def get_episode_status(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for episode environment.

        Args:
            episode_id: Episode identifier

        Returns:
            Status dictionary if episode exists, None otherwise
        """
        if episode_id not in self.active_orchestrators:
            return None

        orchestrator = self.active_orchestrators[episode_id]
        compose_file_path = self.episode_compose_files[episode_id]

        return {
            "episode_id": episode_id,
            "status": "active",
            "compose_file": compose_file_path,
            "domain": self.domain,
            "project_name": orchestrator.project_name if hasattr(orchestrator, "project_name") else None,
        }
