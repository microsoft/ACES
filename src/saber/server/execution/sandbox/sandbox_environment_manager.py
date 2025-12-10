"""Sandbox environment manager for Docker execution environments.

Logging Category: DOCKER

This module manages Docker sandbox environments using static compose files,
providing lifecycle management for episode-specific environments.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

from saber.logging_config import LogCategory, get_saber_logger

from ..exceptions import SandboxExecutionError
from .compose_orchestrator import ComposeOrchestrator
from .environment_config import ComposeEnvironmentConfig

logger = get_saber_logger(LogCategory.DOCKER, __name__)


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
            "Sandbox environment manager initialized",
            extra={
                "event": "sandbox_env_manager_initialized",
                "domain": self.domain,
                "environments_path": str(self.environments_base_path),
                "logs_directory": str(logs_dir),
            },
        )
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

    def create_episode_environment_async(
        self, episode_id: str, sandbox_environment: str, target_episode_id: Optional[str] = None
    ) -> tuple[ComposeOrchestrator, Path]:
        """
        Create episode environment without waiting for health checks.

        Returns immediately after containers start.
        Caller must call wait_for_episode_healthy() separately.

        Args:
            episode_id: Episode identifier
            sandbox_environment: Environment name
            target_episode_id: Optional episode to attach to

        Returns:
            Tuple of (orchestrator, processed_compose_path)

        Raises:
            SandboxExecutionError: If environment creation fails
        """
        if not self._is_ready:
            raise SandboxExecutionError("SandboxManager is not ready yet. Please wait for initialization to complete.")

        if episode_id in self.active_orchestrators:
            raise SandboxExecutionError(f"Environment for episode {episode_id} already exists")

        try:
            compose_file_path = self._get_compose_file_path(sandbox_environment)

            logger.info(
                "Async sandbox environment creation requested",
                extra={
                    "event": "sandbox_env_async_creation_requested",
                    "episode_id": episode_id,
                    "sandbox_environment": sandbox_environment,
                },
            )

            orchestrator = ComposeOrchestrator(logging_config=self.logging_config)

            permanent_network_prefix = f"{self.domain}_permanent_environment_"
            config = ComposeEnvironmentConfig(
                episode_id=episode_id,
                permanent_network_prefix=permanent_network_prefix,
                target_episode_id=target_episode_id,
            )

            # Track the orchestrator and compose file BEFORE starting containers
            # This ensures cleanup can find the orchestrator even if interrupted during container startup
            self.active_orchestrators[episode_id] = orchestrator
            self.episode_compose_files[episode_id] = compose_file_path

            # Start environment WITHOUT health checks
            orchestrator.start_environment_async(str(compose_file_path), config)

            logger.info(
                "Async sandbox environment started (health checks pending)",
                extra={
                    "event": "sandbox_env_async_started",
                    "episode_id": episode_id,
                    "sandbox_environment": sandbox_environment,
                },
            )

            # Return orchestrator and the compose file path that was used
            # The compose file path with resolved variables is needed for health checks
            return orchestrator, compose_file_path

        except Exception as e:
            logger.error(
                "Async sandbox environment creation failed",
                extra={
                    "event": "sandbox_env_async_creation_failed",
                    "episode_id": episode_id,
                    "sandbox_environment": sandbox_environment,
                    "error": str(e),
                },
            )
            raise SandboxExecutionError(f"Failed to create async sandbox environment for episode {episode_id}: {e}")

    def wait_for_episode_healthy(
        self, episode_id: str, timeout_seconds: int = 180, check_interval: float = 2.0
    ) -> None:
        """
        Wait for episode environment to become healthy.

        Must be called after create_episode_environment_async().

        Args:
            episode_id: Episode identifier
            timeout_seconds: Maximum time to wait (default 180s)
            check_interval: Seconds between health checks (default 2s)

        Raises:
            SandboxExecutionError: If episode not found or health checks fail
        """
        orchestrator = self.active_orchestrators.get(episode_id)
        if not orchestrator:
            raise SandboxExecutionError(f"No active environment found for episode {episode_id}")

        # Use the processed compose path stored in orchestrator (has env vars resolved)
        if not hasattr(orchestrator, "processed_compose_path") or not orchestrator.processed_compose_path:
            raise SandboxExecutionError(f"No processed compose path found for episode {episode_id}")

        logger.debug(
            "Waiting for episode environment health",
            extra={
                "event": "sandbox_env_health_wait_start",
                "episode_id": episode_id,
                "timeout_seconds": timeout_seconds,
            },
        )

        try:
            orchestrator.wait_for_healthy(
                compose_file_path=orchestrator.processed_compose_path,
                timeout_seconds=timeout_seconds,
                check_interval=check_interval,
            )

            logger.info(
                "Episode environment healthy",
                extra={
                    "event": "sandbox_env_health_ready",
                    "episode_id": episode_id,
                },
            )
        except Exception as e:
            logger.error(
                "Episode environment health check failed",
                extra={
                    "event": "sandbox_env_health_failed",
                    "episode_id": episode_id,
                    "error": str(e),
                },
            )
            raise SandboxExecutionError(f"Episode {episode_id} environment failed health checks: {e}")

    def get_episode_environment(self, episode_id: str) -> Optional[ComposeOrchestrator]:
        """
        Retrieve orchestrator for the given episode.

        Args:
            episode_id: Episode identifier

        Returns:
            ComposeOrchestrator if episode exists, None otherwise
        """
        return self.active_orchestrators.get(episode_id)

    def get_execution_container_name(self, episode_id: str) -> Optional[str]:
        """
        Get the actual execution container name for an episode.

        Args:
            episode_id: Episode identifier

        Returns:
            Container name if episode exists and has execution service, None otherwise
        """
        orchestrator = self.active_orchestrators.get(episode_id)
        if not orchestrator or not orchestrator.execution_service_name:
            return None

        try:
            return orchestrator._get_actual_container_name(orchestrator.execution_service_name)
        except Exception as e:
            logger.warning(
                "Failed to get execution container name",
                extra={
                    "event": "get_execution_container_name_failed",
                    "episode_id": episode_id,
                    "error": str(e),
                },
            )
            return None

    async def stop_episode_environment(self, episode_id: str) -> bool:
        """
        Stop and clean up episode-specific sandbox environment (async).

        Args:
            episode_id: Episode identifier

        Returns:
            True if environment stopped successfully

        Raises:
            SandboxExecutionError: If episode environment cannot be stopped
        """
        if episode_id not in self.active_orchestrators:
            logger.warning(
                "Sandbox environment stop skipped",
                extra={
                    "event": "sandbox_env_stop_skipped",
                    "episode_id": episode_id,
                },
            )
            return False

        try:
            orchestrator = self.active_orchestrators[episode_id]
            compose_file_path = self.episode_compose_files[episode_id]

            await orchestrator.stop_environment(compose_file_path, episode_id)

            # Remove from active tracking
            del self.active_orchestrators[episode_id]
            del self.episode_compose_files[episode_id]

            logger.info(
                "Sandbox environment stopped",
                extra={
                    "event": "sandbox_env_stopped",
                    "episode_id": episode_id,
                },
            )
            return True

        except Exception as e:
            logger.error(
                "Sandbox environment stop failed",
                extra={
                    "event": "sandbox_env_stop_failed",
                    "episode_id": episode_id,
                    "error": str(e),
                },
            )
            raise SandboxExecutionError(f"Failed to stop sandbox environment for episode {episode_id}: {e}")

    async def cleanup_all_episodes(self) -> None:
        """
        Clean up all active episode environments (async).
        """
        episode_ids = list(self.active_orchestrators.keys())

        # Stop all episodes
        for episode_id in episode_ids:
            try:
                await self.stop_episode_environment(episode_id)
            except Exception as e:
                logger.error(
                    "Sandbox environment cleanup failed",
                    extra={
                        "event": "sandbox_env_cleanup_failed",
                        "episode_id": episode_id,
                        "error": str(e),
                    },
                )

        logger.info(
            "Sandbox environments cleanup completed",
            extra={
                "event": "sandbox_env_cleanup_completed",
                "cleaned_episode_ids": episode_ids,
            },
        )

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
