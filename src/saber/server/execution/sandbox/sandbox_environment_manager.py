"""
Sandbox environment manager for Docker execution environments.

This module manages Docker sandbox environments across sessions,
providing lifecycle management and cleanup capabilities for both
single and multi-container orchestration.
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ....logging_config import get_execution_logger
from ..exceptions import SandboxExecutionError
from .docker_sandbox_environment import DockerSandboxEnvironment
from .environment_spec import SandboxEnvironmentSpec

logger = get_execution_logger(__name__)


class SandboxEnvironmentManager:
    """
    Manager for Docker-based sandbox execution environments.

    Handles creation, tracking, and cleanup of Docker sandbox environments
    across multiple sessions with support for multi-container orchestration.
    """

    def __init__(self, sandbox_config: Dict[str, Any]):
        """
        Initialize the SandboxManager.

        Args:
            sandbox_config: Configuration dictionary for sandbox settings

        Raises:
            SandboxExecutionError: If sandbox configuration is invalid
        """
        self.sandbox_config = sandbox_config
        self.active_environments: Dict[str, DockerSandboxEnvironment] = {}  # episode_id -> Environment
        self._is_ready = False  # Track readiness state

        # Set cleanup behavior
        self.cleanup_on_session_end = sandbox_config.get("cleanup_on_session_end", True)

        logger.info("SandboxManager initializing...")

        try:
            # Mark as ready once initialization is complete
            self._is_ready = True
            logger.info("SandboxManager initialized and ready for session creation")

        except Exception as e:
            logger.error(f"SandboxManager initialization failed: {e}")
            self._is_ready = False
            raise

    def _ensure_docker_images_exist(self, environment_spec: SandboxEnvironmentSpec) -> None:
        """
        Ensure all required Docker images exist, building them if necessary.

        Args:
            environment_spec: Environment specification containing required images

        Raises:
            SandboxExecutionError: If image building fails
        """
        # For MVP: Skip image building during initialization
        # TODO: In the future, dynamically discover and build other images from environments.yaml
        # For now, we assume images (saber-execution, saber-webapp, etc.) exist or
        # will be handled by the environment setup process
        pass

    def _build_docker_image(self, image_name: str, dockerfile_path: str) -> None:
        """
        Build a Docker image from a Dockerfile.

        Args:
            image_name: Name and tag for the built image
            dockerfile_path: Path to Dockerfile relative to repo root

        Raises:
            SandboxExecutionError: If image building fails
        """
        try:
            # When running in container:
            # - /app/src contains the src/ directory
            # - /app/docker contains the docker/ directory
            # - /app is effectively the repo root for our purposes
            repo_root = Path("/app")
            dockerfile_full_path = repo_root / dockerfile_path

            if not dockerfile_full_path.exists():
                raise SandboxExecutionError(f"Dockerfile not found: {dockerfile_full_path}")

            # Build context is the repo root (so COPY paths work correctly)
            build_context = repo_root

            logger.info(f"Building {image_name} from {dockerfile_full_path}")
            logger.info(f"Build context: {build_context}")

            # Use docker build command for better control
            cmd = ["docker", "build", "-t", image_name, "-f", str(dockerfile_full_path), str(build_context)]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))

            if result.returncode != 0:
                logger.error(f"Docker build failed for {image_name}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise SandboxExecutionError(f"Failed to build Docker image {image_name}: {result.stderr}")

            logger.info(f"Successfully built Docker image {image_name}")

        except Exception as e:
            raise SandboxExecutionError(f"Error building Docker image {image_name}: {e}")

    def is_ready(self) -> bool:
        """
        Check if the sandbox manager is ready to create sessions.

        Returns:
            True if ready for session creation, False otherwise
        """
        return self._is_ready

    def wait_for_ready(self, timeout: int = 60) -> bool:
        """
        Wait for the sandbox manager to become ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if became ready within timeout, False if timed out
        """
        start_time = time.time()
        while not self._is_ready and (time.time() - start_time) < timeout:
            time.sleep(0.5)

        return self._is_ready

    def create_episode_environment(
        self,
        episode_id: str,
        environment_spec: SandboxEnvironmentSpec,
    ) -> DockerSandboxEnvironment:
        """
        Create episode-specific sandbox environment.

        Args:
            episode_id: Episode identifier for unique container naming
            environment_spec: Environment specification for container orchestration

        Returns:
            DockerSandboxEnvironment instance

        Raises:
            SandboxExecutionError: If environment for episode already exists or environment cannot be created
        """
        # Check if sandbox manager is ready
        if not self._is_ready:
            raise SandboxExecutionError("SandboxManager is not ready yet. Please wait for initialization to complete.")

        if episode_id in self.active_environments:
            raise SandboxExecutionError(f"Environment for episode {episode_id} already exists")

        try:
            # Ensure required Docker images exist (build if necessary)
            self._ensure_docker_images_exist(environment_spec)

            # Prepare container logging configuration
            container_logging_config = self.sandbox_config.copy()
            container_logging_config.update(
                {
                    "domain": self.sandbox_config.get("domain", "sandbox"),
                    "episode_id": episode_id,
                    "logs_directory": self.sandbox_config.get("logs_directory", "/app/logs"),
                    "enable_logging": self.sandbox_config.get("enable_container_logging", True),
                }
            )

            # Create environment with episode context - session_id can be extracted from episode if needed
            environment = DockerSandboxEnvironment(
                session_id=episode_id,  # Use episode_id as primary identifier
                environment_spec=environment_spec,
                container_logging_config=container_logging_config,
                episode_id=episode_id,
            )

            # Start the environment
            environment.start()

            # Track by episode_id
            self.active_environments[episode_id] = environment

            logger.info(f"Created Docker sandbox environment for episode {episode_id}")
            return environment

        except Exception as e:
            # Clean up if creation failed
            if episode_id in self.active_environments:
                del self.active_environments[episode_id]
            raise SandboxExecutionError(f"Failed to create episode environment: {e}")

    def get_episode_environment(self, episode_id: str) -> Optional[DockerSandboxEnvironment]:
        """
        Get existing Docker sandbox environment for an episode.

        Args:
            episode_id: Episode identifier

        Returns:
            DockerSandboxEnvironment if episode exists, None otherwise
        """
        environment = self.active_environments.get(episode_id)

        # Check if environment services are still healthy
        if environment and not self._is_environment_healthy(environment):
            logger.warning(f"🔥 HEALTH CHECK FAILURE: Environment for episode {episode_id} is unhealthy, removing")
            self.cleanup_episode(episode_id)
            return None

        return environment

    def cleanup_episode(self, episode_id: str) -> None:
        """
        Clean up Docker sandbox environment for an episode.

        Args:
            episode_id: Episode identifier to clean up
        """
        logger.warning(
            f"🔥 CONTAINER TERMINATION INITIATED: SandboxManager.cleanup_episode() called for episode {episode_id}"
        )
        environment = self.active_environments.get(episode_id)
        if not environment:
            logger.warning(
                f"🔥 NO ENVIRONMENT OBJECT: No active environment found for episode {episode_id}, "
                f"checking for orphaned resources"
            )
            # Even if no environment object exists, try to clean up orphaned resources
            self._cleanup_orphaned_episode_resources(episode_id)
            return

        try:
            logger.warning(f"🔥 STOPPING DOCKER ENVIRONMENT: About to call environment.stop() for episode {episode_id}")
            # Stop and clean up the multi-container environment
            environment.stop()
            logger.info(f"Cleaned up Docker sandbox environment for episode {episode_id}")

        except Exception as e:
            logger.error(f"Error cleaning up episode {episode_id}: {e}")
            # Still try to clean up orphaned resources as fallback
            logger.warning(f"🔥 FALLBACK CLEANUP: Attempting to clean orphaned resources for episode {episode_id}")
            self._cleanup_orphaned_episode_resources(episode_id)

        finally:
            # Remove from active environments
            if episode_id in self.active_environments:
                logger.warning(f"🔥 REMOVING EPISODE TRACKING: Deleting episode {episode_id} from active_environments")
                del self.active_environments[episode_id]

    def _cleanup_orphaned_episode_resources(self, episode_id: str) -> None:
        """
        Clean up orphaned Docker resources for an episode even without environment object.

        Args:
            episode_id: Episode identifier to clean up orphaned resources for
        """
        try:
            import subprocess

            # Try to stop any containers with the episode label
            compose_project_name = f"saber-episode-{episode_id}"
            logger.warning(f"🔥 ORPHANED CLEANUP: Attempting to clean up project {compose_project_name}")

            # Use docker compose down with project name to clean up any remaining resources
            cmd = [
                "docker",
                "compose",
                "-p",
                compose_project_name,
                "down",
                "--remove-orphans",
                "--volumes",
                "--timeout",
                "30",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logger.warning(f"🔥 ORPHANED CLEANUP SUCCESS: Cleaned up orphaned resources for episode {episode_id}")
            else:
                logger.warning(
                    f"🔥 ORPHANED CLEANUP PARTIAL: docker compose down returned {result.returncode} "
                    f"for episode {episode_id}"
                )

            if result.stderr:
                logger.debug(f"Orphaned cleanup stderr: {result.stderr}")

        except Exception as e:
            logger.warning(
                f"🔥 ORPHANED CLEANUP ERROR: Failed to clean orphaned resources for episode {episode_id}: {e}"
            )

    def cleanup_all_episodes(self) -> None:
        """
        Clean up all active Docker execution environments.

        This method should be called during shutdown to ensure
        all containers are properly cleaned up.
        """
        if not self.active_environments:
            logger.info("No active episodes to clean up")
            return

        logger.warning(
            f"🔥 MASS CONTAINER TERMINATION: SandboxManager.cleanup_all_episodes() cleaning up "
            f"{len(self.active_environments)} active episodes"
        )

        # Copy episode IDs to avoid modifying dict during iteration
        episode_ids = list(self.active_environments.keys())

        for episode_id in episode_ids:
            try:
                logger.warning(f"🔥 BATCH CLEANUP: Processing episode {episode_id}")
                self.cleanup_episode(episode_id)
            except Exception as e:
                logger.error(f"Error cleaning up episode {episode_id}: {e}")

        logger.info("All episodes cleaned up")

    def list_active_episodes(self) -> List[str]:
        """
        List all active episode IDs.

        Returns:
            List of active episode identifiers
        """
        # Filter out unhealthy episodes
        healthy_episodes = []
        unhealthy_episodes = []

        for episode_id, environment in self.active_environments.items():
            if self._is_environment_healthy(environment):
                healthy_episodes.append(episode_id)
            else:
                unhealthy_episodes.append(episode_id)

        # Clean up unhealthy episodes
        for episode_id in unhealthy_episodes:
            logger.warning(f"🔥 HEALTH CHECK CLEANUP: Cleaning up unhealthy episode {episode_id}")
            self.cleanup_episode(episode_id)

        return healthy_episodes

    def _is_environment_healthy(self, environment: DockerSandboxEnvironment) -> bool:
        """
        Check if a sandbox environment is healthy.

        Args:
            environment: DockerSandboxEnvironment to check

        Returns:
            True if all services are healthy, False otherwise
        """
        try:
            logger.debug("Starting health check for environment...")

            # Check if execution service is healthy
            execution_service = environment.environment_spec.get_execution_service()
            logger.debug(f"Checking execution service: {execution_service}")

            execution_healthy = environment.is_service_healthy(execution_service)
            logger.debug(f"Execution service {execution_service} health status: {execution_healthy}")

            if not execution_healthy:
                logger.warning(f"Execution service {execution_service} is unhealthy")
                return False

            # Check all target services
            logger.debug(f"Checking {len(environment.environment_spec.target_services)} target services...")
            for service in environment.environment_spec.target_services:
                logger.debug(f"Checking target service: {service.name}")
                service_healthy = environment.is_service_healthy(service.name)
                logger.debug(f"Target service {service.name} health status: {service_healthy}")

                if not service_healthy:
                    logger.warning(f"Target service {service.name} is unhealthy")
                    return False

            logger.debug("All services passed health check")
            return True

        except Exception as e:
            logger.warning(f"Error checking environment health: {e}")
            return False

    def get_episode_count(self) -> int:
        """
        Get count of active episodes.

        Returns:
            Number of active episodes
        """
        return len(self.list_active_episodes())

    def get_sandbox_config(self) -> Dict[str, Any]:
        """
        Get sandbox configuration.

        Returns:
            Copy of sandbox configuration dictionary
        """
        return dict(self.sandbox_config)
