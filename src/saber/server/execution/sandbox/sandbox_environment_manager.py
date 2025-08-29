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
        self.active_sessions: Dict[str, DockerSandboxEnvironment] = {}
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

    def create_session_environment(
        self,
        session_id: str,
        environment_spec: SandboxEnvironmentSpec,
    ) -> DockerSandboxEnvironment:
        """
        Create a new Docker sandbox environment for a session.

        Args:
            session_id: Unique identifier for the session
            environment_spec: Environment specification for container orchestration

        Returns:
            DockerSandboxEnvironment instance

        Raises:
            SandboxExecutionError: If session already exists or environment cannot be created
        """
        # Check if sandbox manager is ready
        if not self._is_ready:
            raise SandboxExecutionError("SandboxManager is not ready yet. Please wait for initialization to complete.")

        if session_id in self.active_sessions:
            raise SandboxExecutionError(f"Session {session_id} already has an active environment")

        try:
            # Ensure required Docker images exist (build if necessary)
            self._ensure_docker_images_exist(environment_spec)

            # Prepare container logging configuration
            container_logging_config = self.sandbox_config.copy()
            container_logging_config.update(
                {
                    "domain": self.sandbox_config.get("domain", "sandbox"),
                    "session_id": session_id,
                    "logs_directory": self.sandbox_config.get("logs_directory", "/app/logs"),
                    "enable_logging": self.sandbox_config.get("enable_container_logging", True),
                }
            )

            # Create new environment with specification and logging config
            environment = DockerSandboxEnvironment(
                session_id, environment_spec, container_logging_config=container_logging_config
            )

            # Start the environment
            environment.start()

            # Track the session
            self.active_sessions[session_id] = environment

            logger.info(f"Created Docker sandbox environment for session {session_id}")
            return environment

        except Exception as e:
            # Clean up if creation failed
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            raise SandboxExecutionError(f"Failed to create session environment: {e}")

    def get_session_environment(self, session_id: str) -> Optional[DockerSandboxEnvironment]:
        """
        Get existing Docker sandbox environment for a session.

        Args:
            session_id: Session identifier

        Returns:
            DockerSandboxEnvironment if session exists, None otherwise
        """
        environment = self.active_sessions.get(session_id)

        # Check if environment services are still healthy
        if environment and not self._is_environment_healthy(environment):
            logger.warning(f"🔥 HEALTH CHECK FAILURE: Environment for session {session_id} is unhealthy, removing")
            self.cleanup_session(session_id)
            return None

        return environment

    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up Docker sandbox environment for a session.

        Args:
            session_id: Session identifier to clean up
        """
        logger.warning(
            f"🔥 CONTAINER TERMINATION INITIATED: SandboxManager.cleanup_session() called for session {session_id}"
        )
        environment = self.active_sessions.get(session_id)
        if not environment:
            logger.warning(
                f"🔥 NO ENVIRONMENT OBJECT: No active environment found for session {session_id}, "
                f"checking for orphaned resources"
            )
            # Even if no environment object exists, try to clean up orphaned resources
            self._cleanup_orphaned_session_resources(session_id)
            return

        try:
            logger.warning(f"🔥 STOPPING DOCKER ENVIRONMENT: About to call environment.stop() for session {session_id}")
            # Stop and clean up the multi-container environment
            environment.stop()
            logger.info(f"Cleaned up Docker sandbox environment for session {session_id}")

        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
            # Still try to clean up orphaned resources as fallback
            logger.warning(f"🔥 FALLBACK CLEANUP: Attempting to clean orphaned resources for session {session_id}")
            self._cleanup_orphaned_session_resources(session_id)

        finally:
            # Remove from active sessions
            if session_id in self.active_sessions:
                logger.warning(f"🔥 REMOVING SESSION TRACKING: Deleting session {session_id} from active_sessions")
                del self.active_sessions[session_id]

    def _cleanup_orphaned_session_resources(self, session_id: str) -> None:
        """
        Clean up orphaned Docker resources for a session even without environment object.

        Args:
            session_id: Session identifier to clean up orphaned resources for
        """
        try:
            import subprocess

            # Try to stop any containers with the session label
            compose_project_name = f"saber-session-{session_id}"
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
                logger.warning(f"🔥 ORPHANED CLEANUP SUCCESS: Cleaned up orphaned resources for session {session_id}")
            else:
                logger.warning(
                    f"🔥 ORPHANED CLEANUP PARTIAL: docker compose down returned {result.returncode} "
                    f"for session {session_id}"
                )

            if result.stderr:
                logger.debug(f"Orphaned cleanup stderr: {result.stderr}")

        except Exception as e:
            logger.warning(
                f"🔥 ORPHANED CLEANUP ERROR: Failed to clean orphaned resources for session {session_id}: {e}"
            )

    def cleanup_all_sessions(self) -> None:
        """
        Clean up all active Docker execution environments.

        This method should be called during shutdown to ensure
        all containers are properly cleaned up.
        """
        if not self.active_sessions:
            logger.info("No active sessions to clean up")
            return

        logger.warning(
            f"🔥 MASS CONTAINER TERMINATION: SandboxManager.cleanup_all_sessions() cleaning up "
            f"{len(self.active_sessions)} active sessions"
        )

        # Copy session IDs to avoid modifying dict during iteration
        session_ids = list(self.active_sessions.keys())

        for session_id in session_ids:
            try:
                logger.warning(f"🔥 BATCH CLEANUP: Processing session {session_id}")
                self.cleanup_session(session_id)
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")

        logger.info("All sessions cleaned up")

    def list_active_sessions(self) -> List[str]:
        """
        List all active session IDs.

        Returns:
            List of active session identifiers
        """
        # Filter out unhealthy sessions
        healthy_sessions = []
        unhealthy_sessions = []

        for session_id, environment in self.active_sessions.items():
            if self._is_environment_healthy(environment):
                healthy_sessions.append(session_id)
            else:
                unhealthy_sessions.append(session_id)

        # Clean up unhealthy sessions
        for session_id in unhealthy_sessions:
            logger.warning(f"🔥 HEALTH CHECK CLEANUP: Cleaning up unhealthy session {session_id}")
            self.cleanup_session(session_id)

        return healthy_sessions

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

    def get_session_count(self) -> int:
        """
        Get count of active sessions.

        Returns:
            Number of active sessions
        """
        return len(self.list_active_sessions())

    def get_sandbox_config(self) -> Dict[str, Any]:
        """
        Get sandbox configuration.

        Returns:
            Copy of sandbox configuration dictionary
        """
        return dict(self.sandbox_config)


# Compatibility alias
SandboxManager = SandboxEnvironmentManager
