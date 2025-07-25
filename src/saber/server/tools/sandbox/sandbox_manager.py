"""
Sandbox manager for Docker execution environments.

This module manages Docker execution environments across sessions,
providing lifecycle management and cleanup capabilities.
"""

import logging
from typing import Any, Dict, List, Optional

from ..exceptions import SandboxExecutionError
from .docker_environment import DockerExecutionEnvironment

logger = logging.getLogger(__name__)


class SandboxManager:
    """
    Manager for Docker-based sandbox execution environments.

    Handles creation, tracking, and cleanup of Docker execution environments
    across multiple sessions.
    """

    def __init__(self, sandbox_config: Dict[str, Any]) -> None:
        """
        Initialize sandbox manager.

        Args:
            sandbox_config: Configuration dictionary for sandbox settings

        Raises:
            SandboxExecutionError: If sandbox configuration is invalid
        """
        self.sandbox_config = sandbox_config
        self.active_sessions: Dict[str, DockerExecutionEnvironment] = {}

        # Validate required configuration
        if not sandbox_config.get("enabled", False):
            raise SandboxExecutionError("Sandbox execution is disabled in configuration")

        # Set cleanup behavior
        self.cleanup_on_session_end = sandbox_config.get("cleanup_on_session_end", True)

        logger.info("SandboxManager initialized with Docker execution")

    def create_session_environment(self, session_id: str) -> DockerExecutionEnvironment:
        """
        Create a new Docker execution environment for a session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            DockerExecutionEnvironment instance

        Raises:
            SandboxExecutionError: If session already exists or environment cannot be created
        """
        if session_id in self.active_sessions:
            raise SandboxExecutionError(f"Session {session_id} already has an active environment")

        try:
            # Create new environment
            environment = DockerExecutionEnvironment(session_id, self.sandbox_config)

            # Start the environment
            environment.start()

            # Track the session
            self.active_sessions[session_id] = environment

            logger.info(f"Created Docker environment for session {session_id}")
            return environment

        except Exception as e:
            # Clean up if creation failed
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            raise SandboxExecutionError(f"Failed to create session environment: {e}")

    def get_session_environment(self, session_id: str) -> Optional[DockerExecutionEnvironment]:
        """
        Get existing Docker execution environment for a session.

        Args:
            session_id: Session identifier

        Returns:
            DockerExecutionEnvironment if session exists, None otherwise
        """
        environment = self.active_sessions.get(session_id)

        # Check if environment is still healthy
        if environment and not environment.is_healthy():
            logger.warning(f"Environment for session {session_id} is unhealthy, removing")
            self.cleanup_session(session_id)
            return None

        return environment

    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up Docker execution environment for a session.

        Args:
            session_id: Session identifier to clean up
        """
        environment = self.active_sessions.get(session_id)
        if not environment:
            logger.debug(f"No active environment found for session {session_id}")
            return

        try:
            # Stop and remove the container
            environment.stop()
            logger.info(f"Cleaned up Docker environment for session {session_id}")

        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

        finally:
            # Remove from tracking regardless of cleanup success
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

    def cleanup_all_sessions(self) -> None:
        """
        Clean up all active Docker execution environments.

        This method should be called during shutdown to ensure
        all containers are properly cleaned up.
        """
        if not self.active_sessions:
            logger.info("No active sessions to clean up")
            return

        logger.info(f"Cleaning up {len(self.active_sessions)} active sessions")

        # Copy session IDs to avoid modifying dict during iteration
        session_ids = list(self.active_sessions.keys())

        for session_id in session_ids:
            try:
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
            if environment.is_healthy():
                healthy_sessions.append(session_id)
            else:
                unhealthy_sessions.append(session_id)

        # Clean up unhealthy sessions
        for session_id in unhealthy_sessions:
            logger.warning(f"Cleaning up unhealthy session {session_id}")
            self.cleanup_session(session_id)

        return healthy_sessions

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
