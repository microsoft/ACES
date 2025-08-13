"""
Sandbox manager for Docker execution environments.

This module manages Docker sandbox environments across sessions,
providing lifecycle management and cleanup capabilities for both
single and multi-container orchestration.
"""

import logging
from typing import Any, Dict, List, Optional

from ..exceptions import SandboxExecutionError
from .docker_sandbox_environment import DockerSandboxEnvironment
from .environment_spec import EnvironmentSpec

logger = logging.getLogger(__name__)


class SandboxManager:
    """
    Manager for Docker-based sandbox execution environments.

    Handles creation, tracking, and cleanup of Docker sandbox environments
    across multiple sessions with support for multi-container orchestration.
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
        self.active_sessions: Dict[str, DockerSandboxEnvironment] = {}

        # Set cleanup behavior
        self.cleanup_on_session_end = sandbox_config.get("cleanup_on_session_end", True)

        logger.info("SandboxManager initialized with Docker sandbox support")

    def create_session_environment(
        self, session_id: str, environment_spec: EnvironmentSpec
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
        if session_id in self.active_sessions:
            raise SandboxExecutionError(f"Session {session_id} already has an active environment")

        try:
            # Create new environment with specification
            environment = DockerSandboxEnvironment(session_id, environment_spec)

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
            logger.warning(f"Environment for session {session_id} is unhealthy, removing")
            self.cleanup_session(session_id)
            return None

        return environment

    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up Docker sandbox environment for a session.

        Args:
            session_id: Session identifier to clean up
        """
        environment = self.active_sessions.get(session_id)
        if not environment:
            logger.debug(f"No active environment found for session {session_id}")
            return

        try:
            # Stop and clean up the multi-container environment
            environment.stop()
            logger.info(f"Cleaned up Docker sandbox environment for session {session_id}")

        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

        finally:
            # Remove from active sessions
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
            if self._is_environment_healthy(environment):
                healthy_sessions.append(session_id)
            else:
                unhealthy_sessions.append(session_id)

        # Clean up unhealthy sessions
        for session_id in unhealthy_sessions:
            logger.warning(f"Cleaning up unhealthy session {session_id}")
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
            # Check if execution service is healthy
            execution_service = environment.environment_spec.get_execution_service()
            if not environment.is_service_healthy(execution_service):
                return False

            # Check all target services
            for service in environment.environment_spec.target_services:
                if not environment.is_service_healthy(service.name):
                    return False

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
