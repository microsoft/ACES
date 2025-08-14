"""
Abstract Docker executor base class for shared Docker container management.

This module provides the DockerExecutor abstract base class that handles common
Docker container operations for all Docker-based command executors.
"""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..base import CommandResult, ValidationResult
from ..exceptions import SandboxExecutionError
from ..sandbox.sandbox_manager import SandboxManager
from .base_executors import CommandExecutor

if TYPE_CHECKING:
    from ..sandbox.docker_sandbox_environment import DockerSandboxEnvironment

logger = logging.getLogger(__name__)


class DockerExecutor(CommandExecutor):
    """
    Abstract base class for Docker-based command executors.

    Provides shared Docker container management functionality including:
    - Session-based container environments
    - Container health checks
    - Docker configuration validation
    - Post-execution cleanup
    """

    def __init__(
        self, sandbox_manager: SandboxManager, docker_config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """
        Initialize Docker executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            docker_config: Optional Docker-specific configuration
            **kwargs: Additional arguments passed to parent

        Raises:
            SandboxExecutionError: If sandbox_manager is None or invalid
        """
        super().__init__(**kwargs)

        if sandbox_manager is None:
            raise SandboxExecutionError("sandbox_manager is required for Docker execution")

        self._sandbox_manager = sandbox_manager
        self._docker_config = docker_config or {}

    def get_session_environment(self, session_id: str) -> "DockerSandboxEnvironment":
        """
        Retrieve Docker environment for the given session.

        Args:
            session_id: Session identifier

        Returns:
            DockerSandboxEnvironment for the session

        Raises:
            SandboxExecutionError: If session environment cannot be retrieved
        """
        try:
            environment = self._sandbox_manager.get_session_environment(session_id)
            if not environment:
                raise SandboxExecutionError(f"No environment found for session {session_id}")
            return environment
        except Exception as e:
            raise SandboxExecutionError(f"Failed to get session environment: {e}")

    def ensure_container_ready(self, session_id: str) -> bool:
        """
        Ensure Docker container is ready for command execution.

        Args:
            session_id: Session identifier

        Returns:
            True if container is ready, False otherwise
        """
        try:
            environment = self.get_session_environment(session_id)
            # The sandbox manager handles container readiness internally
            return environment is not None
        except Exception as e:
            logger.error(f"Container readiness check failed for session {session_id}: {e}")
            return False

    def cleanup_execution(self, session_id: str) -> None:
        """
        Perform post-execution cleanup for the session.

        Args:
            session_id: Session identifier to clean up
        """
        try:
            # Let sandbox manager handle the cleanup
            self._sandbox_manager.cleanup_session(session_id)
            logger.debug(f"Cleaned up execution resources for session {session_id}")
        except Exception as e:
            logger.warning(f"Error during execution cleanup for session {session_id}: {e}")

    def validate_docker_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate Docker-specific parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            ValidationResult with Docker-specific validation
        """
        result = ValidationResult.success()

        # Validate working directory if specified
        working_dir = parameters.get("working_dir")
        if working_dir is not None:
            if not isinstance(working_dir, str):
                result.add_error("working_dir must be a string")
            elif not working_dir.startswith("/"):
                result.add_error("working_dir must be an absolute path")

        # Validate timeout
        timeout = parameters.get("timeout")
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                result.add_error("timeout must be a positive number")

        return result

    def get_docker_info(self) -> Dict[str, Any]:
        """
        Get Docker-specific configuration information.

        Returns:
            Dictionary with Docker configuration details
        """
        info: Dict[str, Any] = {
            "execution_environment": "docker_container",
            "timeout": self.get_timeout(),
        }

        try:
            sandbox_config = self._sandbox_manager.get_sandbox_config()
            docker_info: Dict[str, Any] = {}

            # Extract relevant Docker configuration
            for key in ["image", "network_mode", "read_only_root", "user", "resource_limits"]:
                if key in sandbox_config:
                    docker_info[key] = sandbox_config[key]

            info["docker_config"] = docker_info
        except Exception as e:
            logger.warning(f"Could not retrieve Docker configuration: {e}")

        return info

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """
        Execute the command in Docker container.

        Subclasses must implement this method to define their specific execution logic.

        Args:
            parameters: Command-specific parameters
            context: Execution context including session_id

        Returns:
            CommandResult with execution results
        """
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters for Docker execution.

        Combines base parameter validation with Docker-specific validation.

        Args:
            parameters: Parameters to validate

        Returns:
            ValidationResult with comprehensive validation results
        """
        # Run base validation first
        base_result = super().validate_parameters(parameters)

        # Add Docker-specific validation
        docker_result = self.validate_docker_parameters(parameters)

        # Combine results
        combined_result = ValidationResult.success()
        combined_result.errors.extend(base_result.errors)
        combined_result.errors.extend(docker_result.errors)
        combined_result.warnings.extend(base_result.warnings)
        combined_result.warnings.extend(docker_result.warnings)

        # Mark as invalid if there are any errors
        if combined_result.errors:
            combined_result.valid = False

        return combined_result
