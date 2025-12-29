"""
Abstract Docker executor base class for shared Docker container management.

This module provides the DockerExecutor Docker container operations for all
Docker-based command executors.

Logging category: ``LogCategory.DOCKER``.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from ....logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ...base import CommandResult
from ..base import ValidationResult
from ..exceptions import SandboxExecutionError
from ..sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from .base_executors import CommandExecutor

if TYPE_CHECKING:
    from ..sandbox.compose_orchestrator import ComposeOrchestrator

logger = get_saber_logger(LogCategory.DOCKER, __name__)


class DockerExecutor(CommandExecutor):
    """
    Abstract base class for Docker-based command executors.

    Provides shared Docker container management functionality including:
    - Episode-based container environments
    - Container health checks
    - Docker configuration validation
    - Post-execution cleanup
    """

    def __init__(
        self, sandbox_manager: SandboxEnvironmentManager, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """
        Initialize Docker executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            config: Executor configuration dictionary
            **kwargs: Additional arguments passed to parent

        Raises:
            SandboxExecutionError: If sandbox_manager is None or invalid
        """
        super().__init__(config=config, **kwargs)

        if sandbox_manager is None:
            raise SandboxExecutionError("sandbox_manager is required for Docker execution")

        self._sandbox_manager = sandbox_manager

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: dict[str, Any] | None = None,
        additional_params: dict[str, Any] | None = None,
        session_manager: Any | None = None,
        **kwargs: Any,
    ) -> "DockerExecutor":
        """
        Generic factory method for creating executor instances with standardized configuration.

        This method provides a consistent interface for all Docker executors, allowing
        the factory to create instances without knowing specific constructor signatures.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            config: Executor-specific configuration dictionary
            additional_params: Additional parameters specific to this executor type
            session_manager: Optional session manager for cross-episode operations
            **kwargs: Additional keyword arguments

        Returns:
            Configured executor instance
        """
        # Default implementation - subclasses can override for custom initialization
        merged_kwargs = {**kwargs}
        if additional_params:
            merged_kwargs.update(additional_params)

        # Pass config correctly to __init__ (not as docker_config)
        return cls(sandbox_manager=sandbox_manager, config=config, **merged_kwargs)

    def get_episode_environment(self, episode_id: str) -> "ComposeOrchestrator":
        """
        Retrieve Docker environment orchestrator for the given episode.

        Args:
            episode_id: Episode identifier

        Returns:
            ComposeOrchestrator for the episode

        Raises:
            SandboxExecutionError: If episode environment cannot be retrieved
        """
        try:
            environment = self._sandbox_manager.get_episode_environment(episode_id)
            if not environment:
                raise SandboxExecutionError(
                    f"No environment found for episode {episode_id}. Environment must be created before execution."
                )
            return environment
        except Exception as exc:
            log_operation_failure(
                logger,
                "docker_environment_fetch",
                exc,
                episode_id=episode_id,
            )
            raise SandboxExecutionError(f"Failed to get episode environment: {exc}") from exc

    def ensure_container_ready(self, episode_id: str) -> bool:
        """
        Ensure Docker container is ready for command execution.

        Args:
            episode_id: Episode identifier

        Returns:
            True if container is ready, False otherwise
        """
        try:
            environment = self.get_episode_environment(episode_id)
            # The sandbox manager handles container readiness internally
            return environment is not None
        except Exception as exc:
            logger.error(
                "Container readiness check failed",
                extra={
                    "event": "docker_container_readiness_failed",
                    "episode_id": episode_id,
                    "error": str(exc),
                },
            )
            return False

    def cleanup_execution(self, episode_id: str) -> None:
        """
        Perform post-execution cleanup for the episode.

        Args:
            episode_id: episode identifier to clean up
        """
        log_operation_start(logger, "docker_execution_cleanup", episode_id=episode_id)

        try:
            import asyncio

            asyncio.get_event_loop().run_until_complete(self._sandbox_manager.stop_episode_environment(episode_id))
        except Exception as exc:
            log_operation_failure(
                logger,
                "docker_execution_cleanup",
                exc,
                episode_id=episode_id,
            )
            raise SandboxExecutionError(
                f"Failed to clean up execution resources for episode {episode_id}: {exc}"
            ) from exc
        else:
            log_operation_success(
                logger,
                "docker_execution_cleanup",
                episode_id=episode_id,
            )

    def validate_docker_parameters(self, parameters: dict[str, Any]) -> ValidationResult:
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

    def get_docker_info(self) -> dict[str, Any]:
        """
        Get Docker-specific configuration information.

        Returns:
            Dictionary with Docker configuration details
        """
        info: dict[str, Any] = {
            "execution_environment": "docker_container",
            "timeout": self.get_timeout(),
        }

        try:
            sandbox_config = self._sandbox_manager.sandbox_config
            docker_info: dict[str, Any] = {}

            # Extract relevant Docker configuration
            for key in ["image", "network_mode", "read_only_root", "user", "resource_limits"]:
                if key in sandbox_config:
                    docker_info[key] = sandbox_config[key]

            info["docker_config"] = docker_info
        except Exception as exc:
            logger.warning(
                "Could not retrieve Docker configuration",
                extra={
                    "event": "docker_config_inspection_failed",
                    "error": str(exc),
                },
            )

        return info

    def setup_parameters(self, config: dict[str, Any]) -> None:
        """
        Set up Docker executor parameters.

        DockerExecutor is a base class that doesn't define its own parameters.
        Subclasses should override this method to define their specific parameters.

        Args:
            config: The merged configuration dictionary
        """
        pass

    @abstractmethod
    async def execute(self, parameters: dict[str, Any], context: dict[str, Any]) -> CommandResult:
        """
        Execute the command in Docker container.

        Subclasses must implement this method to define their specific execution logic.

        Args:
            parameters: Command-specific parameters
            context: Execution context including episode_id

        Returns:
            CommandResult with execution results
        """
        pass

    async def __call__(self, parameters: dict[str, Any], context: dict[str, Any]) -> CommandResult:
        """
        Call the executor with given parameters and context.

        This provides a more intuitive interface: executor(parameters, context)
        instead of executor.execute(parameters, context).

        Args:
            parameters: Command-specific parameters
            context: Execution context including episode_id

        Returns:
            CommandResult with execution results
        """
        return await self.execute(parameters, context)

    def validate_parameters(self, parameters: dict[str, Any]) -> ValidationResult:
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
