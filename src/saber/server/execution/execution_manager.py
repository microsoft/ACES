"""
ExecutionManager implementation for command execution.

This version provides access to CLI commands with comprehensive
security validation capabilities.

Logging category: EXECUTION.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mcp.types as mcp_types

from ...logging_config import get_execution_logger, log_operation_failure, log_operation_start, log_operation_success
from ..base import Action, CommandResult
from ..benchmarks.task import Task
from .executors.base_executors import CommandExecutor
from .executors.executor_factory import ExecutorFactory
from .sandbox.permanent_environment_manager import PermanentEnvironmentManager
from .sandbox.sandbox_environment_manager import SandboxEnvironmentManager

logger = get_execution_logger(__name__)


class ExecutionManager:
    """
    Execution manager supporting multiple executor types with factory pattern.

    Provides unified interface for executing different types of commands (CLI, Python, etc.)
    with security validation and Docker isolation. Commands are executed sequentially.
    """

    def __init__(self, config_dir: str, session_manager: Optional[Any] = None):
        """
        Initialize ExecutionManager with executor factory and configuration.

        Args:
            config_dir: Path to configuration directory for environment resolution
            session_manager: Optional SessionManager for cross-episode operations

        Task-specific configuration will be provided when sessions are created.
        """
        self._config_dir = config_dir
        self._session_manager = session_manager
        self._permanent_environment_manager: Optional[PermanentEnvironmentManager] = None
        self._sandbox_environment_manager: Optional[SandboxEnvironmentManager] = None
        self._file_copier: Optional[Any] = None  # SandboxFileCopier instance

        # Check for debug mode from environment variable
        self._debug_mode = os.getenv("SABER_DEBUG_MODE", "false").lower() in ("true", "1", "yes")
        if self._debug_mode:
            logger.warning(
                "Debug mode enabled",
                extra={"event": "execution_debug_mode_enabled"},
            )
        else:
            logger.debug(
                "Debug mode disabled",
                extra={"event": "execution_debug_mode_disabled"},
            )

        # Load custom executors from the config directory if provided
        if config_dir:
            self._load_custom_executors(config_dir)

        self._configuration: Dict[str, Any] = {}

        # Executor factory will be created when sandbox manager is available
        self._executor_factory: Optional[ExecutorFactory] = None

        # Episode-specific execution tracking for concurrent commands
        self._active_executions: Dict[str, int] = {}  # episode_id -> count of active executions
        self._max_concurrent_per_episode = 3  # Allow multiple concurrent commands per episode

        # Initialize file copier
        self._initialize_file_copier()

        logger.info(
            "Execution manager initialized",
            extra={
                "event": "execution_manager_initialized",
                "max_concurrent_per_episode": self._max_concurrent_per_episode,
                "debug_mode": self._debug_mode,
            },
        )

    @property
    def executor_factory(self) -> ExecutorFactory:
        """Get the executor factory, creating it if needed."""
        if self._executor_factory is None:
            if self._sandbox_environment_manager is None:
                raise RuntimeError(
                    "Cannot create executor factory: sandbox environment manager not initialized. "
                    "Call configure_for_task() first."
                )
            self._executor_factory = ExecutorFactory(
                sandbox_manager=self._sandbox_environment_manager,
                configuration=self._configuration,
                session_manager=self._session_manager,
            )
        return self._executor_factory

    def is_sandbox_ready(self) -> bool:
        """
        Check if the sandbox manager is ready to create session environments.

        Returns:
            True if sandbox manager is ready, False otherwise
        """
        return self._sandbox_environment_manager is not None and self._sandbox_environment_manager.is_ready()

    def _load_custom_executors(self, config_dir: str) -> None:
        """
        Load custom executors from the configuration directory.

        Looks for Python files ending with '_executor.py' in the config/executors directory
        and loads them to allow registration of custom executors.

        Args:
            config_dir: Path to configuration directory
        """
        try:
            from .executors.executor_registry import load_executors_from_directory

            # Look for custom executors in the executors subdirectory
            executors_dir = os.path.join(config_dir, "executors")

            # If executors directory doesn't exist, also try loading from the config dir directly
            directories_to_check = [executors_dir, config_dir]

            total_successful = 0
            total_failed = 0

            for directory in directories_to_check:
                if not os.path.exists(directory):
                    continue

                logger.debug(
                    "Scanning for custom executors",
                    extra={
                        "event": "custom_executors_scan_directory",
                        "directory": directory,
                    },
                )
                # Load custom executors from the directory
                results = load_executors_from_directory(directory)

                if results:
                    successful_loads = [file for file, result in results.items() if result == "loaded_successfully"]
                    if successful_loads:
                        total_successful += len(successful_loads)
                        logger.info(
                            "Custom executors loaded",
                            extra={
                                "event": "custom_executors_loaded",
                                "directory": directory,
                                "file_count": len(successful_loads),
                            },
                        )
                        for file_path in successful_loads:
                            logger.debug(
                                "Custom executor module loaded",
                                extra={
                                    "event": "custom_executor_loaded_file",
                                    "file": file_path,
                                },
                            )

                    failed_loads = [
                        (file, result) for file, result in results.items() if result != "loaded_successfully"
                    ]
                    if failed_loads:
                        total_failed += len(failed_loads)
                        logger.warning(
                            "Custom executor load failures",
                            extra={
                                "event": "custom_executor_load_failed",
                                "directory": directory,
                                "failure_count": len(failed_loads),
                            },
                        )
                        for file_path, error in failed_loads:
                            logger.warning(
                                "Custom executor file failed to load",
                                extra={
                                    "event": "custom_executor_load_failure_detail",
                                    "file": file_path,
                                    "error": str(error),
                                },
                            )

            if total_successful == 0 and total_failed == 0:
                logger.debug(
                    "No custom executor files discovered",
                    extra={
                        "event": "custom_executors_not_found",
                        "config_dir": config_dir,
                        "executors_dir": executors_dir,
                    },
                )
            else:
                logger.info(
                    "Custom executor loading complete",
                    extra={
                        "event": "custom_executor_load_complete",
                        "successful": total_successful,
                        "failed": total_failed,
                    },
                )

        except ImportError as exc:
            logger.warning(
                "Custom executor registry not available",
                extra={
                    "event": "custom_executor_registry_unavailable",
                    "error": str(exc),
                },
            )
        except Exception as exc:
            log_operation_failure(
                logger,
                "load_custom_executors",
                exc,
                config_dir=config_dir,
            )

    async def step(self, action: Action, context: Optional[Dict[str, Any]] = None) -> CommandResult:
        """
        Execute the action with the appropriate executor.

        Commands are executed concurrently with episode-based limits.

        Args:
            action: Action object containing command and parameters
            context: Optional execution context (should include episode_id)

        Returns:
            CommandResult with execution results
        """
        # Get episode_id from context for episode-specific executor and concurrency tracking
        episode_id = context.get("episode_id") if context else None

        # Track concurrent executions per episode
        if episode_id:
            current_count = self._active_executions.get(episode_id, 0)
            if current_count >= self._max_concurrent_per_episode:
                error_msg = (
                    f"Too many concurrent executions for episode {episode_id} "
                    f"({current_count}/{self._max_concurrent_per_episode})"
                )
                logger.warning(
                    "Episode concurrency limit reached",
                    extra={
                        "event": "episode_concurrency_limit",
                        "episode_id": episode_id,
                        "current": current_count,
                        "limit": self._max_concurrent_per_episode,
                    },
                )
                return CommandResult.error_result(error=error_msg)

            # Increment active execution count for episode
            self._active_executions[episode_id] = current_count + 1
            logger.debug(
                "Episode execution count incremented",
                extra={
                    "event": "episode_execution_count_incremented",
                    "episode_id": episode_id,
                    "active_executions": self._active_executions[episode_id],
                },
            )
        else:
            logger.warning(
                "Episode ID missing for execution step",
                extra={"event": "episode_id_missing_for_step"},
            )

        session_id = context.get("session_id") if context else None
        log_operation_start(
            logger,
            "execute_action",
            episode_id=episode_id,
            session_id=session_id,
            tool_name=action.tool_name,
        )

        try:
            # Get executor directly from action's tool name with episode context
            executor = self.get_executor(action.tool_name, episode_id=episode_id)

            # Use action parameters directly - no mapping needed
            parameters = action.parameters.copy()

            # Validate parameters first
            validation_result = executor.validate_parameters(parameters)
            if not validation_result.valid:
                logger.warning(
                    "Executor parameter validation failed",
                    extra={
                        "event": "executor_parameter_validation_failed",
                        "episode_id": episode_id,
                        "tool_name": action.tool_name,
                        "errors": validation_result.errors,
                    },
                )
                return CommandResult.error_result(
                    error=f"Parameter validation failed: {', '.join(validation_result.errors)}"
                )

            # Execute using the appropriate executor with callable interface
            # This is now truly async and non-blocking
            execution_context = context or {}
            result = await executor(parameters, execution_context)
            log_operation_success(
                logger,
                "execute_action",
                episode_id=episode_id,
                session_id=session_id,
                tool_name=action.tool_name,
            )
            return result

        except Exception as exc:
            log_operation_failure(
                logger,
                "execute_action",
                exc,
                episode_id=episode_id,
                session_id=session_id,
                tool_name=action.tool_name,
            )
            return CommandResult.error_result(error=str(exc))

        finally:
            # Decrement active execution count for episode
            if episode_id and episode_id in self._active_executions:
                self._active_executions[episode_id] -= 1
                if self._active_executions[episode_id] <= 0:
                    del self._active_executions[episode_id]
                logger.debug(
                    "Episode execution count decremented",
                    extra={
                        "event": "episode_execution_count_decremented",
                        "episode_id": episode_id,
                        "active_executions": self._active_executions.get(episode_id, 0),
                    },
                )

    def get_executor(self, executor_type: str, episode_id: Optional[str] = None) -> "CommandExecutor":
        """
        Get a specific executor by type, optionally for a specific episode.

        Args:
            executor_type: Type of executor to retrieve
            episode_id: Optional episode ID to get episode-specific executor

        Returns:
            Executor instance

        Raises:
            ValueError: If executor type is not supported
        """
        return self.executor_factory.get_executor(executor_type, episode_id)

    def _get_episode_executor_factory(self, episode_id: Optional[str] = None) -> ExecutorFactory:
        """
        Get the executor factory (always returns the single shared factory).

        Args:
            episode_id: Episode identifier (unused, kept for compatibility)

        Returns:
            The shared ExecutorFactory instance
        """
        return self.executor_factory

    def get_available_executors(self, episode_id: Optional[str] = None) -> List[str]:
        """
        Get list of available executor types, optionally for a specific episode.

        Args:
            episode_id: Optional episode ID to get episode-specific executors

        Returns:
            List of executor type names
        """
        return self.executor_factory.get_available_executors(episode_id)

    def configure_for_task_async(
        self,
        episode_id: str,
        task: Task,
        session_id: Optional[str] = None,
        target_episode_id: Optional[str] = None,
    ) -> None:
        """
        Configure ExecutionManager for a task/episode WITHOUT waiting for health checks.

        Creates Docker environment but returns immediately after compose up.
        Caller must call wait_for_episode_healthy() separately.

        Args:
            episode_id: Episode identifier
            task: Task object with execution parameters
            session_id: Optional session identifier
            target_episode_id: Optional episode ID to attach network to
        """
        # Resolve environment if specified in task
        if task.environment:
            logger.info(
                "Task requires sandbox environment (async)",
                extra={
                    "event": "task_environment_specified_async",
                    "episode_id": episode_id,
                    "environment": task.environment,
                },
            )

            # Ensure sandbox manager is initialized
            if self._sandbox_environment_manager is None:
                server_dir = Path(self._config_dir).parent
                sandbox_config = {
                    "domain": "excytin_demo",
                    "config_dir": self._config_dir,
                    "logs_dir": str(server_dir / "logs"),
                    "enable_container_logging": True,
                }
                self._sandbox_environment_manager = SandboxEnvironmentManager(sandbox_config)
                self._executor_factory = None

                logger.info(
                    "Sandbox environment manager initialized (async)",
                    extra={
                        "event": "sandbox_manager_initialized_async",
                        "episode_id": episode_id,
                    },
                )

            # Create episode environment WITHOUT health checks
            log_operation_start(
                logger,
                "create_sandbox_environment_async",
                episode_id=episode_id,
                environment=task.environment,
            )
            try:
                # Extract environment name from task.environment
                # Can be str or dict with "base_template" key
                environment_name = (
                    task.environment
                    if isinstance(task.environment, str)
                    else task.environment.get("base_template", str(task.environment))
                )
                self._sandbox_environment_manager.create_episode_environment_async(
                    episode_id, environment_name, target_episode_id
                )
                log_operation_success(
                    logger,
                    "create_sandbox_environment_async",
                    episode_id=episode_id,
                    environment=task.environment,
                )
            except Exception as exc:
                log_operation_failure(
                    logger,
                    "create_sandbox_environment_async",
                    exc,
                    episode_id=episode_id,
                    environment=task.environment,
                )
                raise
        else:
            logger.warning(
                "Task has no sandbox environment",
                extra={
                    "event": "task_environment_missing_async",
                    "episode_id": episode_id,
                },
            )

        # Configure execution settings (same as sync version)
        execution_config = task.execution_config.copy()

        executor_types = self.executor_factory.get_available_executors()
        for executor_type in executor_types:
            config_attr = f"{executor_type}_config"
            if hasattr(task, config_attr):
                config_value = getattr(task, config_attr)
                if config_value:
                    execution_config[executor_type] = config_value

        self._configuration = execution_config

        # Update the factory's configuration and clear cached executors
        # This ensures executors created after this point use the new config
        if self._executor_factory is not None:
            self._executor_factory.update_configuration(execution_config)

        # Derive allowed_executors from executors config if not explicitly set
        # If executors section exists, use its keys as allowed executors
        if "executors" in execution_config and execution_config["executors"]:
            allowed_executors = list(execution_config["executors"].keys())
        elif task.allowed_executors:
            # Fall back to task.allowed_executors if no executors section
            allowed_executors = task.allowed_executors
        else:
            # No explicit configuration - allow all
            allowed_executors = None

        # Register episode configuration
        self.executor_factory.register_episode_configuration(
            episode_id=episode_id,
            allowed_executors=allowed_executors,
            episode_config=execution_config,
        )

        logger.info(
            "Execution manager configured for episode (async - health pending)",
            extra={
                "event": "execution_manager_configured_async",
                "episode_id": episode_id,
                "allowed_executors": allowed_executors,
            },
        )

    async def wait_for_episode_healthy(self, episode_id: str, timeout_seconds: int = 180) -> None:
        """
        Wait for episode environment to become healthy.

        Must be called after configure_for_task_async().

        Args:
            episode_id: Episode identifier
            timeout_seconds: Maximum time to wait for health checks

        Raises:
            RuntimeError: If sandbox manager not initialized or episode not found
        """
        import asyncio

        if self._sandbox_environment_manager is None:
            raise RuntimeError("Sandbox manager not initialized")

        logger.debug(
            "Waiting for episode environment health",
            extra={
                "event": "episode_health_wait_start",
                "episode_id": episode_id,
                "timeout_seconds": timeout_seconds,
            },
        )

        # Run health checks in thread pool to avoid blocking event loop
        await asyncio.to_thread(
            self._sandbox_environment_manager.wait_for_episode_healthy,
            episode_id,
            timeout_seconds=timeout_seconds,
        )

        logger.info(
            "Episode environment healthy",
            extra={
                "event": "episode_health_ready",
                "episode_id": episode_id,
            },
        )

    def get_execution_container_name(self, episode_id: str) -> Optional[str]:
        """
        Get the actual execution container name for an episode.

        Args:
            episode_id: Episode identifier

        Returns:
            Container name if available, None otherwise
        """
        if not self._sandbox_environment_manager:
            return None

        return self._sandbox_environment_manager.get_execution_container_name(episode_id)

    def _initialize_file_copier(self) -> None:
        """Initialize the file copier for copying files to execution containers."""
        from .sandbox.file_copier import SandboxFileCopier

        # Server base directory is parent of config directory
        server_base_dir = Path(self._config_dir).parent
        self._file_copier = SandboxFileCopier(base_dir=server_base_dir)

        logger.info(
            "File copier initialized for execution manager",
            extra={
                "event": "execution_file_copier_initialized",
                "server_base_dir": str(server_base_dir),
            },
        )

    async def copy_initial_files_to_episode(
        self,
        episode_id: str,
        task: Task,
    ) -> None:
        """
        Copy initial files specified in task configuration to episode execution container.

        This method automatically determines the correct execution container name
        and copies files there.

        Args:
            episode_id: The episode ID
            task: Task object with initial_files configuration

        Raises:
            RuntimeError: If file copier is not configured or container name cannot be determined
            Exception: If file copy operation fails after retries
        """
        if not self._file_copier:
            raise RuntimeError("File copier not initialized")

        if not task or not hasattr(task, "initial_files") or not task.initial_files:
            logger.debug(
                "No initial files to copy",
                extra={
                    "event": "no_initial_files",
                    "episode_id": episode_id,
                    "task_id": getattr(task, "task_id", None),
                },
            )
            return

        # Get the execution container name
        container_name = self.get_execution_container_name(episode_id)
        if not container_name:
            raise RuntimeError(
                f"Cannot determine execution container name for episode {episode_id}. "
                "Ensure sandbox environment is running and configured properly."
            )

        logger.info(
            "Copying initial files to episode execution container",
            extra={
                "event": "episode_initial_files_copy_start",
                "episode_id": episode_id,
                "task_id": getattr(task, "task_id", None),
                "file_count": len(task.initial_files),
                "container_name": container_name,
            },
        )

        # Retry logic for file copy (container might not be fully ready immediately after health check)
        max_retries = 5
        retry_delay = 1.0  # Start with 1 second

        for attempt in range(1, max_retries + 1):
            try:
                await self._file_copier.copy_files_to_episode(
                    episode_id=episode_id,
                    file_mappings=task.initial_files,
                    container_name=container_name,
                )

                logger.info(
                    "Initial files copied successfully",
                    extra={
                        "event": "episode_initial_files_copy_success",
                        "episode_id": episode_id,
                        "task_id": getattr(task, "task_id", None),
                        "file_count": len(task.initial_files),
                        "container_name": container_name,
                    },
                )
                return  # Success!

            except Exception as e:
                if attempt < max_retries:
                    import asyncio

                    logger.warning(
                        "Container not ready for file copy, retrying",
                        extra={
                            "event": "episode_initial_files_copy_retry",
                            "episode_id": episode_id,
                            "task_id": getattr(task, "task_id", None),
                            "attempt": attempt,
                            "max_retries": max_retries,
                            "retry_delay": retry_delay,
                            "error": str(e),
                            "container_name": container_name,
                        },
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        "Failed to copy initial files after all retries",
                        extra={
                            "event": "episode_initial_files_copy_failed",
                            "episode_id": episode_id,
                            "task_id": getattr(task, "task_id", None),
                            "attempts": max_retries,
                            "error": str(e),
                            "container_name": container_name,
                        },
                    )
                    raise RuntimeError(f"Failed to copy initial files to container: {e}") from e

    def to_mcp_tools(self, episode_id: Optional[str] = None) -> List[mcp_types.Tool]:
        """
        Convert available executors to MCP format, optionally filtered by episode configuration.

        Args:
            episode_id: Optional episode identifier to filter tools for episode-specific allowed executors

        Returns:
            List containing mcp.types.Tool objects for all executors or episode-specific executors
        """
        return self.executor_factory.get_all_mcp_tools(episode_id)

    def list_commands(self, episode_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available commands from executors, optionally for a specific episode.

        Args:
            episode_id: Optional episode ID to get episode-specific commands

        Returns:
            List of command definitions
        """
        commands = []

        for executor_type in self.executor_factory.get_available_executors(episode_id):
            try:
                executor = self.executor_factory.get_executor(executor_type, episode_id)
                metadata = getattr(executor, "_executor_metadata", {})

                command_info = {
                    "executor_type": executor_type,
                    "name": metadata.get("name", f"{executor_type}_command"),
                    "description": metadata.get("description", f"{executor_type.title()} executor"),
                    "domain": metadata.get("domain", "general"),
                    "security_level": metadata.get("security_level", "medium"),
                    "parameters": list(executor.get_parameters().keys()),
                }

                commands.append(command_info)

            except Exception as e:
                log_operation_failure(
                    logger,
                    "list_executor_commands",
                    e,
                    executor_type=executor_type,
                    episode_id=episode_id,
                )

        return commands

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary.

        Returns:
            Configuration dictionary
        """
        return self._configuration

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active executions (now episode-based).

        Returns:
            Dictionary with execution statistics
        """
        total_active = sum(self._active_executions.values())
        return {
            "total_active_executions": total_active,
            "active_episodes": len(self._active_executions),
            "max_concurrent_per_episode": self._max_concurrent_per_episode,
            "episode_execution_counts": self._active_executions.copy(),
        }

    async def cleanup_episode(self, episode_id: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Clean up episode resources including Docker containers (async).

        Args:
            episode_id: The episode ID to clean up
            context: Additional context for debugging (optional)

        Returns:
            True if cleanup was successful, False otherwise
        """
        log_operation_start(
            logger,
            "cleanup_episode",
            episode_id=episode_id,
            has_context=bool(context),
        )

        cleanup_success = True

        # Clean up episode environment directly through sandbox manager
        try:
            if self._sandbox_environment_manager is not None:
                logger.info(
                    "Sandbox manager cleanup invoked",
                    extra={
                        "event": "sandbox_cleanup_invoked",
                        "episode_id": episode_id,
                    },
                )
                container_cleanup_success = await self._sandbox_environment_manager.stop_episode_environment(episode_id)
                if container_cleanup_success:
                    logger.info(
                        "Episode container cleanup completed",
                        extra={
                            "event": "episode_container_cleanup_completed",
                            "episode_id": episode_id,
                        },
                    )
                else:
                    logger.error(
                        "Episode container cleanup failed",
                        extra={
                            "event": "episode_container_cleanup_failed",
                            "episode_id": episode_id,
                        },
                    )
                    cleanup_success = False
            else:
                logger.warning(
                    "Sandbox manager unavailable for cleanup",
                    extra={
                        "event": "sandbox_manager_missing_for_cleanup",
                        "episode_id": episode_id,
                    },
                )
        except Exception as exc:
            log_operation_failure(
                logger,
                "cleanup_episode_containers",
                exc,
                episode_id=episode_id,
            )
            cleanup_success = False

        # Unregister episode configuration from executor factory
        try:
            self.executor_factory.unregister_episode_configuration(episode_id)
            logger.info(
                "Episode configuration unregistered",
                extra={
                    "event": "episode_configuration_unregistered",
                    "episode_id": episode_id,
                },
            )
        except Exception as exc:
            log_operation_failure(
                logger,
                "unregister_episode_configuration",
                exc,
                episode_id=episode_id,
            )
            cleanup_success = False

        # Clean up episode execution tracking
        if episode_id in self._active_executions:
            try:
                del self._active_executions[episode_id]
                logger.info(
                    "Episode execution tracking cleared",
                    extra={
                        "event": "episode_execution_tracking_cleared",
                        "episode_id": episode_id,
                    },
                )
            except Exception as exc:
                log_operation_failure(
                    logger,
                    "cleanup_episode_execution_tracking",
                    exc,
                    episode_id=episode_id,
                )
                cleanup_success = False

        if cleanup_success:
            log_operation_success(
                logger,
                "cleanup_episode",
                episode_id=episode_id,
            )
        else:
            log_operation_failure(
                logger,
                "cleanup_episode",
                RuntimeError("episode cleanup incomplete"),
                episode_id=episode_id,
            )

        return cleanup_success

    def initialize_permanent_environment_manager(self, config: Dict[str, Any]) -> None:
        """
        Initialize permanent environment manager with unified container lifecycle management.

        Args:
            config: Configuration dictionary for permanent environment settings
        """
        try:
            # Initialize sandbox manager now that we have the domain
            domain = config.get("domain", "excytin_demo")

            log_operation_start(
                logger,
                "initialize_permanent_environment_manager",
                domain=domain,
            )

            # Calculate correct server directory (parent of config directory)
            server_dir = Path(self._config_dir).parent

            sandbox_config = {
                "domain": domain,
                "config_dir": self._config_dir,  # Pass config_dir for proper path resolution
                "logs_dir": str(server_dir / "logs"),  # Logs go to server/logs, not server/config/logs
                "enable_container_logging": True,
            }

            if self._sandbox_environment_manager is None:
                self._sandbox_environment_manager = SandboxEnvironmentManager(sandbox_config)

                # Reset executor factory so it gets recreated with the new sandbox manager
                self._executor_factory = None

                logger.info(
                    "Sandbox environment manager initialized",
                    extra={
                        "event": "sandbox_manager_initialized",
                        "domain": domain,
                    },
                )

            # Initialize permanent environment manager
            self._permanent_environment_manager = PermanentEnvironmentManager(config)

            log_operation_success(
                logger,
                "initialize_permanent_environment_manager",
                domain=domain,
            )

        except Exception as exc:
            log_operation_failure(
                logger,
                "initialize_permanent_environment_manager",
                exc,
                domain=config.get("domain"),
            )
            raise

    def start_permanent_environment(self, environment_spec: Any) -> None:
        """
        DEPRECATED: Start permanent environment through unified container lifecycle management.

        This method is deprecated and will be removed. The environment specification approach
        has been replaced with static compose files. Update your code to use the new file-based
        approach with start_permanent_environment_from_file().

        Args:
            environment_spec: Permanent environment specification (DEPRECATED)

        Raises:
            NotImplementedError: Always raised to enforce migration to new approach
        """
        raise NotImplementedError(
            "start_permanent_environment() is deprecated and no longer supported. "
            "The environment specification system has been removed in favor of static compose files. "
            "Update your code to use start_permanent_environment_from_file() instead."
        )

    async def stop_permanent_environment(self) -> None:
        """
        Stop permanent environment through direct manager call (async).

        Raises:
            RuntimeError: If permanent environment shutdown fails
        """
        if not self._permanent_environment_manager:
            logger.warning(
                "Permanent environment manager not configured",
                extra={"event": "permanent_environment_manager_missing"},
            )
            return

        log_operation_start(
            logger,
            "stop_permanent_environment",
        )
        try:
            await self._permanent_environment_manager.stop_permanent_environment()
            log_operation_success(
                logger,
                "stop_permanent_environment",
            )
        except Exception as exc:
            log_operation_failure(
                logger,
                "stop_permanent_environment",
                exc,
            )
            raise RuntimeError(f"Failed to stop permanent environment: {exc}") from exc

    def is_permanent_environment_running(self) -> bool:
        """
        Check if permanent environment is running.

        Returns:
            True if permanent environment is running, False otherwise
        """
        if not self._permanent_environment_manager:
            return False
        return self._permanent_environment_manager.is_running()

    async def cleanup_all_containers(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Clean up all containers (ephemeral and permanent) for server shutdown (async).

        Args:
            context: Additional context for debugging

        Returns:
            Dictionary with cleanup results
        """
        log_operation_start(
            logger,
            "cleanup_all_containers",
            has_context=bool(context),
        )

        # Clean up all ephemeral episode containers
        try:
            if self._sandbox_environment_manager is not None:
                await self._sandbox_environment_manager.cleanup_all_episodes()
                ephemeral_episodes_cleaned = len(self._sandbox_environment_manager.get_active_episodes())
                ephemeral_success = True
            else:
                logger.warning(
                    "Sandbox manager unavailable for ephemeral cleanup",
                    extra={"event": "sandbox_manager_missing_for_full_cleanup"},
                )
                ephemeral_episodes_cleaned = 0
                ephemeral_success = True  # Not a failure if no manager exists
        except Exception as exc:
            log_operation_failure(
                logger,
                "cleanup_ephemeral_episodes",
                exc,
            )
            ephemeral_episodes_cleaned = 0
            ephemeral_success = False

        # Clean up permanent environment
        permanent_success = True
        try:
            await self.stop_permanent_environment()
        except Exception as exc:
            log_operation_failure(
                logger,
                "cleanup_permanent_environment",
                exc,
            )
            permanent_success = False

        result = {
            "ephemeral_episodes_cleaned": ephemeral_episodes_cleaned,
            "permanent_environment_stopped": permanent_success,
            "total_cleanup_success": ephemeral_success and permanent_success,
            "context": context,
        }

        logger.info(
            "Full cleanup completed",
            extra={
                "event": "full_cleanup_completed",
                "ephemeral_episodes_cleaned": ephemeral_episodes_cleaned,
                "permanent_environment_stopped": permanent_success,
                "total_cleanup_success": result["total_cleanup_success"],
            },
        )

        if result["total_cleanup_success"]:
            log_operation_success(
                logger,
                "cleanup_all_containers",
            )
        else:
            log_operation_failure(
                logger,
                "cleanup_all_containers",
                RuntimeError("full cleanup incomplete"),
            )

        return result

    def cleanup_session(self, session_id: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Clean up all containers and resources for a specific session.

        Args:
            session_id: The session ID to clean up
            context: Additional context for debugging (optional)

        Returns:
            True if cleanup was successful, False otherwise
        """
        log_operation_start(
            logger,
            "cleanup_session",
            session_id=session_id,
            has_context=bool(context),
        )
        try:
            # Clean up session environment if it exists
            if self._sandbox_environment_manager is not None and hasattr(
                self._sandbox_environment_manager, "cleanup_session"
            ):
                self._sandbox_environment_manager.cleanup_session(session_id)
            elif self._sandbox_environment_manager is not None and hasattr(
                self._sandbox_environment_manager, "get_session_environment"
            ):
                # Try to get and stop the session environment
                try:
                    env = self._sandbox_environment_manager.get_session_environment(session_id)
                    if env and hasattr(env, "stop"):
                        env.stop()
                except Exception as exc:
                    logger.warning(
                        "Failed to stop session environment",
                        extra={
                            "event": "session_environment_stop_failed",
                            "session_id": session_id,
                            "error": str(exc),
                        },
                    )

            # Clean up any active executions for this session
            session_episodes = [ep_id for ep_id in self._active_executions.keys() if ep_id.startswith(session_id)]
            for episode_id in session_episodes:
                if episode_id in self._active_executions:
                    del self._active_executions[episode_id]

            log_operation_success(
                logger,
                "cleanup_session",
                session_id=session_id,
            )
            return True
        except Exception as exc:
            log_operation_failure(
                logger,
                "cleanup_session",
                exc,
                session_id=session_id,
            )
            return False

    @property
    def debug_mode(self) -> bool:
        """
        Check if debug mode is enabled.

        Returns:
            True if debug mode is enabled (containers won't be cleaned up)
        """
        return self._debug_mode

    def set_session_manager(self, session_manager: Any) -> None:
        """
        Set or update the session manager.

        Args:
            session_manager: SessionManager instance for cross-episode operations
        """
        self._session_manager = session_manager
        # Update executor factory if it exists
        if self._executor_factory is not None:
            self._executor_factory._session_manager = session_manager
        logger.debug(
            "Session manager set on execution manager",
            extra={"event": "session_manager_set"},
        )
