"""
ExecutionManager implementation for command execution.

This version provides access to CLI commands with comprehensive
security validation capabilities.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Action, CommandResult

# CleanupManager removed - using direct component cleanup
from .executors.docker_executor import DockerExecutor
from .executors.executor_factory import ExecutorFactory
from .sandbox.permanent_environment_manager import PermanentEnvironmentManager
from .sandbox.sandbox_environment_manager import SandboxEnvironmentManager

logger = logging.getLogger(__name__)


class ExecutionManager:
    """
    Execution manager supporting multiple executor types with factory pattern.

    Provides unified interface for executing different types of commands (CLI, Python, etc.)
    with security validation and Docker isolation. Commands are executed sequentially.
    """

    def __init__(self, config_dir: str):
        """
        Initialize ExecutionManager with executor factory and configuration.

        Args:
            config_dir: Path to configuration directory for environment resolution

        Task-specific configuration will be provided when sessions are created.
        """
        self._config_dir = config_dir
        self._permanent_environment_manager: Optional[PermanentEnvironmentManager] = None
        self._sandbox_environment_manager: Optional[SandboxEnvironmentManager] = None

        # Check for debug mode from environment variable
        self._debug_mode = os.getenv("SABER_DEBUG_MODE", "false").lower() in ("true", "1", "yes")
        if self._debug_mode:
            logger.info("SABER Debug Mode ENABLED - Containers will not be cleaned up after execution")
        else:
            logger.debug("SABER Debug Mode disabled - Normal cleanup behavior")

        # Load custom executors from the config directory if provided
        if config_dir:
            self._load_custom_executors(config_dir)

        self._configuration: Dict[str, Any] = {}

        # Executor factory will be created when sandbox manager is available
        self._executor_factory: Optional[ExecutorFactory] = None

        # Episode-specific execution tracking for concurrent commands
        self._active_executions: Dict[str, int] = {}  # episode_id -> count of active executions
        self._max_concurrent_per_episode = 3  # Allow multiple concurrent commands per episode

        logger.info("ExecutionManager initialized for concurrent execution")
        logger.info(f"Max concurrent executions per episode: {self._max_concurrent_per_episode}")

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

        Looks for Python files ending with '_executor.py' in the config directory
        and loads them to allow registration of custom executors.

        Args:
            config_dir: Path to configuration directory
        """
        try:
            from .executors.executor_registry import load_executors_from_directory

            # Load custom executors from the config directory
            results = load_executors_from_directory(config_dir)

            if results:
                successful_loads = [file for file, result in results.items() if result == "loaded_successfully"]
                if successful_loads:
                    logger.info(f"Loaded custom executor definitions from {len(successful_loads)} files")
                    for file_path in successful_loads:
                        logger.debug(f"Loaded custom executors from: {file_path}")

                failed_loads = [(file, result) for file, result in results.items() if result != "loaded_successfully"]
                if failed_loads:
                    logger.warning(f"Failed to load {len(failed_loads)} custom executor files")
                    for file_path, error in failed_loads:
                        logger.warning(f"Failed to load {file_path}: {error}")
            else:
                logger.debug(f"No custom executor files found in {config_dir}")

        except ImportError:
            logger.warning("Custom executor registry not available - custom executors will not be loaded")
        except Exception as e:
            logger.warning(f"Error loading custom executors from {config_dir}: {e}")

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
                return CommandResult.error_result(error=error_msg)

            # Increment active execution count for episode
            self._active_executions[episode_id] = current_count + 1
            logger.debug(f"Episode {episode_id} active executions: {self._active_executions[episode_id]}")
        else:
            logger.warning("No episode_id in context for step execution - proceeding without concurrency limits")

        try:
            # Get executor directly from action's tool name with episode context
            executor = self.get_executor(action.tool_name, episode_id=episode_id)

            # Use action parameters directly - no mapping needed
            parameters = action.parameters.copy()

            # Validate parameters first
            validation_result = executor.validate_parameters(parameters)
            if not validation_result.valid:
                return CommandResult.error_result(
                    error=f"Parameter validation failed: {', '.join(validation_result.errors)}"
                )

            # Execute using the appropriate executor with callable interface
            # This is now truly async and non-blocking
            return await executor(parameters, context or {})

        except Exception as e:
            logger.error(f"Execution failed for episode {episode_id}: {e}")
            return CommandResult.error_result(error=str(e))

        finally:
            # Decrement active execution count for episode
            if episode_id and episode_id in self._active_executions:
                self._active_executions[episode_id] -= 1
                if self._active_executions[episode_id] <= 0:
                    del self._active_executions[episode_id]
                logger.debug(f"Episode {episode_id} active executions: {self._active_executions.get(episode_id, 0)}")

    def get_executor(self, executor_type: str, episode_id: Optional[str] = None) -> DockerExecutor:
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

    def configure_for_task(
        self,
        episode_id: str,
        task: Any,
        session_id: Optional[str] = None,
        target_episode_id: Optional[str] = None,
    ) -> None:
        """
        Configure ExecutionManager for a specific task/episode with optional network attachment.

        Args:
            episode_id: Episode identifier for unique container naming and configuration
            task: Task object containing execution parameters and environment specification
            session_id: Optional session identifier for compatibility/logging
            target_episode_id: Optional episode ID to attach network to
        """
        # Resolve environment if specified in task
        if task.environment:
            logger.info(f"Task specifies environment: {task.environment}")

            # Ensure sandbox manager is initialized
            if self._sandbox_environment_manager is None:
                # Initialize with default configuration for the current task domain
                # Calculate correct server directory (parent of config directory)
                server_dir = Path(self._config_dir).parent

                sandbox_config = {
                    "domain": "excytin_demo",  # Default domain
                    "config_dir": self._config_dir,  # Pass config_dir for proper path resolution
                    "logs_dir": str(server_dir / "logs"),  # Logs go to server/logs, not server/config/logs
                    "enable_container_logging": True,
                }
                self._sandbox_environment_manager = SandboxEnvironmentManager(sandbox_config)

                # Reset executor factory so it gets recreated with the new sandbox manager
                self._executor_factory = None

                logger.info("SandboxEnvironmentManager lazily initialized for task configuration")

            # Create the episode environment using the SandboxManager
            logger.info(
                f"Creating sandbox environment for episode {episode_id}"
                + (f" (attach to {target_episode_id})" if target_episode_id else "")
            )
            try:
                self._sandbox_environment_manager.create_episode_environment(
                    episode_id, task.environment, target_episode_id
                )
                logger.info(f"✅ Created sandbox environment for episode {episode_id}")
            except Exception as e:
                logger.error(f"❌ FAILED to create sandbox environment for episode {episode_id}: {e}")
                raise
        else:
            logger.warning("No environment specified for this task - episode will run without sandbox environment")

        # Start with task's execution config
        execution_config = task.execution_config.copy()

        # Add executor-specific configurations generically
        # Look for any config key that ends with "_config" and maps to an executor type
        executor_types = self.executor_factory.get_available_executors()
        for executor_type in executor_types:
            config_attr = f"{executor_type}_config"
            if hasattr(task, config_attr):
                config_value = getattr(task, config_attr)
                if config_value:
                    execution_config[executor_type] = config_value
                    logger.debug(f"Added {executor_type} configuration from task")

        # Create new configuration for this task
        self._configuration = execution_config

        # Register episode configuration with the single executor factory
        allowed_executors = task.allowed_executors
        episode_config = execution_config

        self.executor_factory.register_episode_configuration(
            episode_id=episode_id, allowed_executors=allowed_executors, episode_config=episode_config
        )

        logger.info(f"ExecutionManager configured for episode {episode_id} with task-specific settings")
        if allowed_executors:
            logger.info(f"Episode {episode_id} restricted to executors: {allowed_executors}")
        if session_id:
            logger.info(f"Episode {episode_id} associated with session {session_id}")

        # Log configured executor types
        configured_executors = [k for k in execution_config.keys() if k in executor_types]
        if configured_executors:
            logger.info(f"Episode {episode_id} configured executor-specific settings for: {configured_executors}")

    def to_mcp_tools(self, episode_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert available executors to MCP format, optionally filtered by episode configuration.

        Args:
            episode_id: Optional episode identifier to filter tools for episode-specific allowed executors

        Returns:
            List containing MCP tool definitions for all executors or episode-specific executors
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
                logger.error(f"Failed to get info for executor {executor_type}: {e}")

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

    def cleanup_episode(self, episode_id: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Clean up episode resources including Docker containers.

        Args:
            episode_id: The episode ID to clean up
            context: Additional context for debugging (optional)

        Returns:
            True if cleanup was successful, False otherwise
        """
        logger.info(f"🔥 EPISODE CLEANUP: ExecutionManager.cleanup_episode() called for episode {episode_id}")

        cleanup_success = True

        # Clean up episode environment directly through sandbox manager
        try:
            if self._sandbox_environment_manager is not None:
                logger.info(f"🧹 Calling sandbox manager cleanup for episode {episode_id}")
                container_cleanup_success = self._sandbox_environment_manager.stop_episode_environment(episode_id)
                if container_cleanup_success:
                    logger.info(f"✅ Episode container cleanup completed for episode {episode_id}")
                else:
                    logger.error(f"❌ Episode container cleanup failed for episode {episode_id}")
                    cleanup_success = False
            else:
                logger.warning(f"⚠️  No sandbox manager available for episode cleanup: {episode_id}")
        except Exception as e:
            logger.error(f"❌ Failed to cleanup episode containers for episode {episode_id}: {e}")
            cleanup_success = False

        # Unregister episode configuration from executor factory
        try:
            self.executor_factory.unregister_episode_configuration(episode_id)
            logger.info(f"✅ Episode configuration unregistered from executor factory for episode {episode_id}")
        except Exception as e:
            logger.error(f"❌ Failed to unregister episode configuration for episode {episode_id}: {e}")
            cleanup_success = False

        # Clean up episode execution tracking
        if episode_id in self._active_executions:
            try:
                del self._active_executions[episode_id]
                logger.info(f"✅ Episode execution tracking cleanup completed for episode {episode_id}")
            except Exception as e:
                logger.error(f"❌ Failed to cleanup episode execution tracking for episode {episode_id}: {e}")
                cleanup_success = False

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

                logger.info(f"SandboxEnvironmentManager initialized for domain: {domain}")

            # Initialize permanent environment manager
            self._permanent_environment_manager = PermanentEnvironmentManager(config)

            logger.info("ExecutionManager: Permanent environment manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize permanent environment manager: {e}")
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

    def stop_permanent_environment(self) -> None:
        """
        Stop permanent environment through direct manager call.

        Raises:
            RuntimeError: If permanent environment shutdown fails
        """
        if not self._permanent_environment_manager:
            logger.warning("No permanent environment manager configured")
            return

        try:
            self._permanent_environment_manager.stop_permanent_environment()
            logger.info("ExecutionManager: Permanent environment stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop permanent environment: {e}")
            raise RuntimeError(f"Failed to stop permanent environment: {e}")

    def is_permanent_environment_running(self) -> bool:
        """
        Check if permanent environment is running.

        Returns:
            True if permanent environment is running, False otherwise
        """
        if not self._permanent_environment_manager:
            return False
        return self._permanent_environment_manager.is_running()

    def cleanup_all_containers(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Clean up all containers (ephemeral and permanent) for server shutdown.

        Args:
            context: Additional context for debugging

        Returns:
            Dictionary with cleanup results
        """
        logger.info("🧹 FULL CLEANUP START: Starting cleanup of all containers (ephemeral + permanent)")

        # Clean up all ephemeral episode containers
        try:
            if self._sandbox_environment_manager is not None:
                self._sandbox_environment_manager.cleanup_all_episodes()
                ephemeral_episodes_cleaned = len(self._sandbox_environment_manager.get_active_episodes())
                ephemeral_success = True
            else:
                logger.warning("⚠️  No sandbox manager available for ephemeral cleanup")
                ephemeral_episodes_cleaned = 0
                ephemeral_success = True  # Not a failure if no manager exists
        except Exception as e:
            logger.error(f"Failed to cleanup ephemeral episodes: {e}")
            ephemeral_episodes_cleaned = 0
            ephemeral_success = False

        # Clean up permanent environment
        permanent_success = True
        try:
            self.stop_permanent_environment()
        except Exception as e:
            logger.error(f"Failed to stop permanent environment: {e}")
            permanent_success = False

        result = {
            "ephemeral_episodes_cleaned": ephemeral_episodes_cleaned,
            "permanent_environment_stopped": permanent_success,
            "total_cleanup_success": ephemeral_success and permanent_success,
            "context": context,
        }

        logger.info(
            f"🧹 FULL CLEANUP COMPLETE: Ephemeral episodes: {ephemeral_episodes_cleaned}, "
            f"Permanent stopped: {permanent_success}, Overall success: {result['total_cleanup_success']}"
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
                except Exception as e:
                    logger.warning(f"Failed to get/stop session environment for {session_id}: {e}")

            # Clean up any active executions for this session
            session_episodes = [ep_id for ep_id in self._active_executions.keys() if ep_id.startswith(session_id)]
            for episode_id in session_episodes:
                if episode_id in self._active_executions:
                    del self._active_executions[episode_id]

            logger.info(f"Successfully cleaned up session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup session {session_id}: {e}")
            return False

    @property
    def debug_mode(self) -> bool:
        """
        Check if debug mode is enabled.

        Returns:
            True if debug mode is enabled (containers won't be cleaned up)
        """
        return self._debug_mode
