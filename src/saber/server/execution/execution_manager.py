"""
ExecutionManager implementation for command execution.

This version provides access to CLI commands with comprehensive
security validation capabilities.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import Action, CommandResult
from .cleanup.cleanup_manager import ContainerCleanupManager
from .cleanup.cleanup_reason import CleanupReason
from .environment_loader import EnvironmentLoader
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
        self._environment_loader = None
        self._permanent_environment_manager: Optional[PermanentEnvironmentManager] = None

        # Initialize environment loader if config directory is provided
        if config_dir:
            environments_path = Path(config_dir) / "environments.yaml"
            if environments_path.exists():

                # Environment loader will be updated when permanent environment manager is initialized
                self._environment_loader = EnvironmentLoader(str(environments_path), None)
                logger.info(f"Environment loader initialized with: {environments_path}")

            # Load custom executors from the same directory
            self._load_custom_executors(config_dir)

        self._configuration: Dict[str, Any] = {}

        # Initialize sandbox manager with basic logging config (will be updated per session)
        initial_sandbox_config = {
            "domain": "execution",
            "logs_directory": str(Path(config_dir) / "logs"),
            "enable_container_logging": True,
        }

        self._sandbox_manager = SandboxEnvironmentManager(initial_sandbox_config)

        # Initialize executor factory with minimal configuration
        self._executor_factory = ExecutorFactory(
            sandbox_manager=self._sandbox_manager,
            configuration=self._configuration,
        )

        # Initialize unified container cleanup manager (permanent manager will be added later)
        self._cleanup_manager = ContainerCleanupManager(
            sandbox_manager=self._sandbox_manager,
            permanent_manager=None,  # Will be set when permanent environment is initialized
        )

        # Session-based execution tracking for concurrent commands
        self._active_executions: Dict[str, int] = {}  # session_id -> count of active executions
        self._max_concurrent_per_session = 3  # Allow multiple concurrent commands per session

        logger.info("ExecutionManager initialized for concurrent execution")
        logger.info(f"Available executor types: {self._executor_factory.get_available_executors()}")
        logger.info(f"Max concurrent executions per session: {self._max_concurrent_per_session}")

    def is_sandbox_ready(self) -> bool:
        """
        Check if the sandbox manager is ready to create session environments.

        Returns:
            True if sandbox manager is ready, False otherwise
        """
        return self._sandbox_manager.is_ready() if self._sandbox_manager else False

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

        Commands are executed concurrently with session-based limits.

        Args:
            action: Action object containing command and parameters
            context: Optional execution context

        Returns:
            CommandResult with execution results
        """
        session_id = context.get("session_id") if context else None

        # Track concurrent executions per session
        if session_id:
            current_count = self._active_executions.get(session_id, 0)
            if current_count >= self._max_concurrent_per_session:
                error_msg = (
                    f"Too many concurrent executions for session {session_id} "
                    f"({current_count}/{self._max_concurrent_per_session})"
                )
                return CommandResult.error_result(error=error_msg)

            # Increment active execution count
            self._active_executions[session_id] = current_count + 1
            logger.debug(f"Session {session_id} active executions: {self._active_executions[session_id]}")

        try:
            # Get executor directly from action's tool name
            executor = self._executor_factory.get_executor(action.tool_name)

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
            logger.error(f"Execution failed: {e}")
            return CommandResult.error_result(error=str(e))

        finally:
            # Decrement active execution count
            if session_id and session_id in self._active_executions:
                self._active_executions[session_id] -= 1
                if self._active_executions[session_id] <= 0:
                    del self._active_executions[session_id]
                logger.debug(f"Session {session_id} active executions: {self._active_executions.get(session_id, 0)}")

    def get_executor(self, executor_type: str) -> DockerExecutor:
        """
        Get a specific executor by type.

        Args:
            executor_type: Type of executor to retrieve

        Returns:
            Executor instance

        Raises:
            ValueError: If executor type is not supported
        """
        return self._executor_factory.get_executor(executor_type)

    def get_available_executors(self) -> List[str]:
        """
        Get list of all available executor types.

        Returns:
            List of executor type names
        """
        return self._executor_factory.get_available_executors()

    def configure_for_task(
        self,
        session_id: str,
        task: Any,
    ) -> None:
        """
        Configure ExecutionManager for a specific task/session.

        Args:
            session_id: Session identifier
            task: Task object containing execution parameters and environment specification
        """
        # Resolve environment if specified in task
        environment_spec = None
        logger.info(f"🔍 DEBUG: Task object: {task}")
        logger.info(f"🔍 DEBUG: Task environment attribute: {getattr(task, 'environment', 'NOT_FOUND')}")

        if task.environment:
            logger.info(f"🔍 DEBUG: Task has environment: {task.environment}")
            if not self._environment_loader:
                logger.error(f"Environment specified but no environment loader available for session {session_id}")
                raise RuntimeError("Environment loader not initialized but environment specified in task")

            try:
                logger.info(f"🔍 DEBUG: About to resolve environment: {task.environment}")
                environment_spec = self._environment_loader.resolve_environment(task.environment)
                logger.info(f"✅ Resolved environment for session {session_id}: {task.environment}")
                logger.info(f"🔍 DEBUG: Environment spec: {environment_spec}")
            except Exception as e:
                logger.error(f"❌ Failed to resolve environment for session {session_id}: {e}")
                raise
        else:
            logger.warning("⚠️ DEBUG: Task has no environment specified")

        # Start with task's execution config
        execution_config = task.execution_config.copy()

        # Add executor-specific configurations generically
        # Look for any config key that ends with "_config" and maps to an executor type
        executor_types = self._executor_factory.get_available_executors()
        for executor_type in executor_types:
            config_attr = f"{executor_type}_config"
            if hasattr(task, config_attr):
                config_value = getattr(task, config_attr)
                if config_value:
                    execution_config[executor_type] = config_value
                    logger.debug(f"Added {executor_type} configuration from task")

        # Create new configuration for this task
        self._configuration = execution_config

        # Update sandbox manager with resolved environment spec and logging config
        sandbox_config = {
            "domain": getattr(task, "domain", "unknown"),
            "logs_directory": str(Path(self._config_dir) / "logs"),
            "enable_container_logging": True,
        }

        logger.info(f"🔍 DEBUG: About to check environment_spec. Value: {environment_spec}")

        if environment_spec:
            logger.info(
                f"🔍 DEBUG: Environment spec found, creating SandboxEnvironmentManager with config: {sandbox_config}"
            )
            self._sandbox_manager = SandboxEnvironmentManager(sandbox_config)

            # Create the session environment immediately
            logger.info(f"🔍 DEBUG: About to call create_session_environment for session {session_id}")
            try:
                self._sandbox_manager.create_session_environment(session_id, environment_spec)
                logger.info(f"✅ Created sandbox environment for session {session_id}")
            except Exception as e:
                logger.error(f"❌ FAILED to create sandbox environment for session {session_id}: {e}")
                logger.error(f"❌ Environment spec was: {environment_spec}")
                logger.error(f"❌ Sandbox config was: {sandbox_config}")
                raise
        else:
            logger.warning("⚠️ No environment_spec found, creating SandboxEnvironmentManager without environment")
            self._sandbox_manager = SandboxEnvironmentManager(sandbox_config)

        allowed_executors = task.allowed_executors
        self._executor_factory = ExecutorFactory(
            sandbox_manager=self._sandbox_manager,
            configuration=self._configuration,
            allowed_executors=allowed_executors,
        )

        logger.info(f"ExecutionManager configured for session {session_id} with task-specific settings")
        if allowed_executors:
            logger.info(f"Restricted to executors: {allowed_executors}")

        # Log configured executor types
        configured_executors = [k for k in self._configuration.keys() if k in executor_types]
        if configured_executors:
            logger.info(f"Configured executor-specific settings for: {configured_executors}")

    def to_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Convert all available executors to MCP format.

        Returns:
            List containing MCP tool definitions for all executors
        """
        return self._executor_factory.get_all_mcp_tools()

    def list_commands(self) -> List[Dict[str, Any]]:
        """
        List all available commands from all executors.

        Returns:
            List of command definitions
        """
        commands = []

        for executor_type in self._executor_factory.get_available_executors():
            try:
                executor = self._executor_factory.get_executor(executor_type)
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

    def get_security_info(self) -> Dict[str, Any]:
        """
        Get security information about the execution environment.

        Returns:
            Dictionary containing security-related information
        """
        return {
            "execution_mode": "docker_sandbox",
            "security_level": "isolated",
            "available_executors": self._executor_factory.get_available_executors(),
            "sandbox_active": bool(self._sandbox_manager.active_sessions),
        }

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary.

        Returns:
            Configuration dictionary
        """
        return self._configuration

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active executions.

        Returns:
            Dictionary with execution statistics
        """
        total_active = sum(self._active_executions.values())
        return {
            "total_active_executions": total_active,
            "active_sessions": len(self._active_executions),
            "max_concurrent_per_session": self._max_concurrent_per_session,
            "session_execution_counts": self._active_executions.copy(),
        }

    def cleanup_session(
        self, session_id: str, reason: Optional[CleanupReason] = None, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Clean up session resources including Docker containers through unified cleanup manager.

        Args:
            session_id: The session ID to clean up
            reason: Standardized cleanup reason
            context: Additional context for debugging

        Returns:
            True if cleanup was successful, False otherwise
        """
        if reason is None:
            reason = CleanupReason.SESSION_TERMINATED

        # Use unified cleanup manager for all container cleanup
        cleanup_success = self._cleanup_manager.cleanup_session(session_id, reason, context)

        # Also clean up our execution tracking
        if session_id in self._active_executions:
            del self._active_executions[session_id]

        return cleanup_success

    def initialize_permanent_environment_manager(self, config: Dict[str, Any]) -> None:
        """
        Initialize permanent environment manager with unified container lifecycle management.

        Args:
            config: Configuration dictionary for permanent environment settings
        """
        try:
            # Initialize permanent environment manager
            self._permanent_environment_manager = PermanentEnvironmentManager(config)

            # Update cleanup manager with permanent environment support
            self._cleanup_manager.permanent_manager = self._permanent_environment_manager

            # Update environment loader with permanent environment connectivity
            if self._environment_loader:
                self._environment_loader.permanent_environment_manager = self._permanent_environment_manager
                logger.info("Environment loader configured with permanent environment connectivity")

            logger.info("ExecutionManager: Permanent environment manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize permanent environment manager: {e}")
            raise

    def start_permanent_environment(self, environment_spec: Any) -> None:
        """
        Start permanent environment through unified container lifecycle management.

        Args:
            environment_spec: Permanent environment specification

        Raises:
            RuntimeError: If permanent environment manager is not initialized or startup fails
        """
        if not self._permanent_environment_manager:
            raise RuntimeError("Permanent environment manager not initialized")

        try:
            # Use ensure_permanent_environments_current for configuration change detection
            self._permanent_environment_manager.ensure_permanent_environments_current(environment_spec)
            logger.info("ExecutionManager: Permanent environment started successfully")
        except Exception as e:
            logger.error(f"ExecutionManager: Failed to start permanent environment: {e}")
            raise RuntimeError(f"Failed to start permanent environment: {e}")

    def stop_permanent_environment(self) -> None:
        """
        Stop permanent environment through unified container lifecycle management.

        Raises:
            RuntimeError: If permanent environment shutdown fails
        """
        if not self._cleanup_manager.stop_permanent_environment():
            raise RuntimeError("Failed to stop permanent environment")

        logger.info("ExecutionManager: Permanent environment stopped successfully")

    def is_permanent_environment_running(self) -> bool:
        """
        Check if permanent environment is running.

        Returns:
            True if permanent environment is running, False otherwise
        """
        return self._cleanup_manager.is_permanent_environment_running()

    def cleanup_all_containers(
        self, reason: Optional[CleanupReason] = None, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Clean up all containers (ephemeral and permanent) for server shutdown.

        Args:
            reason: Standardized cleanup reason (typically SERVER_SHUTDOWN)
            context: Additional context for debugging

        Returns:
            Dictionary with cleanup results
        """
        if reason is None:
            reason = CleanupReason.SERVER_SHUTDOWN

        return self._cleanup_manager.cleanup_all_containers(reason, context)
