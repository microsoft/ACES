"""
ExecutionManager implementation for command execution.

This version provides access to CLI commands with comprehensive
security validation capabilities.
"""

import logging
from typing import Any, Dict, List, Optional

from ..base import Action, CommandResult
from .executors.docker_executor import DockerExecutor
from .executors.executor_factory import ExecutorFactory
from .sandbox.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)


class ExecutionManager:
    """
    Execution manager supporting multiple executor types with factory pattern.

    Provides unified interface for executing different types of commands (CLI, Python, etc.)
    with security validation and Docker isolation. Commands are executed sequentially.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the execution manager

        Args:
            config_dir: Path to configuration directory for environment resolution

        Task-specific configuration will be provided when sessions are created.
        """
        self._config_dir = config_dir
        self._environment_loader = None

        # Initialize environment loader if config directory is provided
        if config_dir:
            from pathlib import Path

            environments_path = Path(config_dir) / "environments.yaml"
            if environments_path.exists():
                from .environment_loader import EnvironmentLoader

                self._environment_loader = EnvironmentLoader(str(environments_path))
                logger.info(f"Environment loader initialized with: {environments_path}")

            # Load custom executors from the same directory
            self._load_custom_executors(config_dir)

        self._configuration: Dict[str, Any] = {}

        # Initialize sandbox manager with empty config (will be updated per session)
        self._sandbox_manager = SandboxManager({})

        # Initialize executor factory with minimal configuration
        self._executor_factory = ExecutorFactory(
            sandbox_manager=self._sandbox_manager,
            configuration=self._configuration,
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
        cleanup_token: Optional[str] = None,
        saber_host_url: Optional[str] = None,
        server_network: Optional[str] = None,
    ) -> None:
        """
        Configure ExecutionManager for a specific task/session.

        Args:
            session_id: Session identifier
            task: Task object containing execution parameters and environment specification
            cleanup_token: Optional cleanup token for container self-termination coordination
            saber_host_url: Optional SABER server URL for orchestrator connectivity
            server_network: Optional server network name for orchestrator connectivity
        """
        # Resolve environment if specified in task
        environment_spec = None
        if task.environment:
            if not self._environment_loader:
                logger.error(f"Environment specified but no environment loader available for session {session_id}")
                raise RuntimeError("Environment loader not initialized but environment specified in task")

            try:
                environment_spec = self._environment_loader.resolve_environment(task.environment)
                logger.debug(f"Resolved environment for session {session_id}: {task.environment}")
            except Exception as e:
                logger.error(f"Failed to resolve environment for session {session_id}: {e}")
                raise

        # Start with task's execution config
        execution_config = task.execution_config.copy()

        # Add cleanup token to environment variables if provided
        if cleanup_token:
            if "environment" not in execution_config:
                execution_config["environment"] = {}
            execution_config["environment"]["SABER_CLEANUP_TOKEN"] = cleanup_token
            execution_config["environment"]["SABER_SESSION_ID"] = session_id
            logger.debug(f"Added cleanup coordination environment variables for session {session_id}")

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

        # Update sandbox manager with resolved environment spec
        if environment_spec:
            self._sandbox_manager = SandboxManager({})

            # Use provided SABER host URL or sensible defaults
            if not saber_host_url:
                import os

                # Option 1: Use explicit environment variable if set
                if os.environ.get("SABER_ORCHESTRATOR_HOST_URL"):
                    saber_host_url = os.environ.get("SABER_ORCHESTRATOR_HOST_URL")
                # Option 2: Default based on environment
                elif os.path.exists("/.dockerenv"):
                    # For containers, use Docker bridge gateway as fallback
                    port = os.environ.get("SABER_PORT", "8000")
                    saber_host_url = f"http://172.17.0.1:{port}"
                else:
                    # Default for non-containerized environments
                    saber_host_url = "http://localhost:8000"

            logger.info(f"Using SABER host URL for orchestrator: {saber_host_url}")
            if server_network:
                logger.info(f"Using server network for orchestrator: {server_network}")

            # Create the session environment immediately with orchestrator integration
            self._sandbox_manager.create_session_environment(
                session_id, environment_spec, cleanup_token, saber_host_url, server_network
            )
            logger.info(f"Created sandbox environment for session {session_id}")
            if cleanup_token:
                logger.info(f"Episode orchestrator will auto-start for session {session_id}")
        else:
            self._sandbox_manager = SandboxManager({})

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

    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up execution resources for a session.

        Args:
            session_id: Session identifier to clean up
        """
        logger.warning(f"🔥 EXECUTION CLEANUP: ExecutionManager.cleanup_session() called for session {session_id}")
        try:
            # Cancel any active executions for this session
            if session_id in self._active_executions:
                logger.warning(
                    f"🔥 ACTIVE EXECUTION CANCEL: Cancelling {self._active_executions[session_id]} "
                    f"active executions for session {session_id}"
                )
                del self._active_executions[session_id]

            logger.warning(
                f"🔥 SANDBOX CLEANUP: About to call sandbox_manager.cleanup_session() for session {session_id}"
            )
            self._sandbox_manager.cleanup_session(session_id)
            logger.info(f"Cleaned up execution resources for session {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up execution resources for session {session_id}: {e}")

    def cleanup_all_sessions(self) -> None:
        """Clean up all execution resources."""
        logger.warning("🔥 EXECUTION CLEANUP ALL: ExecutionManager.cleanup_all_sessions() called")
        try:
            logger.warning("🔥 SANDBOX CLEANUP ALL: About to call sandbox_manager.cleanup_all_sessions()")
            self._sandbox_manager.cleanup_all_sessions()
            logger.warning("🔥 EXECUTOR CLEANUP ALL: About to call executor_factory.cleanup_all_executors()")
            self._executor_factory.cleanup_all_executors()
            logger.info("Cleaned up all execution resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

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
