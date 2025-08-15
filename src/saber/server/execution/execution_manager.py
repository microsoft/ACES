"""
ExecutionManager implementation for command execution.

This version provides access to CLI commands with comprehensive
security validation capabilities.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..base import Action, CommandResult
from .executors.docker_executor import DockerExecutor
from .executors.factory import ExecutorFactory
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

        self._configuration: Dict[str, Any] = {}

        # Initialize sandbox manager with empty config (will be updated per session)
        self._sandbox_manager = SandboxManager({})

        # Initialize executor factory with minimal configuration
        self._executor_factory = ExecutorFactory(
            sandbox_manager=self._sandbox_manager,
            configuration=self._configuration,
        )

        # Sequential execution lock - ensures only one command executes at a time
        self._execution_lock = asyncio.Lock()

        logger.info("ExecutionManager initialized for sequential execution")
        logger.info(f"Available executor types: {self._executor_factory.get_filtered_available_executors()}")

    async def step(self, action: Action, context: Optional[Dict[str, Any]] = None) -> CommandResult:
        """
        Execute the action with the appropriate executor.

        Commands are executed sequentially - only one command can execute at a time.

        Args:
            action: Action object containing command and parameters
            context: Optional execution context

        Returns:
            CommandResult with execution results
        """
        async with self._execution_lock:
            try:
                # Get executor directly from action's tool name
                executor = self._executor_factory.get_executor(action.tool_name)

                # Extract parameters from action
                parameters = {"command": action.command}
                parameters.update(action.parameters)

                # Validate parameters first
                validation_result = executor.validate_parameters(parameters)
                if not validation_result.valid:
                    return CommandResult.error_result(
                        error=f"Parameter validation failed: {', '.join(validation_result.errors)}"
                    )

                # Execute using the appropriate executor
                return await executor.execute(parameters, context or {})

            except Exception as e:
                logger.error(f"Execution failed: {e}")
                return CommandResult.error_result(error=str(e))

    def get_executor(self, executor_type: str) -> DockerExecutor:
        """
        Get a specific executor instance.

        Args:
            executor_type: Type of executor to retrieve

        Returns:
            Executor instance
        """
        return self._executor_factory.get_executor(executor_type)

    def configure_for_task(self, session_id: str, task: Any) -> None:
        """
        Configure ExecutionManager for a specific task/session.

        Args:
            session_id: Session identifier
            task: Task object containing execution parameters and environment specification
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
            # Create the session environment immediately
            self._sandbox_manager.create_session_environment(session_id, environment_spec)
            logger.info(f"Created sandbox environment for session {session_id}")
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
        try:
            self._sandbox_manager.cleanup_session(session_id)
            logger.info(f"Cleaned up execution resources for session {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up execution resources for session {session_id}: {e}")

    def cleanup_all_sessions(self) -> None:
        """Clean up all execution resources."""
        try:
            self._sandbox_manager.cleanup_all_sessions()
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
                metadata = getattr(executor, "_security_command_metadata", {})

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

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        return {
            "execution_mode": "sequential",
            "executor_info": self._executor_factory.get_executor_info(),
        }

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
