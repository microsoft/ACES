"""
Executor factory for dynamic executor creation and management.

This module provides a factory pattern for creating and managing different types
of command executors, supporting scaling to many executor types.
"""

import logging
from typing import Any, Dict, List, Optional

from ..sandbox.sandbox_manager import SandboxManager
from .docker_executor import DockerExecutor
from .executor_registry import executor_registry

logger = logging.getLogger(__name__)


class ExecutorFactory:
    """
    Factory for creating and managing command executors.

    Supports dynamic executor selection and creation with configuration management.
    Uses the global executor registry for discovering available executors.
    """

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        configuration: Optional[Dict[str, Any]] = None,
        allowed_executors: Optional[list[str]] = None,
    ):
        """
        Initialize executor factory.

        Args:
            sandbox_manager: Sandbox manager for Docker operations
            configuration: Execution configuration dictionary
            allowed_executors: Optional list of executor types to enable. If None, all executors are available.
        """
        self._sandbox_manager = sandbox_manager
        self._configuration = configuration or {}
        self._executor_instances: Dict[str, DockerExecutor] = {}

        # Get available executors from the global registry
        available_executors = executor_registry.get_available_executors()

        # Filter executors based on allowed list
        if allowed_executors is not None:
            # Validate that all requested executors are available
            invalid_executors = [ex for ex in allowed_executors if ex not in available_executors]
            if invalid_executors:
                logger.warning(
                    f"Invalid executor types requested: {invalid_executors}. Available: {available_executors}"
                )

            # Create filtered list
            self._allowed_executors = [ex for ex in allowed_executors if ex in available_executors]
            logger.info(f"ExecutorFactory initialized with filtered executors: {self._allowed_executors}")
        else:
            self._allowed_executors = available_executors.copy()
            logger.info(f"ExecutorFactory initialized with all available executors: {self._allowed_executors}")

        logger.info(f"ExecutorFactory initialized with {len(self._allowed_executors)} executor types")

    def get_available_executors(self) -> List[str]:
        """
        Get list of available executor types for this factory instance.

        Returns:
            List of executor type names that this factory can create
        """
        return self._allowed_executors.copy()

    def get_executor(self, executor_type: str, force_new: bool = False) -> DockerExecutor:
        """
        Get or create an executor instance.

        Args:
            executor_type: Type of executor to get/create
            force_new: If True, create a new instance even if one exists

        Returns:
            Configured executor instance

        Raises:
            ValueError: If executor_type is not registered or not allowed
        """
        if executor_type not in self._allowed_executors:
            raise ValueError(
                f"Unknown or disabled executor type: {executor_type}. " f"Available types: {self._allowed_executors}"
            )

        # Return existing instance unless force_new is True
        if not force_new and executor_type in self._executor_instances:
            return self._executor_instances[executor_type]

        # Get executor class from global registry
        try:
            executor_class = executor_registry.get_executor_class(executor_type)
        except KeyError:
            raise ValueError(f"Executor type '{executor_type}' not found in registry")

        logger.debug(f"Creating new {executor_type} executor instance")

        # Get the executor class's default configuration
        default_config = executor_class.get_default_config()

        # Get executor-specific configuration from global configuration
        executor_config = self._configuration.get(executor_type, {})

        # Merge with defaults, giving preference to provided config
        merged_config = {**default_config, **executor_config}

        # Override timeout from global config if available
        global_timeout = self._configuration.get("timeout")
        if global_timeout is not None:
            merged_config["timeout"] = global_timeout

        # Prepare additional parameters for specific executor needs
        additional_params = {}

        # Add CLI-specific parameters if needed
        if executor_type == "cli":
            allowed_commands = self._configuration.get("allowed_commands")
            if allowed_commands is not None:
                additional_params["allowed_commands"] = allowed_commands

        # Use the generic creation method
        executor = executor_class.create_with_config(
            sandbox_manager=self._sandbox_manager, config=merged_config, additional_params=additional_params
        )

        # Cache the instance
        self._executor_instances[executor_type] = executor
        return executor

    def get_all_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get MCP tool definitions for all available executors.

        Returns:
            List of MCP tool definitions
        """
        tools = []

        for executor_type in self._allowed_executors:
            try:
                executor = self.get_executor(executor_type)

                # Get the tool definition with executor type prefix
                tool_schema = executor.to_mcp_schema()
                metadata = getattr(executor, "_executor_metadata", {})

                tool_def = {
                    "name": metadata.get("name", executor_type),
                    "description": metadata.get("description", f"{executor_type.title()} executor"),
                    "inputSchema": tool_schema,
                }

                tools.append(tool_def)

            except Exception as e:
                logger.error(f"Failed to get MCP schema for executor {executor_type}: {e}")

        return tools

    def cleanup_all_executors(self) -> None:
        """Clean up all executor instances."""
        for executor_type, executor in self._executor_instances.items():
            try:
                # Cleanup is handled by the sandbox manager
                logger.debug(f"Cleaned up {executor_type} executor")
            except Exception as e:
                logger.error(f"Error cleaning up {executor_type} executor: {e}")

        self._executor_instances.clear()

    def get_executor_info(self) -> Dict[str, Any]:
        """
        Get information about all available executors.

        Returns:
            Dictionary with executor information
        """
        # Build configurations dynamically from executor classes
        configurations = {}
        for executor_type in self._allowed_executors:
            try:
                executor_class = executor_registry.get_executor_class(executor_type)
                configurations[executor_type] = executor_class.get_default_config()
            except KeyError:
                logger.warning(f"Executor type '{executor_type}' not found in registry")

        info = {
            "available_types": self.get_available_executors(),
            "active_instances": list(self._executor_instances.keys()),
            "registry_size": len(self._allowed_executors),
            "configurations": configurations,
        }

        return info
