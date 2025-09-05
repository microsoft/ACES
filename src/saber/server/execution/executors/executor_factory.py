"""
Executor factory for dynamic executor creation and management.

This module provides a factory pattern for creating and managing different types
of command executors, supporting scaling to many executor types.
"""

import logging
from typing import Any, Dict, List, Optional

from ..sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from .docker_executor import DockerExecutor
from .executor_registry import executor_registry

logger = logging.getLogger(__name__)


class ExecutorFactory:
    """
    Factory for creating and managing command executors with episode-specific configurations.

    Supports dynamic executor selection and creation with configuration management.
    Uses the global executor registry for discovering available executors.
    Maintains episode-specific allowed executor configurations.
    """

    def __init__(
        self,
        sandbox_manager: SandboxEnvironmentManager,
        configuration: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize executor factory.

        Args:
            sandbox_manager: Sandbox manager for Docker operations
            configuration: Base execution configuration dictionary
        """
        self._sandbox_manager = sandbox_manager
        self._configuration = configuration or {}
        self._executor_instances: Dict[str, DockerExecutor] = {}

        # Get all available executors from the global registry
        self._all_available_executors = executor_registry.get_available_executors()

        # Episode-specific configurations: episode_id -> {allowed_executors, config}
        self._episode_configurations: Dict[str, Dict[str, Any]] = {}

        logger.info(f"ExecutorFactory initialized with {len(self._all_available_executors)} total executor types")
        logger.info(f"All available executors: {self._all_available_executors}")

    def register_episode_configuration(
        self,
        episode_id: str,
        allowed_executors: Optional[List[str]] = None,
        episode_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register episode-specific executor configuration.

        Args:
            episode_id: Episode identifier
            allowed_executors: List of executor types allowed for this episode
            episode_config: Episode-specific execution configuration
        """
        # Validate allowed executors
        if allowed_executors is not None:
            invalid_executors = [ex for ex in allowed_executors if ex not in self._all_available_executors]
            if invalid_executors:
                logger.warning(
                    f"Invalid executor types for episode {episode_id}: {invalid_executors}. "
                    f"Available: {self._all_available_executors}"
                )
            # Filter to only valid executors
            valid_allowed_executors = [ex for ex in allowed_executors if ex in self._all_available_executors]
        else:
            # If no specific allowed executors, allow all
            valid_allowed_executors = self._all_available_executors.copy()

        # Store episode configuration
        self._episode_configurations[episode_id] = {
            "allowed_executors": valid_allowed_executors,
            "config": episode_config or {},
        }

        logger.info(
            f"Registered episode {episode_id} with {len(valid_allowed_executors)} "
            f"allowed executors: {valid_allowed_executors}"
        )

    def unregister_episode_configuration(self, episode_id: str) -> None:
        """
        Remove episode-specific configuration.

        Args:
            episode_id: Episode identifier to remove
        """
        if episode_id in self._episode_configurations:
            del self._episode_configurations[episode_id]
            logger.debug(f"Unregistered episode configuration for {episode_id}")

    def get_available_executors(self, episode_id: Optional[str] = None) -> List[str]:
        """
        Get list of available executor types, optionally filtered by episode configuration.

        Args:
            episode_id: Optional episode identifier to get episode-specific executors

        Returns:
            List of executor type names available for the episode or all executors
        """
        if episode_id is None:
            return self._all_available_executors.copy()

        if episode_id in self._episode_configurations:
            allowed_executors: List[str] = self._episode_configurations[episode_id]["allowed_executors"]
            return allowed_executors.copy()

        logger.warning(f"No configuration found for episode {episode_id}, returning all executors")
        return self._all_available_executors.copy()

    def get_executor(
        self, executor_type: str, episode_id: Optional[str] = None, force_new: bool = False
    ) -> DockerExecutor:
        """
        Get or create an executor instance, optionally filtered by episode configuration.

        Args:
            executor_type: Type of executor to get/create
            episode_id: Optional episode identifier for allowed executor filtering
            force_new: If True, create a new instance even if one exists

        Returns:
            Configured executor instance

        Raises:
            ValueError: If executor_type is not registered or not allowed for the episode
        """
        # Check if executor is allowed for this episode
        allowed_executors = self.get_available_executors(episode_id)
        if executor_type not in allowed_executors:
            episode_context = f" for episode {episode_id}" if episode_id else ""
            raise ValueError(
                f"Unknown or disabled executor type: {executor_type}{episode_context}. "
                f"Available types: {allowed_executors}"
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

    def get_all_mcp_tools(self, episode_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get MCP tool definitions for available executors.

        Args:
            episode_id: Optional episode ID to filter tools by episode's allowed executors.
                       If None, returns all tools with a warning.

        Returns:
            List of MCP tool definitions
        """
        tools = []

        # Determine which executors to include
        if episode_id is None:
            logger.warning("No episode_id provided to get_all_mcp_tools, returning all available tools")
            allowed_executors = list(self._all_available_executors)
        else:
            episode_config = self._episode_configurations.get(episode_id)
            if episode_config is None:
                logger.warning(f"No configuration found for episode {episode_id}, returning all available tools")
                allowed_executors = list(self._all_available_executors)
            else:
                allowed_executors = episode_config.get("allowed_executors", [])

        for executor_type in allowed_executors:
            try:
                executor = self.get_executor(executor_type, episode_id=episode_id)

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
        for executor_type in self._all_available_executors:
            try:
                executor_class = executor_registry.get_executor_class(executor_type)
                configurations[executor_type] = executor_class.get_default_config()
            except KeyError:
                logger.warning(f"Executor type '{executor_type}' not found in registry")

        info = {
            "available_types": self.get_available_executors(),
            "active_instances": list(self._executor_instances.keys()),
            "registry_size": len(self._all_available_executors),
            "configurations": configurations,
        }

        return info
