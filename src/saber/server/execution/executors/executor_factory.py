"""Executor factory for dynamic executor creation and management.

Logging Category: EXECUTION

This module provides a factory pattern for creating and managing different types
of command executors, supporting scaling to many executor types.
"""

from typing import Any, Dict, List, Optional

import mcp.types as mcp_types

from saber.logging_config import LogCategory, get_saber_logger

from ..sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from .docker_executor import DockerExecutor
from .executor_registry import executor_registry

logger = get_saber_logger(LogCategory.EXECUTION, __name__)


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

        logger.info(
            "Executor factory initialized",
            extra={
                "event": "executor_factory_initialized",
                "total_executor_types": len(self._all_available_executors),
                "available_executors": self._all_available_executors,
            },
        )

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
                    "Invalid executor types provided for episode",
                    extra={
                        "event": "executor_factory_invalid_episode_executors",
                        "episode_id": episode_id,
                        "invalid_executors": invalid_executors,
                        "available_executors": self._all_available_executors,
                    },
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
            "Episode executor configuration registered",
            extra={
                "event": "executor_factory_episode_registered",
                "episode_id": episode_id,
                "allowed_executor_count": len(valid_allowed_executors),
                "allowed_executors": valid_allowed_executors,
            },
        )

    def unregister_episode_configuration(self, episode_id: str) -> None:
        """
        Remove episode-specific configuration.

        Args:
            episode_id: Episode identifier to remove
        """
        if episode_id in self._episode_configurations:
            del self._episode_configurations[episode_id]
            logger.debug(
                "Episode executor configuration unregistered",
                extra={
                    "event": "executor_factory_episode_unregistered",
                    "episode_id": episode_id,
                },
            )

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

        logger.warning(
            "Episode configuration missing; returning all executors",
            extra={
                "event": "executor_factory_missing_episode_config",
                "episode_id": episode_id,
            },
        )
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

        logger.debug(
            "Creating executor instance",
            extra={
                "event": "executor_factory_instance_creation",
                "executor_type": executor_type,
                "episode_id": episode_id,
                "force_new": force_new,
            },
        )

        # Get the executor class's default configuration
        default_config = executor_class.get_default_config()

        # Get executor-specific configuration from global configuration
        # Look under 'executors' key first, then fall back to root level (for backwards compatibility)
        executors_section = self._configuration.get("executors", {})
        executor_config = executors_section.get(executor_type, self._configuration.get(executor_type, {}))

        # Merge with defaults, giving preference to executor-specific config
        # No global timeout override - each executor must specify its own timeout
        merged_config = {**default_config, **executor_config}

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

    def get_all_mcp_tools(self, episode_id: Optional[str] = None) -> List[mcp_types.Tool]:
        """
        Get MCP tool definitions for available executors as mcp.types.Tool objects.

        Args:
            episode_id: Optional episode ID to filter tools by episode's allowed executors.
                       If None, returns all tools with a warning.

        Returns:
            List of mcp.types.Tool objects ready for FastMCP consumption
        """
        tools = []

        # Determine which executors to include
        if episode_id is None:
            logger.warning(
                "Episode ID missing for MCP tools request; returning all tools",
                extra={
                    "event": "executor_factory_mcp_tools_no_episode",
                },
            )
            allowed_executors = list(self._all_available_executors)
        else:
            episode_config = self._episode_configurations.get(episode_id)
            if episode_config is None:
                logger.warning(
                    "Episode configuration missing for MCP tools; returning all tools",
                    extra={
                        "event": "executor_factory_mcp_tools_missing_episode_config",
                        "episode_id": episode_id,
                    },
                )
                allowed_executors = list(self._all_available_executors)
            else:
                # Derive allowed_executors from stored episode configuration
                # Check for explicit allowed_executors first (backwards compat)
                # Then check for executors section keys
                if "allowed_executors" in episode_config:
                    allowed_executors = episode_config.get("allowed_executors", [])
                elif "executors" in episode_config and episode_config["executors"]:
                    allowed_executors = list(episode_config["executors"].keys())
                else:
                    allowed_executors = []

        for executor_type in allowed_executors:
            try:
                executor = self.get_executor(executor_type, episode_id=episode_id)

                # Get the tool schema as a dictionary
                tool_schema = executor.to_mcp_schema()
                metadata = getattr(executor, "_executor_metadata", {})

                # Convert tool_schema to dict if it's a Pydantic object
                if hasattr(tool_schema, "model_dump"):
                    input_schema_dict = tool_schema.model_dump()
                elif hasattr(tool_schema, "dict"):
                    input_schema_dict = tool_schema.dict()
                else:
                    # Assume it's already a dict or convert to dict
                    input_schema_dict = dict(tool_schema) if hasattr(tool_schema, "__iter__") else {}

                # Create mcp.types.Tool object directly
                mcp_tool = mcp_types.Tool(
                    name=metadata.get("name", executor_type),
                    description=metadata.get("description", f"{executor_type.title()} executor"),
                    inputSchema=input_schema_dict,  # Must be a plain dict for FastMCP compatibility
                )

                tools.append(mcp_tool)

            except Exception as e:
                logger.error(
                    "Failed to build MCP tool schema",
                    extra={
                        "event": "executor_factory_mcp_schema_failed",
                        "executor_type": executor_type,
                        "episode_id": episode_id,
                        "error": str(e),
                    },
                )

        return tools

    def cleanup_all_executors(self) -> None:
        """Clean up all executor instances."""
        for executor_type, executor in self._executor_instances.items():
            try:
                # Cleanup is handled by the sandbox manager
                logger.debug(
                    "Executor cleanup recorded",
                    extra={
                        "event": "executor_factory_cleanup_recorded",
                        "executor_type": executor_type,
                    },
                )
            except Exception as e:
                logger.error(
                    "Executor cleanup failed",
                    extra={
                        "event": "executor_factory_cleanup_failed",
                        "executor_type": executor_type,
                        "error": str(e),
                    },
                )

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
                logger.warning(
                    "Executor type missing from registry during info retrieval",
                    extra={
                        "event": "executor_factory_missing_registry_entry",
                        "executor_type": executor_type,
                    },
                )

        info = {
            "available_types": self.get_available_executors(),
            "active_instances": list(self._executor_instances.keys()),
            "registry_size": len(self._all_available_executors),
            "configurations": configurations,
        }

        return info
