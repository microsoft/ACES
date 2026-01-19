"""Executor factory for dynamic executor creation and management.

Logging Category: EXECUTION

This module provides a factory pattern for creating and managing different types
of command executors, supporting scaling to many executor types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types

from saber.logging_config import LogCategory, get_saber_logger

from ..models import EpisodeConfiguration
from ..sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from .base_executors import CommandExecutor
from .executor_registry import executor_registry

if TYPE_CHECKING:
    from ...session_manager import SessionManager

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
        configuration: dict[str, Any] | None = None,
        session_manager: SessionManager | None = None,
    ):
        """
        Initialize executor factory.

        Args:
            sandbox_manager: Sandbox manager for Docker operations
            configuration: Base execution configuration dictionary (stored by reference, not copied)
            session_manager: Optional session manager for cross-episode access (injected into executors)
        """
        self._sandbox_manager = sandbox_manager
        self._session_manager: SessionManager | None = session_manager
        # Store configuration by reference so updates to the dict are reflected
        # This allows ExecutionManager to update config after factory creation
        self._configuration = configuration if configuration is not None else {}
        # Cache executors by (executor_type, episode_id) tuple to support episode-specific configs
        # episode_id=None is used for schema discovery during MCP registration
        self._executor_instances: dict[tuple[str, str | None], CommandExecutor] = {}

        # Get all available executors from the global registry
        self._all_available_executors = executor_registry.get_available_executors()

        # Episode-specific configurations: episode_id -> EpisodeConfiguration
        self._episode_configurations: dict[str, EpisodeConfiguration] = {}

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
        allowed_executors: list[str] | None = None,
        episode_config: dict[str, Any] | None = None,
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

        # Store episode configuration using typed class
        self._episode_configurations[episode_id] = EpisodeConfiguration(
            allowed_executors=valid_allowed_executors,
            config=episode_config or {},
        )

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
        Remove episode-specific configuration and cached executors.

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

        # Clear cached executor instances for this episode
        keys_to_remove = [
            key
            for key in self._executor_instances.keys()
            if key[1] == episode_id  # key is (executor_type, episode_id)
        ]
        for key in keys_to_remove:
            del self._executor_instances[key]

        if keys_to_remove:
            logger.debug(
                "Episode executor instances cleared from cache",
                extra={
                    "event": "executor_factory_episode_cache_cleared",
                    "episode_id": episode_id,
                    "cleared_count": len(keys_to_remove),
                },
            )

    def get_available_executors(self, episode_id: str | None = None) -> list[str]:
        """
        Get list of available executor types, optionally filtered by episode configuration.

        Args:
            episode_id: Optional episode identifier to get episode-specific executors

        Returns:
            List of executor type names available for the episode or all executors
        """
        if episode_id is None:
            return self._all_available_executors.copy()

        episode_config = self._episode_configurations.get(episode_id)
        if episode_config is not None:
            return episode_config.allowed_executors.copy()

        logger.warning(
            "Episode configuration missing; returning all executors",
            extra={
                "event": "executor_factory_missing_episode_config",
                "episode_id": episode_id,
            },
        )
        return self._all_available_executors.copy()

    def get_executor(
        self, executor_type: str, episode_id: str | None = None, force_new: bool = False
    ) -> CommandExecutor:
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

        # Cache key includes episode_id to support episode-specific configurations
        cache_key = (executor_type, episode_id)

        # Return existing instance unless force_new is True
        if not force_new and cache_key in self._executor_instances:
            return self._executor_instances[cache_key]

        # Get executor class from global registry
        try:
            executor_class = executor_registry.get_executor_class(executor_type)
        except KeyError:
            raise ValueError(f"Executor type '{executor_type}' not found in registry") from None

        logger.debug(
            "Creating executor instance",
            extra={
                "event": "executor_factory_instance_creation",
                "executor_type": executor_type,
                "episode_id": episode_id,
                "force_new": force_new,
            },
        )

        # Get the executor class's typed default configuration
        default_config = executor_class.get_default_config()

        # Collect runtime overrides from episode and factory config
        # Priority: episode config > factory config > defaults
        episode_config_dict: dict[str, Any] = {}
        episode_configuration = self._episode_configurations.get(episode_id) if episode_id else None
        if episode_configuration is not None:
            episode_config_dict = episode_configuration.config

        # Get executor-specific overrides from episode and factory configs
        episode_executors = episode_config_dict.get("executors", {})
        executor_overrides_episode = episode_executors.get(executor_type, {})

        factory_executors = self._configuration.get("executors", {})
        executor_overrides_factory = factory_executors.get(executor_type, self._configuration.get(executor_type, {}))

        # Merge overrides: factory < episode (episode takes precedence)
        merged_overrides = {**executor_overrides_factory, **executor_overrides_episode}

        # Apply overrides to typed config - no dict conversion needed
        typed_config = default_config.with_overrides(merged_overrides)

        # Prepare additional parameters for specific executor needs
        additional_params: dict[str, Any] = {}

        # Add CLI-specific parameters if needed
        if executor_type == "cli":
            allowed_commands = self._configuration.get("allowed_commands")
            if allowed_commands is not None:
                additional_params["allowed_commands"] = allowed_commands

        # Use the generic creation method with session_manager
        # All executors must have create_with_config classmethod
        executor_instance = executor_class.create_with_config(
            sandbox_manager=self._sandbox_manager,
            config=typed_config,
            additional_params=additional_params,
            session_manager=self._session_manager,
        )

        # Cache the instance by (executor_type, episode_id)
        self._executor_instances[cache_key] = executor_instance
        return executor_instance

    def get_all_mcp_tools(self, episode_id: str | None = None) -> list[mcp_types.Tool]:
        """
        Get MCP tool definitions for available executors as mcp.types.Tool objects.

        Args:
            episode_id: Optional episode ID to filter tools by episode's allowed executors.
                       If None, returns all tools with a warning.

        Returns:
            List of mcp.types.Tool objects ready for FastMCP consumption
        """
        tools: list[mcp_types.Tool] = []

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
            episode_configuration = self._episode_configurations.get(episode_id)
            if episode_configuration is None:
                logger.warning(
                    "Episode configuration missing for MCP tools; returning all tools",
                    extra={
                        "event": "executor_factory_mcp_tools_missing_episode_config",
                        "episode_id": episode_id,
                    },
                )
                allowed_executors = list(self._all_available_executors)
            else:
                # Use typed EpisodeConfiguration's allowed_executors
                # Fall back to checking executors section in config dict
                if episode_configuration.allowed_executors:
                    allowed_executors = episode_configuration.allowed_executors.copy()
                elif episode_configuration.config.get("executors"):
                    allowed_executors = list(episode_configuration.config["executors"].keys())
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

    def update_configuration(self, new_configuration: dict[str, Any]) -> None:
        """
        Update the factory's configuration and clear cached executors.

        This is needed when configuration changes after the factory is created,
        such as when a task is loaded with new execution settings.

        Args:
            new_configuration: New configuration dictionary to use
        """
        logger.info(
            "Updating executor factory configuration",
            extra={
                "event": "executor_factory_config_update",
                "old_config_has_executors": "executors" in self._configuration,
                "new_config_has_executors": "executors" in new_configuration,
            },
        )

        # Update the configuration reference
        self._configuration = new_configuration

        # Clear cached executors since they were created with old config
        if self._executor_instances:
            logger.debug(
                "Clearing cached executors due to config update",
                extra={
                    "event": "executor_factory_cache_clear",
                    "cached_executor_count": len(self._executor_instances),
                },
            )
            self._executor_instances.clear()

    def cleanup_all_executors(self) -> None:
        """Clean up all executor instances."""
        for cache_key, _executor in self._executor_instances.items():
            executor_type, episode_id = cache_key
            try:
                # Cleanup is handled by the sandbox manager
                logger.debug(
                    "Executor cleanup recorded",
                    extra={
                        "event": "executor_factory_cleanup_recorded",
                        "executor_type": executor_type,
                        "episode_id": episode_id,
                    },
                )
            except Exception as e:
                logger.error(
                    "Executor cleanup failed",
                    extra={
                        "event": "executor_factory_cleanup_failed",
                        "executor_type": executor_type,
                        "episode_id": episode_id,
                        "error": str(e),
                    },
                )

        self._executor_instances.clear()

    def get_executor_info(self) -> dict[str, Any]:
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

        # Format cache keys for display: "executor_type" or "executor_type@episode_id"
        active_instances = [
            f"{ex_type}@{ep_id}" if ep_id else ex_type for ex_type, ep_id in self._executor_instances.keys()
        ]

        info = {
            "available_types": self.get_available_executors(),
            "active_instances": active_instances,
            "registry_size": len(self._all_available_executors),
            "configurations": configurations,
        }

        return info
