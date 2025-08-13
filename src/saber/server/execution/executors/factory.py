"""
Executor factory for dynamic executor creation and management.

This module provides a factory pattern for creating and managing different types
of command executors, supporting scaling to many executor types.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from ..sandbox.sandbox_manager import SandboxManager
from .cli import CLIExecutor
from .docker_executor import DockerExecutor
from .python_executor import PythonExecutor

if TYPE_CHECKING:
    from ..execution_manager import ExecutionConfiguration

logger = logging.getLogger(__name__)


class ExecutorFactory:
    """
    Factory for creating and managing command executors.

    Supports dynamic executor selection and creation with configuration management.
    Designed to scale to many (10s) of executor types.
    """

    # Registry of available executor types
    _executor_registry: Dict[str, Type[DockerExecutor]] = {
        "cli": CLIExecutor,
        "python": PythonExecutor,
    }

    # Default configurations for each executor type
    _default_configs: Dict[str, Dict[str, Any]] = {
        "cli": {
            "default_shell_mode": False,
            "timeout": 300.0,
        },
        "python": {
            "allowed_modules": [
                "os",
                "sys",
                "json",
                "csv",
                "datetime",
                "time",
                "random",
                "math",
                "re",
                "collections",
                "itertools",
                "functools",
                "requests",
                "urllib",
                "base64",
                "hashlib",
                "subprocess",
            ],
            "script_templates": {},
            "timeout": 600.0,  # Python scripts may take longer
        },
    }

    def __init__(self, sandbox_manager: SandboxManager, configuration: Optional["ExecutionConfiguration"] = None):
        """
        Initialize executor factory.

        Args:
            sandbox_manager: Sandbox manager for Docker operations
            configuration: Full execution configuration manager (each executor extracts what it needs)
        """
        self._sandbox_manager = sandbox_manager
        self._configuration = configuration
        self._executor_instances: Dict[str, DockerExecutor] = {}

        logger.info(f"ExecutorFactory initialized with {len(self._executor_registry)} executor types")

    @classmethod
    def register_executor(cls, executor_type: str, executor_class: Type[DockerExecutor]) -> None:
        """
        Register a new executor type.

        Args:
            executor_type: String identifier for the executor
            executor_class: Executor class (must inherit from DockerExecutor)
        """
        if not issubclass(executor_class, DockerExecutor):
            raise ValueError(f"Executor class must inherit from DockerExecutor, got: {executor_class}")

        cls._executor_registry[executor_type] = executor_class
        logger.info(f"Registered executor type: {executor_type}")

    @classmethod
    def unregister_executor(cls, executor_type: str) -> None:
        """
        Unregister an executor type.

        Args:
            executor_type: String identifier for the executor to remove
        """
        if executor_type in cls._executor_registry:
            del cls._executor_registry[executor_type]
            logger.info(f"Unregistered executor type: {executor_type}")

    @classmethod
    def get_available_executors(cls) -> List[str]:
        """
        Get list of available executor types.

        Returns:
            List of registered executor type names
        """
        return list(cls._executor_registry.keys())

    def get_executor(self, executor_type: str, force_new: bool = False) -> DockerExecutor:
        """
        Get or create an executor instance.

        Args:
            executor_type: Type of executor to get/create
            force_new: If True, create a new instance even if one exists

        Returns:
            Configured executor instance

        Raises:
            ValueError: If executor_type is not registered
        """
        if executor_type not in self._executor_registry:
            raise ValueError(
                f"Unknown executor type: {executor_type}. " f"Available types: {list(self._executor_registry.keys())}"
            )

        # Return existing instance unless force_new is True
        if not force_new and executor_type in self._executor_instances:
            return self._executor_instances[executor_type]

        # Create new executor instance
        executor_class = self._executor_registry[executor_type]

        logger.debug(f"Creating new {executor_type} executor instance")

        if executor_type == "cli":
            # Pass CLI-specific configuration section and security config
            cli_config = self._configuration.get_section("cli") if self._configuration else {}
            allowed_commands = self._configuration.get_allowed_commands() if self._configuration else None
            timeout = self._configuration.get_execution_timeout() if self._configuration else 300.0

            executor = executor_class(
                sandbox_manager=self._sandbox_manager,
                cli_config=cli_config,
                allowed_commands=allowed_commands,
                timeout=timeout,
            )
        elif executor_type == "python":
            # Pass Python-specific configuration section
            python_config = self._configuration.get_section("python") if self._configuration else {}
            timeout = self._configuration.get_execution_timeout() if self._configuration else 600.0

            executor = executor_class(
                sandbox_manager=self._sandbox_manager, python_config=python_config, timeout=timeout
            )
        else:
            # Generic creation for custom executor types - use default config
            default_config = self._default_configs.get(executor_type, {})
            timeout = (
                self._configuration.get_execution_timeout()
                if self._configuration
                else default_config.get("timeout", 300.0)
            )

            executor = executor_class(
                sandbox_manager=self._sandbox_manager, docker_config=default_config, timeout=timeout
            )

        # Cache the instance
        self._executor_instances[executor_type] = executor
        return executor

    def create_executor_for_command(self, command: str, action_type: Optional[str] = None) -> DockerExecutor:
        """
        Create appropriate executor based on command content or action type.

        Args:
            command: Command string to analyze
            action_type: Optional explicit action type

        Returns:
            Appropriate executor instance
        """
        # If action_type is explicitly provided, use it
        if action_type and action_type in self._executor_registry:
            return self.get_executor(action_type)

        # Analyze command to determine best executor
        executor_type = self._analyze_command(command)
        return self.get_executor(executor_type)

    def _analyze_command(self, command: str) -> str:
        """
        Analyze command string to determine appropriate executor type.

        Args:
            command: Command string to analyze

        Returns:
            Executor type name
        """
        command = command.strip().lower()

        # Python script patterns
        python_indicators = [
            "python",
            "python3",
            "pip",
            "pytest",
            "jupyter",
            ".py",
            "import ",
            "def ",
            "class ",
            "if __name__",
        ]

        if any(indicator in command for indicator in python_indicators):
            return "python"

        # Default to CLI executor for shell commands
        return "cli"

    def get_all_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get MCP tool definitions for all available executors.

        Returns:
            List of MCP tool definitions
        """
        tools = []

        for executor_type in self._executor_registry:
            try:
                executor = self.get_executor(executor_type)

                # Get the tool definition with executor type prefix
                tool_schema = executor.to_mcp_schema()
                metadata = getattr(executor, "_security_command_metadata", {})

                tool_def = {
                    "name": f"{executor_type}_{metadata.get('name', 'command')}",
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
        info = {
            "available_types": self.get_available_executors(),
            "active_instances": list(self._executor_instances.keys()),
            "registry_size": len(self._executor_registry),
            "configurations": self._default_configs.copy(),
        }

        return info
