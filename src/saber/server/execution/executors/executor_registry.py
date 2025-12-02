"""Executor Registry for SABER Framework.

Logging Category: EXECUTION

This module provides a unified registration system for all executors,
both standard framework executors and user-defined custom executors.
All executors must register themselves through this registry.
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict, Type

from saber.logging_config import LogCategory, get_saber_logger

from .base_executors import CommandExecutor

logger = get_saber_logger(LogCategory.EXECUTION, __name__)


class ExecutorRegistry:
    """
    Registry for managing all executor registration in the SABER framework.

    Provides a unified system for registering both standard framework executors
    and user-defined custom executors. All executors must register through this system.
    """

    def __init__(self) -> None:
        """Initialize the executor registry."""
        self._registered_executors: Dict[str, Type[CommandExecutor]] = {}
        self._registration_sources: Dict[str, str] = {}  # Track where each executor came from
        logger.info(
            "Executor registry initialized",
            extra={"event": "executor_registry_initialized"},
        )

    def register_executor_class(
        self, executor_type: str, executor_class: Type[CommandExecutor], source: str = "external"
    ) -> None:
        """
        Register an executor class.

        Args:
            executor_type: String identifier for the executor
            executor_class: Executor class (must inherit from CommandExecutor)
            source: Source description for tracking/debugging

        Raises:
            ValueError: If executor_class doesn't inherit from CommandExecutor
            RuntimeError: If executor_type is already registered
        """
        if not issubclass(executor_class, CommandExecutor):
            raise ValueError(f"Executor class must inherit from CommandExecutor, got: {executor_class}")

        if executor_type in self._registered_executors:
            existing_source = self._registration_sources.get(executor_type, "unknown")
            raise RuntimeError(f"Executor type '{executor_type}' is already registered from source: {existing_source}")

        # Validate the executor class has required methods
        self._validate_executor_class(executor_class)

        # Register in our registry
        self._registered_executors[executor_type] = executor_class
        self._registration_sources[executor_type] = source

        logger.info(
            "Executor registered",
            extra={
                "event": "executor_registered",
                "executor_type": executor_type,
                "source": source,
                "executor_class": executor_class.__name__,
            },
        )

    def register_executor_from_file(self, executor_type: str, file_path: str, class_name: str) -> None:
        """
        Register a custom executor from an external Python file.

        Args:
            executor_type: String identifier for the executor
            file_path: Path to the Python file containing the executor
            class_name: Name of the executor class in the file

        Raises:
            FileNotFoundError: If file doesn't exist
            ImportError: If file cannot be imported or class not found
            ValueError: If class doesn't inherit from DockerExecutor
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Executor file not found: {file_path}")

        if not file_path_obj.suffix == ".py":
            raise ValueError(f"Executor file must be a .py file, got: {file_path}")

        try:
            # Load the module dynamically
            module_name = f"custom_executor_{executor_type}_{file_path_obj.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for: {file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the executor class
            if not hasattr(module, class_name):
                raise ImportError(f"Class '{class_name}' not found in file: {file_path}")

            executor_class = getattr(module, class_name)

            # Register the executor
            self.register_executor_class(executor_type, executor_class, f"file:{file_path}")

        except Exception as e:
            logger.error(
                "Executor registration from file failed",
                extra={
                    "event": "executor_registration_from_file_failed",
                    "executor_type": executor_type,
                    "file_path": file_path,
                    "class_name": class_name,
                    "error": str(e),
                },
            )
            raise

    def load_executors_from_directory(self, directory_path: str) -> Dict[str, str]:
        """
        Load all executor definition files from a directory.

        Looks for Python files that define custom executors and loads them.
        Does NOT automatically register - files should use register_executor().

        Args:
            directory_path: Path to directory containing executor definition files

        Returns:
            Dictionary mapping loaded files to any errors encountered
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        results = {}
        executor_files = list(directory.glob("*executor*.py"))  # More flexible pattern

        logger.info(
            "Executor definition files discovery started",
            extra={
                "event": "executor_definition_discovery_started",
                "directory_path": directory_path,
                "candidate_count": len(executor_files),
            },
        )

        for file_path in executor_files:
            try:
                # Load the module - it should register itself using the hook functions
                module_name = f"custom_executor_module_{file_path.stem}"
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    results[str(file_path)] = "Could not create module spec"
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                results[str(file_path)] = "loaded_successfully"
                logger.info(
                    "Executor definition file loaded",
                    extra={
                        "event": "executor_definition_loaded",
                        "file_path": str(file_path),
                    },
                )

            except Exception as e:
                error_msg = f"Failed to load: {e}"
                results[str(file_path)] = error_msg
                logger.warning(
                    "Executor definition file load failed",
                    extra={
                        "event": "executor_definition_load_failed",
                        "file_path": str(file_path),
                        "error": str(e),
                    },
                )

        logger.info(
            "Executor definition files discovery completed",
            extra={
                "event": "executor_definition_discovery_completed",
                "directory_path": directory_path,
                "successful_count": len([r for r in results.values() if r == "loaded_successfully"]),
                "total_files": len(executor_files),
            },
        )
        return results

    def unregister_executor(self, executor_type: str) -> None:
        """
        Unregister an executor.

        Args:
            executor_type: String identifier for the executor to remove
        """
        if executor_type in self._registered_executors:
            del self._registered_executors[executor_type]
            if executor_type in self._registration_sources:
                del self._registration_sources[executor_type]
            logger.info(
                "Executor unregistered",
                extra={
                    "event": "executor_unregistered",
                    "executor_type": executor_type,
                },
            )
        else:
            logger.warning(
                "Executor unregister skipped",
                extra={
                    "event": "executor_unregistered_unknown",
                    "executor_type": executor_type,
                },
            )

    def list_registered_executors(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered executors with their metadata.

        Returns:
            Dictionary mapping executor_type to metadata
        """
        result = {}
        for executor_type, executor_class in self._registered_executors.items():
            result[executor_type] = {
                "class_name": executor_class.__name__,
                "module": executor_class.__module__,
                "source": self._registration_sources.get(executor_type, "unknown"),
                "default_config": executor_class.get_default_config(),
                "docstring": executor_class.__doc__ or "No documentation available",
            }
        return result

    def get_executor_class(self, executor_type: str) -> Type[CommandExecutor]:
        """
        Get an executor class by type.

        Args:
            executor_type: String identifier for the executor

        Returns:
            Executor class

        Raises:
            KeyError: If executor_type is not registered
        """
        if executor_type not in self._registered_executors:
            available = list(self._registered_executors.keys())
            raise KeyError(f"Executor type '{executor_type}' not registered. Available: {available}")
        return self._registered_executors[executor_type]

    def get_available_executors(self) -> list[str]:
        """
        Get list of all registered executor types.

        Returns:
            List of executor type strings
        """
        return list(self._registered_executors.keys())

    def clear_all_executors(self) -> None:
        """
        Unregister all executors.

        Warning: This will clear ALL executors, including standard ones.
        Use with caution.
        """
        executor_types = list(self._registered_executors.keys())
        for executor_type in executor_types:
            self.unregister_executor(executor_type)
        logger.info(
            "All executors cleared",
            extra={
                "event": "executors_cleared",
                "cleared_count": len(executor_types),
            },
        )

    def _validate_executor_class(self, executor_class: Type[CommandExecutor]) -> None:
        """
        Validate that an executor class has the required interface.

        Args:
            executor_class: Executor class to validate

        Raises:
            ValueError: If class doesn't have required methods
        """
        required_methods = ["execute", "to_mcp_schema", "setup_parameters"]
        missing_methods = []

        for method_name in required_methods:
            if not hasattr(executor_class, method_name):
                missing_methods.append(method_name)

        if missing_methods:
            raise ValueError(f"Executor class {executor_class.__name__} is missing required methods: {missing_methods}")


# Global registry instance
executor_registry = ExecutorRegistry()


def register_executor(executor_type: str, executor_class: Type[CommandExecutor], source: str = "external") -> None:
    """
    Hook function for registering executors.

    This is the main entry point for all executor registration, both standard framework
    executors and user-defined custom executors.

    Args:
        executor_type: String identifier for the executor (e.g., "cli", "python", "java")
        executor_class: Executor class that inherits from CommandExecutor
        source: Source description for tracking (e.g., "standard", "external", "file:path")

    Example:
        from saber.server.execution.executors.executor_registry import register_executor
        from my_executor import JavaExecutor

        register_executor("java", JavaExecutor, "external")
    """
    executor_registry.register_executor_class(executor_type, executor_class, source)


def register_executor_from_file(executor_type: str, file_path: str, class_name: str) -> None:
    """
    Hook function for registering executors from external Python files.

    Args:
        executor_type: String identifier for the executor
        file_path: Path to the Python file containing the executor
        class_name: Name of the executor class in the file

    Example:
        register_executor_from_file("java", "/path/to/java_executor.py", "JavaExecutor")
    """
    executor_registry.register_executor_from_file(executor_type, file_path, class_name)


def load_executors_from_directory(directory_path: str) -> Dict[str, str]:
    """
    Hook function for loading all executor definitions from a directory.

    This loads all *executor*.py files from the specified directory.
    The files themselves should use register_executor() to register their executors.

    Args:
        directory_path: Path to directory containing executor files

    Returns:
        Dictionary mapping file paths to load results

    Example:
        # Load all executors from the pentest demo server directory
        load_executors_from_directory("/path/to/pentest_demo/server/executors")
    """
    return executor_registry.load_executors_from_directory(directory_path)


def get_executor_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all registered executors.

    Returns:
        Dictionary mapping executor_type to metadata
    """
    return executor_registry.list_registered_executors()


def get_executor_class(executor_type: str) -> Type[CommandExecutor]:
    """
    Get an executor class by type.

    Args:
        executor_type: String identifier for the executor

    Returns:
        Executor class
    """
    return executor_registry.get_executor_class(executor_type)


def get_available_executors() -> list[str]:
    """
    Get list of all registered executor types.

    Returns:
        List of executor type strings
    """
    return executor_registry.get_available_executors()
