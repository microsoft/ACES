"""
Custom Executor Registry for SABER Framework.

This module provides functionality for users to register custom executors
from external Python modules, enabling extensibility without modifying core framework code.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, Type

from .executors.docker_executor import DockerExecutor
from .executors.factory import ExecutorFactory

logger = logging.getLogger(__name__)


class CustomExecutorRegistry:
    """
    Registry for managing custom executor registration from external Python modules.

    Enables users to define custom executors in external Python files and register them
    with the SABER execution framework through simple hook functions.
    """

    def __init__(self) -> None:
        """Initialize the custom executor registry."""
        self._custom_executors: Dict[str, Type[DockerExecutor]] = {}
        self._registration_sources: Dict[str, str] = {}  # Track where each executor came from
        logger.info("CustomExecutorRegistry initialized")

    def register_executor_class(
        self, executor_type: str, executor_class: Type[DockerExecutor], source: str = "external"
    ) -> None:
        """
        Register a custom executor class directly.

        Args:
            executor_type: String identifier for the executor
            executor_class: Executor class (must inherit from DockerExecutor)
            source: Source description for tracking/debugging

        Raises:
            ValueError: If executor_class doesn't inherit from DockerExecutor
            RuntimeError: If executor_type is already registered
        """
        if not issubclass(executor_class, DockerExecutor):
            raise ValueError(f"Executor class must inherit from DockerExecutor, got: {executor_class}")

        if executor_type in self._custom_executors:
            existing_source = self._registration_sources.get(executor_type, "unknown")
            raise RuntimeError(f"Executor type '{executor_type}' is already registered from source: {existing_source}")

        # Validate the executor class has required methods
        self._validate_executor_class(executor_class)

        # Register with both our registry and the factory
        self._custom_executors[executor_type] = executor_class
        self._registration_sources[executor_type] = source
        ExecutorFactory.register_executor(executor_type, executor_class)

        logger.info(f"Registered custom executor '{executor_type}' from source: {source}")

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
            logger.error(f"Failed to register executor from file {file_path}: {e}")
            raise

    def load_executors_from_directory(self, directory_path: str) -> Dict[str, str]:
        """
        Load all executor definition files from a directory.

        Looks for Python files that define custom executors and loads them.
        Does NOT automatically register - files should use register_custom_executor().

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

        logger.info(f"Loading {len(executor_files)} executor definition files from {directory_path}")

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
                logger.info(f"Loaded executor definitions from {file_path}")

            except Exception as e:
                error_msg = f"Failed to load: {e}"
                results[str(file_path)] = error_msg
                logger.warning(f"Failed to load executor file {file_path}: {e}")

        logger.info(
            f"Loaded {len([r for r in results.values() if r == 'loaded_successfully'])} executor definition files"
        )
        return results

    def unregister_executor(self, executor_type: str) -> None:
        """
        Unregister a custom executor.

        Args:
            executor_type: String identifier for the executor to remove
        """
        if executor_type in self._custom_executors:
            del self._custom_executors[executor_type]
            if executor_type in self._registration_sources:
                del self._registration_sources[executor_type]
            ExecutorFactory.unregister_executor(executor_type)
            logger.info(f"Unregistered custom executor: {executor_type}")
        else:
            logger.warning(f"Attempted to unregister unknown executor: {executor_type}")

    def list_custom_executors(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered custom executors with their metadata.

        Returns:
            Dictionary mapping executor_type to metadata
        """
        result = {}
        for executor_type, executor_class in self._custom_executors.items():
            result[executor_type] = {
                "class_name": executor_class.__name__,
                "module": executor_class.__module__,
                "source": self._registration_sources.get(executor_type, "unknown"),
                "default_config": executor_class.get_default_config(),
                "docstring": executor_class.__doc__ or "No documentation available",
            }
        return result

    def clear_all_custom_executors(self) -> None:
        """
        Unregister all custom executors.

        Note: This only affects executors registered through this registry,
        not the built-in executors (cli, python).
        """
        custom_types = list(self._custom_executors.keys())
        for executor_type in custom_types:
            self.unregister_executor(executor_type)
        logger.info(f"Cleared {len(custom_types)} custom executors")

    def _validate_executor_class(self, executor_class: Type[DockerExecutor]) -> None:
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
custom_executor_registry = CustomExecutorRegistry()


def register_custom_executor(executor_type: str, executor_class: Type[DockerExecutor]) -> None:
    """
    Hook function for registering custom executors from external modules.

    This is the main entry point for external code to register custom executors.
    Custom executor files should call this function to register their executor classes.

    Args:
        executor_type: String identifier for the executor (e.g., "java", "nodejs")
        executor_class: Executor class that inherits from DockerExecutor

    Example:
        from saber.server.execution.custom_executor_registry import register_custom_executor
        from my_executor import JavaExecutor

        register_custom_executor("java", JavaExecutor)
    """
    custom_executor_registry.register_executor_class(executor_type, executor_class)


def register_executor_from_file(executor_type: str, file_path: str, class_name: str) -> None:
    """
    Hook function for registering custom executors from external Python files.

    Args:
        executor_type: String identifier for the executor
        file_path: Path to the Python file containing the executor
        class_name: Name of the executor class in the file

    Example:
        register_executor_from_file("java", "/path/to/java_executor.py", "JavaExecutor")
    """
    custom_executor_registry.register_executor_from_file(executor_type, file_path, class_name)


def load_custom_executors_from_directory(directory_path: str) -> Dict[str, str]:
    """
    Hook function for loading all custom executor definitions from a directory.

    This loads all *_executor.py files from the specified directory.
    The files themselves should use register_custom_executor() to register their executors.

    Args:
        directory_path: Path to directory containing custom executor files

    Returns:
        Dictionary mapping file paths to load results

    Example:
        # Load all custom executors from the pentest demo server directory
        load_custom_executors_from_directory("/path/to/pentest_demo/server/executors")
    """
    return custom_executor_registry.load_executors_from_directory(directory_path)


def get_custom_executor_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all registered custom executors.

    Returns:
        Dictionary mapping executor_type to metadata
    """
    return custom_executor_registry.list_custom_executors()
