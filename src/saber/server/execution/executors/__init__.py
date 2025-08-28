"""
Executor implementations for command execution.
"""

# Import standard registry to ensure executors are registered
from . import standard_registry  # noqa: F401
from .base_executors import CommandExecutor
from .docker_executor import DockerExecutor
from .executor_factory import ExecutorFactory

# Import executor registry for external access
from .executor_registry import (
    executor_registry,
    get_available_executors,
    get_executor_class,
    get_executor_info,
    register_executor,
)

# Re-export the standard executors for backward compatibility
from .standard_registry import CLIExecutor, PythonExecutor, SQLExecutor

__all__ = [
    "CommandExecutor",
    "DockerExecutor",
    "CLIExecutor",
    "PythonExecutor",
    "SQLExecutor",
    "ExecutorFactory",
    "executor_registry",
    "register_executor",
    "get_available_executors",
    "get_executor_class",
    "get_executor_info",
]
