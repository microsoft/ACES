"""
Executor implementations for command execution.
"""

# Import standard registry to ensure executors are registered
# Import copilot registry for GitHub Copilot CLI-compatible executors
from . import (
    copilot_registry,  # noqa: F401
    standard_registry,  # noqa: F401
)
from .base_executors import CommandExecutor

# Re-export the copilot executors
from .copilot_registry import CreateExecutor, EditExecutor, GrepExecutor, ViewExecutor
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
from .standard_registry import BashExecutor, PythonExecutor

# TODO: Fix SQLExecutor _max_rows attribute issue
# from .standard_registry import SQLExecutor

__all__ = [
    "CommandExecutor",
    "DockerExecutor",
    "BashExecutor",
    "PythonExecutor",
    # "SQLExecutor",  # temporarily removed
    # Copilot-compatible executors
    "ViewExecutor",
    "CreateExecutor",
    "EditExecutor",
    "GrepExecutor",
    "ExecutorFactory",
    "executor_registry",
    "register_executor",
    "get_available_executors",
    "get_executor_class",
    "get_executor_info",
]
