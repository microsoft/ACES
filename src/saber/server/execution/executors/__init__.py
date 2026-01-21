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
from .copilot_registry import COPILOT_EXECUTOR_TYPES, CreateExecutor, EditExecutor, GrepExecutor, ViewExecutor
from .docker_executor import DockerExecutor
from .executor_factory import ExecutorFactory

# Import executor registry for external access
from .executor_registry import (
    add_executor_to_group,
    executor_registry,
    expand_executor_list,
    get_available_executors,
    get_available_groups,
    get_executor_class,
    get_executor_info,
    get_group_executors,
    is_executor_group,
    register_executor,
    register_executor_group,
)

# Re-export the standard executors for backward compatibility
from .standard_registry import STANDARD_EXECUTOR_TYPES, BashExecutor, PythonExecutor

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
    # Registry functions
    "executor_registry",
    "register_executor",
    "register_executor_group",
    "add_executor_to_group",
    "get_available_executors",
    "get_available_groups",
    "get_executor_class",
    "get_executor_info",
    "get_group_executors",
    "expand_executor_list",
    "is_executor_group",
    # Executor type lists
    "COPILOT_EXECUTOR_TYPES",
    "STANDARD_EXECUTOR_TYPES",
]
