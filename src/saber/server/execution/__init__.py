"""
Command execution and validators for SABER.
"""

from ..base import CommandResult
from .base import Parameter, ParameterType, ValidationResult
from .exceptions import (
    DomainNotFoundError,
    DuplicateToolError,
    ExecutionManagerError,
    ParameterValidationError,
    ToolExecutionError,
    ToolNotFoundError,
)
from .execution_manager import ExecutionManager
from .executors.executor_registry import (
    get_executor_info,
    load_executors_from_directory,
    register_executor,
    register_executor_from_file,
)

__all__ = [
    "CommandResult",
    "ValidationResult",
    "Parameter",
    "ParameterType",
    "ExecutionManagerError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ParameterValidationError",
    "DuplicateToolError",
    "DomainNotFoundError",
    "ExecutionManager",
    # Executor registration hooks
    "register_executor",
    "register_executor_from_file",
    "load_executors_from_directory",
    "get_executor_info",
]
