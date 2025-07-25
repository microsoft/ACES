"""
Exception classes for the ExecutionManager system.

This module defines all custom exceptions used throughout the tool execution framework.
"""


class ExecutionManagerError(Exception):
    """Base exception for ExecutionManager errors."""

    pass


class ToolNotFoundError(ExecutionManagerError):
    """Raised when a requested tool is not found."""

    pass


class ToolExecutionError(ExecutionManagerError):
    """Raised when tool execution fails."""

    pass


class ParameterValidationError(ExecutionManagerError):
    """Raised when parameter validation fails."""

    pass


class DuplicateToolError(ExecutionManagerError):
    """Raised when attempting to register a tool with duplicate name."""

    pass


class DomainNotFoundError(ExecutionManagerError):
    """Raised when a requested domain is not found."""

    pass
