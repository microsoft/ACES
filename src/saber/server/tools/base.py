"""
Base classes and data models for the ExecutionManager system.

This module provides core abstractions, parameter types, result types,
and data models used throughout the ExecutionManager system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union


class ParameterType(str, Enum):
    """Supported parameter types for tool parameters."""

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NUMBER = "number"


@dataclass
class Parameter:
    """Represents a tool parameter with validation rules."""

    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None  # For string validation

    def validate_value(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a parameter value against this parameter's constraints.

        Returns:
            tuple: (is_valid, error_message)
        """
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' is required"
            return True, None

        # Type validation
        if self.type == ParameterType.STRING and not isinstance(value, str):
            return False, f"Parameter '{self.name}' must be a string"
        elif self.type == ParameterType.INTEGER and not isinstance(value, int):
            return False, f"Parameter '{self.name}' must be an integer"
        elif self.type == ParameterType.BOOLEAN and not isinstance(value, bool):
            return False, f"Parameter '{self.name}' must be a boolean"
        elif self.type == ParameterType.ARRAY and not isinstance(value, list):
            return False, f"Parameter '{self.name}' must be an array"
        elif self.type == ParameterType.OBJECT and not isinstance(value, dict):
            return False, f"Parameter '{self.name}' must be an object"
        elif self.type == ParameterType.NUMBER and not isinstance(value, (int, float)):
            return False, f"Parameter '{self.name}' must be a number"

        # Enum validation
        if self.enum_values and value not in self.enum_values:
            return False, f"Parameter '{self.name}' must be one of {self.enum_values}"

        # Range validation for numbers
        if self.type in (ParameterType.INTEGER, ParameterType.NUMBER):
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be <= {self.max_value}"

        # Pattern validation for strings
        if self.type == ParameterType.STRING and self.pattern:
            import re

            if not re.match(self.pattern, value):
                return False, f"Parameter '{self.name}' does not match required pattern"

        return True, None


@dataclass
class ToolResult:
    """Result of tool execution."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls, data: Any, execution_time: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> "ToolResult":
        """Create a successful tool result."""
        return cls(success=True, data=data, execution_time=execution_time, metadata=metadata or {})

    @classmethod
    def error_result(
        cls, error: str, execution_time: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> "ToolResult":
        """Create an error tool result."""
        return cls(success=False, error=error, execution_time=execution_time, metadata=metadata or {})


@dataclass
class ValidationResult:
    """Result of validation operations."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def success(cls, warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(valid=True, warnings=warnings or [])

    @classmethod
    def failure(cls, errors: List[str], warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(valid=False, errors=errors, warnings=warnings or [])

    def add_error(self, error: str) -> None:
        """Add an error to the validation result."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(warning)


# Decorator for marking security tools
def security_tool(
    domain: str,
    name: str,
    description: str,
    author: str = "Unknown",
) -> Callable[[Type], Type]:
    """
    Decorator to mark a ToolExecutor class as a security tool.

    Args:
        domain: Security domain (malware, threat_investigation, forensics)
        name: Tool name
        description: Tool description
        author: Tool author
    """

    def decorator(cls: Type) -> Type:
        # Store metadata on the class
        cls._security_tool_metadata = {
            "domain": domain,
            "name": name,
            "description": description,
            "author": author,
        }
        return cls

    return decorator
