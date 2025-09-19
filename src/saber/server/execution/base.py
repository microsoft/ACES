"""
Base classes and data models for the ExecutionManager system.

This module provides core abstractions, parameter types, result types,
and data models used throughout the ExecutionManager command execution system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Union


class ParameterType(str, Enum):
    """Supported parameter types for command parameters."""

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NUMBER = "number"


@dataclass
class Parameter:
    """Represents a command parameter with validation rules."""

    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None  # For string validation
    items: Optional[dict] = None  # For array type validation

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

        # Array items validation
        if self.type == ParameterType.ARRAY and self.items and isinstance(value, list):
            item_type = self.items.get("type")
            if item_type == "string":
                for i, item in enumerate(value):
                    if not isinstance(item, str):
                        return False, f"Parameter '{self.name}' array item {i} must be a string"
            elif item_type == "integer":
                for i, item in enumerate(value):
                    if not isinstance(item, int):
                        return False, f"Parameter '{self.name}' array item {i} must be an integer"
            elif item_type == "number":
                for i, item in enumerate(value):
                    if not isinstance(item, (int, float)):
                        return False, f"Parameter '{self.name}' array item {i} must be a number"

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
