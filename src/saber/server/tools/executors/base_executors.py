"""
Base executor implementations for common tool patterns.

This module provides the base ToolExecutor class for implementing custom tools.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from ..base import Parameter, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Base implementation for tool executors with common functionality.
    """

    def __init__(self, timeout: Optional[float] = None, *args: Any, **kwargs: Any) -> None:
        """
        Initialize tool executor.

        Args:
            timeout: Execution timeout in seconds
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self._timeout = timeout
        self._parameters: Dict[str, Parameter] = {}

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given parameters and context.

        Args:
            parameters: Tool-specific parameters
            context: Execution context (session_id, task_id, etc.)

        Returns:
            ToolResult containing execution results
        """
        pass

    def get_timeout(self) -> Optional[float]:
        """
        Get the execution timeout for this tool in seconds.

        Returns:
            Timeout in seconds, or None for no timeout
        """
        return self._timeout or 300.0  # Default 5 minutes

    def get_parameters(self) -> Dict[str, Parameter]:
        """Get the tool parameters."""
        return self._parameters.copy()

    def add_parameter(self, parameter: Parameter) -> None:
        """Add a parameter to the tool."""
        self._parameters[parameter.name] = parameter

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters against the tool's parameter definitions.

        Args:
            parameters: Parameters to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult.success()

        # Check for required parameters
        for param_name, param_def in self._parameters.items():
            if param_def.required and param_name not in parameters:
                result.add_error(f"Required parameter '{param_name}' is missing")
                continue

            if param_name in parameters:
                is_valid, error_msg = param_def.validate_value(parameters[param_name])
                if not is_valid and error_msg:
                    result.add_error(error_msg)

        # Check for unknown parameters
        for param_name in parameters:
            if param_name not in self._parameters:
                result.add_warning(f"Unknown parameter '{param_name}' will be ignored")

        return result

    def to_mcp_schema(self) -> Dict[str, Any]:
        """
        Convert tool parameters to MCP tool schema format.

        Returns:
            Dictionary containing MCP-compatible parameter schema
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param_def in self._parameters.items():
            # Build parameter schema
            param_schema: Dict[str, Any] = {
                "type": param_def.type.value,
                "description": param_def.description,
            }

            # Add default value if present
            if param_def.default is not None:
                param_schema["default"] = param_def.default

            # Add enum values if present
            if param_def.enum_values:
                param_schema["enum"] = param_def.enum_values

            # Add range constraints for numbers
            if param_def.type.value in ("integer", "number"):
                if param_def.min_value is not None:
                    param_schema["minimum"] = param_def.min_value
                if param_def.max_value is not None:
                    param_schema["maximum"] = param_def.max_value

            # Add pattern for strings
            if param_def.type.value == "string" and param_def.pattern:
                param_schema["pattern"] = param_def.pattern

            properties[param_name] = param_schema

            if param_def.required:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
