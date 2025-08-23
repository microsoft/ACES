"""
Base executor implementations for common command patterns.

This module provides the base CommandExecutor class for implementing custom command executors.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ...base import CommandResult
from ..base import Parameter, ValidationResult

logger = logging.getLogger(__name__)


class CommandExecutor(ABC):
    """
    Base implementation for command executors with common functionality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, *args: Any, **kwargs: Any) -> None:
        """
        Initialize command executor.

        Args:
            config: Executor configuration dictionary
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Merge provided config with defaults
        default_config = self.get_default_config()
        self._config = {**default_config, **(config or {})}
        self._parameters: Dict[str, Parameter] = {}

        # Allow subclasses to set up their specific parameters
        self.setup_parameters(self._config)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default configuration for this executor type.

        Subclasses should override this method to provide their specific default configurations.

        Returns:
            Dictionary containing default configuration values
        """
        return {
            "timeout": 300.0,  # Default 5 minutes
        }

    @abstractmethod
    def setup_parameters(self, config: Dict[str, Any]) -> None:
        """
        Set up executor-specific parameters.

        This method is called during initialization and allows each executor
        to define its specific parameters using add_parameter().

        Args:
            config: The merged configuration dictionary containing both default
                   and user-provided configuration values

        Subclasses must implement this method to define their parameters.
        """
        pass

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """
        Execute the command with given parameters and context.

        Args:
            parameters: Command-specific parameters
            context: Execution context (session_id, task_id, etc.)

        Returns:
            CommandResult containing execution results
        """
        pass

    async def __call__(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """
        Call the executor with given parameters and context.

        This provides a more intuitive interface: executor(parameters, context)
        instead of executor.execute(parameters, context).

        Args:
            parameters: Command-specific parameters
            context: Execution context (session_id, task_id, etc.)

        Returns:
            CommandResult containing execution results
        """
        return await self.execute(parameters, context)

    def get_timeout(self) -> float:
        """
        Get the execution timeout for this command in seconds.

        Returns:
            Timeout in seconds from configuration
        """
        timeout = self._config.get("timeout")
        if timeout is None:
            logger.warning(f"No timeout configured for {self.__class__.__name__}, using default 300.0 seconds")
            return 300.0
        return float(timeout)

    def get_parameters(self) -> Dict[str, Parameter]:
        """Get the command parameters."""
        return self._parameters.copy()

    def add_parameter(self, parameter: Parameter) -> None:
        """Add a parameter to the command."""
        self._parameters[parameter.name] = parameter

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters against the command's parameter definitions.

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
