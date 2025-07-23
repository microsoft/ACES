"""
Base classes and data models for the ToolRegistry system.

This module defines the core abstractions for security tools, executors,
and data models used throughout the ToolRegistry system.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast


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


class ToolExecutor(ABC):
    """Abstract base class for tool executors."""

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

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters before execution.

        Args:
            parameters: Parameters to validate

        Returns:
            ValidationResult indicating if parameters are valid
        """
        pass

    def get_timeout(self) -> Optional[float]:
        """
        Get the execution timeout for this tool in seconds.

        Returns:
            Timeout in seconds, or None for no timeout
        """
        return 300.0  # Default 5 minutes


@dataclass
class SecurityTool:
    """Represents a security tool with its metadata and executor."""

    name: str
    domain: str
    description: str
    version: str
    author: str
    parameters: Dict[str, Parameter]
    executor: Any  # Will be validated as ToolExecutor in validate()
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        """
        Validate the tool configuration.

        Returns:
            ValidationResult indicating if tool is valid
        """
        result = ValidationResult.success()

        # Validate basic fields
        if not self.name:
            result.add_error("Tool name cannot be empty")
        if not self.domain:
            result.add_error("Tool domain cannot be empty")
        if not self.description:
            result.add_error("Tool description cannot be empty")
        if not self.version:
            result.add_error("Tool version cannot be empty")
        if not self.author:
            result.add_error("Tool author cannot be empty")

        # Validate domain is one of the supported domains
        supported_domains = {"malware", "threat_investigation", "forensics"}
        if self.domain not in supported_domains:
            result.add_error(f"Domain '{self.domain}' not in supported domains: {supported_domains}")

        # Validate parameters
        for param_name, param in self.parameters.items():
            if param_name != param.name:
                result.add_error(f"Parameter name mismatch: key '{param_name}' != param.name '{param.name}'")

        # Validate executor
        if not isinstance(self.executor, ToolExecutor):
            result.add_error("Executor must be an instance of ToolExecutor")

        return result

    def to_mcp_tool(self) -> Dict[str, Any]:
        """
        Convert to MCP tool format.

        Returns:
            Dictionary representing the tool in MCP format
        """
        # Convert parameters to MCP format
        mcp_parameters: Dict[str, Any] = {}
        required_params: List[str] = []

        for param in self.parameters.values():
            param_schema: Dict[str, Any] = {"type": param.type.value, "description": param.description}

            if param.enum_values:
                param_schema["enum"] = param.enum_values
            if param.min_value is not None:
                param_schema["minimum"] = param.min_value
            if param.max_value is not None:
                param_schema["maximum"] = param.max_value
            if param.pattern:
                param_schema["pattern"] = param.pattern
            if param.default is not None:
                param_schema["default"] = param.default

            mcp_parameters[param.name] = param_schema

            if param.required:
                required_params.append(param.name)

        return {
            "name": self.name,
            "description": f"{self.description} (Domain: {self.domain}, Version: {self.version})",
            "inputSchema": {"type": "object", "properties": mcp_parameters, "required": required_params},
        }

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with parameter validation.

        Args:
            parameters: Tool parameters
            context: Execution context

        Returns:
            ToolResult with execution results
        """
        if not self.enabled:
            return ToolResult.error_result("Tool is disabled")

        # Validate parameters
        validation_result = cast(ToolExecutor, self.executor).validate_parameters(parameters)
        if not validation_result.valid:
            return ToolResult.error_result(f"Parameter validation failed: {', '.join(validation_result.errors)}")

        # Execute with timing
        start_time = time.time()
        try:
            result = await cast(ToolExecutor, self.executor).execute(parameters, context)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult.error_result(f"Tool execution failed: {str(e)}", execution_time)


# Exception classes for ToolRegistry
class ToolRegistryError(Exception):
    """Base exception for ToolRegistry errors."""

    pass


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested tool is not found."""

    pass


class ToolExecutionError(ToolRegistryError):
    """Raised when tool execution fails."""

    pass


class ParameterValidationError(ToolRegistryError):
    """Raised when parameter validation fails."""

    pass


class DuplicateToolError(ToolRegistryError):
    """Raised when attempting to register a tool with duplicate name."""

    pass


class DomainNotFoundError(ToolRegistryError):
    """Raised when a requested domain is not found."""

    pass


# Decorator for marking security tools
def security_tool(
    domain: str,
    name: str,
    description: str,
    version: str = "1.0.0",
    author: str = "Unknown",
    tags: Optional[List[str]] = None,
) -> Callable[[Type], Type]:
    """
    Decorator to mark a ToolExecutor class as a security tool.

    Args:
        domain: Security domain (malware, threat_investigation, forensics)
        name: Tool name
        description: Tool description
        version: Tool version
        author: Tool author
        tags: Optional tags for categorization
    """

    def decorator(cls: Type) -> Type:
        # Store metadata on the class
        cls._security_tool_metadata = {
            "domain": domain,
            "name": name,
            "description": description,
            "version": version,
            "author": author,
            "tags": tags or [],
        }
        return cls

    return decorator
