"""
Base executor implementations for common command patterns.

This module provides the base CommandExecutor class for implementing custom command executors.

Logging category: ``LogCategory.TASK_EXEC``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic

if TYPE_CHECKING:
    from ...session_manager import SessionManager
    from ..sandbox.sandbox_environment_manager import SandboxEnvironmentManager

from ....logging_config import LogCategory, get_saber_logger
from ....models import MCPInputSchema, MCPPropertySchema
from ...base import CommandResult
from ..base import ExecutionContext, ExecutorParameters, P, Parameter, ValidationResult
from ..models import ExecutorConfig

logger = get_saber_logger(LogCategory.TASK_EXEC, __name__)


def normalize_context(context: ExecutionContext | dict[str, Any]) -> ExecutionContext:
    """
    Normalize context to ExecutionContext type.

    Provides backward compatibility for tests and code that still pass dict contexts.

    Args:
        context: Either an ExecutionContext instance or a dict

    Returns:
        ExecutionContext instance
    """
    if isinstance(context, ExecutionContext):
        return context
    return ExecutionContext.from_dict(context)


def normalize_parameters(
    params: P | dict[str, Any],
    params_class: type[P],
) -> P:
    """
    Normalize parameters to the expected parameter dataclass type.

    Provides backward compatibility for tests and code that still pass dict parameters.

    Args:
        params: Either a parameter dataclass instance or a dict
        params_class: The parameter dataclass class to convert to

    Returns:
        Parameter dataclass instance
    """
    if isinstance(params, params_class):
        return params
    if isinstance(params, dict):
        return params_class.from_dict(params)
    # Already correct type (could be a subclass)
    if isinstance(params, ExecutorParameters):
        return params
    raise TypeError(f"Expected {params_class.__name__} or dict, got {type(params).__name__}")


class CommandExecutor(ABC, Generic[P]):
    """
    Base implementation for command executors with common functionality.

    Type parameter P is the executor's specific parameter dataclass type.
    """

    def __init__(
        self,
        config: ExecutorConfig | None = None,
        session_manager: SessionManager | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize command executor.

        Args:
            config: Typed executor configuration
            session_manager: Optional session manager for cross-episode operations
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Use provided config or get default from subclass
        self._config = config if config is not None else self.get_default_config()
        self._parameters: dict[str, Parameter] = {}

        # Session manager for cross-episode operations
        self._session_manager: SessionManager | None = session_manager

        # Allow subclasses to set up their specific parameters
        self.setup_parameters(self._config)

    @classmethod
    def get_default_config(cls) -> ExecutorConfig:
        """
        Get default configuration for this executor type.

        Subclasses should override this method to provide their specific default configurations.

        Returns:
            ExecutorConfig with default values
        """
        return ExecutorConfig(timeout=300.0)

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: ExecutorConfig | None = None,
        additional_params: dict[str, Any] | None = None,
        session_manager: SessionManager | None = None,
        **kwargs: Any,
    ) -> CommandExecutor[P]:
        """
        Generic factory method for creating executor instances with standardized configuration.

        This method provides a consistent interface for all executors, allowing
        the factory to create instances without knowing specific constructor signatures.

        Subclasses can override this method for custom initialization logic.

        Args:
            sandbox_manager: Sandbox manager for executor operations
            config: Executor-specific configuration dictionary
            additional_params: Additional parameters specific to this executor type
            session_manager: Optional session manager for cross-episode operations
            **kwargs: Additional keyword arguments

        Returns:
            Configured executor instance
        """
        # Default implementation - subclasses can override for custom initialization
        merged_kwargs = {**kwargs}
        if additional_params:
            merged_kwargs.update(additional_params)

        # Pass all relevant parameters to the constructor
        return cls(
            config=config,
            session_manager=session_manager,
            sandbox_manager=sandbox_manager,
            **merged_kwargs,
        )

    @abstractmethod
    def setup_parameters(self, config: ExecutorConfig) -> None:
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

    @classmethod
    @abstractmethod
    def get_parameters_class(cls) -> type[ExecutorParameters]:
        """
        Get the parameter dataclass type for this executor.

        This is used by ExecutionManager to convert dict → typed params
        before calling execute().

        Returns:
            The parameter dataclass class (e.g., BashParameters)
        """
        pass

    @abstractmethod
    async def execute(self, parameters: P, context: ExecutionContext) -> CommandResult:
        """
        Execute the command with given parameters and context.

        Args:
            parameters: Strongly-typed parameter dataclass instance
            context: Strongly-typed execution context

        Returns:
            CommandResult containing execution results
        """
        pass

    async def __call__(self, parameters: P, context: ExecutionContext) -> CommandResult:
        """
        Call the executor with given parameters and context.

        This provides a more intuitive interface: executor(parameters, context)
        instead of executor.execute(parameters, context).

        Args:
            parameters: Strongly-typed parameter dataclass instance
            context: Strongly-typed execution context

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
        return self._config.timeout

    def get_parameters(self) -> dict[str, Parameter]:
        """Get the command parameters."""
        return self._parameters.copy()

    def add_parameter(self, parameter: Parameter) -> None:
        """Add a parameter to the command."""
        self._parameters[parameter.name] = parameter

    def validate_parameters(self, parameters: P) -> ValidationResult:
        """
        Validate typed parameters against the command's parameter definitions.

        Args:
            parameters: Strongly-typed parameter dataclass instance

        Returns:
            ValidationResult
        """
        result = ValidationResult.success()

        # Check for required parameters and type validation
        for param_name, param_def in self._parameters.items():
            param_value = getattr(parameters, param_name, None)

            if param_def.required and param_value is None:
                result.add_error(f"Required parameter '{param_name}' is missing")
                continue

            if param_value is not None:
                is_valid, error_msg = param_def.validate_value(param_value)
                if not is_valid and error_msg:
                    result.add_error(error_msg)

        return result

    def to_mcp_schema(self) -> MCPInputSchema:
        """
        Convert tool parameters to MCP tool schema format.

        Returns:
            MCPInputSchema containing typed parameter schema
        """
        properties: dict[str, MCPPropertySchema] = {}
        required: list[str] = []

        for param_name, param_def in self._parameters.items():
            # Build parameter schema with conditional field setting to avoid None values
            property_schema_kwargs: dict[str, Any] = {
                "type": param_def.type.value,
                "description": param_def.description,
            }

            # Only set optional fields if they have actual values
            if param_def.default is not None:
                property_schema_kwargs["default"] = param_def.default
            if param_def.enum_values is not None:
                property_schema_kwargs["enum"] = param_def.enum_values

            property_schema = MCPPropertySchema(**property_schema_kwargs)

            # Add range constraints for numbers
            if param_def.type.value in ("integer", "number"):
                if param_def.min_value is not None:
                    property_schema.minimum = param_def.min_value
                if param_def.max_value is not None:
                    property_schema.maximum = param_def.max_value

            # Add pattern for strings
            if param_def.type.value == "string" and param_def.pattern:
                property_schema.pattern = param_def.pattern

            # Add items for arrays
            if param_def.type.value == "array" and param_def.items:
                property_schema.items = param_def.items

            properties[param_name] = property_schema

            if param_def.required:
                required.append(param_name)

        return MCPInputSchema(
            type="object",
            properties=properties,
            required=required,
        )
