"""
Base executor implementations for common command patterns.

This module provides the base CommandExecutor class for implementing custom command executors.

Logging category: ``LogCategory.TASK_EXEC``.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..sandbox.sandbox_environment_manager import SandboxEnvironmentManager

from ....logging_config import LogCategory, get_saber_logger
from ....models import MCPInputSchema, MCPPropertySchema
from ...base import CommandResult
from ..base import Parameter, ValidationResult

logger = get_saber_logger(LogCategory.TASK_EXEC, __name__)


class CommandExecutor(ABC):
    """
    Base implementation for command executors with common functionality.
    """

    def __init__(
        self, config: dict[str, Any] | None = None, session_manager: Any | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize command executor.

        Args:
            config: Executor configuration dictionary
            session_manager: Optional session manager for cross-episode operations
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Merge provided config with defaults
        default_config = self.get_default_config()
        self._config = {**default_config, **(config or {})}
        self._parameters: dict[str, Parameter] = {}

        # Session manager for cross-episode operations
        self._session_manager: Any | None = session_manager

        # Allow subclasses to set up their specific parameters
        self.setup_parameters(self._config)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Get default configuration for this executor type.

        Subclasses should override this method to provide their specific default configurations.

        Returns:
            Dictionary containing default configuration values
        """
        return {
            "timeout": 300.0,  # Default 5 minutes
        }

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: "SandboxEnvironmentManager",
        config: dict[str, Any] | None = None,
        additional_params: dict[str, Any] | None = None,
        session_manager: Any | None = None,
        **kwargs: Any,
    ) -> "CommandExecutor":
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
    def setup_parameters(self, config: dict[str, Any]) -> None:
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
    async def execute(self, parameters: dict[str, Any], context: dict[str, Any]) -> CommandResult:
        """
        Execute the command with given parameters and context.

        Args:
            parameters: Command-specific parameters
            context: Execution context (episode_id, task_id, etc.)

        Returns:
            CommandResult containing execution results
        """
        pass

    async def __call__(self, parameters: dict[str, Any], context: dict[str, Any]) -> CommandResult:
        """
        Call the executor with given parameters and context.

        This provides a more intuitive interface: executor(parameters, context)
        instead of executor.execute(parameters, context).

        Args:
            parameters: Command-specific parameters
            context: Execution context (episode_id, task_id, etc.)

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
            logger.warning(
                "Executor timeout missing; using default",
                extra={
                    "event": "executor_timeout_default",
                    "executor_type": self.__class__.__name__,
                    "default_timeout": 300.0,
                },
            )
            return 300.0
        return float(timeout)

    def get_parameters(self) -> dict[str, Parameter]:
        """Get the command parameters."""
        return self._parameters.copy()

    def add_parameter(self, parameter: Parameter) -> None:
        """Add a parameter to the command."""
        self._parameters[parameter.name] = parameter

    def validate_parameters(self, parameters: dict[str, Any]) -> ValidationResult:
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
