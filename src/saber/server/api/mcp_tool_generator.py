"""
MCP Tool Generator Utility.

Logging Category: MCP_API

This module provides utilities for dynamically generating MCP-compatible tool functions
from executor schemas. It handles the conversion from **kwargs-based functions to
properly typed functions that the FastMCP library can register.
"""

from typing import Any, Callable, Dict, List, Optional

from saber.logging_config import LogCategory, get_saber_logger

from ...models.mcp import MCPToolSchema

logger = get_saber_logger(LogCategory.MCP_API, __name__)


class MCPToolGenerator:
    """
    Utility class for generating MCP-compatible tool functions dynamically.

    This class takes executor schemas and creates properly typed async functions
    that can be registered with FastMCP, avoiding the **kwargs limitation.
    """

    def __init__(self) -> None:
        """Initialize the MCP tool generator."""
        self.logger = logger

    def create_executor_tool(self, executor_name: str, mcp_schema: MCPToolSchema, handler_func: Callable) -> Callable:
        """
        Create a dynamically typed MCP tool function for an executor.

        Args:
            executor_name: Name of the executor (e.g., 'cli', 'python', 'xss_testing')
            mcp_schema: Typed MCPToolSchema object from executor registration
            handler_func: Function to call with the parameters (e.g., handle_call_tool)

        Returns:
            Async function with proper typing that can be registered with FastMCP
        """
        # Extract parameter information from the typed schema
        input_schema = mcp_schema.inputSchema
        properties = input_schema.properties
        required = input_schema.required

        if not self.validate_mcp_tool_schema(mcp_schema):
            raise ValueError(f"Invalid MCP schema for executor '{executor_name}'")

        # Build function signature components - no need for Context parameter
        params = []
        param_names = []
        for param_name, param_def in properties.items():
            param_names.append(param_name)
            # Get Python type from JSON schema type - param_def is now MCPPropertySchema
            param_type = self._json_type_to_python_type(param_def.type)

            # Check if parameter is required or optional
            if param_name in required:
                # Required parameter - no default value
                params.append(f"{param_name}: {param_type}")
            else:
                # Optional parameter - use Optional type and default to None
                if param_def.default is not None:
                    # Use the actual default value from schema
                    default_repr = repr(param_def.default)
                    params.append(f"{param_name}: Optional[{param_type}] = {default_repr}")
                else:
                    # Default to None for optional parameters without explicit default
                    params.append(f"{param_name}: Optional[{param_type}] = None")

        # Context parameters are now included in the schema properties, so no need to add them separately
        all_params = params

        # Generate the function code
        function_signature = f"async def {executor_name}({', '.join(all_params)}) -> str:"

        # Build parameter dictionary construction
        param_dict_items = []
        for param_name in param_names:
            param_dict_items.append(f"'{param_name}': {param_name}")

        param_dict_construction = "{" + ", ".join(param_dict_items) + "}"

        # Build property defaults mapping for runtime access
        property_defaults = {}
        for prop_name, prop_def in properties.items():
            property_defaults[prop_name] = getattr(prop_def, "default", None)

        # Complete function body
        function_body = f'''
{function_signature}
    """Execute a {executor_name} command in the SABER sandbox environment."""
    # Build parameters dict, excluding None values and default values that weren't explicitly set
    parameters = {{}}
    provided_params = {{k: v for k, v in {param_dict_construction}.items() if v is not None}}

    # Only include parameters that were explicitly provided or are required
    for param_name, param_value in provided_params.items():
        # Always include required parameters
        if param_name in {required}:
            parameters[param_name] = param_value
        else:
            # For optional parameters, only include if they differ from default or no default exists
            default_val = property_defaults.get(param_name)
            if default_val is None or param_value != default_val:
                parameters[param_name] = param_value

    result = await handler_func('{executor_name}', parameters)

    # Extract the text content from the CallToolResult object (from FastMCP)
    if hasattr(result, 'is_error') and result.is_error:
        import json
        # Handle CallToolResult with error
        if hasattr(result, 'content') and result.content and len(result.content) > 0:
            error_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
        else:
            error_text = "Unknown error"
        return json.dumps({{"success": False, "error": error_text}})
    else:
        # Handle successful CallToolResult
        if hasattr(result, 'content') and result.content and len(result.content) > 0:
            result_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
        else:
            result_text = ""
        return str(result_text)
'''

        # Create the function dynamically
        namespace = {
            "handler_func": handler_func,
            "property_defaults": property_defaults,
            "Optional": Optional,
            "str": str,
            "int": int,
            "bool": bool,
            "float": float,
            "List": List,
            "Dict": Dict,
            "Any": Any,
        }

        try:
            exec(function_body, namespace)
            dynamic_function = namespace[executor_name]

            # Type check - ensure we got a callable
            if not callable(dynamic_function):
                raise RuntimeError(f"Generated object for '{executor_name}' is not callable")

            # Set proper metadata
            if hasattr(dynamic_function, "__name__"):
                dynamic_function.__name__ = executor_name
            if hasattr(dynamic_function, "__doc__"):
                dynamic_function.__doc__ = f"Execute a {executor_name} command in the SABER sandbox environment."

            logger.debug(
                "MCP tool function generated",
                extra={
                    "event": "mcp_tool_function_generated",
                    "executor_name": executor_name,
                    "function_signature": function_signature,
                    "parameter_count": len(param_names),
                },
            )
            # Cast to Callable to satisfy mypy - dynamic_function is callable due to check above
            return dynamic_function

        except Exception as e:
            logger.error(
                "Failed to generate MCP tool function",
                extra={
                    "event": "mcp_tool_function_generation_failed",
                    "executor_name": executor_name,
                    "error": str(e),
                },
            )
            logger.debug(
                "Generated MCP function body",
                extra={
                    "event": "mcp_tool_function_body_debug",
                    "executor_name": executor_name,
                    "function_body": function_body,
                },
            )
            raise RuntimeError(f"Dynamic function generation failed for executor '{executor_name}': {e}")

    def _json_type_to_python_type(self, json_type: str) -> str:
        """
        Convert JSON schema type to Python type annotation string.

        Args:
            json_type: JSON schema type (e.g., 'string', 'integer', 'boolean')

        Returns:
            Python type annotation string
        """
        type_mapping = {
            "string": "str",
            "integer": "int",
            "boolean": "bool",
            "number": "float",
            "array": "List[str]",  # Assume string arrays for simplicity
            "object": "Dict[str, Any]",
        }

        return type_mapping.get(json_type, "str")

    def _convert_schema_type_to_python(self, param_def: Dict[str, Any]) -> str:
        """
        Convert JSON schema parameter definition to Python type annotation string.

        Args:
            param_def: JSON schema parameter definition

        Returns:
            Python type annotation string
        """
        json_type = param_def.get("type", "string")
        return self._json_type_to_python_type(json_type)

    def _get_default_value(self, param_def: Dict[str, Any]) -> str:
        """
        Get the default value for a parameter as a string suitable for function signature.

        Args:
            param_def: JSON schema parameter definition

        Returns:
            Default value as string for function signature
        """
        default_value = param_def.get("default")
        if default_value is not None:
            if isinstance(default_value, str):
                return f"'{default_value}'"
            else:
                return str(default_value)
        else:
            param_type = param_def.get("type", "string")
            if param_type == "string":
                return "None"
            elif param_type in ["integer", "number"]:
                return "None"
            elif param_type == "boolean":
                return "None"
            else:
                return "None"

    def validate_mcp_tool_schema(self, mcp_schema: MCPToolSchema) -> bool:
        """
        Validate that a typed MCPToolSchema is suitable for dynamic function generation.

        Args:
            mcp_schema: Typed MCPToolSchema object

        Returns:
            True if schema is valid for function generation
        """
        # Basic validation - the Pydantic model already ensures structure
        if not mcp_schema.name or not mcp_schema.description:
            return False

        # Validate input schema
        input_schema = mcp_schema.inputSchema
        if not input_schema.properties:
            return False

        # Check that all properties have valid types
        for prop_name, prop_def in input_schema.properties.items():
            if not prop_def.type:
                return False
            # Ensure type is supported for function generation
            if prop_def.type not in ["string", "integer", "number", "boolean", "array", "object"]:
                return False

        return True

    def validate_mcp_schema(self, mcp_schema: Dict[str, Any]) -> bool:
        """
        Validate that an MCP schema is suitable for dynamic function generation.

        Args:
            mcp_schema: MCP schema dictionary

        Returns:
            True if schema is valid for function generation
        """
        # inputSchema is required
        if "inputSchema" not in mcp_schema:
            return False

        input_schema = mcp_schema.get("inputSchema", {})
        if not isinstance(input_schema, dict):
            return False

        # properties is required
        if "properties" not in input_schema:
            return False

        properties = input_schema.get("properties", {})
        if not isinstance(properties, dict):
            return False

        # Check that all properties have valid types
        for prop_name, prop_def in properties.items():
            if not isinstance(prop_def, dict):
                return False
            if "type" not in prop_def:
                return False

        return True
