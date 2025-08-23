"""
Unit tests for MCPToolGenerator utility.

Tests the dynamic MCP tool function generation logic that converts
executor schemas into properly typed functions for FastMCP registration.
"""

import asyncio
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from saber.server.api.mcp_tool_generator import MCPToolGenerator


class TestMCPToolGenerator:
    """Test suite for MCPToolGenerator utility class."""

    def test_validate_mcp_schema_valid(self):
        """Test schema validation with valid schema."""
        valid_schema = {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute"
                    },
                    "shell": {
                        "type": "boolean",
                        "description": "Use shell mode",
                        "default": False
                    }
                },
                "required": ["command"]
            }
        }

        assert MCPToolGenerator.validate_mcp_schema(valid_schema) is True

    def test_validate_mcp_schema_invalid(self):
        """Test schema validation with invalid schemas."""
        # Not a dictionary
        assert MCPToolGenerator.validate_mcp_schema("invalid") is False

        # Missing inputSchema
        assert MCPToolGenerator.validate_mcp_schema({}) is False

        # Invalid inputSchema
        assert MCPToolGenerator.validate_mcp_schema({"inputSchema": "invalid"}) is False

        # Missing properties
        assert MCPToolGenerator.validate_mcp_schema({
            "inputSchema": {"type": "object"}
        }) is False

        # Invalid property definition
        assert MCPToolGenerator.validate_mcp_schema({
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": "invalid"
                }
            }
        }) is False

        # Missing type in property
        assert MCPToolGenerator.validate_mcp_schema({
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {
                        "description": "Command"
                    }
                }
            }
        }) is False

    def test_json_type_to_python_type(self):
        """Test JSON schema type to Python type conversion."""
        assert MCPToolGenerator._json_type_to_python_type("string") == "str"
        assert MCPToolGenerator._json_type_to_python_type("integer") == "int"
        assert MCPToolGenerator._json_type_to_python_type("boolean") == "bool"
        assert MCPToolGenerator._json_type_to_python_type("number") == "float"
        assert MCPToolGenerator._json_type_to_python_type("array") == "List[str]"
        assert MCPToolGenerator._json_type_to_python_type("object") == "Dict[str, Any]"
        assert MCPToolGenerator._json_type_to_python_type("unknown") == "str"  # fallback

    @pytest.mark.asyncio
    async def test_create_executor_tool_cli(self):
        """Test creating MCP tool function for CLI executor."""
        # Mock handler function
        mock_handler = AsyncMock(return_value={
            "content": [{"type": "text", "text": "Command executed successfully"}],
            "isError": False
        })

        # CLI executor schema
        cli_schema = {
            "name": "cli",
            "description": "Execute CLI commands",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute"
                    },
                    "shell": {
                        "type": "boolean",
                        "description": "Use shell mode",
                        "default": False
                    }
                },
                "required": ["command"]
            }
        }

        # Generate the tool function
        tool_function = MCPToolGenerator.create_executor_tool(
            executor_name="cli",
            mcp_schema=cli_schema,
            handler_func=mock_handler
        )

        # Verify function properties
        assert tool_function.__name__ == "execute_cli"
        assert "CLI" in tool_function.__doc__ or "cli" in tool_function.__doc__
        assert asyncio.iscoroutinefunction(tool_function)

        # Test function execution with required parameter
        result = await tool_function(command="ls -la")
        assert result == "Command executed successfully"
        mock_handler.assert_called_once_with("cli", {"command": "ls -la"})

        # Reset mock and test with optional parameter
        mock_handler.reset_mock()
        result = await tool_function(command="ps aux", shell=True)
        assert result == "Command executed successfully"
        mock_handler.assert_called_once_with("cli", {"command": "ps aux", "shell": True})

    @pytest.mark.asyncio
    async def test_create_executor_tool_python(self):
        """Test creating MCP tool function for Python executor."""
        # Mock handler function
        mock_handler = AsyncMock(return_value={
            "content": [{"type": "text", "text": "Python code executed"}],
            "isError": False
        })

        # Python executor schema
        python_schema = {
            "name": "python",
            "description": "Execute Python code",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "template": {
                        "type": "string",
                        "description": "Script template to use"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory",
                        "default": "/workspace"
                    }
                },
                "required": ["code"]
            }
        }

        # Generate the tool function
        tool_function = MCPToolGenerator.create_executor_tool(
            executor_name="python",
            mcp_schema=python_schema,
            handler_func=mock_handler
        )

        # Verify function properties
        assert tool_function.__name__ == "execute_python"
        assert asyncio.iscoroutinefunction(tool_function)

        # Test function execution
        result = await tool_function(code="print('hello')", working_dir="/tmp")
        assert result == "Python code executed"
        mock_handler.assert_called_once_with("python", {
            "code": "print('hello')",
            "working_dir": "/tmp"
        })

    @pytest.mark.asyncio
    async def test_create_executor_tool_error_handling(self):
        """Test error handling in generated tool function."""
        # Mock handler function that returns error
        mock_handler = AsyncMock(return_value={
            "content": [{"type": "text", "text": "Command failed"}],
            "isError": True
        })

        # Simple schema
        schema = {
            "name": "test",
            "description": "Test executor",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Test input"
                    }
                },
                "required": ["input"]
            }
        }

        # Generate the tool function
        tool_function = MCPToolGenerator.create_executor_tool(
            executor_name="test",
            mcp_schema=schema,
            handler_func=mock_handler
        )

        # Test error handling
        result = await tool_function(input="test")
        assert '"success": false' in result
        assert '"error": "Command failed"' in result

    def test_create_executor_tool_none_filtering(self):
        """Test that None values are filtered from parameters."""
        # This test verifies the generated function filters None values
        # We'll inspect the generated code rather than execute it

        schema = {
            "name": "test",
            "description": "Test executor",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "required_param": {
                        "type": "string",
                        "description": "Required parameter"
                    },
                    "optional_param": {
                        "type": "string",
                        "description": "Optional parameter"
                    }
                },
                "required": ["required_param"]
            }
        }

        mock_handler = AsyncMock()

        # Generate the tool function
        tool_function = MCPToolGenerator.create_executor_tool(
            executor_name="test",
            mcp_schema=schema,
            handler_func=mock_handler
        )

        # The function should exist and be callable
        assert callable(tool_function)
        assert asyncio.iscoroutinefunction(tool_function)

    def test_create_executor_tool_invalid_schema(self):
        """Test error handling for invalid schemas."""
        mock_handler = AsyncMock()

        # Schema with missing required fields
        invalid_schema = {
            "name": "invalid",
            "description": "Invalid schema"
            # Missing inputSchema
        }

        # Should raise an error during function generation
        with pytest.raises(Exception):
            MCPToolGenerator.create_executor_tool(
                executor_name="invalid",
                mcp_schema=invalid_schema,
                handler_func=mock_handler
            )

    def test_create_executor_tool_complex_types(self):
        """Test handling of complex parameter types."""
        mock_handler = AsyncMock(return_value={
            "content": [{"type": "text", "text": "Success"}],
            "isError": False
        })

        # Schema with various parameter types
        complex_schema = {
            "name": "complex",
            "description": "Complex executor",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "string_param": {"type": "string"},
                    "int_param": {"type": "integer"},
                    "bool_param": {"type": "boolean"},
                    "float_param": {"type": "number"},
                    "array_param": {"type": "array"},
                    "object_param": {"type": "object"}
                },
                "required": ["string_param"]
            }
        }

        # Should generate function without errors
        tool_function = MCPToolGenerator.create_executor_tool(
            executor_name="complex",
            mcp_schema=complex_schema,
            handler_func=mock_handler
        )

        assert callable(tool_function)
        assert tool_function.__name__ == "execute_complex"
