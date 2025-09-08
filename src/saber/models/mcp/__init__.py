"""
SABER MCP Models - Model Context Protocol specific models.

These models define the MCP API contract for tool discovery and execution.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MCPPropertySchema(BaseModel):
    """Schema definition for a single property in MCP input schema."""

    type: str = Field(description="JSON Schema type (string, integer, number, boolean, array, object)")
    description: Optional[str] = Field(None, description="Property description")
    title: Optional[str] = Field(None, description="Property title (from JSON Schema)")
    default: Optional[Any] = Field(None, description="Default value for the property")
    enum: Optional[List[Any]] = Field(None, description="Allowed values for enum properties")
    minimum: Optional[Union[int, float]] = Field(None, description="Minimum value for numeric properties")
    maximum: Optional[Union[int, float]] = Field(None, description="Maximum value for numeric properties")
    pattern: Optional[str] = Field(None, description="Regex pattern for string properties")


class MCPInputSchema(BaseModel):
    """JSON Schema definition for MCP tool input parameters."""

    type: str = Field(default="object", description="Schema type (always 'object' for tool parameters)")
    properties: Dict[str, MCPPropertySchema] = Field(description="Parameter property definitions")
    required: List[str] = Field(default_factory=list, description="List of required parameter names")


class MCPToolSchema(BaseModel):
    """Schema definition for an MCP tool."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    inputSchema: MCPInputSchema = Field(description="Typed schema for tool input parameters")


class MCPToolListResponse(BaseModel):
    """Response model for MCP tool listing."""

    tools: List[MCPToolSchema] = Field(description="Available tools")
    session_id: Optional[str] = Field(None, description="Session context")
    episode_id: Optional[str] = Field(None, description="Episode context")


class MCPToolCallResponse(BaseModel):
    """Response model for MCP tool execution."""

    content: List[Dict[str, Any]] = Field(description="Tool execution result content")
    isError: bool = Field(default=False, description="Whether the call resulted in an error")


class MCPErrorResponse(BaseModel):
    """Response model for MCP errors."""

    error: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class MCPToolCallRequest(BaseModel):
    """Request model for MCP tool execution with strict typing."""

    tool_name: str = Field(description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Tool-specific arguments (validated against tool schema)"
    )
    episode_id: str = Field(description="Episode ID for request context")
    task_id: Optional[str] = Field(None, description="Optional task ID for request context")
    timeout: Optional[float] = Field(None, description="Optional timeout override for this specific call")
    context: Optional[Dict[str, str]] = Field(None, description="Additional execution context metadata")


class SessionContext(BaseModel):
    """Session context for MCP requests."""

    session_id: str = Field(description="SABER session ID")
    episode_id: Optional[str] = Field(None, description="SABER episode ID (added when episode starts)")
    task_id: Optional[str] = Field(None, description="SABER task ID")
    client_id: str = Field(description="Client identifier")


# MCP-specific exceptions
class MCPError(Exception):
    """Base exception for MCP operations."""

    pass


class MCPConnectionError(MCPError):
    """MCP connection related errors."""

    pass


class MCPToolNotFoundError(MCPError):
    """Tool not found in MCP server."""

    pass


class MCPExecutionError(MCPError):
    """Tool execution failed."""

    pass


class MCPTimeoutError(MCPError):
    """Request timeout during MCP operation."""

    pass


# Type aliases for convenience
MCPTool = MCPToolSchema  # Alias for consistency
