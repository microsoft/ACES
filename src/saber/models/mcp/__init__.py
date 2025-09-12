"""
SABER MCP Models - Model Context Protocol specific models.

These models define the MCP API contract for tool discovery and execution.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class OrchestrationEnvironment(str, Enum):
    """
    Enumeration of supported orchestration environments for SABER MCP connections.

    This enum identifies which system is orchestrating the agent's interaction
    with the SABER MCP server, enabling environment-specific behavior and logging.
    """

    INSPECT = "inspect"
    """Inspect AI framework orchestration - agents running under inspect_ai"""

    STANDALONE = "standalone"
    """Standalone client orchestration - direct MCP client connections"""

    def __str__(self) -> str:
        """Return the enum value as string for logging."""
        return self.value

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string value is a valid orchestration environment."""
        try:
            cls(value)
            return True
        except ValueError:
            return False

    @classmethod
    def get_valid_values(cls) -> list[str]:
        """Get list of all valid orchestration environment values."""
        return [env.value for env in cls]


@dataclass
class RequestHeaders:
    """
    Parsed HTTP headers for MCP requests.

    This dataclass standardizes header extraction and validation,
    providing a single object that can be passed around instead of
    multiple return values.
    """

    session_id: Optional[str]
    episode_id: Optional[str]
    orchestration_env: OrchestrationEnvironment
    task_id: Optional[str] = None
    client_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the headers after initialization."""
        # orchestration_env is mandatory, so if it's None, we should raise an error
        # This should not happen since we validate during parsing, but good to be explicit
        if self.orchestration_env is None:
            raise ValueError("orchestration_env is mandatory but was None")

    @property
    def has_session_context(self) -> bool:
        """Check if session context is available."""
        return self.session_id is not None

    @property
    def has_episode_context(self) -> bool:
        """Check if episode context is available."""
        return self.episode_id is not None

    @property
    def context_summary(self) -> str:
        """Get a summary string of the available context."""
        parts = []
        if self.session_id:
            parts.append(f"session:{self.session_id}")
        if self.episode_id:
            parts.append(f"episode:{self.episode_id}")
        if self.task_id:
            parts.append(f"task:{self.task_id}")
        parts.append(f"orchestration:{self.orchestration_env}")
        return f"[{', '.join(parts)}]"


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
