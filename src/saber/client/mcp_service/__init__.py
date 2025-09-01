"""
SABER MCP Service

Shared MCP service that provides standard MCP protocol access to containerized agents
while routing requests to the appropriate SABER server sessions.

This service acts as a proxy between standard MCP clients (like anthropic/mcp-python)
and the SABER server, enabling zero-code-change agent compatibility.
"""

from .agent_registry import AgentSessionRegistry
from .main import create_app
from .mcp_proxy import MCPProxy

__all__ = ["create_app", "AgentSessionRegistry", "MCPProxy"]
