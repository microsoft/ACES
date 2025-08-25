#!/usr/bin/env python3
"""
SABER MCP Client

Lightweight MCP client wrapper using FastMCP for secure tool execution
with the SABER server. Provides session-aware communication with proper
header handling for session mapping.
"""

from typing import Any, Dict, Optional

from fastmcp import Client
from fastmcp.client.transports import SSETransport
from saber.base import MCPHeaders


class SABERMCPClient:
    """
    Lightweight MCP client wrapper using FastMCP.

    Provides session-aware communication with SABER server's MCP API
    for secure tool discovery and execution within isolated sandboxes.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        client_id: str = "saber-client",
    ) -> None:
        """
        Initialize MCP client.

        Args:
            base_url: Base URL of SABER MCP server
            session_id: Session ID for server mapping
            task_id: Task ID for context
            client_id: Client identifier
        """
        self.base_url = base_url
        self.client: Optional[Client] = None
        self.session_id = session_id
        self.task_id = task_id
        self.client_id = client_id
        # Use session_id as client_id for mapping if available
        self.mcp_client_id = session_id if session_id else client_id

    async def __aenter__(self) -> "SABERMCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> bool:
        """Connect to MCP server using FastMCP.Client with custom headers."""
        try:
            # Create FastMCP client for HTTP endpoint with session headers
            sse_url = f"{self.base_url}/sse"

            # Create headers for session mapping
            headers = {}
            if self.session_id:
                headers[MCPHeaders.SESSION_ID] = self.session_id
            if self.task_id:
                headers[MCPHeaders.TASK_ID] = self.task_id
            headers[MCPHeaders.CLIENT_ID] = self.client_id

            transport = SSETransport(url=sse_url, headers=headers)
            self.client = Client(transport)

            # Connect using context manager
            await self.client.__aenter__()

            # Test the connection with ping
            await self.client.ping()

            print(f"✅ Connected to MCP server at {sse_url} with session_id: {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to MCP server: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        try:
            if self.client:
                await self.client.__aexit__(None, None, None)
                self.client = None
            print("✅ Disconnected from MCP server")
        except Exception as e:
            print(f"⚠️  Error disconnecting from MCP: {e}")

    # FastMCP pass-through methods - leverage FastMCP's built-in capabilities

    async def ping(self) -> Any:
        """Test connection to server."""
        if not self.client:
            raise Exception("Not connected to MCP server")
        return await self.client.ping()

    async def list_tools(self) -> Any:
        """List available tools - uses FastMCP's built-in implementation."""
        if not self.client:
            raise Exception("Not connected to MCP server")
        return await self.client.list_tools()

    async def list_resources(self) -> Any:
        """List available resources - uses FastMCP's built-in implementation."""
        if not self.client:
            raise Exception("Not connected to MCP server")
        return await self.client.list_resources()

    async def list_prompts(self) -> Any:
        """List available prompts - uses FastMCP's built-in implementation."""
        if not self.client:
            raise Exception("Not connected to MCP server")
        return await self.client.list_prompts()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool - uses FastMCP's built-in implementation."""
        if not self.client:
            raise Exception("Not connected to MCP server")
        return await self.client.call_tool(tool_name, arguments)
