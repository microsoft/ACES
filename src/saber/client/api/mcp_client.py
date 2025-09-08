"""
Enhanced MCP Client for SABER with FastMCP and strict typing.

This replaces the previous implementation with:
1. FastMCP integration with SSE transport
2. Strict typing with MCPToolCallRequest/Response
3. Session context via HTTP headers
4. Proper error handling and connection management
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastmcp import Client
from fastmcp.client.transports import SSETransport

from ...models import HTTPHeaders
from ...models.mcp import (
    MCPConnectionError,
    MCPError,
    MCPExecutionError,
    MCPTimeoutError,
    MCPTool,
    MCPToolCallRequest,
    MCPToolCallResponse,
    MCPToolNotFoundError,
    SessionContext,
)
from ..models import MCPConfig

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Enhanced MCP client using FastMCP with strict typing and session context.

    Features:
    - FastMCP integration with SSE transport
    - Session context via HTTP headers
    - Typed requests and responses
    - Proper connection lifecycle management
    - Comprehensive error handling
    """

    def __init__(self, config: MCPConfig):
        """
        Initialize MCP client.

        Args:
            config: MCP client configuration
        """
        self.config = config
        self.client: Optional[Client] = None
        self._session_context: Optional[SessionContext] = None
        self._connected = False

        logger.debug(f"Initialized MCP client for: {config.base_url}")

    async def connect(self, session_context: SessionContext) -> None:
        """
        Connect to SABER MCP server with session context.

        Args:
            session_context: Session context for header injection

        Raises:
            MCPConnectionError: Connection failed
        """
        if self._connected:
            await self.disconnect()

        try:
            self._session_context = session_context

            # Build headers for connection including episode context
            headers = {
                HTTPHeaders.SESSION_ID: session_context.session_id,
                HTTPHeaders.CLIENT_ID: session_context.client_id,
            }

            # Add episode context headers (required for episode-scoped MCP clients)
            if session_context.episode_id:
                headers[HTTPHeaders.EPISODE_ID] = session_context.episode_id
            else:
                # Fail fast - episode ID is required for episode-scoped clients
                raise ValueError("Episode ID is required for MCP client connection")

            if session_context.task_id:
                headers[HTTPHeaders.TASK_ID] = session_context.task_id
            # Create FastMCP client
            self.client = self._create_client(headers)

            # Connect using context manager
            await self.client.__aenter__()

            # Note: Removed immediate ping() - FastMCP handles initialization sequence internally
            # The server was rejecting immediate ping before initialization was complete

            self._connected = True
            logger.info("MCP client connected successfully")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise MCPConnectionError(f"Connection failed: {e}") from e

    def _create_client(self, headers: Dict[str, str]) -> Client:
        """Create FastMCP client with SSE transport.

        Args:
            headers: HTTP headers for the transport

        Returns:
            Configured FastMCP client
        """
        transport = SSETransport(url=self.config.get_sse_url(), headers=headers)
        return Client(transport)

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        if not self._connected:
            return

        try:
            if self.client:
                await self.client.__aexit__(None, None, None)
                self.client = None

            self._connected = False
            self._session_context = None
            logger.info("MCP client disconnected")

        except Exception as e:
            logger.warning(f"Error during MCP disconnect: {e}")

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self.client is not None

    @property
    def session_context(self) -> Optional[SessionContext]:
        """Get current session context."""
        return self._session_context

    def _build_request_headers(self, episode_id: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, str]:
        """
        Build headers for individual MCP requests with episode/task context.

        Args:
            episode_id: Optional episode ID override
            task_id: Optional task ID override

        Returns:
            Headers dict for the request
        """
        if not self._session_context:
            return {}

        headers = {
            HTTPHeaders.SESSION_ID: self._session_context.session_id,
            HTTPHeaders.CLIENT_ID: self._session_context.client_id,
        }

        # Episode ID is required for MCP operations
        if episode_id:
            headers[HTTPHeaders.EPISODE_ID] = episode_id
        else:
            raise ValueError("Episode ID is required for MCP requests")

        # Task ID is optional
        if task_id:
            headers[HTTPHeaders.TASK_ID] = task_id

        return headers

    async def discover_tools(self, episode_id: str, task_id: Optional[str] = None) -> List[MCPTool]:
        """
        Discover available tools with episode context.

        Args:
            episode_id: Episode ID for request context (required)
            task_id: Optional task ID for request context

        Returns:
            List of available tools

        Raises:
            MCPConnectionError: Not connected to server
            MCPError: Discovery failed
            ValueError: Episode ID not provided
        """
        if not self.is_connected():
            raise MCPConnectionError("MCP client not connected")

        try:
            logger.debug("Discovering MCP tools with episode context")

            # Use FastMCP's built-in list_tools
            if self.client is None:
                raise RuntimeError("MCP client not connected")

            # Note: FastMCP client doesn't support custom headers for list_tools
            tools_result = await self.client.list_tools()

            # Convert to typed tools
            tools = []
            if hasattr(tools_result, "tools"):
                tools_data = tools_result.tools
            elif hasattr(tools_result, "__iter__"):
                # tools_result is already a list of tools
                tools_data = tools_result
            else:
                tools_data = []

            for tool_data in tools_data:
                if hasattr(tool_data, "model_dump"):
                    tool = MCPTool.model_validate(tool_data.model_dump())
                else:
                    tool = MCPTool.model_validate(tool_data)
                tools.append(tool)

            logger.info(f"Discovered {len(tools)} MCP tools")
            return tools

        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            raise MCPError(f"Tool discovery failed: {e}") from e

    async def execute_tool(self, request: MCPToolCallRequest) -> MCPToolCallResponse:
        """
        Execute a tool with typed request and response.

        Args:
            request: Typed tool execution request (includes episode_id and task_id)

        Returns:
            Typed tool execution response

        Raises:
            MCPConnectionError: Not connected to server
            MCPToolNotFoundError: Tool not found
            MCPExecutionError: Tool execution failed
            MCPTimeoutError: Request timed out
            ValueError: Episode ID not provided in request
        """
        if not self.is_connected():
            raise MCPConnectionError("MCP client not connected")

        try:
            logger.debug(f"Executing MCP tool: {request.tool_name}")

            # Determine timeout (request override or config default)
            timeout = request.timeout or self.config.timeout

            # Use FastMCP's built-in call_tool
            if self.client is None:
                raise RuntimeError("MCP client not connected")

            # FastMCP call_tool signature: call_tool(name: str, arguments: dict | None = None)
            result = await asyncio.wait_for(
                self.client.call_tool(request.tool_name, request.arguments), timeout=timeout
            )

            # Convert CallToolResult to MCPToolCallResponse
            if hasattr(result, "content") and hasattr(result, "is_error"):
                # Handle FastMCP CallToolResult
                content_dicts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        content_dicts.append({"type": "text", "text": item.text})
                    else:
                        content_dicts.append({"type": "text", "text": str(item)})

                response = MCPToolCallResponse(content=content_dicts, isError=result.is_error)
            else:
                # Fallback: try to validate as-is (for backwards compatibility)
                response = MCPToolCallResponse.model_validate(result)

            logger.info(f"Tool execution completed: {request.tool_name}")
            return response

        except asyncio.TimeoutError as e:
            logger.error(f"Tool execution timeout: {request.tool_name} after {timeout}s")
            raise MCPTimeoutError(f"Tool execution timeout: {e}") from e
        except Exception as e:
            # Check if it's a "tool not found" error
            if "not found" in str(e).lower() or "unknown tool" in str(e).lower():
                logger.error(f"Tool not found: {request.tool_name}")
                raise MCPToolNotFoundError(f"Tool not found: {request.tool_name}") from e
            else:
                logger.error(f"Tool execution failed: {request.tool_name} - {e}")
                raise MCPExecutionError(f"Tool execution failed: {e}") from e

    async def health_check(self) -> bool:
        """
        Check MCP server health.

        Returns:
            True if server is healthy, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            if self.client is None:
                return False
            await self.client.ping()
            return True
        except Exception as e:
            logger.warning(f"MCP health check failed: {e}")
            return False

    # Context manager support
    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry - requires separate connect() call."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
