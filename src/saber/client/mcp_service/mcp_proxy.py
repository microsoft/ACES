#!/usr/bin/env python3
"""
MCP Proxy for SABER Client

Proxies MCP requests between agent containers and SABER MCP server using FastMCP.
This replaces the old HTTP-based approach with proper FastMCP SSE connections.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from fastmcp import Client
from fastmcp.client.transports import SSETransport

from ...base import MCPHeaders
from .agent_registry import AgentSession, AgentSessionRegistry

logger = logging.getLogger(__name__)


# Response models for internal use
class MCPResponse:
    """Response wrapper for MCP proxy operations."""

    def __init__(self, result: Any = None, error: Optional[Dict[str, Any]] = None):
        self.result = result
        self.error = error


class MCPProxy:
    """
    Proxies MCP requests between agent containers and SABER MCP server.

    This class handles:
    - Forwarding standard MCP protocol requests (list_tools, call_tool, etc.)
    - Injecting session headers for proper SABER server routing
    - Response forwarding and error handling
    - Connection management to SABER MCP server using FastMCP
    - UI adapter integration for real-time tool call monitoring
    """

    def __init__(
        self,
        session_registry: AgentSessionRegistry,
        saber_mcp_url: str = "http://localhost:8001",
        timeout: float = 30.0,
    ):
        """
        Initialize MCP proxy.

        Args:
            session_registry: Registry for agent session mapping
            saber_mcp_url: URL of SABER MCP server
            timeout: Request timeout in seconds
        """
        self.session_registry = session_registry
        self.saber_mcp_url = saber_mcp_url
        self.timeout = timeout
        self._client_pool: Dict[str, Client] = {}  # session_id -> Client cache

        logger.info(f"🔧 MCP Proxy initialized: {saber_mcp_url}")

    async def aclose(self) -> None:
        """Close all FastMCP client connections."""
        for client in self._client_pool.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing FastMCP client: {e}")
        self._client_pool.clear()

    async def _get_client_for_session(self, session: AgentSession) -> Client:
        """
        Get or create FastMCP client for a session.

        Args:
            session: Agent session with routing information

        Returns:
            FastMCP client configured for the session
        """
        # Include episode_id in session key for episode-first architecture
        episode_key = session.saber_episode_id or "no-episode"
        session_key = f"{session.saber_session_id}_{episode_key}_{session.task_id or 'default'}"

        if session_key not in self._client_pool:
            # Create new FastMCP client for this session
            sse_url = f"{self.saber_mcp_url}/sse"

            # Create headers for session mapping with episode-first architecture
            headers = {
                MCPHeaders.SESSION_ID: session.saber_session_id,
                MCPHeaders.CLIENT_ID: session.agent_id,
            }

            # EPISODE-FIRST: Include episode_id header (REQUIRED)
            if session.saber_episode_id:
                headers[MCPHeaders.EPISODE_ID] = session.saber_episode_id
            else:
                # FAIL FAST: Episode ID is required for episode-first architecture
                logger.warning(
                    f"⚠️ No episode_id for session {session.saber_session_id} - this may cause server-side failures"
                )

            if session.task_id:
                headers[MCPHeaders.TASK_ID] = session.task_id

            transport = SSETransport(url=sse_url, headers=headers)
            client = Client(transport)

            # Connect using context manager
            await client.__aenter__()

            # Test the connection
            try:
                await client.ping()
                logger.debug(
                    f"✅ FastMCP client connected for session {session.saber_session_id}, "
                    f"episode {session.saber_episode_id}"
                )
            except Exception as e:
                logger.warning(
                    f"FastMCP ping failed for session {session.saber_session_id}, "
                    f"episode {session.saber_episode_id}: {e}"
                )
                # Continue anyway, the connection might still work for tools

            self._client_pool[session_key] = client

        return self._client_pool[session_key]

    async def list_tools(self, agent_id: str) -> MCPResponse:
        """
        Proxy list_tools request to SABER server using FastMCP.

        Args:
            agent_id: Agent container ID

        Returns:
            MCPResponse with available tools or error
        """
        try:
            # Get session for routing
            session = await self.session_registry.get_session(agent_id)
            if not session:
                logger.error(f"❌ List tools failed: Agent {agent_id} not registered in session registry")
                return MCPResponse(result=None, error={"code": -32000, "message": f"Agent {agent_id} not registered"})

            # Get FastMCP client for this session
            client = await self._get_client_for_session(session)

            # Call list_tools using FastMCP
            logger.debug(f"🔗 Calling SABER server list_tools for session {session.saber_session_id}")
            tools = await client.list_tools()

            tool_count = len(tools.tools) if hasattr(tools, "tools") else (len(tools) if isinstance(tools, list) else 0)
            logger.info(f"🔗 SABER server response: {tool_count} tools available for {agent_id}")

            # Convert FastMCP response to our MCPResponse format
            if hasattr(tools, "tools"):
                tools_list = [tool.model_dump() for tool in tools.tools]
            else:
                tools_list = tools if isinstance(tools, list) else []

            return MCPResponse(result=tools_list)

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling list_tools for agent {agent_id}")
            return MCPResponse(result=None, error={"code": -32000, "message": "Request timeout"})
        except Exception as e:
            logger.error(f"Error calling list_tools for agent {agent_id}: {e}")
            return MCPResponse(result=None, error={"code": -32000, "message": f"Internal error: {str(e)}"})

    async def call_tool(self, agent_id: str, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """
        Proxy call_tool request to SABER server using FastMCP.

        Args:
            agent_id: Agent container ID
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            MCPResponse with tool result or error
        """
        # Generate unique call ID for tracking
        call_id = str(uuid.uuid4())

        logger.info("🚨🚨🚨 EXECUTE_TOOL ENDPOINT HIT!!! 🚨🚨🚨")
        logger.info(f"Tool: {tool_name}, Agent: {agent_id}, Call ID: {call_id}")

        try:
            # Get session for routing
            session = await self.session_registry.get_session(agent_id)
            if not session:
                logger.error(f"❌ Tool call failed: Agent {agent_id} not registered in session registry")
                return MCPResponse(result=None, error={"code": -32000, "message": f"Agent {agent_id} not registered"})

            # Get FastMCP client for this session
            client = await self._get_client_for_session(session)

            # Call tool using FastMCP
            logger.info(f"🔗 Calling SABER server: {tool_name} for session {session.saber_session_id}")
            result = await client.call_tool(tool_name, arguments)

            result_size = len(str(result)) if result else 0
            logger.info(f"🔗 SABER server response: {tool_name} completed for {agent_id} ({result_size} chars)")

            # Convert FastMCP response to our MCPResponse format
            if hasattr(result, "content"):
                # Handle FastMCP ToolResult format
                content_list = []
                for content in result.content:
                    if hasattr(content, "text"):
                        content_list.append({"type": "text", "text": content.text})
                    elif hasattr(content, "model_dump"):
                        content_list.append(content.model_dump())
                    else:
                        content_list.append({"type": "text", "text": str(content)})

                response_data = {"content": content_list, "isError": getattr(result, "isError", False)}
            else:
                # Handle raw response
                response_data = result if isinstance(result, dict) else {"result": result}

            return MCPResponse(result=response_data)

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling tool {tool_name} for agent {agent_id}")
            return MCPResponse(result=None, error={"code": -32000, "message": "Request timeout"})
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} for agent {agent_id}: {e}")
            return MCPResponse(result=None, error={"code": -32000, "message": f"Internal error: {str(e)}"})

    async def list_resources(self, agent_id: str) -> MCPResponse:
        """
        Proxy list_resources request to SABER server using FastMCP.

        Args:
            agent_id: Agent container ID

        Returns:
            MCPResponse with available resources or error
        """
        try:
            # Get session for routing
            session = await self.session_registry.get_session(agent_id)
            if not session:
                return MCPResponse(result=None, error={"code": -32000, "message": f"Agent {agent_id} not registered"})

            # Get FastMCP client for this session
            client = await self._get_client_for_session(session)

            # Call list_resources using FastMCP
            resources = await client.list_resources()

            logger.debug(f"Listed resources for agent {agent_id}")

            # Convert FastMCP response to our MCPResponse format
            if hasattr(resources, "resources"):
                resources_list = [resource.model_dump() for resource in resources.resources]
            else:
                resources_list = resources if isinstance(resources, list) else []

            return MCPResponse(result=resources_list)

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling list_resources for agent {agent_id}")
            return MCPResponse(result=None, error={"code": -32000, "message": "Request timeout"})
        except Exception as e:
            logger.error(f"Error calling list_resources for agent {agent_id}: {e}")
            return MCPResponse(result=None, error={"code": -32000, "message": f"Internal error: {str(e)}"})

    async def ping(self, agent_id: str) -> MCPResponse:
        """
        Proxy ping request to SABER server using FastMCP.

        Args:
            agent_id: Agent container ID

        Returns:
            MCPResponse with ping result or error
        """
        try:
            # Get session for routing
            session = await self.session_registry.get_session(agent_id)
            if not session:
                return MCPResponse(result=None, error={"code": -32000, "message": f"Agent {agent_id} not registered"})

            # Get FastMCP client for this session
            client = await self._get_client_for_session(session)

            # Call ping using FastMCP
            result = await client.ping()

            logger.debug(f"Pinged SABER server for agent {agent_id}")

            return MCPResponse(result={"pong": True, "raw_result": result})

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling ping for agent {agent_id}")
            return MCPResponse(result=None, error={"code": -32000, "message": "Request timeout"})
        except Exception as e:
            logger.error(f"Error calling ping for agent {agent_id}: {e}")
            return MCPResponse(result=None, error={"code": -32000, "message": f"Internal error: {str(e)}"})

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of MCP proxy and SABER server connectivity.

        Returns:
            Dict with health status information
        """
        try:
            # Basic health info
            health_info = {
                "mcp_proxy_healthy": True,
                "saber_mcp_url": self.saber_mcp_url,
                "active_connections": len(self._client_pool),
                "timeout": self.timeout,
            }

            # Try to get session count
            try:
                session_stats = await self.session_registry.get_session_stats()
                total_sessions = session_stats.get("total_sessions", 0)
                health_info.update(
                    {
                        "total_sessions": total_sessions,
                        "session_registry_healthy": True,
                    }
                )
            except Exception as e:
                health_info.update(
                    {
                        "session_registry_healthy": False,
                        "session_registry_error": str(e),
                    }
                )

            # Check if we can reach SABER server (if we have any active sessions)
            saber_server_healthy = True
            saber_server_error = None

            if self._client_pool:
                # Try ping with one of the existing clients
                try:
                    client = next(iter(self._client_pool.values()))
                    await client.ping()
                except Exception as e:
                    saber_server_healthy = False
                    saber_server_error = str(e)

            health_info.update(
                {
                    "saber_server_healthy": saber_server_healthy,
                    "saber_server_error": saber_server_error,
                }
            )

            return health_info

        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {
                "mcp_proxy_healthy": False,
                "error": str(e),
                "saber_mcp_url": self.saber_mcp_url,
            }
