"""
MCP Proxy for SABER Service

Proxies standard MCP protocol requests from agent containers to the SABER MCP server,
handling session routing and response forwarding.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp
from saber.base import MCPHeaders

from .agent_registry import AgentSessionRegistry

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """Represents an MCP request from an agent."""

    method: str
    params: Dict[str, Any]
    agent_id: str
    request_id: Optional[str] = None


@dataclass
class MCPResponse:
    """Represents an MCP response to an agent."""

    result: Any
    error: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


class MCPProxy:
    """
    Proxies MCP requests between agent containers and SABER MCP server.

    This class handles:
    - Forwarding standard MCP protocol requests (list_tools, call_tool, etc.)
    - Injecting session headers for proper SABER server routing
    - Response forwarding and error handling
    - Connection management to SABER MCP server
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
            session_registry: Agent session registry for routing
            saber_mcp_url: URL of SABER MCP server
            timeout: Request timeout in seconds
        """
        self.session_registry = session_registry
        self.saber_mcp_url = saber_mcp_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._client_session: Optional[aiohttp.ClientSession] = None

    def _get_client(self) -> aiohttp.ClientSession:
        if self._client_session is None or self._client_session.closed:
            self._client_session = aiohttp.ClientSession(timeout=self.timeout)
        return self._client_session

    async def aclose(self) -> None:
        if self._client_session and not self._client_session.closed:
            await self._client_session.close()

    async def list_tools(self, agent_id: str) -> MCPResponse:
        """
        Proxy list_tools request to SABER server.

        Args:
            agent_id: Agent container ID

        Returns:
            MCPResponse with available tools or error
        """
        try:
            # Get session for routing
            session = await self.session_registry.get_session(agent_id)
            if not session:
                return MCPResponse(result=None, error={"code": -32000, "message": f"Agent {agent_id} not registered"})

            # Build headers for SABER server routing
            headers = {
                MCPHeaders.SESSION_ID: session.saber_session_id,
                MCPHeaders.CLIENT_ID: agent_id,
                "Content-Type": "application/json",
            }

            if session.task_id:
                headers[MCPHeaders.TASK_ID] = session.task_id

            # Forward request to SABER MCP server
            client = self._get_client()
            async with client.post(
                f"{self.saber_mcp_url}/mcp/list_tools",
                headers=headers,
                json={"jsonrpc": "2.0", "method": "list_tools", "params": {}},
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    return MCPResponse(result=data.get("result", []))
                else:
                    error_text = await response.text()
                    logger.error(f"SABER server error for list_tools: {response.status} - {error_text}")
                    return MCPResponse(
                        result=None, error={"code": -32000, "message": f"SABER server error: {response.status}"}
                    )

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling list_tools for agent {agent_id}")
            return MCPResponse(result=None, error={"code": -32000, "message": "Request timeout"})
        except Exception as e:
            logger.error(f"Error calling list_tools for agent {agent_id}: {e}")
            return MCPResponse(result=None, error={"code": -32000, "message": f"Internal error: {str(e)}"})

    async def call_tool(self, agent_id: str, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """
        Proxy call_tool request to SABER server.

        Args:
            agent_id: Agent container ID
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            MCPResponse with tool result or error
        """
        try:
            # Get session for routing
            session = await self.session_registry.get_session(agent_id)
            if not session:
                return MCPResponse(result=None, error={"code": -32000, "message": f"Agent {agent_id} not registered"})

            # Build headers for SABER server routing
            headers = {
                MCPHeaders.SESSION_ID: session.saber_session_id,
                MCPHeaders.CLIENT_ID: agent_id,
                "Content-Type": "application/json",
            }

            if session.task_id:
                headers[MCPHeaders.TASK_ID] = session.task_id

            # Build MCP call_tool request
            request_data = {
                "jsonrpc": "2.0",
                "method": "call_tool",
                "params": {"name": tool_name, "arguments": arguments},
            }

            # Forward request to SABER MCP server
            client = self._get_client()
            async with client.post(
                f"{self.saber_mcp_url}/mcp/call_tool", headers=headers, json=request_data
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    if "error" in data:
                        return MCPResponse(result=None, error=data["error"])
                    else:
                        return MCPResponse(result=data.get("result"))
                else:
                    error_text = await response.text()
                    logger.error(f"SABER server error for call_tool({tool_name}): " f"{response.status} - {error_text}")
                    return MCPResponse(
                        result=None, error={"code": -32000, "message": f"SABER server error: {response.status}"}
                    )

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling tool {tool_name} for agent {agent_id}")
            return MCPResponse(result=None, error={"code": -32000, "message": "Request timeout"})
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} for agent {agent_id}: {e}")
            return MCPResponse(result=None, error={"code": -32000, "message": f"Internal error: {str(e)}"})

    async def list_resources(self, agent_id: str) -> MCPResponse:
        """
        Proxy list_resources request to SABER server.

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

            # Build headers for SABER server routing
            headers = {
                MCPHeaders.SESSION_ID: session.saber_session_id,
                MCPHeaders.CLIENT_ID: agent_id,
                "Content-Type": "application/json",
            }

            if session.task_id:
                headers[MCPHeaders.TASK_ID] = session.task_id

            # Forward request to SABER MCP server
            client = self._get_client()
            async with client.post(
                f"{self.saber_mcp_url}/mcp/list_resources",
                headers=headers,
                json={"jsonrpc": "2.0", "method": "list_resources", "params": {}},
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    return MCPResponse(result=data.get("result", []))
                else:
                    error_text = await response.text()
                    logger.error(f"SABER server error for list_resources: " f"{response.status} - {error_text}")
                    return MCPResponse(
                        result=None, error={"code": -32000, "message": f"SABER server error: {response.status}"}
                    )

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling list_resources for agent {agent_id}")
            return MCPResponse(result=None, error={"code": -32000, "message": "Request timeout"})
        except Exception as e:
            logger.error(f"Error calling list_resources for agent {agent_id}: {e}")
            return MCPResponse(result=None, error={"code": -32000, "message": f"Internal error: {str(e)}"})

    async def ping(self, agent_id: str) -> MCPResponse:
        """
        Handle ping request (local response, no forwarding needed).

        Args:
            agent_id: Agent container ID

        Returns:
            MCPResponse with pong
        """
        # Verify agent is registered
        session = await self.session_registry.get_session(agent_id)
        if not session:
            return MCPResponse(result=None, error={"code": -32000, "message": f"Agent {agent_id} not registered"})

        return MCPResponse(result={"pong": True})

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of MCP proxy and connection to SABER server.

        Returns:
            Health status dictionary
        """
        try:
            # Test connection to SABER MCP server
            client = self._get_client()
            async with client.get(f"{self.saber_mcp_url}/health") as response:
                saber_healthy = response.status == 200
        except Exception as e:
            logger.warning(f"SABER server health check failed: {e}")
            saber_healthy = False

        # Get session stats
        session_stats = await self.session_registry.get_session_stats()

        return {
            "mcp_proxy_healthy": True,
            "saber_server_healthy": saber_healthy,
            "saber_mcp_url": self.saber_mcp_url,
            **session_stats,
        }
