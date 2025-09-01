#!/usr/bin/env python3
"""
MCP Client Factory

Factory for creating standard MCP clients that connect to the SABER sidecar.
Supports multiple MCP client libraries (anthropic/mcp-python, custom implementations).
"""

import logging
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


class MCPClientFactory:
    """
    Factory for creating MCP clients using standard libraries.

    Creates MCP clients that connect to the SABER sidecar service,
    which proxies requests to the SABER MCP server with proper session mapping.
    """

    def __init__(self) -> None:
        """Initialize MCP client factory."""
        pass

    async def create_client(
        self,
        sidecar_url: str,
        session_id: str,
        task_id: Optional[str] = None,
        client_id: str = "agent-container",
        library: str = "anthropic",
    ) -> Any:
        """
        Create MCP client using specified library.

        Args:
            sidecar_url: URL of the SABER MCP sidecar service
            session_id: Session ID for server mapping
            task_id: Optional task ID for context
            client_id: Client identifier
            library: MCP library to use ("anthropic", "httpx", "custom")

        Returns:
            Connected MCP client
        """
        if library == "anthropic":
            return await self._create_anthropic_client(sidecar_url, session_id, task_id, client_id)
        elif library == "httpx":
            return await self._create_httpx_client(sidecar_url, session_id, task_id, client_id)
        elif library == "custom":
            return await self._create_custom_client(sidecar_url, session_id, task_id, client_id)
        else:
            raise ValueError(f"Unsupported MCP library: {library}")

    async def _create_anthropic_client(
        self, sidecar_url: str, session_id: str, task_id: Optional[str], client_id: str
    ) -> Any:
        """Create MCP client using anthropic/mcp-python library."""
        try:
            # For now, use a simple HTTP-based approach since anthropic/mcp-python
            # primarily supports stdio transport. We'll create a compatible wrapper.
            return await self._create_http_mcp_client(sidecar_url, session_id, task_id, client_id)

        except ImportError:
            logger.warning("anthropic/mcp-python not available, falling back to HTTP client")
            return await self._create_http_mcp_client(sidecar_url, session_id, task_id, client_id)

    async def _create_httpx_client(
        self, sidecar_url: str, session_id: str, task_id: Optional[str], client_id: str
    ) -> Any:
        """Create MCP client using httpx for HTTP transport."""
        return await self._create_http_mcp_client(sidecar_url, session_id, task_id, client_id)

    async def _create_custom_client(
        self, sidecar_url: str, session_id: str, task_id: Optional[str], client_id: str
    ) -> Any:
        """Create custom MCP client implementation."""
        return await self._create_http_mcp_client(sidecar_url, session_id, task_id, client_id)

    async def _create_http_mcp_client(
        self, sidecar_url: str, session_id: str, task_id: Optional[str], client_id: str
    ) -> "StandardMCPClient":
        """Create HTTP-based MCP client for sidecar communication."""
        client = StandardMCPClient(sidecar_url=sidecar_url, session_id=session_id, task_id=task_id, client_id=client_id)

        await client.connect()
        return client


class StandardMCPClient:
    """
    Standard MCP client implementation for SABER sidecar communication.

    Provides a clean MCP interface that agents can use without SABER-specific code.
    Communicates with the SABER sidecar which handles session mapping and proxying.
    """

    def __init__(
        self, sidecar_url: str, session_id: str, task_id: Optional[str] = None, client_id: str = "agent-container"
    ):
        """Initialize standard MCP client."""
        self.sidecar_url = sidecar_url.rstrip("/")
        self.session_id = session_id
        self.task_id = task_id
        self.client_id = client_id
        self.connected = False

        # Import httpx here to avoid import errors if not available
        try:
            import httpx

            self.http_client = httpx.AsyncClient(timeout=30.0)
        except ImportError:
            raise ImportError("httpx required for HTTP MCP client. Install with: pip install httpx")

    async def connect(self) -> None:
        """Connect to MCP sidecar and register session."""
        try:
            # Register session with sidecar
            register_url = f"{self.sidecar_url}/admin/sessions"
            register_data = {
                "agent_id": self.client_id,
                "saber_session_id": self.session_id,
                "task_id": self.task_id,
            }
            response = await self.http_client.post(register_url, json=register_data)
            response.raise_for_status()

            # Test connection with list_tools
            await self.list_tools()

            self.connected = True
            logger.info(f"✅ Connected to MCP sidecar at {self.sidecar_url}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP sidecar: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MCP sidecar."""
        try:
            if self.connected:
                # Unregister session (best effort)
                unregister_url = f"{self.sidecar_url}/admin/sessions/{self.client_id}"
                try:
                    await self.http_client.delete(unregister_url)
                except Exception:
                    pass

            await self.http_client.aclose()
            self.connected = False
            logger.info("🧹 Disconnected from MCP sidecar")

        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")

    async def list_tools(self) -> list[Dict[str, Any]]:
        """List available tools from SABER server via sidecar."""
        if not self.connected:
            raise RuntimeError("MCP client not connected")

        url = f"{self.sidecar_url}/mcp/list_tools"
        headers = self._get_session_headers()

        response = await self.http_client.post(
            url,
            headers=headers,
            json={"jsonrpc": "2.0", "method": "list_tools", "params": {}},
        )
        response.raise_for_status()

        result = response.json()
        # JSON-RPC envelope
        return cast(List[Dict[str, Any]], result.get("result", []))

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on SABER server via sidecar."""
        if not self.connected:
            raise RuntimeError("MCP client not connected")

        url = f"{self.sidecar_url}/mcp/call_tool"
        headers = self._get_session_headers()
        data = {
            "jsonrpc": "2.0",
            "method": "call_tool",
            "params": {"name": name, "arguments": arguments},
        }

        response = await self.http_client.post(url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()

        # JSON-RPC envelope
        if "error" in result and result["error"]:
            error_msg = result["error"].get("message", "Tool execution failed")
            raise RuntimeError(f"Tool {name} failed: {error_msg}")
        return cast(Dict[str, Any], result.get("result", {}))

    async def list_resources(self) -> list[Dict[str, Any]]:
        """List available resources from SABER server via sidecar."""
        if not self.connected:
            raise RuntimeError("MCP client not connected")

        url = f"{self.sidecar_url}/mcp/list_resources"
        headers = self._get_session_headers()

        response = await self.http_client.post(
            url,
            headers=headers,
            json={"jsonrpc": "2.0", "method": "list_resources", "params": {}},
        )
        response.raise_for_status()

        result = response.json()
        # JSON-RPC envelope
        return cast(List[Dict[str, Any]], result.get("result", []))

    def _get_session_headers(self) -> Dict[str, str]:
        """Get headers for session mapping."""
        headers = {"X-Session-ID": self.session_id, "X-Client-ID": self.client_id}

        if self.task_id:
            headers["X-Task-ID"] = self.task_id

        return headers

    async def __aenter__(self) -> "StandardMCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
