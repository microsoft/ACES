#!/usr/bin/env python3
"""
SABER MCP Client

Lightweight MCP client wrapper using FastMCP for secure tool execution
with the SABER server. Provides session-aware communication with proper
header handling for session mapping.

Includes MonitoredMCPClient wrapper for episode termination detection.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Protocol

from fastmcp import Client
from fastmcp.client.transports import SSETransport
from saber.base import MCPHeaders

logger = logging.getLogger(__name__)


class MCPMonitorCallback(Protocol):
    """Protocol for harness callbacks on MCP responses."""

    async def on_mcp_response(self, tool_name: str, arguments: Dict[str, Any], result: Any) -> None:
        """Called after each MCP tool response."""
        ...

    async def on_mcp_error(self, tool_name: str, arguments: Dict[str, Any], error: Exception) -> None:
        """Called when MCP tool call fails."""
        ...


class MCPClient:
    """
    Lightweight MCP client wrapper using FastMCP.

    Provides pure MCP communication with server for tool discovery
    and execution. No episode monitoring or additional logic.
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

    async def __aenter__(self) -> "MCPClient":
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


class SABERMCPClient:
    """
    Enhanced MCP Client with episode termination monitoring.

    Wraps the barebones MCPClient to provide episode termination detection
    and harness callbacks while maintaining clean MCP communication for agents.

    Implements HARD TERMINATION: when episode is terminated, all subsequent
    tool calls are immediately blocked with CancelledError.
    """

    def __init__(self, raw_mcp_client: MCPClient, monitor_callback: Optional[MCPMonitorCallback] = None):
        """
        Initialize SABER MCP client with monitoring.

        Args:
            raw_mcp_client: The underlying MCPClient instance
            monitor_callback: Optional callback for monitoring responses
        """
        self._client = raw_mcp_client
        self._callback = monitor_callback
        self._episode_terminated = False
        self._termination_reason: Optional[str] = None

        # Server-side termination detection via step counting
        self._step_count = 0
        self._max_steps: Optional[int] = None

    # Delegate all connection methods to underlying client
    async def connect(self) -> None:
        """Connect to MCP server."""
        await self._client.connect()

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        await self._client.disconnect()

    async def ping(self) -> Any:
        """Test connection to server."""
        return await self._client.ping()

    async def list_tools(self) -> Any:
        """List available tools."""
        return await self._client.list_tools()

    async def list_resources(self) -> Any:
        """List available resources."""
        return await self._client.list_resources()

    async def list_prompts(self) -> Any:
        """List available prompts."""
        return await self._client.list_prompts()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool and monitor the response.

        IMPLEMENTS HARD TERMINATION: Blocks all tool calls after episode termination.
        Uses proactive step counting to detect server-side termination.
        """
        # HARD TERMINATION: Block immediately if episode already terminated
        if self._episode_terminated:
            error_msg = f"Episode already terminated ({self._termination_reason}). Refusing tool call: {tool_name}"
            logger.warning(f"🚫 {error_msg}")
            raise asyncio.CancelledError(error_msg)

        # PROACTIVE TERMINATION: Check if we're about to exceed max steps
        if self._max_steps and self._step_count >= self._max_steps:
            self._episode_terminated = True
            self._termination_reason = "max_steps_reached_proactive"
            logger.warning(
                f"🛑 PROACTIVE TERMINATION: Max steps ({self._max_steps}) reached, blocking tool call: {tool_name}"
            )
            raise asyncio.CancelledError(f"Episode terminated proactively: max steps ({self._max_steps}) reached")

        # Increment step counter BEFORE making the call
        self._step_count += 1

        try:
            # Pure MCP communication - no modification
            result = await self._client.call_tool(tool_name, arguments)

            # Check for episode termination signals in response
            if tool_name == "end_episode":
                # Agent explicitly ended episode
                self._episode_terminated = True
                self._termination_reason = "agent_completed"
                logger.info("🏁 Agent called end_episode - episode terminated")
            else:
                # Check server response for termination signals
                termination_info = self._check_server_termination(result)
                if termination_info["terminated"]:
                    self._episode_terminated = True
                    self._termination_reason = termination_info["reason"]
                    logger.info(f"🛑 Server terminated episode: {self._termination_reason}")

                    # IMMEDIATE TERMINATION: Don't wait for next tool call
                    logger.info("🔥 Episode terminated - subsequent tool calls will be blocked")

            # Notify harness callback (non-blocking)
            if self._callback:
                try:
                    await self._callback.on_mcp_response(tool_name, arguments, result)
                except Exception as e:
                    logger.warning(f"Monitor callback failed: {e}")

            # Return clean result to agent
            return result

        except asyncio.CancelledError:
            # Re-raise cancellation immediately - don't mask it
            raise
        except Exception as e:
            # Check if this is an episode termination error
            error_msg = str(e).lower()
            if "no active episode" in error_msg:
                self._episode_terminated = True
                self._termination_reason = "server_terminated"
                logger.info(f"🛑 Episode terminated (detected from error): {e}")

                # Notify harness callback about the error
                if self._callback:
                    try:
                        await self._callback.on_mcp_error(tool_name, arguments, e)
                    except Exception as callback_error:
                        logger.warning(f"Monitor callback failed: {callback_error}")

                # IMMEDIATE TERMINATION: Raise CancelledError instead of returning error response
                logger.info("🔥 Raising CancelledError due to server termination error")
                raise asyncio.CancelledError(f"Episode terminated by server: {e}")

            # Notify harness callback about the error
            if self._callback:
                try:
                    await self._callback.on_mcp_error(tool_name, arguments, e)
                except Exception as callback_error:
                    logger.warning(f"Monitor callback failed: {callback_error}")

            # Re-raise non-termination exceptions
            raise

    def _check_server_termination(self, mcp_result: Any) -> Dict[str, Any]:
        """Check MCP response for server-side episode termination signals."""
        terminated = False
        reason = None

        if isinstance(mcp_result, dict):
            # Check for metadata in the top-level response
            metadata = mcp_result.get("metadata", {})
            if isinstance(metadata, dict):
                if metadata.get("episode_terminated", False):
                    terminated = True
                    reason = metadata.get("termination_reason", "server_terminated")
                    return {"terminated": terminated, "reason": reason}

            # Check content for termination signals (fallback)
            content = mcp_result.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")

                        # Look for explicit server termination signal
                        if "[EPISODE_TERMINATED:" in text:
                            terminated = True
                            # Extract reason from [EPISODE_TERMINATED: reason] format
                            try:
                                start = text.index("[EPISODE_TERMINATED:") + len("[EPISODE_TERMINATED:")
                                end = text.index("]", start)
                                reason = text[start:end].strip()
                            except (ValueError, IndexError):
                                reason = "server_terminated"
                            break

                        # Fallback: Look for other termination keywords in response text
                        elif any(
                            signal in text.lower()
                            for signal in [
                                "episode_terminated",
                                "max_steps_reached",
                                "episode_timeout",
                                "episode_failed",
                            ]
                        ):
                            terminated = True
                            if "max_steps" in text.lower():
                                reason = "max_steps_reached"
                            elif "timeout" in text.lower():
                                reason = "timeout"
                            elif "failed" in text.lower() or "error" in text.lower():
                                reason = "error"
                            else:
                                reason = "server_terminated"
                            break

        return {"terminated": terminated, "reason": reason}

    # Properties for harness to check episode state
    @property
    def episode_terminated(self) -> bool:
        """Check if episode has been terminated."""
        return self._episode_terminated

    @property
    def termination_reason(self) -> Optional[str]:
        """Get the reason for episode termination."""
        return self._termination_reason

    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count

    def set_max_steps(self, max_steps: int) -> None:
        """Set maximum steps for proactive termination."""
        self._max_steps = max_steps
        logger.info(f"🎯 MCP Client configured for proactive termination at {max_steps} steps")

    def reset_episode_state(self) -> None:
        """Reset episode termination state for new episode."""
        self._episode_terminated = False
        self._termination_reason = None
        self._step_count = 0
        self._max_steps = None
