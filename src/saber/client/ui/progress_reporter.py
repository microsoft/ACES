#!/usr/bin/env python3
"""
Tool Call Progress Reporter

Real-time progress reporting for MCP tool calls from sidecar to harness.
This enables live UI updates during tool execution.
"""

import logging
from typing import Optional

import aiohttp

from .interfaces import MCPToolCall, ToolCallProgressReporter

logger = logging.getLogger(__name__)


class SSEProgressReporter(ToolCallProgressReporter):
    """
    Reports tool call progress via HTTP/SSE from MCP sidecar to harness progress server.

    This class sends real-time tool call progress from the MCP sidecar
    to a simple HTTP server running in the harness for immediate UI updates.

    Architecture: MCP Sidecar → HTTP → Harness Progress Server → UI Manager
    """

    def __init__(
        self,
        harness_progress_url: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: float = 5.0,
    ):
        """
        Initialize SSE progress reporter.

        Args:
            harness_progress_url: Base URL of harness progress server
            session_id: Session ID for context
            agent_id: Agent ID for context
            timeout: HTTP request timeout in seconds
        """
        self.harness_progress_url = harness_progress_url.rstrip("/")
        self.session_id = session_id
        self.agent_id = agent_id
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(f"📊 SSE progress reporter initialized: {harness_progress_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def _send_progress_event(self, event_type: str, tool_call: MCPToolCall) -> None:
        """Send progress event to harness progress server."""
        try:
            session = await self._get_session()

            # Prepare tool call data for our progress server format
            tool_call_data = {
                "id": tool_call.call_id,
                "name": tool_call.tool_name,
                "arguments": tool_call.input_args,
                "status": tool_call.status.value if tool_call.status else None,
                "result": tool_call.output,
                "error": tool_call.error,
                "start_time": tool_call.start_time.isoformat() if tool_call.start_time else None,
                "end_time": tool_call.end_time.isoformat() if tool_call.end_time else None,
                "execution_time_ms": tool_call.execution_time_ms,
                "progress_message": getattr(tool_call, "progress_message", None),
                "session_id": self.session_id,
                "agent_id": self.agent_id,
            }

            # Send to specific progress server endpoint based on event type
            endpoint_map = {
                "tool_call_start": "/progress/tool_call_start",
                "tool_call_complete": "/progress/tool_call_complete",
            }

            endpoint = endpoint_map.get(event_type)
            if not endpoint:
                logger.warning(f"Unknown event type: {event_type}")
                return

            progress_url = f"{self.harness_progress_url}{endpoint}"
            async with session.post(progress_url, json=tool_call_data) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ Progress event failed: {response.status} - {await response.text()}")
                else:
                    logger.debug(f"📤 Progress event sent: {event_type} for {tool_call.tool_name}")

            # Send to harness progress endpoint
            progress_url = f"{self.harness_progress_url}/api/progress/tool-call"
            async with session.post(progress_url, json=tool_call_data) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ Progress event failed: {response.status} - {await response.text()}")
                else:
                    logger.debug(f"📊 Progress event sent: {event_type} for {tool_call.tool_name}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to send progress event: {e}")

    async def tool_call_start(self, tool_call: MCPToolCall) -> None:
        """Report that a tool call has started."""
        await self._send_progress_event("tool_call_start", tool_call)

    async def tool_call_complete(self, tool_call: MCPToolCall) -> None:
        """Report that a tool call has completed."""
        await self._send_progress_event("tool_call_complete", tool_call)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# For backward compatibility, keep the DirectUIProgressReporter as an alias
DirectUIProgressReporter = SSEProgressReporter
