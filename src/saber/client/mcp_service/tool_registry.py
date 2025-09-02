#!/usr/bin/env python3
"""
Tool Registry for SABER MCP Sidecar

Manages tool discovery and caching from SABER MCP server to enable function injection
in agent containers. Tools are scoped per session+episode combination since tool
availability is dynamic based on the specific task being executed.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from .mcp_proxy import MCPProxy

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """Information about available tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    timeout: Optional[int] = 30
    cached_at: Optional[datetime] = None


class ToolRegistry:
    """
    Manages tool discovery and caching from SABER MCP server.

    Tools are scoped per session+episode combination since:
    - Sessions provide the broader context
    - Episodes define the specific task and its tool requirements

    Responsibilities:
    - Discover available tools from SABER server per session+episode
    - Cache tool metadata with configurable TTL
    - Handle tool schema updates and notifications
    - Provide tool metadata for agent function generation
    """

    def __init__(self, mcp_proxy: MCPProxy, cache_ttl: timedelta = timedelta(minutes=5), auto_refresh: bool = True):
        self.mcp_proxy = mcp_proxy
        self.cache_ttl = cache_ttl
        self.auto_refresh = auto_refresh

        # Tool cache: (session_id, episode_id) -> tool_name -> ToolInfo
        self._tool_cache: Dict[Tuple[str, str], Dict[str, ToolInfo]] = {}
        self._cache_timestamps: Dict[Tuple[str, str], datetime] = {}
        self._refresh_tasks: Dict[Tuple[str, str], asyncio.Task] = {}

        self.logger = logging.getLogger(__name__)

    async def get_tools_for_session_episode(
        self,
        session_id: str,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, ToolInfo]:
        """
        Get available tools for a specific session+episode combination.

        Args:
            session_id: SABER session identifier
            episode_id: Episode identifier (if None, uses "default")
            task_id: Task identifier for additional context
            force_refresh: Skip cache and fetch from server

        Returns:
            Dictionary mapping tool names to metadata
        """
        # Use "default" if no episode_id provided for backwards compatibility
        effective_episode_id = episode_id or "default"
        cache_key = (session_id, effective_episode_id)

        # Check cache validity
        if not force_refresh and self._is_cache_valid(cache_key):
            return self._tool_cache.get(cache_key, {})

        # Discover tools from SABER server
        try:
            tools = await self._discover_tools_from_server(session_id, effective_episode_id, task_id, agent_id)
            self._update_cache(cache_key, tools)

            # Start auto-refresh task if enabled
            if self.auto_refresh:
                self._schedule_refresh(cache_key, session_id, effective_episode_id, task_id)

            return tools

        except Exception as e:
            self.logger.error(f"Failed to discover tools for session {session_id}, episode {effective_episode_id}: {e}")
            # Return cached tools as fallback
            return self._tool_cache.get(cache_key, {})

    async def _discover_tools_from_server(
        self, session_id: str, episode_id: str, task_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> Dict[str, ToolInfo]:
        """Discover tools from SABER MCP server via proxy."""
        # Use the actual registered agent_id, fall back to artificial one for backwards compatibility
        if agent_id:
            effective_agent_id = agent_id
        else:
            effective_agent_id = f"tool-discovery-{session_id}-{episode_id}"

        try:
            # Use MCP proxy to get tools for this session+episode context
            self.logger.info(
                f"Discovering tools for session {session_id}, episode {episode_id} via agent_id {effective_agent_id}"
            )
            response = await self.mcp_proxy.list_tools(effective_agent_id)

            self.logger.info(f"MCP response: result={type(response.result)}, error={response.error}")
            if response.error:
                raise RuntimeError(f"MCP error: {response.error}")

            tools = {}
            # Handle the response format from MCPProxy
            if isinstance(response.result, list):
                result_tools = response.result
            elif isinstance(response.result, dict) and "tools" in response.result:
                result_tools = response.result["tools"]
            else:
                self.logger.warning(f"Unexpected tools response format: {type(response.result)}")
                result_tools = []

            self.logger.info(f"Parsed {len(result_tools)} tools from response")

            for tool_data in result_tools:
                # Handle both dictionary format and Tool object format
                if hasattr(tool_data, "name"):
                    # FastMCP Tool object
                    tool_name = tool_data.name
                    tool_description = getattr(tool_data, "description", "")
                    tool_schema = getattr(tool_data, "inputSchema", {})
                    if hasattr(tool_schema, "dict"):
                        tool_schema = tool_schema.dict()
                elif isinstance(tool_data, dict):
                    # Dictionary format
                    tool_name = tool_data["name"]
                    tool_description = tool_data.get("description", "")
                    tool_schema = tool_data.get("inputSchema", {})
                else:
                    self.logger.warning(f"Unexpected tool format: {type(tool_data)} - {tool_data}")
                    continue

                metadata = ToolInfo(
                    name=tool_name,
                    description=tool_description,
                    parameters=tool_schema,
                    timeout=30,  # Default timeout
                    cached_at=datetime.utcnow(),
                )
                tools[metadata.name] = metadata

            self.logger.info(f"Discovered {len(tools)} tools for session {session_id}, episode {episode_id}")
            return tools

        except Exception as e:
            self.logger.error(f"Tool discovery failed for session {session_id}, episode {episode_id}: {e}")
            raise

    def _is_cache_valid(self, cache_key: Tuple[str, str]) -> bool:
        """Check if cached tools are still valid."""
        if cache_key not in self._cache_timestamps:
            return False

        cache_age = datetime.utcnow() - self._cache_timestamps[cache_key]
        return cache_age < self.cache_ttl

    def _update_cache(self, cache_key: Tuple[str, str], tools: Dict[str, ToolInfo]) -> None:
        """Update tool cache for session+episode."""
        self._tool_cache[cache_key] = tools
        self._cache_timestamps[cache_key] = datetime.utcnow()

    def _schedule_refresh(
        self, cache_key: Tuple[str, str], session_id: str, episode_id: str, task_id: Optional[str] = None
    ) -> None:
        """Schedule automatic cache refresh."""
        # Cancel existing refresh task
        if cache_key in self._refresh_tasks:
            self._refresh_tasks[cache_key].cancel()

        # Schedule new refresh
        async def refresh_task() -> None:
            await asyncio.sleep(self.cache_ttl.total_seconds())
            try:
                # Note: refresh task doesn't have access to agent_id, falls back to artificial ID
                await self.get_tools_for_session_episode(session_id, episode_id, task_id, force_refresh=True)
            except Exception as e:
                self.logger.warning(f"Auto-refresh failed for session {session_id}, episode {episode_id}: {e}")

        self._refresh_tasks[cache_key] = asyncio.create_task(refresh_task())

    async def execute_tool(
        self,
        session_id: str,
        episode_id: Optional[str],
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool via MCP proxy.

        Args:
            session_id: SABER session identifier
            episode_id: Episode identifier
            tool_name: Name of tool to execute
            arguments: Tool arguments
            timeout: Execution timeout override

        Returns:
            Tool execution result
        """
        effective_episode_id = episode_id or "default"
        # Use the actual registered agent_id, fall back to artificial one for backwards compatibility
        if agent_id:
            effective_agent_id = agent_id
        else:
            effective_agent_id = f"tool-executor-{session_id}-{effective_episode_id}"

        # Get tool metadata for validation
        tools = await self.get_tools_for_session_episode(session_id, episode_id, agent_id=agent_id)
        if tool_name not in tools:
            raise ValueError(
                f"Tool '{tool_name}' not available for session {session_id}, episode {effective_episode_id}"
            )

        tool_metadata = tools[tool_name]
        effective_timeout = timeout or tool_metadata.timeout

        try:
            # Execute via MCP proxy
            response = await asyncio.wait_for(
                self.mcp_proxy.call_tool(effective_agent_id, tool_name, arguments), timeout=effective_timeout
            )

            if response.error:
                return {"success": False, "error": response.error, "result": None}

            return {"success": True, "result": response.result, "error": None}

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": {"code": -32603, "message": f"Tool execution timed out after {effective_timeout}s"},
                "result": None,
            }
        except Exception as e:
            return {"success": False, "error": {"code": -32603, "message": str(e)}, "result": None}

    async def cleanup_session_episode(self, session_id: str, episode_id: Optional[str] = None) -> None:
        """Clean up cached data for a session+episode."""
        effective_episode_id = episode_id or "default"
        cache_key = (session_id, effective_episode_id)

        # Cancel refresh task
        if cache_key in self._refresh_tasks:
            self._refresh_tasks[cache_key].cancel()
            del self._refresh_tasks[cache_key]

        # Remove from cache
        self._tool_cache.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)

        self.logger.info(f"Cleaned up tool cache for session {session_id}, episode {effective_episode_id}")

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up all cached data for a session (all episodes)."""
        # Find all cache keys for this session
        keys_to_remove = [key for key in self._tool_cache.keys() if key[0] == session_id]

        for cache_key in keys_to_remove:
            # Cancel refresh task
            if cache_key in self._refresh_tasks:
                self._refresh_tasks[cache_key].cancel()
                del self._refresh_tasks[cache_key]

            # Remove from cache
            self._tool_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)

        self.logger.info(f"Cleaned up tool cache for session {session_id} ({len(keys_to_remove)} episodes)")

    # Backwards compatibility method
    async def get_tools_for_session(self, session_id: str, force_refresh: bool = False) -> Dict[str, ToolInfo]:
        """
        Backwards compatibility method for existing code.
        Uses default episode_id.
        """
        return await self.get_tools_for_session_episode(
            session_id=session_id, episode_id="default", force_refresh=force_refresh
        )
