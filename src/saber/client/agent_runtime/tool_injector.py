#!/usr/bin/env python3
"""
Tool Injector for SABER Agent Runtime

Replaces the MCP client with function injection. Fetches tool metadata from the sidecar
and creates local proxy functions that agents can call as native Python functions.
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""

    pass


class SidecarConnectionError(Exception):
    """Exception raised when sidecar communication fails."""

    pass


class ToolInjector:
    """
    Manages tool discovery and function injection for agent runtime.

    Responsibilities:
    - Connect to sidecar and fetch tool metadata
    - Generate async proxy functions for each tool
    - Inject functions into agent execution context
    - Handle errors and retries for tool execution
    """

    def __init__(
        self,
        sidecar_url: str,
        session_id: str,
        agent_id: str,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.sidecar_url = sidecar_url.rstrip("/")
        self.session_id = session_id
        self.episode_id = episode_id
        self.task_id = task_id
        self.agent_id = agent_id
        self.timeout = timeout
        self.max_retries = max_retries

        self.tools: Dict[str, Callable] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        self.session: Optional[aiohttp.ClientSession] = None

        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize tool injector and discover available tools."""
        self.session = aiohttp.ClientSession()

        try:
            await self._discover_tools()
            self._generate_proxy_functions()
            self.logger.info(f"Initialized {len(self.tools)} tools for agent {self.agent_id}")

        except Exception as e:
            self.logger.error(f"Failed to initialize tool injector: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _discover_tools(self) -> None:
        """Discover available tools from sidecar."""
        headers = {"X-Saber-Session-Id": self.session_id, "X-Saber-Agent-Id": self.agent_id}

        # Add episode and task context if available
        if self.episode_id:
            headers["X-Saber-Episode-Id"] = self.episode_id
        if self.task_id:
            headers["X-Saber-Task-Id"] = self.task_id

        try:
            if not self.session:
                raise SidecarConnectionError("HTTP session not available")
            async with self.session.get(f"{self.sidecar_url}/tools", headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise SidecarConnectionError(f"Failed to get tools: HTTP {response.status}: {error_text}")

                data = await response.json()
                self.tool_metadata = data.get("tools", {})

                self.logger.info(f"Discovered {len(self.tool_metadata)} tools from sidecar")

        except aiohttp.ClientError as e:
            raise SidecarConnectionError(f"Network error communicating with sidecar: {e}")
        except Exception as e:
            raise SidecarConnectionError(f"Unexpected error discovering tools: {e}")

    def _generate_proxy_functions(self) -> None:
        """Generate async proxy functions for each discovered tool."""
        for tool_name, metadata in self.tool_metadata.items():
            proxy_function = self._create_tool_proxy(tool_name, metadata)
            self.tools[tool_name] = proxy_function

    def _create_tool_proxy(self, tool_name: str, metadata: Dict[str, Any]) -> Callable[..., Any]:
        """Create an async proxy function for a specific tool."""

        @wraps(self._execute_tool_with_retry)
        async def tool_proxy(**kwargs: Any) -> Any:
            """Auto-generated proxy function for tool execution."""
            return await self._execute_tool_with_retry(tool_name, kwargs)

        # Set function metadata for introspection
        tool_proxy.__name__ = tool_name
        tool_proxy.__doc__ = metadata.get("description", f"Execute {tool_name} tool")

        # Generate type annotations from JSON schema (simplified)
        tool_proxy.__annotations__ = self._generate_type_annotations(metadata.get("parameters", {}))

        return tool_proxy

    def _generate_type_annotations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate type annotations from JSON schema (simplified implementation)."""
        annotations = {}

        properties = parameters.get("properties", {})
        for param_name, param_schema in properties.items():
            param_type = param_schema.get("type", "any")

            # Simple type mapping (can be extended)
            type_map = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": List,
                "object": Dict[str, Any],
            }

            annotations[param_name] = type_map.get(param_type, Any)

        annotations["return"] = Any  # Tool return type is dynamic
        return annotations

    async def _execute_tool_with_retry(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute tool with retry logic and error handling."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await self._execute_tool(tool_name, arguments)

                if result.get("success", False):
                    return result.get("result")
                else:
                    error = result.get("error", "Unknown error")
                    raise ToolExecutionError(f"Tool {tool_name} failed: {error}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = (2**attempt) * 0.1
                    self.logger.warning(f"Tool {tool_name} attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break
            except Exception as e:
                # Don't retry on non-network errors
                raise ToolExecutionError(f"Tool {tool_name} execution error: {e}")

        # All retries exhausted
        raise SidecarConnectionError(f"Tool {tool_name} failed after {self.max_retries + 1} attempts: {last_error}")

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call via sidecar."""
        self.logger.info(f"🔗 ToolInjector._execute_tool called: {tool_name} via {self.sidecar_url}/execute_tool")

        headers = {
            "X-Saber-Session-Id": self.session_id,
            "X-Saber-Agent-Id": self.agent_id,
            "Content-Type": "application/json",
        }

        # Add episode and task context if available
        if self.episode_id:
            headers["X-Saber-Episode-Id"] = self.episode_id
        if self.task_id:
            headers["X-Saber-Task-Id"] = self.task_id

        payload = {"tool_name": tool_name, "arguments": arguments, "timeout": self.timeout}
        self.logger.info(f"📤 Sending POST to {self.sidecar_url}/execute_tool with payload: {payload}")

        try:
            if not self.session:
                raise SidecarConnectionError("HTTP session not available")
            async with self.session.post(
                f"{self.sidecar_url}/execute_tool",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout + 5),  # Add buffer
            ) as response:
                self.logger.info(f"📥 Received response: HTTP {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"❌ HTTP error: {response.status}: {error_text}")
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=error_text,
                    )

                result = await response.json()
                self.logger.info(f"✅ Tool execution result: {result}")
                return result if isinstance(result, dict) else {"result": result}

        except asyncio.TimeoutError:
            self.logger.error(f"⏰ Tool {tool_name} timed out after {self.timeout}s")
            raise asyncio.TimeoutError(f"Tool {tool_name} timed out after {self.timeout}s")

    def inject_into_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Inject tool functions into agent execution context."""
        # Add all tools as direct functions
        for tool_name, tool_function in self.tools.items():
            context[tool_name] = tool_function

        # Add utility functions
        context["get_available_tools"] = lambda: list(self.tools.keys())
        context["get_tool_info"] = lambda name: self.tool_metadata.get(name, {})

        return context

    async def list_tools(self) -> List[str]:
        """List available tools - backwards compatibility method."""
        return list(self.tools.keys())

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool - backwards compatibility method."""
        self.logger.info(f"🚀 ToolInjector.call_tool called: {tool_name} with args: {arguments}")

        if tool_name not in self.tools:
            self.logger.error(f"❌ Tool '{tool_name}' not available. Available tools: {list(self.tools.keys())}")
            raise ToolExecutionError(f"Tool '{tool_name}' not available")

        self.logger.info(f"🔧 Calling tool proxy function for {tool_name}")
        # Call the injected proxy function
        result = await self.tools[tool_name](**arguments)
        self.logger.info(f"✅ Tool {tool_name} execution completed")
        return result

    async def refresh_tools(self) -> None:
        """Refresh tool discovery from sidecar."""
        try:
            await self._discover_tools()
            self._generate_proxy_functions()
            self.logger.info(f"Refreshed tools for agent {self.agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to refresh tools: {e}")
            # Continue with existing tools
