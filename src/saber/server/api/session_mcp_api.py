"""
SessionMCPAPI implementation for SABER domain server.

The SessionMCPAPI handles Model Context Protocol server functionality,
providing tool discovery and tool execution only. All other operations
are handled by SessionRestAPI.
"""

import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from ..base import Action, CommandResult

logger = logging.getLogger(__name__)


class SessionMCPAPI:
    """
    Model Context Protocol handler for SessionManager.

    Handles ONLY tool discovery and tool execution via MCP protocol.
    All other operations (session management, episodes, policy) are handled by SessionRestAPI.
    """

    def __init__(self, session_manager: Any, host: str = "0.0.0.0", port: int = 3001) -> None:
        """
        Initialize the SessionMCPAPI.

        Args:
            session_manager: SessionManager instance to delegate to
            host: MCP server host address
            port: MCP server port
        """
        self.session_manager = session_manager
        self.host = host
        self.port = port
        self.mcp_server: Optional[FastMCP] = None
        self.active_mcp_sessions: Dict[str, str] = {}  # MCP session -> SessionManager session mapping

        logger.info(f"SessionMCPAPI initialized for domain '{session_manager.domain_name}' on {host}:{port}")

    async def start_mcp_server(self) -> None:
        """Start the MCP server."""
        try:
            self.mcp_server = FastMCP(f"SABER-{self.session_manager.domain_name}-MCP")

            # Register MCP handlers
            self._setup_mcp_handlers()

            logger.info(f"Starting SABER {self.session_manager.domain_name} MCP server on {self.host}:{self.port}")

            # Use run_async instead of run to work within existing asyncio loop
            await self.mcp_server.run_async(transport="sse", host=self.host, port=self.port)

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise

    async def shutdown_mcp_server(self) -> None:
        """Shutdown the MCP server."""
        try:
            if self.mcp_server:
                await self.mcp_server.close()  # type: ignore[attr-defined]
                self.mcp_server = None

            # Clear session mappings
            self.active_mcp_sessions.clear()

            logger.info("MCP server shutdown complete")

        except Exception as e:
            logger.error(f"Error during MCP server shutdown: {e}")

    def _setup_mcp_handlers(self) -> None:
        """Setup MCP protocol handlers."""
        if not self.mcp_server:
            raise RuntimeError("MCP server not initialized")

        # Register tools for each available executor type
        from ..execution.executors.factory import ExecutorFactory

        # Get available executors and register tools for each
        available_executors = ExecutorFactory.get_available_executors()
        logger.info(f"Registering MCP tools for executors: {available_executors}")

        # Dynamically register a tool for each executor
        for executor_name in available_executors:
            self._register_executor_tool(executor_name)

    def _register_executor_tool(self, executor_name: str) -> None:
        """
        Dynamically register an MCP tool for a specific executor.

        Args:
            executor_name: The name of the executor (e.g., 'cli', 'python')
        """
        tool_name = f"execute_{executor_name}"

        # Create the async function for this executor
        async def executor_tool(command: str, session_id: str, parameters: Optional[dict] = None) -> str:
            f"""Execute a {executor_name} command in the SABER sandbox environment.

            Args:
                command: The {executor_name} command/code to execute
                session_id: Session ID for context
                parameters: Optional parameters for the {executor_name} executor

            Returns:
                Command execution result as JSON
            """
            try:
                action = Action(tool_name=executor_name, command=command, parameters=parameters or {})
                result = await self.session_manager.execute_command(session_id, action)

                if result.success:
                    import json

                    return (
                        json.dumps(result.data)
                        if result.data
                        else json.dumps({"success": True, "output": f"{executor_name.title()} executed successfully"})
                    )
                else:
                    return json.dumps({"success": False, "error": result.error})

            except Exception as e:
                logger.error(f"Error executing {executor_name} command: {e}")
                import json

                return json.dumps({"success": False, "error": str(e)})

        # Set proper function metadata for the tool
        executor_tool.__name__ = tool_name
        executor_tool.__doc__ = f"Execute a {executor_name} command in the SABER sandbox environment."

        # Register the tool with the MCP server
        if self.mcp_server is not None:
            self.mcp_server.tool(name=tool_name)(executor_tool)

        logger.debug(f"Registered MCP tool: {tool_name}")

    async def handle_list_tools(self) -> List[Dict[str, Any]]:
        """
        Handle MCP tool discovery.

        Returns:
            List of available MCP tools from ExecutionManager
        """
        try:
            # Get tools from execution manager
            tools: List[Dict[str, Any]] = self.session_manager.execution_manager.to_mcp_tools()

            logger.debug(f"Returning {len(tools)} tools for MCP discovery")
            return tools

        except Exception as e:
            logger.error(f"Error handling list_tools: {e}")
            return []

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP tool execution.

        Args:
            name: Tool name to execute
            arguments: Tool arguments

        Returns:
            MCP-formatted execution result
        """
        try:
            # Extract session context from arguments
            session_context = self._map_session_context(arguments)
            session_id = session_context.get("session_id")

            if not session_id:
                return self._convert_to_mcp_result(
                    CommandResult.error_result(error="Missing session_id in tool arguments")
                )

            # Convert tool call to Action
            action = self._convert_to_action(name, arguments)

            # Execute command through SessionManager
            command_result = await self.session_manager.execute_command(session_id, action)

            # Convert result to MCP format
            return self._convert_to_mcp_result(command_result)

        except Exception as e:
            logger.error(f"Error handling call_tool {name}: {e}")
            return self._convert_to_mcp_result(CommandResult.error_result(error=f"Tool execution failed: {str(e)}"))

    def _map_session_context(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map MCP arguments to session context.

        Args:
            arguments: MCP tool arguments

        Returns:
            Dictionary with session context information
        """
        # Extract session_id from arguments
        # MCP clients should include session_id in their tool calls
        session_id = arguments.get("session_id")

        if not session_id:
            # Try to extract from common argument patterns
            session_id = arguments.get("context", {}).get("session_id")

        return {
            "session_id": session_id,
            "mcp_client": arguments.get("client_id", "unknown"),
        }

    def _convert_to_action(self, tool_name: str, arguments: Dict[str, Any]) -> Action:
        """
        Convert MCP tool call to Action object.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments

        Returns:
            Action object for execution
        """
        # Extract command from arguments
        command = arguments.get("command", "")

        # Extract nested parameters, or use empty dict if not provided
        # Filter out session context and command from parameters
        if "parameters" in arguments and isinstance(arguments["parameters"], dict):
            parameters = arguments["parameters"]
        else:
            # If no nested parameters, filter out known session/command fields
            parameters = {
                k: v for k, v in arguments.items() if k not in ["session_id", "client_id", "context", "command"]
            }

        return Action(tool_name=tool_name, command=command, parameters=parameters)

    def _convert_to_mcp_result(self, command_result: CommandResult) -> Dict[str, Any]:
        """
        Convert CommandResult to MCP-compatible result.

        Args:
            command_result: Result from command execution

        Returns:
            MCP-formatted result dictionary
        """
        if command_result.success:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(command_result.data) if command_result.data else "Command executed successfully",
                    }
                ],
                "isError": False,
            }
        else:
            return {"content": [{"type": "text", "text": f"Error: {command_result.error}"}], "isError": True}
