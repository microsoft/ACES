"""
SessionMCPAPI implementation for SABER domain server.

The SessionMCPAPI handles Model Context Protocol server functionality,
providing tool discovery and tool execution only. All other operations
are handled by SessionRestAPI.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from ..base import Action, CommandResult
from ..execution.executors.factory import ExecutorFactory

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

        # Get available executors and register tools for each
        available_executors = ExecutorFactory.get_available_executors()
        logger.info(f"Registering MCP tools for executors: {available_executors}")

        # Dynamically register a tool for each executor
        for executor_name in available_executors:
            self._register_executor_tool(executor_name)

        # Register hardcoded MCP API tools
        self._register_hardcoded_tools()

    def _register_hardcoded_tools(self) -> None:
        """Register hardcoded MCP API tools with the FastMCP server."""
        if not self.mcp_server:
            raise RuntimeError("MCP server not initialized")

        # Register end_episode tool
        async def end_episode_tool(session_id: str, arguments: Optional[str] = None) -> str:
            """End the current episode and optionally record a discovered flag/target/objective."""
            try:
                args = {"session_id": session_id}
                if arguments:
                    args["result"] = arguments  # Map arguments to result internally
                mcp_result = await self._handle_end_episode_call(args)
                # Extract the text content from the MCP result
                if mcp_result.get("isError", False):
                    return json.dumps({"success": False, "error": mcp_result["content"][0]["text"]})
                else:
                    return str(mcp_result["content"][0]["text"])
            except Exception as e:
                logger.error(f"Error in end_episode tool: {e}")
                return json.dumps({"success": False, "error": str(e)})

        # Set proper function metadata and register with FastMCP
        end_episode_tool.__name__ = "end_episode"
        self.mcp_server.tool(name="end_episode")(end_episode_tool)

        logger.debug("Registered hardcoded MCP tool: end_episode")

    def _register_executor_tool(self, executor_name: str) -> None:
        """
        Dynamically register an MCP tool for a specific executor.

        Args:
            executor_name: The name of the executor (e.g., 'cli', 'python')
        """
        tool_name = f"execute_{executor_name}"

        # Create the async function for this executor
        async def executor_tool(session_id: str, arguments: str, parameters: Optional[dict] = None) -> str:
            f"""Execute a {executor_name} command in the SABER sandbox environment.

            Args:
                session_id: Session ID for context
                arguments: The {executor_name} command/code to execute
                parameters: Optional parameters for the {executor_name} executor

            Returns:
                Command execution result as JSON
            """
            # Delegate to the standard handle_call_tool method
            mcp_args = {"session_id": session_id, "arguments": arguments}
            if parameters:
                mcp_args.update(parameters)

            result = await self.handle_call_tool(executor_name, mcp_args)

            # Extract the text content from the MCP result
            if result.get("isError", False):
                return json.dumps({"success": False, "error": result["content"][0]["text"]})
            else:
                return str(result["content"][0]["text"])

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
            List of available MCP tools from ExecutionManager plus hardcoded MCP tools
        """
        try:
            # Get tools from execution manager
            tools: List[Dict[str, Any]] = self.session_manager.execution_manager.to_mcp_tools()

            # Add hardcoded MCP API tools
            hardcoded_tools = [
                {
                    "name": "end_episode",
                    "description": "End the current episode and optionally record a discovered flag/target/objective",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string", "description": "Session ID for context"},
                            "arguments": {
                                "type": "string",
                                "description": (
                                    "Optional flag, target, or objective that was " "discovered during the episode"
                                ),
                            },
                        },
                        "required": ["session_id"],
                    },
                }
            ]

            # Combine executor tools and hardcoded tools
            all_tools = tools + hardcoded_tools

            logger.debug(
                f"Returning {len(all_tools)} tools ({len(tools)} executor tools + "
                f"{len(hardcoded_tools)} hardcoded tool) for MCP discovery"
            )
            return all_tools

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
            # Handle hardcoded MCP API tools
            if name == "end_episode":
                return await self._handle_end_episode_call(arguments)

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

    async def _handle_end_episode_call(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the hardcoded end_episode tool call.

        Args:
            arguments: Tool arguments (should contain session_id and optional result)

        Returns:
            MCP-formatted result confirming episode end
        """
        try:
            # Extract session context
            session_context = self._map_session_context(arguments)
            session_id = session_context.get("session_id")

            if not session_id:
                return self._convert_to_mcp_result(
                    CommandResult.error_result(error="Missing session_id in end_episode arguments")
                )

            # Extract optional result/flag/objective
            result = arguments.get("arguments", "")

            # If a result was provided, save it as an action in the episode
            if result:
                logger.info(f"Episode ending with result: {result}")
                # Create an action to record the discovered result
                result_action = Action(
                    tool_name="episode_result",
                    arguments=f"Episode completed with result: {result}",
                    parameters={"result": result, "episode_end": True},
                )
                # Execute the result action to save it in the episode
                await self.session_manager.execute_command(session_id, result_action)

            # End the episode through SessionManager
            await self.session_manager.end_episode(session_id)

            # Prepare success message
            success_message = "Episode ended successfully"
            if result:
                success_message += f" with result: {result}"

            logger.info(f"Episode ended for session {session_id}")

            # Return success result
            command_result = CommandResult.success_result(data=success_message)
            return self._convert_to_mcp_result(command_result)

        except Exception as e:
            logger.error(f"Error handling end_episode tool call: {e}")
            return self._convert_to_mcp_result(CommandResult.error_result(error=f"Failed to end episode: {str(e)}"))

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
        # Extract arguments for the tool (command, code, etc.)
        tool_arguments = arguments.get("arguments", "")

        # Extract nested parameters, or use empty dict if not provided
        # Filter out session context and arguments from parameters
        if "parameters" in arguments and isinstance(arguments["parameters"], dict):
            parameters = arguments["parameters"]
        else:
            # If no nested parameters, filter out known session/arguments fields
            parameters = {
                k: v for k, v in arguments.items() if k not in ["session_id", "client_id", "context", "arguments"]
            }

        # Add the arguments directly to parameters
        if tool_arguments:
            parameters["arguments"] = tool_arguments

        return Action(tool_name=tool_name, arguments=tool_arguments, parameters=parameters)

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
