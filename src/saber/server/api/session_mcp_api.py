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
from fastmcp.server.dependencies import get_http_headers

from ...base import MCPHeaders
from ..base import Action, CommandResult
from .mcp_tool_generator import MCPToolGenerator

logger = logging.getLogger(__name__)


class SessionMCPAPI:
    """
    Model Context Protocol handler for SessionManager.

    Handles ONLY tool discovery and tool execution via MCP protocol.
    All other operations (session management, episodes, policy) are handled by SessionRestAPI.
    """

    def __init__(self, session_manager: Any, host: str = "0.0.0.0", port: int = 8001) -> None:
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
        self.tool_generator = MCPToolGenerator()

        logger.info(f"SessionMCPAPI initialized for domain '{session_manager.domain_name}' on {host}:{port}")

    async def _get_session_from_headers(self) -> Optional[str]:
        """
        Get SABER session ID from HTTP headers.

        Returns:
            SABER session ID if found, None otherwise
        """
        try:
            headers = get_http_headers()

            # Try case-insensitive header lookup
            session_id_from_header = None
            for header_name, header_value in headers.items():
                if header_name.lower() == MCPHeaders.SESSION_ID.lower():
                    session_id_from_header = header_value
                    break

            if session_id_from_header:
                logger.info(f"✅ Found session_id in header: {session_id_from_header}")
                return str(session_id_from_header)
            else:
                logger.info(f"❌ No {MCPHeaders.SESSION_ID} header found. Available headers: {list(headers.keys())}")

            return None

        except Exception as e:
            logger.error(f"Error getting session from HTTP headers: {e}")
            return None

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
            logger.info("MCP server shutdown complete")

        except Exception as e:
            logger.error(f"Error during MCP server shutdown: {e}")

    def _setup_mcp_handlers(self) -> None:
        """Setup MCP protocol handlers."""
        if not self.mcp_server:
            raise RuntimeError("MCP server not initialized")

        # Get available executors and register tools for each
        available_executors = self.session_manager.execution_manager.get_available_executors()
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

        # Register end_episode tool using @decorator syntax
        @self.mcp_server.tool  # type: ignore[misc]
        async def end_episode(submission: Optional[str] = None) -> str:
            """End the current episode and optionally record a discovered flag/target/objective."""
            try:
                # Get session from HTTP headers
                session_id = await self._get_session_from_headers()
                if not session_id:
                    return json.dumps({"success": False, "error": "No SABER session mapped to MCP request"})

                # Build arguments for end episode call
                args = {}
                if submission:
                    args["parameters"] = {"submission": submission}

                mcp_result = await self._handle_end_episode_call(args, session_id)

                # Extract the text content from the MCP result
                if mcp_result.get("isError", False):
                    return json.dumps({"success": False, "error": mcp_result["content"][0]["text"]})
                else:
                    return str(mcp_result["content"][0]["text"])

            except Exception as e:
                logger.error(f"Error in end_episode tool: {e}")
                return json.dumps({"success": False, "error": str(e)})

        logger.debug("Registered hardcoded MCP tool: end_episode")

    def _register_executor_tool(self, executor_name: str) -> None:
        """
        Dynamically register an MCP tool for a specific executor.

        Args:
            executor_name: The name of the executor (e.g., 'cli', 'python')
        """
        try:
            # Get the executor instance to extract the MCP schema
            executor_instance = self.session_manager.execution_manager.get_executor(executor_name)
            input_schema = executor_instance.to_mcp_schema()
            metadata = getattr(executor_instance, "_executor_metadata", {})

            # Build the full MCP schema structure
            mcp_schema = {
                "name": metadata.get("name", executor_name),
                "description": metadata.get("description", f"{executor_name.title()} executor"),
                "inputSchema": input_schema,
            }

            # Validate the schema
            if not self.tool_generator.validate_mcp_schema(mcp_schema):
                logger.error(f"Invalid MCP schema for executor '{executor_name}'")
                return

            # Generate the dynamic tool function using templated code
            executor_tool = self.tool_generator.create_executor_tool(
                executor_name=executor_name, mcp_schema=mcp_schema, handler_func=self.handle_call_tool
            )

            # Register the tool with the MCP server
            if self.mcp_server is not None:
                self.mcp_server.tool(name=executor_name)(executor_tool)

            logger.debug(f"Registered MCP tool: {executor_name}")

        except Exception as e:
            logger.error(f"Failed to register MCP tool for executor '{executor_name}': {e}")
            raise

    async def handle_list_tools(self) -> List[Dict[str, Any]]:
        """
        Handle MCP tool discovery.

        Returns:
            List of available MCP tools from ExecutionManager plus hardcoded MCP tools
        """
        try:
            # Get tools from execution manager
            tools: List[Dict[str, Any]] = self.session_manager.execution_manager.to_mcp_tools()

            # Add hardcoded MCP API tools (without session_id in schema)
            hardcoded_tools = [
                {
                    "name": "end_episode",
                    "description": "End the current episode and optionally record a discovered flag/target/objective",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "submission": {
                                "type": "string",
                                "description": "Optional flag, target, or objective discovered during episode",
                            },
                        },
                        "required": [],
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
        Handle MCP tool execution using get_http_headers.

        Args:
            name: Tool name to execute
            arguments: Tool arguments (clean, no session_id)

        Returns:
            MCP-formatted execution result
        """
        try:
            # Get session from HTTP headers
            session_id = await self._get_session_from_headers()
            if not session_id:
                return self._convert_to_mcp_result(
                    CommandResult.error_result(error="No SABER session mapped to MCP request")
                )

            # Execute action through SessionManager
            action = self._convert_to_action(name, arguments)
            command_result = await self.session_manager.execute_action(session_id, action)

            # Convert result to MCP format
            return self._convert_to_mcp_result(command_result)

        except Exception as e:
            logger.error(f"Error handling call_tool {name}: {e}")
            return self._convert_to_mcp_result(CommandResult.error_result(error=f"Tool execution failed: {str(e)}"))

    async def _handle_end_episode_call(
        self, arguments: Dict[str, Any], session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle the hardcoded end_episode tool call.

        Args:
            arguments: Tool arguments (clean, no session_id)
            session_id: SABER session ID from Context (can be None for legacy support)

        Returns:
            MCP-formatted result confirming episode end
        """
        try:
            # If session_id not provided, try to get it from arguments or headers
            if not session_id:
                if "session_id" in arguments:
                    session_id = arguments["session_id"]
                else:
                    session_id = await self._get_session_from_headers()

            if not session_id:
                return self._convert_to_mcp_result(
                    CommandResult.error_result(error="No SABER session mapped to MCP request")
                )

            # Extract optional result/flag/objective from parameters.submission
            result = ""
            if "parameters" in arguments and isinstance(arguments["parameters"], dict):
                result = arguments["parameters"].get("submission", "")

            # If a result was provided, save it as an action in the episode
            if result:
                logger.info(f"Episode ending with result: {result}")
                # Create an action to record the discovered result
                result_action = Action(
                    tool_name="episode_result",
                    parameters={
                        "submission": result,
                        "episode_end": True,
                        "arguments": f"Episode completed with result: {result}",
                    },
                )
                # Execute the result action to save it in the episode
                await self.session_manager.execute_action(session_id, result_action)

            # End the episode through SessionManager
            logger.warning(f"🔥 MCP END EPISODE: Agent called end_episode tool for session {session_id}")
            self.session_manager.end_episode(session_id, "agent_completed")

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

    def _convert_to_action(self, tool_name: str, arguments: Dict[str, Any]) -> Action:
        """
        Convert MCP tool call to Action object.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments (clean, no session_id)

        Returns:
            Action object for execution
        """
        # Filter out session_id from arguments for the Action parameters
        filtered_arguments = {k: v for k, v in arguments.items() if k != "session_id"}
        return Action(tool_name=tool_name, parameters=filtered_arguments)

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
