"""
SessionMCPAPI implementation for SABER domain server.

The SessionMCPAPI handles Model Context Protocol server functionality,
providing tool discovery and tool execution only. All other operations
are handled by SessionRestAPI.
"""

import json
import logging
import time
import uuid

# Forward declaration to avoid circular imports
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

from ...models import HTTPHeaders
from ...models.mcp import MCPInputSchema, MCPPropertySchema, MCPToolCallResponse, MCPToolListResponse, MCPToolSchema
from ..base import Action, CommandResult
from .mcp_tool_generator import MCPToolGenerator

if TYPE_CHECKING:
    from ..session_manager import SessionManager

logger = logging.getLogger(__name__)


class SessionMCPAPI:
    """
    Model Context Protocol handler for SessionManager.

    Handles ONLY tool discovery and tool execution via MCP protocol.
    All other operations (session management, episodes, policy) are handled by SessionRestAPI.
    """

    def __init__(self, session_manager: "SessionManager", host: str = "0.0.0.0", port: int = 8001) -> None:
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
                if header_name.lower() == HTTPHeaders.SESSION_ID.lower():
                    session_id_from_header = header_value
                    break

            if session_id_from_header:
                logger.info(f"✅ Found session_id in header: {session_id_from_header}")
                return str(session_id_from_header)
            else:
                logger.info(f"❌ No {HTTPHeaders.SESSION_ID} header found. Available headers: {list(headers.keys())}")

            return None

        except Exception as e:
            logger.error(f"Error getting session from HTTP headers: {e}")
            return None

    async def _get_episode_from_headers(self) -> Optional[str]:
        """
        Get SABER episode ID from HTTP headers.

        Returns:
            SABER episode ID if found, None otherwise
        """
        try:
            headers = get_http_headers()

            # Try case-insensitive header lookup
            episode_id_from_header = None
            for header_name, header_value in headers.items():
                if header_name.lower() == HTTPHeaders.EPISODE_ID.lower():
                    episode_id_from_header = header_value
                    break

            if episode_id_from_header:
                logger.info(f"✅ Found episode_id in header: {episode_id_from_header}")
                return str(episode_id_from_header)
            else:
                logger.debug(f"No {HTTPHeaders.EPISODE_ID} header found. Available headers: {list(headers.keys())}")

            return None

        except Exception as e:
            logger.error(f"Error getting episode from HTTP headers: {e}")
            return None

    async def _get_session_and_episode_from_headers(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get both SABER session ID and episode ID from HTTP headers.

        Returns:
            Tuple of (session_id, episode_id) if found, (None, None) otherwise
        """
        session_id = await self._get_session_from_headers()
        episode_id = await self._get_episode_from_headers()
        return session_id, episode_id

    async def start_mcp_server(self) -> None:
        """Start the MCP server."""
        try:
            self.mcp_server = FastMCP(f"SABER-{self.session_manager.domain_name}-MCP")

            # Register MCP handlers
            self._setup_mcp_handlers()

            logger.info(f"Starting SABER {self.session_manager.domain_name} MCP server on {self.host}:{self.port}")

            # Use HTTP transport for production deployment (exposes /mcp endpoint)
            await self.mcp_server.run_async(transport="http", host=self.host, port=self.port)

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise

    async def shutdown_mcp_server(self) -> None:
        """Shutdown the MCP server."""
        try:
            if self.mcp_server:
                await self.mcp_server.close()
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
        async def end_episode(submission: str = "") -> str:
            """End the current episode and optionally record a discovered flag/target/objective."""
            try:
                # Get session and episode from HTTP headers
                session_id, episode_id = await self._get_session_and_episode_from_headers()
                if not session_id:
                    return json.dumps({"success": False, "error": "No SABER session mapped to MCP request"})
                if not episode_id:
                    return json.dumps(
                        {"success": False, "error": "No SABER episode ID in headers - episode context required"}
                    )

                # Build arguments for end episode call
                args = {}
                if submission:
                    args["parameters"] = {"submission": submission}

                mcp_result = await self._handle_end_episode_call(args, session_id, episode_id)

                # Extract the text content from the typed MCP result
                if mcp_result.isError:
                    return json.dumps({"success": False, "error": mcp_result.content[0]["text"]})
                else:
                    return str(mcp_result.content[0]["text"])

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
            # Get the executor instance to extract the MCP schema (no episode context during registration)
            executor_instance = self.session_manager.execution_manager.get_executor(executor_name, episode_id=None)
            input_schema = executor_instance.to_mcp_schema()
            metadata = getattr(executor_instance, "_executor_metadata", {})

            # Build the typed MCP tool schema
            mcp_tool_schema = MCPToolSchema(
                name=metadata.get("name", executor_name),
                description=metadata.get("description", f"{executor_name.title()} executor"),
                inputSchema=input_schema,
            )

            # Validate the typed schema
            if not self.tool_generator.validate_mcp_tool_schema(mcp_tool_schema):
                logger.error(f"Invalid MCP schema for executor '{executor_name}'")
                return

            # Generate the dynamic tool function using typed schema
            executor_tool = self.tool_generator.create_executor_tool(
                executor_name=executor_name, mcp_schema=mcp_tool_schema, handler_func=self.handle_call_tool
            )

            # Register the tool with the MCP server
            if self.mcp_server is not None:
                self.mcp_server.tool(name=executor_name)(executor_tool)

            logger.debug(f"Registered MCP tool: {executor_name}")

        except Exception as e:
            logger.error(f"Failed to register MCP tool for executor '{executor_name}': {e}")
            raise

    async def handle_list_tools(self) -> MCPToolListResponse:
        """
        Handle MCP tool discovery with episode-specific context.

        Returns:
            MCPToolListResponse with available MCP tools from ExecutionManager filtered by episode's
            allowed executors plus hardcoded MCP tools
        """
        try:
            # Get episode context from headers
            session_id, episode_id = await self._get_session_and_episode_from_headers()

            # Get tools from execution manager with episode context
            tools_data: List[Dict[str, Any]] = self.session_manager.execution_manager.to_mcp_tools(episode_id)

            # Add hardcoded MCP API tools (without session_id in schema)
            hardcoded_tools_data = [
                MCPToolSchema(
                    name="end_episode",
                    description="End the current episode and optionally record a discovered flag/target/objective",
                    inputSchema=MCPInputSchema(
                        type="object",
                        properties={
                            "submission": MCPPropertySchema(
                                type="string",
                                description="Optional flag, target, or objective discovered during episode",
                                title=None,
                                default="",  # Use empty string instead of None
                                enum=None,
                                minimum=None,
                                maximum=None,
                                pattern=None,
                            )
                        },
                        required=[],  # Keep as optional since it has a default
                    ),
                )
            ]

            # Convert executor tools to typed MCPToolSchema objects
            executor_mcp_tools = [
                MCPToolSchema(name=tool["name"], description=tool["description"], inputSchema=tool["inputSchema"])
                for tool in tools_data
            ]

            # Combine executor tools and hardcoded tools (hardcoded_tools_data already contains MCPToolSchema objects)
            mcp_tools = executor_mcp_tools + hardcoded_tools_data

            logger.debug(
                f"Returning {len(mcp_tools)} tools ({len(tools_data)} executor tools + "
                f"{len(hardcoded_tools_data)} hardcoded tool) for MCP discovery"
                f"{f' for episode {episode_id}' if episode_id else ' (no episode context)'}"
            )

            return MCPToolListResponse(tools=mcp_tools, session_id=session_id, episode_id=episode_id)

        except Exception as e:
            logger.error(f"Error handling list_tools: {e}")
            return MCPToolListResponse(tools=[], session_id=None, episode_id=None)

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolCallResponse:
        """
        Handle MCP tool execution using get_http_headers.

        Args:
            name: Tool name to execute
            arguments: Tool arguments (clean, no session_id)

        Returns:
            MCPToolCallResponse with execution result
        """
        # Get session and episode from HTTP headers first
        session_id, episode_id = await self._get_session_and_episode_from_headers()
        if not session_id:
            return MCPToolCallResponse(
                content=[{"type": "text", "text": "Error: No SABER session mapped to MCP request"}], isError=True
            )

        # Episode ID is required for multi-episode architecture
        if not episode_id:
            return MCPToolCallResponse(
                content=[{"type": "text", "text": "Error: No SABER episode ID in headers - episode context required"}],
                isError=True,
            )

        # Get task_id from the specific episode, not the session
        episode = self.session_manager.get_episode_by_id(episode_id)
        if not episode:
            return MCPToolCallResponse(
                content=[{"type": "text", "text": f"Error: Episode {episode_id} not found"}], isError=True
            )
        task_id = episode.task_id

        # Get step information for progress tracking from the specific episode
        current_step = len(episode.steps) + 1  # +1 because we're about to start a new step
        max_steps = episode.max_steps

        # Generate unique call ID for tracking
        call_id = str(uuid.uuid4())
        start_time = time.time()

        # Publish tool started event with task_id and step info
        tool_event_publisher = self.session_manager.get_tool_event_publisher()
        if tool_event_publisher:
            try:
                await tool_event_publisher.publish_tool_started(
                    session_id=session_id,
                    episode_id=episode_id,
                    tool_name=name,
                    arguments=arguments,
                    call_id=call_id,
                    task_id=task_id,
                    current_step=current_step,
                    max_steps=max_steps,
                )
                logger.debug(
                    f"Published tool_call_started event: {name} (call_id={call_id}, "
                    f"task_id={task_id}, episode_id={episode_id}, step={current_step}/{max_steps})"
                )
            except Exception as e:
                logger.error(f"Failed to publish tool_call_started event: {e}")
                # Continue execution - event publishing failure shouldn't break tool execution

        try:
            # Execute action through SessionManager with explicit episode_id
            action = self._convert_to_action(name, arguments)
            command_result = await self.session_manager.execute_action(session_id, episode_id, action)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Publish tool completed event
            tool_event_publisher = self.session_manager.get_tool_event_publisher()
            if tool_event_publisher:
                try:
                    # Get updated step info after tool execution from the specific episode
                    updated_episode = self.session_manager.get_episode_by_id(episode_id)
                    completed_step = len(updated_episode.steps) if updated_episode else current_step

                    await tool_event_publisher.publish_tool_completed(
                        session_id=session_id,
                        episode_id=episode_id,
                        tool_name=name,
                        call_id=call_id,
                        success=command_result.success,
                        arguments=arguments,
                        result=(
                            command_result.data.get("output")
                            if command_result.success and command_result.data
                            else command_result.stdout
                        ),
                        error=command_result.error if not command_result.success else None,
                        execution_time_ms=execution_time_ms,
                        task_id=task_id,
                        current_step=completed_step,
                        max_steps=max_steps,
                    )
                    logger.debug(
                        f"Published tool_call_completed event: {name} "
                        f"({'success' if command_result.success else 'failed'}, "
                        f"task_id={task_id}, episode_id={episode_id}, step={completed_step}/{max_steps})"
                    )
                except Exception as e:
                    logger.error(f"Failed to publish tool_call_completed event: {e}")

            # Convert result to MCP format using typed response
            return self._convert_to_mcp_result(command_result)

        except Exception as e:
            # Calculate execution time for failed call
            execution_time_ms = (time.time() - start_time) * 1000

            # Publish tool failed event
            tool_event_publisher = self.session_manager.get_tool_event_publisher()
            if tool_event_publisher:
                try:
                    await tool_event_publisher.publish_tool_completed(
                        session_id=session_id,
                        episode_id=episode_id,
                        tool_name=name,
                        call_id=call_id,
                        success=False,
                        arguments=arguments,  # Include the original arguments
                        result=None,
                        error=str(e),
                        execution_time_ms=execution_time_ms,
                        task_id=task_id,
                        current_step=current_step,
                        max_steps=max_steps,
                    )
                    logger.debug(
                        f"Published tool_call_completed (failed) event: {name} "
                        f"(task_id={task_id}, episode_id={episode_id}, step={current_step}/{max_steps})"
                    )
                except Exception as pub_error:
                    logger.error(f"Failed to publish tool_call_completed (failed) event: {pub_error}")

            logger.error(f"Error handling call_tool {name}: {e}")
            return MCPToolCallResponse(
                content=[{"type": "text", "text": f"Error: Tool execution failed: {str(e)}"}], isError=True
            )

    async def _handle_end_episode_call(
        self, arguments: Dict[str, Any], session_id: str, episode_id: str
    ) -> MCPToolCallResponse:
        """
        Handle the hardcoded end_episode tool call.

        Args:
            arguments: Tool arguments (clean, no session_id)
            session_id: SABER session ID from headers
            episode_id: SABER episode ID from headers

        Returns:
            MCPToolCallResponse confirming episode end
        """
        try:
            if not session_id:
                return MCPToolCallResponse(
                    content=[{"type": "text", "text": "Error: No SABER session mapped to MCP request"}], isError=True
                )

            if not episode_id:
                return MCPToolCallResponse(
                    content=[{"type": "text", "text": "Error: No SABER episode ID - episode context required"}],
                    isError=True,
                )

            # Extract optional result/flag/objective from parameters.submission
            result = ""
            if "parameters" in arguments and isinstance(arguments["parameters"], dict):
                result = arguments["parameters"].get("submission", "")

            # If there's a result, record it as an action before ending the episode
            if result:
                logger.info(f"Recording episode result: {result}")

                # Create an action to record the episode result
                result_action = Action(
                    tool_name="episode_result", parameters={"submission": result, "episode_end": True}
                )

                # Execute the action to record it
                await self.session_manager.execute_action(session_id, episode_id, result_action)

            # End the episode through SessionManager, passing the result
            logger.warning(
                f"🔥 MCP END EPISODE: Agent called end_episode tool for session {session_id}, episode {episode_id}"
            )
            if result:
                logger.info(f"Episode ending with result: {result}")
                await self.session_manager.end_episode(session_id, episode_id, "agent_completed", result)
            else:
                await self.session_manager.end_episode(session_id, episode_id, "agent_completed")

            # Prepare success message
            success_message = "Episode ended successfully"
            if result:
                success_message += f" with result: {result}"

            logger.info(f"Episode ended for session {session_id}")

            # Return success result with typed response
            return MCPToolCallResponse(content=[{"type": "text", "text": success_message}], isError=False)

        except Exception as e:
            logger.error(f"Error handling end_episode tool call: {e}")
            return MCPToolCallResponse(
                content=[{"type": "text", "text": f"Error: Failed to end episode: {str(e)}"}], isError=True
            )

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

    def _convert_to_mcp_result(self, command_result: CommandResult) -> MCPToolCallResponse:
        """
        Convert CommandResult to MCP-compatible result with episode termination signals.

        Args:
            command_result: Result from command execution

        Returns:
            MCPToolCallResponse with optional termination metadata
        """
        if command_result.success:
            # Check if we need to add episode termination signals
            result_text = str(command_result.data) if command_result.data else "Command executed successfully"

            # Add episode termination signal if present in command result metadata
            if hasattr(command_result, "metadata") and command_result.metadata:
                if command_result.metadata.get("episode_terminated"):
                    termination_reason = command_result.metadata.get("termination_reason", "server_terminated")
                    result_text += f"\n[EPISODE_TERMINATED: {termination_reason}]"

            return MCPToolCallResponse(content=[{"type": "text", "text": result_text}], isError=False)
        else:
            return MCPToolCallResponse(
                content=[{"type": "text", "text": f"Error: {command_result.error}"}], isError=True
            )
