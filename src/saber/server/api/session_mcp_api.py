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

from ...models import EvalSubmission, HTTPHeaders, OrchestrationEnvironment, RequestHeaders
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

    async def _get_headers(self) -> RequestHeaders:
        """
        Parse and validate all required headers from the HTTP request.

        This method extracts all SABER-specific headers and validates them,
        returning a standardized RequestHeaders object for use throughout
        the MCP request processing pipeline.

        Returns:
            RequestHeaders object with parsed and validated header values

        Raises:
            ValueError: If mandatory headers are missing or invalid
        """
        try:
            headers = get_http_headers()
            available_headers = list(headers.keys())

            # Extract session ID (optional)
            session_id = None
            for header_name, header_value in headers.items():
                if header_name.lower() == HTTPHeaders.SESSION_ID.lower():
                    session_id = str(header_value)
                    logger.info(f"✅ Found session_id in header: {session_id}")
                    break

            if not session_id:
                logger.info(f"❌ No {HTTPHeaders.SESSION_ID} header found. Available headers: {available_headers}")

            # Extract episode ID (optional)
            episode_id = None
            for header_name, header_value in headers.items():
                if header_name.lower() == HTTPHeaders.EPISODE_ID.lower():
                    episode_id = str(header_value)
                    logger.info(f"✅ Found episode_id in header: {episode_id}")
                    break

            if not episode_id:
                logger.debug(f"No {HTTPHeaders.EPISODE_ID} header found. Available headers: {available_headers}")

            # Extract orchestration environment (MANDATORY)
            orchestration_env_value = None
            for header_name, header_value in headers.items():
                if header_name.lower() == HTTPHeaders.ORCHESTRATION_ENV.lower():
                    orchestration_env_value = header_value
                    break

            if not orchestration_env_value:
                raise ValueError(
                    f"MANDATORY header {HTTPHeaders.ORCHESTRATION_ENV} not found. "
                    f"Available headers: {available_headers}. "
                    f"Valid values: {OrchestrationEnvironment.get_valid_values()}"
                )

            # Validate the orchestration environment value
            if not OrchestrationEnvironment.is_valid(orchestration_env_value):
                raise ValueError(
                    f"Invalid orchestration environment '{orchestration_env_value}' in header "
                    f"{HTTPHeaders.ORCHESTRATION_ENV}. Valid values: {OrchestrationEnvironment.get_valid_values()}"
                )

            orchestration_env = OrchestrationEnvironment(orchestration_env_value)
            logger.info(f"✅ Found orchestration_env in header: {orchestration_env}")

            # Extract task ID (optional)
            task_id = None
            for header_name, header_value in headers.items():
                if header_name.lower() == HTTPHeaders.TASK_ID.lower():
                    task_id = str(header_value)
                    break

            # Extract client ID (optional)
            client_id = None
            for header_name, header_value in headers.items():
                if header_name.lower() == HTTPHeaders.CLIENT_ID.lower():
                    client_id = str(header_value)
                    break

            # Create and return the RequestHeaders object
            request_headers = RequestHeaders(
                session_id=session_id,
                episode_id=episode_id,
                orchestration_env=orchestration_env,
                task_id=task_id,
                client_id=client_id,
            )

            logger.debug(f"Parsed request headers: {request_headers.context_summary}")
            return request_headers

        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            logger.error(f"Error parsing headers: {e}")
            raise ValueError(f"Failed to parse request headers: {e}") from e

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
                # FastMCP doesn't require explicit cleanup
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
                # Get parsed headers
                headers = await self._get_headers()
                if not headers.has_session_context:
                    return json.dumps({"success": False, "error": "No SABER session mapped to MCP request"})
                if not headers.has_episode_context:
                    return json.dumps(
                        {"success": False, "error": "No SABER episode ID in headers - episode context required"}
                    )

                # At this point, session_id and episode_id are guaranteed to be non-None
                assert headers.session_id is not None
                assert headers.episode_id is not None

                # Build arguments for end episode call
                args = {}
                if submission:
                    args["parameters"] = {"submission": submission}

                mcp_result = await self._handle_end_episode_call(
                    args, headers.session_id, headers.episode_id, headers.orchestration_env
                )

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
            # Get parsed headers
            headers = await self._get_headers()

            # Get tools from execution manager with episode context
            tools_data: List[Dict[str, Any]] = self.session_manager.execution_manager.to_mcp_tools(headers.episode_id)

            # Convert executor tools to typed MCPToolSchema objects
            executor_mcp_tools = [
                MCPToolSchema(name=tool["name"], description=tool["description"], inputSchema=tool["inputSchema"])
                for tool in tools_data
            ]

            # Only add hardcoded MCP API tools for STANDALONE orchestration
            # For INSPECT orchestration, we rely on the framework's native capabilities
            hardcoded_tools_data = []
            if headers.orchestration_env == OrchestrationEnvironment.STANDALONE:
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

            # Combine executor tools and hardcoded tools (if any)
            mcp_tools = executor_mcp_tools + hardcoded_tools_data

            logger.debug(
                f"Returning {len(mcp_tools)} tools ({len(tools_data)} executor tools + "
                f"{len(hardcoded_tools_data)} hardcoded tool{'s' if len(hardcoded_tools_data) != 1 else ''}) "
                f"for MCP discovery"
                f"{f' for episode {headers.episode_id}' if headers.has_episode_context else ' (no episode context)'}"
                f" {headers.context_summary}"
            )

            return MCPToolListResponse(tools=mcp_tools, session_id=headers.session_id, episode_id=headers.episode_id)

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
        # Get parsed headers first
        headers = await self._get_headers()
        if not headers.has_session_context:
            return MCPToolCallResponse(
                content=[{"type": "text", "text": "Error: No SABER session mapped to MCP request"}], isError=True
            )

        # Episode ID is required for multi-episode architecture
        if not headers.has_episode_context:
            return MCPToolCallResponse(
                content=[{"type": "text", "text": "Error: No SABER episode ID in headers - episode context required"}],
                isError=True,
            )

        # At this point, session_id and episode_id are guaranteed to be non-None
        assert headers.session_id is not None
        assert headers.episode_id is not None

        # Get task_id from the specific episode, not the session
        episode = self.session_manager.get_episode_by_id(headers.episode_id)
        if not episode:
            return MCPToolCallResponse(
                content=[{"type": "text", "text": f"Error: Episode {headers.episode_id} not found"}], isError=True
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
                    session_id=headers.session_id,
                    episode_id=headers.episode_id,
                    tool_name=name,
                    arguments=arguments,
                    call_id=call_id,
                    task_id=task_id,
                    current_step=current_step,
                    max_steps=max_steps,
                )
                logger.debug(
                    f"Published tool_call_started event: {name} (call_id={call_id}, "
                    f"task_id={task_id}, episode_id={headers.episode_id}, step={current_step}/{max_steps}, "
                    f"orchestration: {headers.orchestration_env})"
                )
            except Exception as e:
                logger.error(f"Failed to publish tool_call_started event: {e}")
                # Continue execution - event publishing failure shouldn't break tool execution

        try:
            # Execute action through SessionManager with explicit episode_id
            action = self._convert_to_action(name, arguments)
            command_result = await self.session_manager.execute_action(headers.session_id, headers.episode_id, action)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Publish tool completed event
            tool_event_publisher = self.session_manager.get_tool_event_publisher()
            if tool_event_publisher:
                try:
                    # Get updated step info after tool execution from the specific episode
                    updated_episode = self.session_manager.get_episode_by_id(headers.episode_id)
                    completed_step = len(updated_episode.steps) if updated_episode else current_step

                    await tool_event_publisher.publish_tool_completed(
                        session_id=headers.session_id,
                        episode_id=headers.episode_id,
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
                        f"task_id={task_id}, episode_id={headers.episode_id}, step={completed_step}/{max_steps}, "
                        f"orchestration: {headers.orchestration_env})"
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
                        session_id=headers.session_id,
                        episode_id=headers.episode_id,
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
                        f"(task_id={task_id}, episode_id={headers.episode_id}, step={current_step}/{max_steps}, "
                        f"orchestration: {headers.orchestration_env})"
                    )
                except Exception as pub_error:
                    logger.error(f"Failed to publish tool_call_completed (failed) event: {pub_error}")

            logger.error(f"Error handling call_tool {name}: {e}")
            return MCPToolCallResponse(
                content=[{"type": "text", "text": f"Error: Tool execution failed: {str(e)}"}], isError=True
            )

    async def _handle_end_episode_call(
        self, arguments: Dict[str, Any], session_id: str, episode_id: str, orchestration_env: OrchestrationEnvironment
    ) -> MCPToolCallResponse:
        """
        Handle the hardcoded end_episode tool call.

        This function processes the agent's submission by:
        1. Recording the submission as a step in the episode
        2. Ending the episode with the submission for evaluation
        3. Returning the submission as text for React agent scoring

        The React agent expects the tool result text to contain the actual answer.

        Args:
            arguments: Tool arguments containing submission parameter (clean, no session_id)
            session_id: SABER session ID from headers
            episode_id: SABER episode ID from headers
            orchestration_env: Orchestration environment from headers

        Returns:
            MCPToolCallResponse with submission as text content
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
            logger.info(
                f"Handling end_episode call for session {session_id}, episode {episode_id}, arguments: {arguments} "
                f"[orchestration: {orchestration_env}]"
            )
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
                f"🔥 MCP END EPISODE: Agent called end_episode tool for session {session_id}, episode {episode_id} "
                f"[orchestration: {orchestration_env}]"
            )
            if result:
                logger.info(f"Episode ending with result: {result}")
                # Create EvalSubmission object from MCP string submission
                mcp_submission = EvalSubmission(
                    episode_id=episode_id,
                    task_id="unknown",  # MCP doesn't provide task_id context
                    model="mcp_agent",
                    choices=[],
                    submission=result,
                    tokens={},
                    time=0.0,
                )
                await self.session_manager.end_episode(session_id, episode_id, "agent_completed", mcp_submission)
            else:
                await self.session_manager.end_episode(session_id, episode_id, "agent_completed")

            logger.info(f"Episode ended for session {session_id}")

            # Return the submission as text for React agent to extract as the answer
            # The React agent expects result.text to contain the actual answer for scoring
            response_text = result if result else "No submission provided"

            # Return submission result with typed response
            return MCPToolCallResponse(content=[{"type": "text", "text": response_text}], isError=False)

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
