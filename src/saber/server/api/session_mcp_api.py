"""
SessionMCPAPI implementation for SABER domain server.

The SessionMCPAPI handles Model Context Protocol server functionality,
providing tool discovery and tool execution only. All other operations
are handled by SessionRestAPI.

Logging category: MCP_API.
"""

import json

# Forward declaration to avoid circular imports
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import mcp.types as mcp_types
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

from ...logging_config import get_api_logger, log_operation_failure, log_operation_start, log_operation_success
from ...models import EvalSubmission, HTTPHeaders, OrchestrationEnvironment, RequestHeaders
from ...models.mcp import MCPToolCallResponse, MCPToolListResponse, MCPToolSchema
from ..base import Action, CommandResult
from .mcp_tool_generator import MCPToolGenerator

if TYPE_CHECKING:
    from ..session_manager import SessionManager

logger = get_api_logger(__name__, is_mcp=True)


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

        logger.info(
            "MCP API initialized",
            extra={
                "event": "mcp_api_initialized",
                "domain": session_manager.domain_name,
                "host": host,
                "port": port,
            },
        )

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
        available_headers: list[str] | None = None
        try:
            headers = get_http_headers()
            available_headers = list(headers.keys())

            # Extract session ID (optional)
            session_id = next(
                (str(value) for name, value in headers.items() if name.lower() == HTTPHeaders.SESSION_ID.lower()),
                None,
            )

            # Extract episode ID (optional)
            episode_id = next(
                (str(value) for name, value in headers.items() if name.lower() == HTTPHeaders.EPISODE_ID.lower()),
                None,
            )

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
            logger.debug(
                "Resolved orchestration environment",
                extra={
                    "event": "mcp_orchestration_env_resolved",
                    "orchestration_env": orchestration_env.value,
                },
            )

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

            logger.debug(
                "Parsed MCP headers",
                extra={
                    "event": "mcp_headers_parsed",
                    "session_id": request_headers.session_id,
                    "episode_id": request_headers.episode_id,
                    "orchestration_env": request_headers.orchestration_env.value,
                    "task_id": request_headers.task_id,
                    "client_id": request_headers.client_id,
                },
            )
            return request_headers

        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as exc:
            extras: Dict[str, Any] = {"event": "mcp_headers_parse_failed"}
            if available_headers is not None:
                extras["available_headers"] = available_headers
            log_operation_failure(
                logger,
                "parse_mcp_headers",
                exc,
                **extras,
            )
            raise ValueError(f"Failed to parse request headers: {exc}") from exc

    async def start_mcp_server(self) -> None:
        """Start the MCP server."""
        try:
            log_operation_start(
                logger,
                "start_mcp_server",
                domain=self.session_manager.domain_name,
                host=self.host,
                port=self.port,
            )
            self.mcp_server = FastMCP(f"SABER-{self.session_manager.domain_name}-MCP")

            # Register MCP handlers
            self._setup_mcp_handlers()

            # Use HTTP transport for production deployment (exposes /mcp endpoint)
            await self.mcp_server.run_async(transport="http", host=self.host, port=self.port)

            log_operation_success(
                logger,
                "start_mcp_server",
                domain=self.session_manager.domain_name,
                host=self.host,
                port=self.port,
            )
        except Exception as e:
            log_operation_failure(
                logger,
                "start_mcp_server",
                e,
                domain=self.session_manager.domain_name,
                host=self.host,
                port=self.port,
            )
            raise

    async def shutdown_mcp_server(self) -> None:
        """Shutdown the MCP server."""
        try:
            if self.mcp_server:
                # FastMCP doesn't require explicit cleanup
                self.mcp_server = None
            log_operation_success(
                logger,
                "shutdown_mcp_server",
                domain=self.session_manager.domain_name,
            )

        except Exception as e:
            log_operation_failure(
                logger,
                "shutdown_mcp_server",
                e,
                domain=self.session_manager.domain_name,
            )

    def _setup_mcp_handlers(self) -> None:
        """Setup MCP protocol handlers."""
        if not self.mcp_server:
            raise RuntimeError("MCP server not initialized")

        # Get available executors and register tools for each
        available_executors = self.session_manager.execution_manager.get_available_executors()
        logger.info(
            "Registering MCP tools",
            extra={
                "event": "mcp_register_tools",
                "executor_count": len(available_executors),
                "executors": available_executors,
            },
        )

        # Dynamically register a tool for each executor
        for executor_name in available_executors:
            self._register_executor_tool(executor_name)

        # Register hardcoded MCP API tools
        self._register_hardcoded_tools()

        # Override FastMCP's list_tools handler with episode-specific filtering
        self._override_list_tools_handler()

    def _register_hardcoded_tools(self) -> None:
        """Register hardcoded MCP API tools with the FastMCP server."""
        if not self.mcp_server:
            raise RuntimeError("MCP server not initialized")

        # Register end_episode tool using @decorator syntax
        @self.mcp_server.tool  # type: ignore[misc]
        async def end_episode(
            submission: str = "",
            __saber_assistant_message__: str | None = None,
            __saber_reasoning__: str | None = None,
        ) -> str:
            """End the current episode and optionally record a discovered flag/target/objective."""
            try:
                # Get parsed headers
                headers = await self._get_headers()
                if not headers.has_session_context:
                    logger.warning(
                        "end_episode called without session context",
                        extra={"event": "end_episode_no_session"},
                    )
                    return json.dumps(
                        {
                            "success": False,
                            "error": "No session context available. Agent may have failed during initialization.",
                        }
                    )
                if not headers.has_episode_context:
                    logger.warning(
                        "end_episode called without episode context",
                        extra={
                            "event": "end_episode_no_episode",
                            "session_id": headers.session_id,
                        },
                    )
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                "No episode context available. The episode may have failed to be created, "
                                "or the sample failed early."
                            ),
                        }
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

            except Exception as exc:
                log_operation_failure(
                    logger,
                    "mcp_end_episode_tool",
                    exc,
                )
                return json.dumps({"success": False, "error": str(exc)})

        logger.debug(
            "Registered hardcoded MCP tool",
            extra={"event": "mcp_tool_registered", "tool_name": "end_episode"},
        )

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

            # Add SABER context parameters to the schema
            # These are optional parameters that the client can inject with assistant messages and reasoning
            from saber.models.mcp import MCPPropertySchema

            # Create new properties dict with existing properties plus our context parameters
            updated_properties = dict(input_schema.properties or {})
            updated_properties["__saber_assistant_message__"] = MCPPropertySchema(
                type="string",
                description="Optional: Agent's assistant message for context",
            )
            updated_properties["__saber_reasoning__"] = MCPPropertySchema(
                type="string",
                description="Optional: Agent's reasoning content for context",
            )

            # Create a new schema with updated properties
            input_schema = input_schema.model_copy(update={"properties": updated_properties})

            # Build the typed MCP tool schema
            mcp_tool_schema = MCPToolSchema(
                name=metadata.get("name", executor_name),
                description=metadata.get("description", f"{executor_name.title()} executor"),
                inputSchema=input_schema,
            )

            # Validate the typed schema
            if not self.tool_generator.validate_mcp_tool_schema(mcp_tool_schema):
                logger.error(
                    "Invalid MCP schema for executor",
                    extra={
                        "event": "mcp_executor_schema_invalid",
                        "executor_name": executor_name,
                    },
                )
                return

            # Generate the dynamic tool function using typed schema
            executor_tool = self.tool_generator.create_executor_tool(
                executor_name=executor_name, mcp_schema=mcp_tool_schema, handler_func=self.handle_call_tool
            )

            # Register the tool with the MCP server
            if self.mcp_server is not None:
                self.mcp_server.tool(name=executor_name)(executor_tool)

            logger.debug(
                "Registered MCP tool",
                extra={
                    "event": "mcp_tool_registered",
                    "tool_name": executor_name,
                },
            )

        except Exception as exc:
            log_operation_failure(
                logger,
                "register_mcp_executor_tool",
                exc,
                executor_name=executor_name,
            )
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
            executor_tools: List[mcp_types.Tool] = self.session_manager.execution_manager.to_mcp_tools(
                headers.episode_id
            )

            # Only add hardcoded MCP API tools for STANDALONE orchestration
            # For INSPECT orchestration, we rely on the framework's native capabilities
            hardcoded_tools_data = []
            if headers.orchestration_env == OrchestrationEnvironment.STANDALONE:
                hardcoded_tools_data = [
                    mcp_types.Tool(
                        name="end_episode",
                        description="End the current episode and optionally record a discovered flag/target/objective",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "submission": {
                                    "type": "string",
                                    "description": "Optional flag, target, or objective discovered during episode",
                                    "default": "",  # Use empty string as default
                                }
                            },
                            "required": [],  # Keep as optional since it has a default
                        },
                    )
                ]

            # Combine executor tools and hardcoded tools (if any)
            mcp_tools = executor_tools + hardcoded_tools_data

            logger.debug(
                "Returning MCP tools",
                extra={
                    "event": "mcp_tools_listed",
                    "total_tools": len(mcp_tools),
                    "executor_tool_count": len(executor_tools),
                    "hardcoded_tool_count": len(hardcoded_tools_data),
                    "session_id": headers.session_id,
                    "episode_id": headers.episode_id,
                    "orchestration_env": headers.orchestration_env.value,
                },
            )

            return MCPToolListResponse(tools=mcp_tools, session_id=headers.session_id, episode_id=headers.episode_id)

        except Exception as exc:
            log_operation_failure(
                logger,
                "list_mcp_tools",
                exc,
            )
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
        # DEBUG: Log incoming MCP call
        logger.info(
            f"🔍 SERVER: MCP tool call received: {name}",
            extra={
                "event": "mcp_tool_call_received",
                "tool_name": name,
                "argument_keys": list(arguments.keys()),
                "has_saber_assistant": "__saber_assistant_message__" in arguments,
                "has_saber_reasoning": "__saber_reasoning__" in arguments,
                "full_arguments": arguments,
            },
        )

        # Get parsed headers first
        headers = await self._get_headers()
        if not headers.has_session_context:
            logger.warning(
                "Tool call attempted without session context",
                extra={
                    "event": "mcp_tool_call_no_session",
                    "tool_name": name,
                },
            )
            return MCPToolCallResponse(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Error: No SABER session available. This typically indicates "
                            "the agent failed to initialize properly."
                        ),
                    }
                ],
                isError=True,
            )

        # Episode ID is required for multi-episode architecture
        if not headers.has_episode_context:
            logger.warning(
                "Tool call attempted without episode context",
                extra={
                    "event": "mcp_tool_call_no_episode",
                    "tool_name": name,
                    "session_id": headers.session_id,
                },
            )
            return MCPToolCallResponse(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Error: No episode context available. This usually means the sample failed "
                            "before the episode could be created, or the agent failed during initialization."
                        ),
                    }
                ],
                isError=True,
            )

        # At this point, session_id and episode_id are guaranteed to be non-None
        assert headers.session_id is not None
        assert headers.episode_id is not None

        # Get episode for validation
        episode = self.session_manager.get_episode_by_id(headers.episode_id)
        if not episode:
            logger.warning(
                "Tool call attempted with non-existent episode",
                extra={
                    "event": "mcp_tool_call_episode_not_found",
                    "tool_name": name,
                    "session_id": headers.session_id,
                    "episode_id": headers.episode_id,
                },
            )
            return MCPToolCallResponse(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Error: Episode {headers.episode_id} not found. "
                            "The episode may have been terminated or cleaned up."
                        ),
                    }
                ],
                isError=True,
            )

        try:
            # Execute action through SessionManager with explicit episode_id
            action = self._convert_to_action(name, arguments)
            command_result = await self.session_manager.execute_action(headers.session_id, headers.episode_id, action)

            # Convert result to MCP format using typed response
            return self._convert_to_mcp_result(command_result)

        except Exception as exc:
            # Provide more helpful error context based on exception type
            error_context = ""
            if "timeout" in str(exc).lower():
                error_context = " (timeout occurred)"
            elif "permission" in str(exc).lower() or "denied" in str(exc).lower():
                error_context = " (permission denied)"
            elif "connection" in str(exc).lower():
                error_context = " (connection error)"
            elif "not found" in str(exc).lower():
                error_context = " (resource not found)"

            log_operation_failure(
                logger,
                "call_mcp_tool",
                exc,
                tool_name=name,
                session_id=headers.session_id,
                episode_id=headers.episode_id,
                error_context=error_context,
            )
            return MCPToolCallResponse(
                content=[{"type": "text", "text": f"Error: Tool execution failed{error_context}: {exc}"}], isError=True
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
                logger.warning(
                    "_handle_end_episode_call called without session_id",
                    extra={"event": "handle_end_episode_no_session"},
                )
                return MCPToolCallResponse(
                    content=[
                        {
                            "type": "text",
                            "text": "Error: No session context available. Agent initialization may have failed.",
                        }
                    ],
                    isError=True,
                )

            if not episode_id:
                logger.warning(
                    "_handle_end_episode_call called without episode_id",
                    extra={
                        "event": "handle_end_episode_no_episode",
                        "session_id": session_id,
                    },
                )
                return MCPToolCallResponse(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Error: No episode context available. The sample likely failed "
                                "before the episode could be created."
                            ),
                        }
                    ],
                    isError=True,
                )

            log_operation_start(
                logger,
                "mcp_end_episode",
                session_id=session_id,
                episode_id=episode_id,
                orchestration_env=orchestration_env.value,
            )

            # Extract optional result/flag/objective from parameters.submission
            result = ""
            if "parameters" in arguments and isinstance(arguments["parameters"], dict):
                result = arguments["parameters"].get("submission", "")

            # If there's a result, record it as an action before ending the episode
            if result:
                logger.debug(
                    "Recording MCP submission before episode end",
                    extra={
                        "event": "mcp_end_episode_submission_recorded",
                        "session_id": session_id,
                        "episode_id": episode_id,
                    },
                )

                # Create an action to record the episode result
                result_action = Action(
                    tool_name="episode_result",
                    parameters={"submission": result, "episode_end": True},
                    reasoning=None,
                    assistant_message=None,
                )

                # Execute the action to record it
                await self.session_manager.execute_action(session_id, episode_id, result_action)

            # End the episode through SessionManager, passing the result
            if result:
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

            log_operation_success(
                logger,
                "mcp_end_episode",
                session_id=session_id,
                episode_id=episode_id,
                orchestration_env=orchestration_env.value,
                submission_present=bool(result),
            )

            # Return the submission as text for React agent to extract as the answer
            # The React agent expects result.text to contain the actual answer for scoring
            response_text = result if result else "No submission provided"

            # Return submission result with typed response
            return MCPToolCallResponse(content=[{"type": "text", "text": response_text}], isError=False)

        except Exception as exc:
            log_operation_failure(
                logger,
                "mcp_end_episode",
                exc,
                session_id=session_id,
                episode_id=episode_id,
                orchestration_env=orchestration_env.value,
            )
            return MCPToolCallResponse(
                content=[{"type": "text", "text": f"Error: Failed to end episode: {exc}"}], isError=True
            )

    def _convert_to_action(self, tool_name: str, arguments: Dict[str, Any]) -> Action:
        """
        Convert MCP tool call to Action object.

        Extracts both assistant message and reasoning from special parameters and stores them
        separately in the Action fields. These parameters are stripped from the tool
        parameters to keep execution clean.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments (may include __saber_assistant_message__ and __saber_reasoning__)

        Returns:
            Action object for execution with context extracted
        """
        # DEBUG: Log incoming arguments
        logger.info(
            f"🔍 SERVER: _convert_to_action called for {tool_name}",
            extra={
                "event": "convert_to_action_called",
                "tool_name": tool_name,
                "argument_keys": list(arguments.keys()),
                "has_saber_assistant": "__saber_assistant_message__" in arguments,
                "has_saber_reasoning": "__saber_reasoning__" in arguments,
            },
        )

        # Extract context if present (injected by context injection)
        assistant_message = arguments.get("__saber_assistant_message__")
        reasoning = arguments.get("__saber_reasoning__")

        # DEBUG: Log extraction results
        logger.info(
            f"🔍 SERVER: Extracted context - assistant_msg={len(assistant_message) if assistant_message else 0} chars, "
            f"reasoning={len(reasoning) if reasoning else 0} chars",
            extra={
                "event": "context_extracted",
                "has_assistant_message": assistant_message is not None,
                "has_reasoning": reasoning is not None,
                "assistant_preview": assistant_message[:100] if assistant_message else None,
                "reasoning_preview": reasoning[:100] if reasoning else None,
            },
        )

        # Filter out session_id and saber context parameters from arguments
        filtered_arguments = {
            k: v
            for k, v in arguments.items()
            if k not in ("session_id", "__saber_assistant_message__", "__saber_reasoning__")
        }

        action = Action(
            tool_name=tool_name, parameters=filtered_arguments, assistant_message=assistant_message, reasoning=reasoning
        )

        if assistant_message or reasoning:
            logger.info(
                f"✅ SERVER: Action created WITH context for {tool_name}",
                extra={
                    "event": "action_with_context",
                    "tool_name": tool_name,
                    "has_assistant_message": assistant_message is not None,
                    "has_reasoning": reasoning is not None,
                },
            )
        else:
            logger.warning(f"⚠️ SERVER: Action created WITHOUT context for {tool_name}")

        return action

    def _convert_to_mcp_result(self, command_result: CommandResult) -> MCPToolCallResponse:
        """
        Convert CommandResult to MCP-compatible result with direct access to execution data.

        Following SABER best practices: fail fast, explicit state, no defensive programming.
        Returns raw command execution data for direct client access.

        Args:
            command_result: Result from command execution

        Returns:
            MCPToolCallResponse with structured command execution data
        """
        # Build structured result with direct access to command data
        result_data = {
            "stdout": command_result.stdout,
            "stderr": command_result.stderr,
            "exit_code": command_result.exit_code,
        }

        # Add episode termination signal if present
        if command_result.metadata.get("episode_terminated"):
            result_data["episode_terminated"] = True
            result_data["termination_reason"] = command_result.metadata.get("termination_reason", "server_terminated")

        return MCPToolCallResponse(
            content=[{"type": "application/json", "data": result_data}], isError=not command_result.success
        )

    def _override_list_tools_handler(self) -> None:
        """Override FastMCP's default list_tools handler with episode-specific filtering."""
        try:
            if not self.mcp_server:
                raise RuntimeError("MCP server not initialized")

            # Access FastMCP's internal _mcp_server and override the list_tools handler
            if hasattr(self.mcp_server, "_mcp_server"):
                # Register our custom handler to override the default
                self.mcp_server._mcp_server.list_tools()(self._custom_list_tools_handler)
                logger.info(
                    "Overrode FastMCP list_tools handler",
                    extra={"event": "mcp_list_tools_override_enabled"},
                )
            else:
                logger.warning(
                    "FastMCP internal structure changed; list_tools override unavailable",
                    extra={"event": "mcp_list_tools_override_unavailable"},
                )

        except Exception as exc:
            log_operation_failure(
                logger,
                "override_mcp_list_tools_handler",
                exc,
            )
            logger.warning(
                "Tool discovery will fall back to global registration",
                extra={"event": "mcp_list_tools_override_fallback"},
            )

    async def _custom_list_tools_handler(self) -> List[mcp_types.Tool]:
        """
        Custom list_tools handler that provides episode-specific tool filtering.

        This method overrides FastMCP's default tool discovery to return only the tools
        that are available for the current episode context based on HTTP headers.

        Returns:
            List of mcp.types.Tool objects filtered by episode context
        """
        try:
            # Call our existing handle_list_tools method which has episode filtering logic
            response = await self.handle_list_tools()
            return response.tools

        except Exception as exc:
            log_operation_failure(
                logger,
                "custom_list_tools_handler",
                exc,
            )
            # Fall back to empty tools list on error
            return []
