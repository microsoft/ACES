"""
SABER Agent Base Infrastructure

Provides common SABER functionality that all agents need:
- Episode management with tool call limits from server
- Dynamic prompt fetching from policy endpoint
- MCP tool discovery and execution
- Session and context management
- Fail-fast error handling

Following SABER best practices:
- No backwards compatibility
- Fail-fast design
- Type-safe implementation
- Async context management
"""

import logging
from typing import Any, List, Optional

from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageSystem
from inspect_ai.tool import Tool
from inspect_ai.util import store

from ...models import EpisodeCreateResponse
from ...models.mcp import MCPToolCallRequest, MCPToolSchema
from ..client_session import ClientSessionManager

logger = logging.getLogger(__name__)


class SABERAgentError(Exception):
    """Base exception for SABER agent errors."""

    pass


class SABERAgentContext:
    """
    SABER agent context manager for common infrastructure.

    Handles:
    - Episode creation and management
    - MCP tool discovery and conversion
    - Dynamic prompt fetching
    - Session lifecycle management
    - Fail-fast error handling
    """

    def __init__(
        self,
        session_manager: ClientSessionManager,
        agent_id: str,
        default_prompt: str = "You are a security domain agent.",
    ):
        """Initialize SABER agent context.

        Args:
            session_manager: SABER session manager
            agent_id: Agent identifier
            default_prompt: Default prompt if policy fetch fails
        """
        self.session_manager = session_manager
        self.agent_id = agent_id
        self.default_prompt = default_prompt
        self.episode: Optional[EpisodeCreateResponse] = None
        self.enhanced_prompt: str = default_prompt
        self.mcp_tools: List[Tool] = []

    async def __aenter__(self) -> "SABERAgentContext":
        """Initialize SABER context and episode."""
        await self._initialize_saber_context()
        await self._fetch_dynamic_prompt()
        await self._discover_mcp_tools()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup SABER context and end episode."""
        if self.episode and self.session_manager:
            try:
                # Determine result based on exception
                if exc_type is None:
                    result = "success"
                    reason = "completed"
                else:
                    result = str(exc_val) if exc_val else "error"
                    reason = "error"

                await self.session_manager.end_episode(
                    self.episode.session_id, self.episode.episode_id, reason=reason, result=result
                )
            except Exception as e:
                logger.error(f"Failed to end episode: {e}")
                # Don't raise - cleanup failures shouldn't break the main flow

    def get_enhanced_state(self, state: AgentState) -> AgentState:
        """Enhance state with SABER prompt if available.

        Args:
            state: Original agent state

        Returns:
            Enhanced agent state with SABER prompt
        """
        if self.enhanced_prompt == self.default_prompt:
            return state  # No enhancement needed

        enhanced_messages = list(state.messages)
        enhanced_messages.insert(0, ChatMessageSystem(content=self.enhanced_prompt))
        enhanced_state = AgentState(messages=enhanced_messages)
        enhanced_state.output = state.output
        return enhanced_state

    def get_all_tools(self, base_tools: List[Tool]) -> List[Tool]:
        """Combine base tools with discovered MCP tools.

        Args:
            base_tools: Base inspect_ai tools

        Returns:
            Combined tool list
        """
        return list(base_tools) + self.mcp_tools

    async def _initialize_saber_context(self) -> None:
        """Initialize SABER context in the task store."""
        # Configure inspect_ai debug logging
        self._configure_inspect_ai_debug_logging()

        # Get SABER context from store
        task_store = store()

        # Store session manager
        task_store.set("saber_session_manager", self.session_manager)

        if self.session_manager is None:
            raise SABERAgentError("Session manager is required but not provided")

        session_id = self.session_manager.get_current_session_id()
        if session_id is None:
            raise SABERAgentError("Session ID not available from session manager")
        task_store.set("saber_session_id", session_id)

        # Get task_id from current sample metadata
        from inspect_ai.solver._task_state import sample_state

        current_state = sample_state()
        if current_state is None:
            raise SABERAgentError("Current task state is not available")

        task_id = current_state.metadata.get("task_id")
        if task_id is None:
            raise SABERAgentError("Task ID not found in sample metadata")
        task_store.set("saber_task_id", task_id)
        logger.info(f"Using task_id: {task_id}")

        try:
            self.episode = await self.session_manager.create_episode(session_id, task_id)
            task_store.set("saber_current_episode", self.episode)
            logger.info(f"Created SABER episode: {self.episode.episode_id}")
        except Exception as e:
            logger.error(f"Failed to create SABER episode: {e}")
            raise SABERAgentError(f"Episode creation failed: {e}") from e

        # Mark as initialized
        task_store.set("saber_initialized", True)
        task_store.set("saber_agent_id", self.agent_id)

    async def _fetch_dynamic_prompt(self) -> None:
        """Fetch dynamic prompt from policy endpoint."""
        if not self.episode:
            return

        try:
            policy_response = await self.session_manager.get_policy_response(
                self.episode.session_id, self.episode.episode_id
            )
            if policy_response and policy_response.prompt:
                self.enhanced_prompt = policy_response.prompt
                logger.info("Using dynamic prompt from policy endpoint")
        except Exception as e:
            logger.warning(f"Failed to fetch policy prompt: {e}")
            # Continue with default prompt

    async def _discover_mcp_tools(self) -> None:
        """Discover and convert MCP tools to inspect_ai Tool objects."""
        if not self.episode:
            return

        try:
            # Get task_id from context for MCP client connection
            task_store = store()
            task_id = task_store.get("saber_task_id")

            # Ensure MCP client exists for this episode (fail fast if connection fails)
            await self.session_manager.ensure_episode_mcp_client(self.episode.episode_id, task_id)

            # Get MCP tools from session manager
            mcp_tools = await self.session_manager.list_mcp_tools(self.episode.episode_id)

            # Convert to inspect_ai tools
            for mcp_tool in mcp_tools:
                inspect_tool = await self._convert_mcp_tool(mcp_tool)
                if inspect_tool:
                    self.mcp_tools.append(inspect_tool)

            logger.info(f"Discovered {len(self.mcp_tools)} MCP tools")

        except Exception as e:
            logger.error(f"Failed to discover MCP tools: {e}")
            # Re-raise - failing to create MCP client should fail the episode
            raise SABERAgentError(f"MCP tool discovery failed: {e}") from e

    async def _convert_mcp_tool(self, mcp_tool: MCPToolSchema) -> Optional[Tool]:
        """Convert an MCP tool to an inspect_ai Tool object."""
        from inspect_ai.tool import tool  # noqa: F401

        try:
            # Create tool execution function
            async def tool_executor(**kwargs: Any) -> str:
                # Get current episode from store
                task_store = store()
                episode_data = task_store.get("saber_current_episode")
                if not episode_data:
                    raise SABERAgentError("No active SABER episode for tool execution")

                # Create MCP tool request
                request = MCPToolCallRequest(
                    tool_name=mcp_tool.name,
                    arguments=kwargs,
                    episode_id=episode_data.episode_id,
                    task_id=episode_data.task_id,
                    timeout=None,
                    context={
                        "agent_id": task_store.get("saber_agent_id"),
                        "episode_id": episode_data.episode_id,
                        "session_id": episode_data.session_id,
                    },
                )

                # Execute via session manager
                response = await self.session_manager.execute_mcp_tool(request)

                # Handle response
                if response.isError:
                    error_msg = "Unknown error"
                    if response.content and len(response.content) > 0:
                        error_msg = str(response.content[0].get("text", error_msg))
                    raise RuntimeError(f"Tool execution failed: {error_msg}")

                # Return result
                if response.content and len(response.content) > 0:
                    return str(response.content[0].get("text", ""))
                return ""

            # Build tool function following web_browser pattern
            return self._build_inspect_ai_tool(mcp_tool, tool_executor)

        except Exception as e:
            logger.error(f"Failed to convert MCP tool {mcp_tool.name}: {e}")
            return None

    def _build_inspect_ai_tool(self, mcp_tool: MCPToolSchema, tool_executor: Any) -> Optional[Tool]:
        """Build inspect_ai Tool from MCP tool schema."""
        from typing import cast

        from inspect_ai.tool import tool  # noqa: F401

        try:
            # Prepare parameter information from schema
            required_params = mcp_tool.inputSchema.required
            properties = mcp_tool.inputSchema.properties

            # Build parameter documentation and function signature
            param_docs = []
            param_assignments = []
            func_params = []

            for param_name, param_info in properties.items():
                param_type = param_info.type
                param_desc = param_info.description or param_name.title()
                is_required = param_name in required_params

                # Convert JSON schema types to Python types
                py_type = "str"
                if param_type == "integer":
                    py_type = "int"
                elif param_type == "boolean":
                    py_type = "bool"
                elif param_type == "number":
                    py_type = "float"

                # Build parameter documentation
                status = "required" if is_required else "optional"
                param_docs.append(f"    {param_name} ({status}): {param_desc}")

                # Build parameter assignment for function body
                param_assignments.append(f"args['{param_name}'] = {param_name}")

                # Build function parameter with type annotation and default
                if is_required:
                    func_params.append(f"{param_name}: {py_type}")
                else:
                    # For optional parameters, provide a sensible default
                    if py_type == "str":
                        func_params.append(f"{param_name}: {py_type} = ''")
                    elif py_type == "int":
                        func_params.append(f"{param_name}: {py_type} = 0")
                    elif py_type == "bool":
                        func_params.append(f"{param_name}: {py_type} = False")
                    elif py_type == "float":
                        func_params.append(f"{param_name}: {py_type} = 0.0")
                    else:
                        func_params.append(f"{param_name}: {py_type} = None")

            # Handle case where there are no parameters
            if not func_params:
                func_signature = ""
                param_assignment_code = "pass"
            else:
                func_signature = ", ".join(func_params)
                param_assignment_code = "; ".join(param_assignments)

            # Create a valid Python function name (replace invalid characters)
            safe_func_name = mcp_tool.name.replace("-", "_").replace(".", "_")

            # Create the tool function following the exact web_browser pattern
            function_code = f'''
@tool
def {safe_func_name}() -> Tool:
    """Execute a {mcp_tool.name} command in the SABER sandbox environment"""

    async def execute({func_signature}) -> str:
        """Execute a {mcp_tool.name} command in the SABER sandbox environment

Args:
{chr(10).join(param_docs)}

Returns:
    str: Tool execution result
        """
        logger.debug(f"SABER DEBUG: Executing {mcp_tool.name} with args: {{locals()}}")
        # Collect all parameters into kwargs for the executor
        args = {{}}
        {param_assignment_code}
        result = await tool_executor(**args)
        logger.debug(f"SABER DEBUG: {mcp_tool.name} result: {{result}}")
        return result

    return execute
'''

            logger.debug(f"Generated function code for {mcp_tool.name}:")
            logger.debug(function_code)

            # Execute the generated code in the current namespace
            namespace = {
                "tool_executor": tool_executor,
                "tool": tool,
                "Tool": Tool,
                "logger": logger,
            }
            exec(function_code, namespace)

            # Get the created factory function and call it to get the actual tool
            factory_func = namespace[safe_func_name]
            actual_tool = factory_func()

            # Type check and return the tool
            if isinstance(actual_tool, Tool) or callable(actual_tool):
                return cast(Tool, actual_tool)
            else:
                logger.error(f"Generated tool is not a valid Tool type: {type(actual_tool)}")
                return None

        except Exception as e:
            logger.error(f"Failed to build inspect_ai tool for {mcp_tool.name}: {e}")
            return None

    def _configure_inspect_ai_debug_logging(self) -> None:
        """Configure inspect_ai loggers to DEBUG level for SABER debugging."""
        import logging

        # Set the specific inspect_ai tool loggers to DEBUG
        inspect_ai_tool_logger = logging.getLogger("inspect_ai.tool._tool_def")
        inspect_ai_tool_logger.setLevel(logging.DEBUG)

        inspect_ai_info_logger = logging.getLogger("inspect_ai.tool._tool_info")
        inspect_ai_info_logger.setLevel(logging.DEBUG)

        logger.debug("SABER DEBUG: Configured inspect_ai tool loggers to DEBUG level")
