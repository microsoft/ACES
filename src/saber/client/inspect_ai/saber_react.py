"""
SABER React Agent - CTF-style agent with SABER integration

This implementation follows the inspect_ai CTF agent pattern while adding
SABER infrastructure for security domain benchmarking:

1. Uses @agent decorator for proper registration
2. Returns a react agent enhanced with SABER capabilities
3. Uses inspect_ai's store() for context management
4. Manages SABER episodes and MCP tools
5. Integrates with SABER session management

Architecture:
- Simple function-based agent (no factories)
- SABER context via inspect_ai's store
- Episode management for tool call limits
- MCP tool discovery and integration
- Session lifecycle management
"""

import logging
from typing import Any, List, Optional

from inspect_ai.agent import Agent, AgentState, agent, react
from inspect_ai.model import ChatMessageSystem
from inspect_ai.tool import Tool
from inspect_ai.util import store

from ...models import EpisodeCreateResponse
from ...models.mcp import MCPToolCallRequest, MCPToolSchema
from ..client_session import ClientSessionManager

logger = logging.getLogger(__name__)


# SABER DEBUG: Explicitly configure inspect_ai loggers for debug output
def configure_inspect_ai_debug_logging() -> None:
    """Configure inspect_ai loggers to DEBUG level for SABER debugging."""
    import logging

    # Set the specific inspect_ai tool loggers to DEBUG
    inspect_ai_tool_logger = logging.getLogger("inspect_ai.tool._tool_def")
    inspect_ai_tool_logger.setLevel(logging.DEBUG)

    inspect_ai_info_logger = logging.getLogger("inspect_ai.tool._tool_info")
    inspect_ai_info_logger.setLevel(logging.DEBUG)

    logger.debug("SABER DEBUG: Configured inspect_ai tool loggers to DEBUG level")


@agent  # type: ignore[misc]
def saber_react(
    attempts: int = 1,
    # SABER-specific parameters
    agent_id: str = "saber_react",
    session_manager: Optional[ClientSessionManager] = None,
) -> Agent:
    """SABER-enhanced ReAct agent for security domain benchmarking.

    This agent enhances the standard react agent with SABER infrastructure:
    - Episode management with tool call limits from server
    - Dynamic prompt fetching from policy endpoint
    - MCP tool discovery and execution
    - Session and context management

    Args:
        attempts: Number of attempts for the react agent
        agent_id: SABER agent identifier
        session_manager: SABER session manager instance

    Returns:
        Agent function compatible with inspect_ai
    """

    # Default SABER prompt for security domain
    SABER_PROMPT = "Default prompt"

    async def saber_enhanced_agent(state: AgentState, tools: List[Tool]) -> AgentState:
        """Enhanced agent implementation with SABER integration."""

        # SABER DEBUG: Configure inspect_ai logging first
        configure_inspect_ai_debug_logging()

        # Get SABER context from store
        task_store = store()
        await _initialize_saber_context(task_store, session_manager, agent_id)

        # Get current episode context
        episode: EpisodeCreateResponse = task_store.get("saber_current_episode")
        if not episode:
            raise RuntimeError("No SABER episode found in context")

        # Connect MCP session
        current_session_manager = task_store.get("saber_session_manager")
        enhanced_prompt = SABER_PROMPT
        try:
            policy_response = await current_session_manager.get_policy_response(episode.session_id, episode.episode_id)
            if policy_response and policy_response.prompt:
                enhanced_prompt = policy_response.prompt
                logger.info("Using dynamic prompt from policy endpoint")
        except Exception as e:
            logger.warning(f"Failed to fetch policy prompt: {e}")

        # Discover MCP tools
        mcp_tools = []
        if current_session_manager:
            try:
                # Get task_id from context for MCP client connection
                task_id = task_store.get("saber_task_id")
                mcp_tools = await _discover_mcp_tools(current_session_manager, episode.episode_id, task_id)
            except Exception as e:
                logger.error(f"Failed to discover MCP tools: {e}")
                # Re-raise to fail episode if MCP is required
                raise

        # Combine all tools
        all_tools = list(tools) + mcp_tools

        # Enhance state with SABER prompt
        enhanced_state = state
        if enhanced_prompt != SABER_PROMPT:  # Only add if we have a custom prompt
            enhanced_messages = list(state.messages)
            enhanced_messages.insert(0, ChatMessageSystem(content=enhanced_prompt))
            enhanced_state = AgentState(messages=enhanced_messages)
            enhanced_state.output = state.output

        try:
            # Create react agent with enhanced configuration
            react_agent = react(
                name="SABER React Agent",
                description="SABER-enhanced ReAct agent for security domain",
                prompt=enhanced_prompt,
                tools=all_tools,
                attempts=attempts,
            )

            # Execute the react agent
            result = await react_agent(enhanced_state)

            # End episode with success
            if current_session_manager:
                await current_session_manager.end_episode(
                    episode.session_id, episode.episode_id, reason="completed", result="success"
                )

            return result

        except Exception as e:
            # End episode with error
            if current_session_manager:
                await current_session_manager.end_episode(
                    episode.session_id, episode.episode_id, reason="error", result=str(e)
                )
            raise
        finally:
            # MCP cleanup is handled automatically in end_episode()
            pass

    return saber_enhanced_agent


async def _initialize_saber_context(
    task_store: Any, session_manager: Optional[ClientSessionManager], agent_id: str
) -> None:
    """Initialize SABER context in the task store."""

    # Import sample_state to access current sample metadata
    from inspect_ai.solver._task_state import sample_state

    # Store session manager
    task_store.set("saber_session_manager", session_manager)

    if session_manager is None:
        raise ValueError("Session manager is required but not provided")

    session_id = session_manager.get_current_session_id()
    if session_id is None:
        raise ValueError("Session ID not available from session manager")
    task_store.set("saber_session_id", session_id)

    # Get task_id from current sample metadata
    current_state = sample_state()
    if current_state is None:
        raise ValueError("Current task state is not available")

    task_id = current_state.metadata.get("task_id")
    if task_id is None:
        raise ValueError("Task ID not found in sample metadata")
    task_store.set("saber_task_id", task_id)
    logger.info(f"Using task_id: {task_id}")

    try:
        episode: EpisodeCreateResponse = await session_manager.create_episode(session_id, task_id)
        task_store.set("saber_current_episode", episode)
        logger.info(f"Created SABER episode: {episode.episode_id}")
    except Exception as e:
        logger.error(f"Failed to create SABER episode: {e}")
        raise

    # Mark as initialized
    task_store.set("saber_initialized", True)
    task_store.set("saber_agent_id", agent_id)


async def _discover_mcp_tools(
    session_manager: ClientSessionManager, episode_id: str, task_id: Optional[str] = None
) -> List[Tool]:
    """Discover and convert MCP tools to inspect_ai Tool objects."""

    try:
        # Ensure MCP client exists for this episode (fail fast if connection fails)
        await session_manager.ensure_episode_mcp_client(episode_id, task_id)

        # Get MCP tools from session manager
        mcp_tools = await session_manager.list_mcp_tools(episode_id)

        # Convert to inspect_ai tools
        inspect_tools = []
        for mcp_tool in mcp_tools:
            inspect_tool = await _convert_mcp_tool(mcp_tool, session_manager)
            if inspect_tool:
                inspect_tools.append(inspect_tool)

        logger.info(f"Discovered {len(inspect_tools)} MCP tools")
        return inspect_tools

    except Exception as e:
        logger.error(f"Failed to discover MCP tools: {e}")
        # Re-raise - failing to create MCP client should fail the episode
        raise


async def _convert_mcp_tool(mcp_tool: MCPToolSchema, session_manager: ClientSessionManager) -> Optional[Tool]:
    """Convert an MCP tool to an inspect_ai Tool object."""

    from inspect_ai.tool import tool

    try:
        # Create tool execution function
        async def tool_executor(**kwargs: Any) -> str:
            # Get current episode from store
            task_store = store()
            episode_data = task_store.get("saber_current_episode")
            if not episode_data:
                raise RuntimeError("No active SABER episode for tool execution")

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
            response = await session_manager.execute_mcp_tool(request)

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

        # Prepare parameter information from schema
        required_params = mcp_tool.inputSchema.required
        properties = mcp_tool.inputSchema.properties

        # Build parameter documentation
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

        # Build the complete docstring in inspect_ai format
        description = mcp_tool.description or f"{mcp_tool.name.title()} tool"
        # Remove trailing period if present for inspect_ai compatibility
        if description.endswith("."):
            description = description[:-1]

        # Create function signature
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
        # This creates a factory function decorated with @tool that returns an async function
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

        # Get the created factory function
        factory_func = namespace[safe_func_name]

        logger.debug(f"Created factory function: {factory_func}")
        logger.debug(f"Factory function type: {type(factory_func)}")
        logger.debug(f"Factory function name: {getattr(factory_func, '__name__', None)}")

        # Check if the factory has registry info (it should since it was decorated)
        if hasattr(factory_func, "__registry_info__"):
            logger.debug(f"Factory has registry info: {factory_func.__registry_info__}")
        else:
            logger.debug("Factory has no registry info")

        # Call the factory function to get the actual tool (async execute function)
        if not callable(factory_func):
            logger.error(f"Factory function {safe_func_name} is not callable")
            return None

        actual_tool = factory_func()
        logger.debug(f"Actual tool from factory: {actual_tool}")
        logger.debug(f"Actual tool type: {type(actual_tool)}")

        import asyncio

        logger.debug(f"Actual tool is coroutine function: {asyncio.iscoroutinefunction(actual_tool)}")

        # Type check and return the tool
        from typing import cast

        if isinstance(actual_tool, Tool) or callable(actual_tool):
            return cast(Tool, actual_tool)  # Return the actual tool (execute function) that inspect_ai can call
        else:
            logger.error(f"Generated tool is not a valid Tool type: {type(actual_tool)}")
            return None

    except Exception as e:
        logger.error(f"Failed to convert MCP tool {mcp_tool.name}: {e}")
        return None
