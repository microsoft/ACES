"""Claude Code SDK agent implementation for SABER.

This agent uses Anthropic's Claude Code SDK (claude-code-sdk) to run an agentic loop
that interacts with SABER's MCP tools in the sandbox environment.

IMPORTANT: This agent requires:
1. The Claude Code CLI: npm install -g @anthropic-ai/claude-code
2. The Python SDK: pip install claude-code-sdk (or uv sync --extra claude-code)
3. ANTHROPIC_API_KEY environment variable

Key features:
- Integrates with Inspect AI's agent/solver protocol
- Uses SABER MCP tools (NOT Claude Code's default tools like Bash, Write, etc.)
- Records all assistant messages and tool calls in the Inspect AI transcript
- Properly handles message conversion between Claude Code SDK and Inspect AI formats

How it works:
1. Claude Code SDK manages its own conversation loop internally via the CLI
2. We configure it to ONLY use SABER's MCP tools (default tools are disabled)
3. MCP tools are provided via HTTP endpoint pointing to SABER's MCP server
4. All messages are converted to Inspect AI format and recorded in transcript
5. ModelEvent and ToolEvent are created for each interaction

Tool Integration:
- Default Claude Code tools (Bash, Write, Read, etc.) are DISABLED via disallowed_tools
- Only SABER MCP tools are available via allowed_tools list
- Tools are named with prefix: mcp__saber__<tool_name>

Message Types (from claude-code-sdk):
- SystemMessage: Initialization metadata (subtype, data)
- AssistantMessage: Model responses with content blocks (TextBlock, ThinkingBlock, ToolUseBlock)
- UserMessage: Tool results as ToolResultBlock in content list
- ResultMessage: Final message with session_id, num_turns, duration_ms, usage, result
"""

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any

from inspect_ai.agent._agent import AgentState
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput, ModelUsage
from inspect_ai.tool import ToolCall
from inspect_ai.tool._tool_call import ToolCallError

from .....logging_config import LogCategory, get_saber_logger
from ....integration.tools import saber_tools
from ..tools import (
    get_saber_mcp_url_and_headers,
    record_model_event,
    record_tool_event_from_call,
)

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Check if claude-code-sdk is available
CLAUDE_SDK_AVAILABLE = False
try:
    import claude_code_sdk  # noqa: F401

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Message Conversion Utilities
# =============================================================================


def _convert_inspect_messages_to_claude_code(
    messages: list[ChatMessage],
) -> list[dict[str, Any]]:
    """Convert Inspect AI ChatMessages to Claude Code SDK message format.

    Claude Code SDK expects messages in a specific format for conversation history.
    This function converts Inspect AI's ChatMessage types to that format.

    Args:
        messages: List of Inspect AI ChatMessage objects

    Returns:
        List of message dicts compatible with Claude Code SDK
    """
    converted = []
    for msg in messages:
        if isinstance(msg, ChatMessageSystem):
            # System messages are handled separately via system_prompt option
            continue
        elif isinstance(msg, ChatMessageUser):
            converted.append(
                {
                    "role": "user",
                    "content": msg.text if isinstance(msg.content, str) else msg.text,
                }
            )
        elif isinstance(msg, ChatMessageAssistant):
            content_parts = []
            # Add text content
            if msg.text:
                content_parts.append({"type": "text", "text": msg.text})
            # Add tool use blocks if present
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_parts.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function,
                            "input": tc.arguments,
                        }
                    )
            converted.append(
                {
                    "role": "assistant",
                    "content": content_parts if content_parts else msg.text,
                }
            )
        elif isinstance(msg, ChatMessageTool):
            # Tool results
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.text,
                            "is_error": msg.error is not None,
                        }
                    ],
                }
            )
    return converted


def _convert_claude_code_message_to_inspect(
    message: Any,
    model_name: str = "claude-code",
) -> tuple[list[ChatMessage], ModelOutput | None]:
    """Convert a Claude Code SDK message to Inspect AI ChatMessage(s).

    Handles AssistantMessage, UserMessage, and extracts tool calls.

    Args:
        message: A message from Claude Code SDK (AssistantMessage, UserMessage, etc.)
        model_name: Model name to use for assistant messages

    Returns:
        Tuple of (list of ChatMessages, optional ModelOutput for assistant messages)
    """
    messages: list[ChatMessage] = []
    output: ModelOutput | None = None

    # Import types from Claude Code SDK
    try:
        from claude_code_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
            UserMessage,
        )
    except ImportError:
        logger.error("claude-code-sdk not installed. Install with: pip install claude-code-sdk")
        raise

    if isinstance(message, AssistantMessage):
        # Extract text and tool calls from content blocks
        text_parts = []
        tool_calls = []

        for block in message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingBlock):
                # ThinkingBlock contains reasoning - include in text for transcript
                if hasattr(block, "thinking"):
                    text_parts.append(f"[Thinking: {block.thinking}]")
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        function=block.name,
                        arguments=block.input,
                        type="function",
                    )
                )

        content = "\n".join(text_parts) if text_parts else ""
        assistant_msg = ChatMessageAssistant(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            model=model_name,
            source="generate",
        )
        messages.append(assistant_msg)

        # Create ModelOutput for transcript
        output = ModelOutput(
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    message=assistant_msg,
                    stop_reason="tool_calls" if tool_calls else "stop",
                )
            ],
        )

    elif isinstance(message, UserMessage):
        content = message.content if isinstance(message.content, str) else str(message.content)
        messages.append(ChatMessageUser(content=content))

    elif isinstance(message, SystemMessage):
        # System messages are metadata, log but don't add to conversation
        logger.debug(f"Claude Code system message: {message.subtype}", extra={"data": message.data})

    elif isinstance(message, ResultMessage):
        # Final result message - extract completion if present
        logger.info(
            "Claude Code result message received",
            extra={
                "session_id": message.session_id,
                "duration_ms": message.duration_ms,
                "num_turns": message.num_turns,
                "total_cost_usd": message.total_cost_usd,
                "is_error": message.is_error,
            },
        )

    return messages, output


def _create_tool_result_message(
    tool_use_id: str,
    function_name: str,
    result: str | dict[str, Any],
    is_error: bool = False,
) -> ChatMessageTool:
    """Create a ChatMessageTool from a tool execution result.

    Args:
        tool_use_id: The ID of the tool use being responded to
        function_name: Name of the tool function
        result: The result content
        is_error: Whether the result is an error

    Returns:
        ChatMessageTool with the result
    """
    content = result if isinstance(result, str) else str(result)
    error = ToolCallError("unknown", content) if is_error else None

    return ChatMessageTool(
        content=content,
        tool_call_id=tool_use_id,
        function=function_name,
        error=error,
    )


# =============================================================================
# MCP Tool Bridge - HTTP Configuration
# =============================================================================


async def _get_saber_tool_names() -> list[str]:
    """Get the list of SABER tool names from the sandbox.

    Returns:
        List of tool names available from SABER MCP server
    """
    try:
        tool_source = saber_tools()
        tools = await tool_source.tools()
        return [getattr(t, "name", getattr(t, "__name__", "unnamed")) for t in tools]
    except Exception as e:
        logger.warning(f"Could not get SABER tool names: {e}")
        return []


# =============================================================================
# Claude Code Agent Implementation
# =============================================================================


async def _run_claude_code_agent(
    state: AgentState,
    instruction_prompt: str,
    assistant_prompt: str,
    submit_prompt: str,
    continue_prompt: str,
    submit_enabled: bool = True,
) -> AgentState:
    """Run the Claude Code agent loop.

    This function:
    1. Sets up Claude Code SDK with SABER MCP tools (no default tools)
    2. Runs the agentic conversation loop
    3. Records all messages and tool calls in the Inspect AI transcript
    4. Extracts submission from submit tool call (if present)
    5. Returns the final AgentState with updated messages

    Note: We don't manually track submission state. The submit tool is just
    another MCP tool - after the loop completes, we check if it was called
    and extract the answer from its arguments (following React agent pattern).

    Args:
        state: Initial agent state with conversation history
        instruction_prompt: Main instructions for the agent
        assistant_prompt: Assistant behavior prompt
        submit_prompt: Instructions about submission
        continue_prompt: Message shown to continue the agent
        submit_enabled: Whether to enable submission capability

    Returns:
        Updated AgentState after agent execution
    """
    try:
        from claude_code_sdk import (
            AssistantMessage,
            ClaudeCodeOptions,
            ClaudeSDKClient,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            UserMessage,
        )
    except ImportError as e:
        raise ImportError(
            "claude-code-sdk is required for the claude_code agent. Install with: pip install claude-code-sdk"
        ) from e

    # Build the system prompt
    system_prompt_parts = [instruction_prompt]
    if assistant_prompt:
        system_prompt_parts.append(assistant_prompt)
    if submit_enabled and submit_prompt:
        system_prompt_parts.append(submit_prompt)
    system_prompt = "\n\n".join(system_prompt_parts)

    # Add system message to state if not already present
    has_system = any(isinstance(m, ChatMessageSystem) for m in state.messages)
    if not has_system:
        state.messages.insert(0, ChatMessageSystem(content=system_prompt))

    # Get SABER MCP URL and headers from sandbox
    mcp_url, mcp_headers = get_saber_mcp_url_and_headers()

    # Create HTTP MCP server config for Claude Code CLI
    # The CLI will connect to SABER's MCP server via HTTP (POST + SSE response)
    from claude_code_sdk.types import McpHttpServerConfig

    mcp_server_config: McpHttpServerConfig = {
        "type": "http",
        "url": mcp_url,
        "headers": mcp_headers,
    }

    logger.info(
        "Configuring Claude Code agent with HTTP MCP",
        extra={
            "system_prompt_length": len(system_prompt),
            "mcp_url": mcp_url,
            "submit_enabled": submit_enabled,
        },
    )

    # Configure Claude Code options
    # CRITICAL: We disable ALL default Claude Code tools (Bash, Write, etc.)
    # Claude Code will auto-discover SABER MCP tools from the mcp_servers config
    # The can_use_tool callback enforces restrictions at runtime for all tool use
    disallowed_tools = [
        # File system tools
        "Bash",
        "Write",
        "Edit",
        "MultiEdit",
        "Read",
        "Glob",
        "Grep",
        "LS",
        # Web tools
        "WebFetch",
        "WebSearch",
        # Note/todo tools
        "TodoRead",
        "TodoWrite",
        "NotebookRead",
        "NotebookEdit",
        # MCP introspection tools (agent should use SABER tools directly)
        "ListMcpResources",
        "ReadMcpResource",
        "Skill",
        # Plan approval (SABER manages task flow)
        "Plan",
        # Note: Task is NOT disabled - we allow subagents, but can_use_tool callback
        # enforces that subagents can only use mcp__saber__* tools
    ]

    # Create a can_use_tool callback to enforce tool restrictions at runtime
    # This applies to BOTH the main agent AND any subagents spawned by Task tool
    from claude_code_sdk.types import (
        PermissionResultAllow,
        PermissionResultDeny,
        ToolPermissionContext,
    )

    disallowed_tools_set = set(disallowed_tools)

    async def enforce_tool_restrictions(
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Enforce tool restrictions for main agent and subagents.

        This callback is invoked for EVERY tool use, including those by subagents.
        It ensures that only SABER MCP tools are allowed, regardless of how
        the tool was invoked.
        """
        # Allow SABER MCP tools (mcp__saber__*)
        if tool_name.startswith("mcp__saber__"):
            logger.debug(f"Tool '{tool_name}' allowed (mcp__saber__ prefix)")
            return PermissionResultAllow()

        # Block explicitly disallowed tools
        if tool_name in disallowed_tools_set:
            logger.warning(
                f"Tool '{tool_name}' DENIED - in disallowed_tools list",
                extra={"tool_name": tool_name, "source": "can_use_tool"},
            )
            return PermissionResultDeny(
                message=f"Tool '{tool_name}' is not available. Use SABER MCP tools instead (mcp__saber__*).",
                interrupt=False,
            )

        # Block all other non-MCP tools (including those used by subagents)
        logger.warning(
            f"Tool '{tool_name}' DENIED - not a SABER MCP tool",
            extra={"tool_name": tool_name, "source": "can_use_tool"},
        )
        return PermissionResultDeny(
            message=f"Tool '{tool_name}' is not available in SABER. Use SABER MCP tools (mcp__saber__*).",
            interrupt=False,
        )

    # Set up debug logging to file
    # The SDK requires both debug-to-stderr in extra_args AND a file object in debug_stderr
    import os

    log_dir = os.environ.get("SABER_CLAUDE_CODE_LOG_DIR", "logs/claude_code")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    debug_log_path = os.path.join(log_dir, f"claude_code_debug_{timestamp}.log")
    debug_log_file = open(debug_log_path, "w")
    logger.info(f"Claude Code debug logs will be written to: {debug_log_path}")

    options = ClaudeCodeOptions(
        system_prompt=system_prompt,
        disallowed_tools=disallowed_tools,
        mcp_servers={"saber": mcp_server_config},
        # NOTE: No allowed_tools - let Claude Code auto-discover MCP tools
        # The can_use_tool callback enforces that only mcp__saber__* tools are used
        permission_mode="bypassPermissions",  # SABER manages permissions
        extra_args={"debug-to-stderr": None},  # Enable debug output
        debug_stderr=debug_log_file,  # Write debug to file
        can_use_tool=enforce_tool_restrictions,  # Enforce restrictions on ALL tool use including subagents
    )

    # Build initial prompt from existing messages
    # Extract the last user message as the initial prompt
    initial_prompt = ""
    for msg in reversed(state.messages):
        if isinstance(msg, ChatMessageUser):
            initial_prompt = msg.text
            break

    if not initial_prompt:
        initial_prompt = "Please begin the task described in your instructions."

    model_name = "claude-code"
    final_completion = ""
    current_tool_calls: dict[str, ToolCall] = {}  # Track tool calls by ID

    # Create an async generator for the initial prompt
    # This is required for can_use_tool callback to work - SDK requires streaming mode
    async def initial_prompt_stream() -> AsyncIterator[dict[str, Any]]:
        """Yield the initial prompt as a streaming message."""
        yield {
            "type": "user",
            "message": {"role": "user", "content": initial_prompt},
            "parent_tool_use_id": None,
            "session_id": "saber-session",
        }

    try:
        # Use connect() with AsyncIterable to enable can_use_tool callback
        # The context manager pattern calls connect() with None which bypasses the callback
        client = ClaudeSDKClient(options=options)
        await client.connect(initial_prompt_stream())

        try:
            async for message in client.receive_messages():
                msg_type = type(message).__name__
                logger.debug(f"Claude Code message type: {msg_type}")

                if isinstance(message, AssistantMessage):
                    # Convert and record assistant message
                    inspect_messages, model_output = _convert_claude_code_message_to_inspect(message, model_name)

                    for msg in inspect_messages:
                        state.messages.append(msg)

                        # Track tool calls for later matching with results
                        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                current_tool_calls[tc.id] = tc

                    if model_output:
                        # Record ModelEvent in transcript
                        record_model_event(
                            model=model_name,
                            input_messages=list(state.messages[:-1]),
                            output=model_output,
                        )
                        state.output = model_output

                    # Extract final text for completion
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            final_completion = block.text

                elif isinstance(message, UserMessage):
                    # User messages contain tool results as ToolResultBlock
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                tool_use_id = block.tool_use_id
                                content = block.content if isinstance(block.content, str) else str(block.content)
                                is_error = block.is_error or False

                                # Create ChatMessageTool
                                tool_call = current_tool_calls.get(tool_use_id)
                                function_name = tool_call.function if tool_call else "unknown"

                                tool_msg = _create_tool_result_message(
                                    tool_use_id=tool_use_id,
                                    function_name=function_name,
                                    result=content,
                                    is_error=is_error,
                                )
                                state.messages.append(tool_msg)

                                # Record ToolEvent in transcript
                                if tool_call:
                                    record_tool_event_from_call(
                                        tool_call=tool_call,
                                        result=content,
                                        is_error=is_error,
                                        message_id=tool_msg.id,
                                    )
                    elif isinstance(message.content, str):
                        # Regular user message (string content)
                        user_msg = ChatMessageUser(content=message.content)
                        state.messages.append(user_msg)

                elif isinstance(message, SystemMessage):
                    logger.debug(
                        f"Claude Code system message: {message.subtype}",
                        extra={"data": message.data},
                    )

                elif isinstance(message, ResultMessage):
                    # Final message - extract result if present
                    if message.result:
                        final_completion = message.result

                    logger.info(
                        "Claude Code agent completed",
                        extra={
                            "session_id": message.session_id,
                            "num_turns": message.num_turns,
                            "duration_ms": message.duration_ms,
                            "total_cost_usd": message.total_cost_usd,
                            "is_error": message.is_error,
                        },
                    )

                    # Update model output with usage if available
                    if message.usage and state.output:
                        state.output.usage = ModelUsage(
                            input_tokens=message.usage.get("input_tokens", 0),
                            output_tokens=message.usage.get("output_tokens", 0),
                            total_tokens=message.usage.get("total_tokens", 0),
                        )

                    # Break after ResultMessage - it's the final message
                    break
        finally:
            # Always disconnect the client
            await client.disconnect()

    except Exception as e:
        logger.error(f"Claude Code agent error: {e}", exc_info=True)
        # Add error to final output
        final_completion = f"Error: {str(e)}"
    finally:
        # Close the debug log file
        debug_log_file.close()
        logger.info(f"Claude Code debug log closed: {debug_log_path}")

    # Check for submission by looking at tool messages (like React agent does)
    # The submit tool's argument 'answer' contains the submitted answer
    submitted_answer = _extract_submission_from_messages(state.messages)
    if submitted_answer:
        final_completion = submitted_answer
        logger.info("Found submission in tool calls", extra={"answer_length": len(submitted_answer)})

    # Ensure we have a final output with completion
    if state.output:
        state.output.completion = final_completion
    else:
        state.output = ModelOutput(
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=final_completion, model=model_name),
                    stop_reason="stop",
                )
            ],
        )
        state.output.completion = final_completion

    return state


def _extract_submission_from_messages(
    messages: list[ChatMessage],
    submit_tool_name: str = "submit",
) -> str | None:
    """Extract submission answer from tool messages (React agent pattern).

    Searches for a successful call to the submit tool and extracts the answer.
    This follows the same pattern as React's submission() function.

    Args:
        messages: List of chat messages to search
        submit_tool_name: Name of the submit tool (default: "submit")

    Returns:
        The submitted answer string, or None if no submission found
    """
    # First, find tool calls to submit
    for msg in messages:
        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            for tc in msg.tool_calls:
                # Check for submit tool (may be prefixed with mcp__saber__)
                if tc.function == submit_tool_name or tc.function.endswith(f"__{submit_tool_name}"):
                    # The answer is in the arguments
                    answer = tc.arguments.get("answer")
                    if answer:
                        return str(answer) if answer is not None else None

    # Also check tool result messages (the tool returns the answer)
    for msg in messages:
        if isinstance(msg, ChatMessageTool):
            if msg.function == submit_tool_name or (msg.function and msg.function.endswith(f"__{submit_tool_name}")):
                if msg.error is None and msg.text:
                    return str(msg.text)

    return None


def create_agent(**kwargs: Any) -> Callable[..., Any]:
    """Create a Claude Code agent with SABER integration.

    This agent:
    - Uses the Claude Code SDK for agentic execution
    - Extracts four prompts from sample metadata (instruction, assistant, submit, continue)
    - Uses ONLY saber_tools() from the MCP sandbox (default Claude Code tools are DISABLED)
    - Records all messages and tool calls in the Inspect AI transcript

    Args:
        **kwargs: Additional parameters (reserved for future use)

    Returns:
        Agent factory function configured for SABER

    Raises:
        ImportError: If claude-agent-sdk is not installed

    Usage:
        This function is called by the task factory with metadata-extracted prompts.
        The actual prompts are injected at runtime from sample metadata.
    """
    if not CLAUDE_SDK_AVAILABLE:
        raise ImportError(
            "The 'claude_code' agent requires the 'claude-code-sdk' package. "
            "Install with: pip install claude-code-sdk (or uv sync --extra claude-code)"
        )

    def create_with_prompts(
        instruction_prompt: str,
        assistant_prompt: str,
        submit_prompt: str,
        continue_prompt: str,
        transcript_config: dict | None = None,
        submit: bool | None = None,
    ) -> Any:
        """Inner factory that receives prompts from task execution.

        Args:
            instruction_prompt: The main instructions for the agent
            assistant_prompt: Assistant behavior prompt
            submit_prompt: Instructions about submission
            continue_prompt: Message shown after each step to guide the agent
            transcript_config: Optional transcript configuration dict
            submit: Whether to enable the submit tool. None=True (default), False=disabled.
        """
        submit_enabled = submit if submit is not None else True

        logger.debug(
            "Creating Claude Code agent with SABER prompts",
            extra={
                "instruction_length": len(instruction_prompt),
                "assistant_length": len(assistant_prompt),
                "submit_length": len(submit_prompt),
                "continue_length": len(continue_prompt),
                "submit_enabled": submit_enabled,
            },
        )

        async def execute(state: AgentState) -> AgentState:
            """Execute the Claude Code agent."""
            return await _run_claude_code_agent(
                state=state,
                instruction_prompt=instruction_prompt,
                assistant_prompt=assistant_prompt,
                submit_prompt=submit_prompt,
                continue_prompt=continue_prompt,
                submit_enabled=submit_enabled,
            )

        return execute

    return create_with_prompts


__all__ = ["create_agent"]
