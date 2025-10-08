"""
SABER Context Injection - Tool Params Patching Approach

This module patches the tool_params() function to inject context directly
into the parameters dict, avoiding wrapper bypass issues.

Architecture:
- Monkey-patch tool_params() to detect SABER MCP tools
- Inject __saber_assistant_message__ and __saber_reasoning__ into params dict
- Tool receives context as regular parameters (no wrapper needed)
- Server-side code already extracts these special params

Advantages:
- No wrapper bypass issues
- Simpler implementation
- Works with any tool calling mechanism
- Minimal invasive changes
"""

from typing import Any, Callable, cast

from inspect_ai._util.content import ContentReasoning

# Import and store original functions for monkey-patching and testing
from inspect_ai.model._call_tools import execute_tools as original_execute_tools
from inspect_ai.model._call_tools import tool_params as original_tool_params
from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


# ============================================================================
# Context Storage - Shared between execute_tools and tool_params
# ============================================================================


class _ContextStorage:
    """Thread-safe storage for assistant context during tool execution."""

    def __init__(self) -> None:
        self.assistant_message: str | None = None
        self.reasoning: str | None = None

    def set_context(self, assistant_message: str | None, reasoning: str | None) -> None:
        """Store context for upcoming tool calls."""
        self.assistant_message = assistant_message
        self.reasoning = reasoning

    def clear_context(self) -> None:
        """Clear stored context after tool execution."""
        self.assistant_message = None
        self.reasoning = None

    def has_context(self) -> bool:
        """Check if context is available."""
        return bool(self.assistant_message or self.reasoning)


# Global context storage
_context = _ContextStorage()


# ============================================================================
# Patched tool_params Function
# ============================================================================


def saber_tool_params(input: dict[str, Any], func: Callable[..., Any]) -> dict[str, Any]:
    """
    Enhanced tool_params that injects SABER context.

    This function wraps inspect_ai's tool_params to inject assistant messages
    and reasoning into SABER MCP tool calls.

    Args:
        input: Tool call arguments from model
        func: Tool callable

    Returns:
        Enhanced parameters dict with injected context
    """
    # Get original parameters
    params = cast(dict[str, Any], original_tool_params(input, func))

    # Only inject if we have context and this is a SABER tool
    if _context.has_context() and is_saber_mcp_tool(func):
        # Inject assistant message if available
        if _context.assistant_message:
            params["__saber_assistant_message__"] = _context.assistant_message

        # Inject reasoning if available
        if _context.reasoning:
            params["__saber_reasoning__"] = _context.reasoning

        logger.debug(
            "Context injected into tool parameters",
            extra={
                "event": "context_injection",
                "func_name": getattr(func, "__name__", "unknown"),
                "has_assistant_message": _context.assistant_message is not None,
                "has_reasoning": _context.reasoning is not None,
            },
        )

    return params


# ============================================================================
# Context Extraction Functions
# ============================================================================


def extract_assistant_content(message: ChatMessageAssistant) -> str | None:
    """Extract text content from assistant message."""
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            # Skip reasoning content - that's extracted separately
            if isinstance(item, ContentReasoning):
                continue
            # Extract text from various content types
            if hasattr(item, "text") and isinstance(item.text, str):
                text_parts.append(item.text)
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts) if text_parts else None

    # Try to extract from object attributes
    if hasattr(content, "text") and isinstance(content.text, str):
        return str(content.text)

    return None


def extract_reasoning_content(message: ChatMessageAssistant) -> str | None:
    """Extract reasoning content from assistant message (o1/o3 models)."""
    # Check reasoning attribute first (o1 models)
    if hasattr(message, "reasoning") and message.reasoning:
        return str(message.reasoning)

    # Check for ContentReasoning in content list
    if isinstance(message.content, list):
        for item in message.content:
            if isinstance(item, ContentReasoning):
                # ContentReasoning has a 'reasoning' attribute (not 'text')
                if hasattr(item, "reasoning"):
                    return str(item.reasoning)

    return None


def is_saber_mcp_tool(func: Callable[..., Any]) -> bool:
    """
    Detect if tool is a SABER MCP server tool.

    Args:
        func: Tool callable to check

    Returns:
        True if tool is a SABER MCP tool
    """
    # Check function name
    func_name = getattr(func, "__name__", "")
    if func_name and "saber" in func_name.lower():
        return True

    # Check for MCP server indicators
    if hasattr(func, "_server_config"):
        return True

    # Check module name
    module = getattr(func, "__module__", "")
    if module and "mcp" in module.lower():
        # Assume all MCP tools in SABER context are SABER tools
        # (inspect_ai doesn't use MCP for built-in tools)
        return True

    # Check for __qualname__ containing MCP indicators
    qualname = getattr(func, "__qualname__", "")
    if qualname and "MCP" in qualname:
        return True

    return False


# ============================================================================
# Execute Tools Context Capture
# ============================================================================


async def saber_execute_tools(
    messages: list[ChatMessage],
    tools: Any,
    max_output: int | None = None,
) -> Any:
    """
    Enhanced execute_tools that captures context and stores it.

    Context is stored in _context and will be injected by saber_tool_params.

    Args:
        messages: Current message list
        tools: Available tools
        max_output: Maximum output length

    Returns:
        ExecuteToolsResult from original execute_tools
    """
    try:
        # Capture assistant context if available
        if messages and isinstance(messages[-1], ChatMessageAssistant):
            assistant_message = messages[-1]

            # Extract content and reasoning
            assistant_content = extract_assistant_content(assistant_message)
            reasoning_content = extract_reasoning_content(assistant_message)

            # Store for tool_params to use
            if assistant_content or reasoning_content:
                _context.set_context(assistant_content, reasoning_content)
                logger.debug(
                    "Assistant context captured for tool injection",
                    extra={
                        "event": "context_captured",
                        "has_assistant_content": assistant_content is not None,
                        "has_reasoning": reasoning_content is not None,
                    },
                )

        # Call original execute_tools (our patched tool_params will inject context)
        result = await original_execute_tools(messages, tools, max_output)

        # Clear context after execution
        _context.clear_context()

        return result

    except Exception as e:
        logger.error(f"Error in saber_execute_tools: {e}", exc_info=True)
        _context.clear_context()
        # Fallback to original on error
        return await original_execute_tools(messages, tools, max_output)
