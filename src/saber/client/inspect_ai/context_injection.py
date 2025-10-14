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

Thread Safety:
- Uses contextvars for async-safe context isolation
- Each async context gets its own context storage
- No cross-contamination between concurrent agent executions
"""

import contextvars
import re
from contextlib import contextmanager
from typing import Any, Callable, Final, Iterator, TypeVar, cast

from inspect_ai._util.content import ContentReasoning

# Import and store original functions for monkey-patching and testing
from inspect_ai.model._call_tools import execute_tools as original_execute_tools
from inspect_ai.model._call_tools import tool_params as original_tool_params
from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


# ============================================================================
# Constants and Configuration
# ============================================================================

# Maximum context size (10KB)
MAX_CONTEXT_SIZE: Final[int] = 10_000

# Truncation marker
TRUNCATION_MARKER: Final[str] = "\n...[truncated due to size limit]"

# Explicit tool registration
SABER_MCP_TOOLS: set[str] = {
    "end_episode",
}


class ContextInjectionConfig:
    """Configuration for context injection behavior."""

    def __init__(self) -> None:
        self.enabled: bool = True
        self.registered_tools: set[str] = SABER_MCP_TOOLS.copy()
        self.inject_reasoning: bool = True
        self.inject_assistant_message: bool = True
        self.max_context_size: int = MAX_CONTEXT_SIZE

    def register_tool(self, tool_name: str) -> None:
        """Register a tool for context injection."""
        self.registered_tools.add(tool_name)

    def unregister_tool(self, tool_name: str) -> None:
        """Remove a tool from context injection."""
        self.registered_tools.discard(tool_name)


# Global config instance
_config = ContextInjectionConfig()


class ContextInjectionMetrics:
    """Simple metrics tracking for context injection."""

    def __init__(self) -> None:
        self.captures_attempted = 0
        self.captures_succeeded = 0
        self.captures_failed = 0
        self.injections_attempted = 0
        self.injections_succeeded = 0
        self.truncations = 0

    def record_capture_attempt(self, success: bool) -> None:
        self.captures_attempted += 1
        if success:
            self.captures_succeeded += 1
        else:
            self.captures_failed += 1

    def record_injection_attempt(self, success: bool) -> None:
        self.injections_attempted += 1
        if success:
            self.injections_succeeded += 1

    def record_truncation(self) -> None:
        self.truncations += 1

    def get_stats(self) -> dict:
        return {
            "captures_attempted": self.captures_attempted,
            "captures_succeeded": self.captures_succeeded,
            "captures_failed": self.captures_failed,
            "injections_attempted": self.injections_attempted,
            "injections_succeeded": self.injections_succeeded,
            "truncations": self.truncations,
            "capture_success_rate": (
                self.captures_succeeded / self.captures_attempted if self.captures_attempted > 0 else 0.0
            ),
        }


# Global metrics instance
_metrics = ContextInjectionMetrics()


# ============================================================================
# Context Storage - Thread-safe using contextvars
# ============================================================================


class _ContextStorage:
    """Async-safe storage for assistant context during tool execution.

    This class stores context (assistant message and reasoning) for a single
    async execution context. Using contextvars ensures that concurrent agent
    executions don't interfere with each other.
    """

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


# Context variable for async-safe context isolation
# Each async task/context gets its own _ContextStorage instance
_context_var: contextvars.ContextVar[_ContextStorage | None] = contextvars.ContextVar(
    "_saber_context_storage", default=None
)


def _get_context() -> _ContextStorage:
    """Get or create context storage for current async context."""
    context = _context_var.get()
    if context is None:
        context = _ContextStorage()
        _context_var.set(context)
    return context


# ============================================================================
# Content Sanitization and Truncation
# ============================================================================


def _sanitize_text(text: str) -> str:
    """
    Sanitize text content for safe injection.

    - Removes control characters (except newlines, tabs, carriage returns)
    - Normalizes whitespace
    - Ensures valid UTF-8

    Args:
        text: Raw text content

    Returns:
        Sanitized text safe for injection
    """
    # Remove control characters except \n, \t, \r
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple consecutive newlines (max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def _truncate_text(text: str, max_size: int) -> str:
    """
    Truncate text to maximum size if needed.

    Args:
        text: Text to truncate
        max_size: Maximum size in characters

    Returns:
        Truncated text with marker if truncation occurred
    """
    if len(text) <= max_size:
        return text

    # Truncate and add marker
    truncate_at = max_size - len(TRUNCATION_MARKER)
    truncated = text[:truncate_at] + TRUNCATION_MARKER

    _metrics.record_truncation()

    logger.warning(
        "Context content truncated",
        extra={
            "event": "context_truncated",
            "original_size": len(text),
            "truncated_size": len(truncated),
            "max_size": max_size,
        },
    )

    return truncated


def _sanitize_and_truncate(text: str | None, max_size: int = MAX_CONTEXT_SIZE) -> str | None:
    """
    Sanitize and truncate text content.

    Args:
        text: Text to process
        max_size: Maximum size in characters

    Returns:
        Processed text or None if input was None/empty
    """
    if not text:
        return None

    # Sanitize first
    sanitized = _sanitize_text(text)

    if not sanitized:
        return None

    # Then truncate if needed
    return _truncate_text(sanitized, max_size)


# ============================================================================
# Scoped Monkey-Patching Context Manager
# ============================================================================


@contextmanager
def saber_context_injection_patch() -> Iterator[None]:
    """
    Context manager for scoped SABER context injection patching.

    Temporarily patches inspect_ai's execute_tools and tool_params to enable
    context capture and injection. Restores original functions on exit.

    Usage:
        with saber_context_injection_patch():
            # Context injection is active
            result = await agent.run(task)
        # Original functions restored
    """
    import inspect_ai.agent._react
    import inspect_ai.model._call_tools

    # Store original functions
    original_react_execute = inspect_ai.agent._react.execute_tools
    original_call_tools_execute = inspect_ai.model._call_tools.execute_tools
    original_tool_params_func = inspect_ai.model._call_tools.tool_params

    try:
        # Apply patches
        inspect_ai.agent._react.execute_tools = saber_execute_tools
        inspect_ai.model._call_tools.execute_tools = saber_execute_tools
        inspect_ai.model._call_tools.tool_params = saber_tool_params

        logger.info("🔧 SABER context injection patches applied")
        yield

    finally:
        # Always restore, even on exception
        inspect_ai.agent._react.execute_tools = original_react_execute
        inspect_ai.model._call_tools.execute_tools = original_call_tools_execute
        inspect_ai.model._call_tools.tool_params = original_tool_params_func

        logger.info("🔧 SABER context injection patches removed")


# ============================================================================
# Public Configuration API
# ============================================================================


def configure_context_injection(
    enabled: bool | None = None,
    tools: set[str] | None = None,
    inject_reasoning: bool | None = None,
    inject_assistant_message: bool | None = None,
    max_context_size: int | None = None,
) -> None:
    """
    Configure context injection behavior.

    Args:
        enabled: Enable/disable context injection globally
        tools: Set of tool names to inject context into
        inject_reasoning: Whether to inject reasoning content
        inject_assistant_message: Whether to inject assistant messages
        max_context_size: Maximum size of context in bytes
    """
    if enabled is not None:
        _config.enabled = enabled
    if tools is not None:
        _config.registered_tools = tools
    if inject_reasoning is not None:
        _config.inject_reasoning = inject_reasoning
    if inject_assistant_message is not None:
        _config.inject_assistant_message = inject_assistant_message
    if max_context_size is not None:
        _config.max_context_size = max_context_size


# TypeVar for the saber_tool decorator
_F = TypeVar("_F", bound=Callable[..., Any])


def saber_tool(func: _F) -> _F:
    """
    Decorator to explicitly mark a tool for SABER context injection.

    Usage:
        @saber_tool
        async def my_custom_tool(**kwargs):
            ...
    """
    func._saber_context_injection = True  # type: ignore[attr-defined]
    return func


def get_context_injection_metrics() -> dict:
    """Get current context injection metrics."""
    return _metrics.get_stats()


def reset_context_injection_metrics() -> None:
    """Reset metrics (useful for testing)."""
    global _metrics
    _metrics = ContextInjectionMetrics()


def reset_tool_discovery() -> None:
    """
    Reset tool discovery state (useful for testing).

    Currently a no-op as tool discovery doesn't use caching.
    This function exists for test compatibility and future extensibility.
    """
    # No state to reset in current implementation
    # Tool discovery is stateless and checks _config.registered_tools directly
    pass


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

    # Get context for current async execution
    context = _get_context()

    func_name = getattr(func, "__name__", "unknown")

    # Check if we should inject
    if not _config.enabled:
        logger.warning(f"🔍 Context injection DISABLED globally for {func_name}")
        return params

    if not context.has_context():
        logger.debug(f"🔍 No context available to inject for {func_name}")
        return params

    if not is_saber_mcp_tool(func):
        logger.debug(f"🔍 Tool {func_name} is NOT a SABER MCP tool, skipping injection")
        return params

    try:
        injected_any = False

        # Inject based on config
        if _config.inject_assistant_message and context.assistant_message:
            sanitized = _sanitize_and_truncate(context.assistant_message, _config.max_context_size)
            if sanitized:
                params["__saber_assistant_message__"] = sanitized
                injected_any = True
                logger.info(f"🔍 Injected assistant_message ({len(sanitized)} chars) into {func_name}")

        if _config.inject_reasoning and context.reasoning:
            sanitized = _sanitize_and_truncate(context.reasoning, _config.max_context_size)
            if sanitized:
                params["__saber_reasoning__"] = sanitized
                injected_any = True
                logger.info(f"🔍 Injected reasoning ({len(sanitized)} chars) into {func_name}")

        if injected_any:
            _metrics.record_injection_attempt(success=True)
            logger.debug(
                f"✅ Context successfully injected into {func_name}",
                extra={
                    "event": "context_injection",
                    "func_name": func_name,
                    "has_assistant_message": "__saber_assistant_message__" in params,
                    "has_reasoning": "__saber_reasoning__" in params,
                    "param_keys": list(params.keys()),
                },
            )
        else:
            logger.warning(f"⚠️ No context was injected into {func_name} despite having context")

    except Exception as e:
        logger.error(
            f"Failed to inject context into tool params: {e}",
            exc_info=True,
            extra={
                "event": "context_injection_failed",
                "tool_name": func_name,
            },
        )
        # Don't raise - return params without context rather than breaking tool execution

    return params


# ============================================================================
# Context Extraction Functions
# ============================================================================


def extract_assistant_content(message: ChatMessageAssistant) -> str | None:
    """Extract text content from assistant message with sanitization."""
    content = message.content

    if isinstance(content, str):
        return _sanitize_and_truncate(content)

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

        if text_parts:
            combined = "\n".join(text_parts)
            return _sanitize_and_truncate(combined)
        return None

    # Try to extract from object attributes
    if hasattr(content, "text") and isinstance(content.text, str):
        return _sanitize_and_truncate(str(content.text))

    return None


def extract_reasoning_content(message: ChatMessageAssistant) -> str | None:
    """Extract reasoning content from assistant message (o1/o3 models) with sanitization."""
    # Check reasoning attribute first (o1 models)
    if hasattr(message, "reasoning") and message.reasoning:
        return _sanitize_and_truncate(str(message.reasoning))

    # Check for ContentReasoning in content list
    if isinstance(message.content, list):
        for item in message.content:
            if isinstance(item, ContentReasoning):
                # ContentReasoning has a 'reasoning' attribute (not 'text')
                if hasattr(item, "reasoning"):
                    return _sanitize_and_truncate(str(item.reasoning))

    return None


def is_saber_mcp_tool(func: Callable[..., Any]) -> bool:
    """
    Detect if tool should receive context injection.

    Detects tools from SABER MCP server using multiple indicators:
    1. Explicit registration in SABER_MCP_TOOLS
    2. @saber_tool decorator
    3. _server_config attribute (dynamically generated MCP tools)

    Args:
        func: Tool callable to check

    Returns:
        True if tool is registered for context injection
    """
    if not _config.enabled:
        return False

    func_name = getattr(func, "__name__", "")
    func_module = getattr(func, "__module__", "")

    # Check explicit registration
    if func_name in _config.registered_tools:
        logger.debug(f"🔍 {func_name} found in registered tools")
        return True

    # Check @saber_tool decorator
    if hasattr(func, "_saber_context_injection") and func._saber_context_injection:
        logger.debug(f"🔍 {func_name} has @saber_tool decorator")
        return True

    # Check if tool is from inspect_ai.tool._mcp module (MCP tools)
    # This catches tools created by mcp_server_http() that don't have _server_config
    if func_module and func_module.startswith("inspect_ai.tool._mcp"):
        logger.debug(f"🔍 {func_name} is from {func_module} - treating as SABER MCP tool")
        return True

    logger.debug(f"🔍 {func_name} is NOT a SABER MCP tool")
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

    Context is stored in contextvars and will be injected by saber_tool_params.

    Args:
        messages: Current message list
        tools: Available tools
        max_output: Maximum output length

    Returns:
        ExecuteToolsResult from original execute_tools
    """
    # Get context for current async execution
    context = _get_context()

    try:
        # Capture assistant context if available
        if messages and isinstance(messages[-1], ChatMessageAssistant):
            try:
                assistant_message = messages[-1]

                # Extract content and reasoning
                assistant_content = extract_assistant_content(assistant_message)
                reasoning_content = extract_reasoning_content(assistant_message)

                # Store for tool_params to use
                if assistant_content or reasoning_content:
                    context.set_context(assistant_content, reasoning_content)
                    _metrics.record_capture_attempt(success=True)

                    logger.debug(
                        "✅ Context captured and stored successfully",
                        extra={
                            "event": "context_captured",
                            "has_assistant_content": assistant_content is not None,
                            "has_reasoning": reasoning_content is not None,
                            "assistant_length": len(assistant_content) if assistant_content else 0,
                            "reasoning_length": len(reasoning_content) if reasoning_content else 0,
                        },
                    )
                else:
                    logger.debug("⚠️ No assistant content or reasoning extracted to store")

            except Exception as e:
                _metrics.record_capture_attempt(success=False)
                logger.error(
                    f"❌ Failed to capture assistant context: {e}",
                    exc_info=True,
                    extra={"event": "context_capture_failed"},
                )
                # Continue execution but without context
                # Don't fail the entire tool execution due to context capture issues

        # Call original execute_tools (our patched tool_params will inject context)
        result = await original_execute_tools(messages, tools, max_output)

        return result

    finally:
        # ALWAYS clear context in finally block
        logger.debug("🔍 Clearing context in finally block")
        context.clear_context()
