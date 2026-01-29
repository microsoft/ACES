"""Shared utilities for SABER agent implementations.

This module provides common helper functions used across multiple agent
implementations (Copilot, Claude Code, etc.) for transcript recording,
sandbox access, and system message building.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from inspect_ai.event._model import ModelEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.log._transcript import transcript
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.model._chat_message import ToolCall, ToolCallError
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput, ModelUsage
from inspect_ai.util import sandbox

from ....logging_config import LogCategory, get_saber_logger
from ....models.headers import HTTPHeaders

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage

logger = get_saber_logger(LogCategory.AGENT, __name__)


# =============================================================================
# Sandbox Utilities
# =============================================================================


def get_saber_sandbox(sandbox_name: str = "saber") -> Any:
    """Get the SABER sandbox instance, unwrapped from any proxy.

    Inspect AI wraps sandbox environments in a proxy. This function
    retrieves the sandbox and unwraps it to get the actual
    SABERSandboxEnvironment instance.

    Args:
        sandbox_name: Name of the sandbox to retrieve (default: "saber")

    Returns:
        The actual SABERSandboxEnvironment instance
    """
    sb = sandbox(sandbox_name)
    # Unwrap proxy if needed
    return sb._sandbox if hasattr(sb, "_sandbox") else sb


def get_mcp_client_from_sandbox(sandbox_name: str = "saber") -> Any:
    """Extract MCP client from the SABER sandbox.

    Args:
        sandbox_name: Name of the sandbox (default: "saber")

    Returns:
        The MCP client instance from the sandbox
    """
    actual_sandbox = get_saber_sandbox(sandbox_name)
    return actual_sandbox._mcp_client


def get_saber_mcp_url_and_headers(sandbox_name: str = "saber") -> tuple[str, dict[str, str]]:
    """Get the SABER MCP URL and headers from the sandbox.

    Builds the MCP endpoint URL and required headers for making
    requests to the SABER MCP server.

    Args:
        sandbox_name: Name of the sandbox (default: "saber")

    Returns:
        Tuple of (mcp_url, headers_dict)
    """
    actual_sandbox = get_saber_sandbox(sandbox_name)

    # Get the MCP URL from sandbox
    mcp_url = f"{actual_sandbox._mcp_url}/mcp"

    # Build headers from sandbox session info
    headers: dict[str, str] = {}
    if actual_sandbox._session_id:
        headers[HTTPHeaders.SESSION_ID] = actual_sandbox._session_id
    if actual_sandbox._primary_episode_id:
        headers[HTTPHeaders.EPISODE_ID] = actual_sandbox._primary_episode_id
    if actual_sandbox._task_id:
        headers[HTTPHeaders.TASK_ID] = actual_sandbox._task_id
    headers[HTTPHeaders.ORCHESTRATION_ENV] = "inspect_ai"
    headers["Accept"] = "application/json, text/event-stream"
    headers["Content-Type"] = "application/json"

    logger.debug(
        "SABER MCP configuration",
        extra={
            "mcp_url": mcp_url,
            "headers": list(headers.keys()),
            "session_id": actual_sandbox._session_id,
            "episode_id": actual_sandbox._primary_episode_id,
        },
    )

    return mcp_url, headers


# =============================================================================
# System Message Building
# =============================================================================


def build_system_message(
    assistant_prompt: str,
    submit_prompt: str,
    submit_enabled: bool,
) -> str:
    """Build a system message from assistant and submit prompts.

    Combines the assistant behavior prompt with submission instructions
    (if enabled) into a single system message string.

    Args:
        assistant_prompt: The assistant behavior/persona instructions
        submit_prompt: Instructions about task submission
        submit_enabled: Whether submission is enabled

    Returns:
        Combined system message content
    """
    parts = []
    if assistant_prompt:
        parts.append(assistant_prompt)
    if submit_prompt and submit_enabled:
        parts.append(submit_prompt)
    return "\n\n".join(parts)


# =============================================================================
# Transcript Recording
# =============================================================================


def record_model_event(
    model: str,
    input_messages: list[ChatMessage],
    output: ModelOutput,
    config: dict[str, Any] | None = None,
) -> None:
    """Record a ModelEvent in the Inspect AI transcript.

    Creates and records a ModelEvent for assistant responses, making
    them visible in the Inspect AI log viewer.

    Args:
        model: Name of the model that generated the response
        input_messages: The input messages sent to the model
        output: The model's output (ModelOutput with choices)
        config: Optional generation config parameters
    """
    event = ModelEvent(
        model=model,
        input=input_messages,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(**(config or {})),
        output=output,
        completed=datetime.now(timezone.utc),
    )
    transcript()._event(event)


def record_model_event_from_content(
    model: str,
    input_messages: list[Any],
    assistant_content: str,
    tool_calls: list[ToolCall] | None = None,
) -> None:
    """Record a ModelEvent from raw content (convenience wrapper).

    Creates a ModelOutput from the provided content and tool calls,
    then records it as a ModelEvent.

    Args:
        model: Name of the model
        input_messages: The input messages
        assistant_content: Text content of the assistant response
        tool_calls: Optional list of tool calls made
    """
    assistant_message = ChatMessageAssistant(
        content=assistant_content,
        tool_calls=tool_calls if tool_calls else None,
    )

    output = ModelOutput(
        model=model,
        choices=[
            ChatCompletionChoice(
                message=assistant_message,
                stop_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=ModelUsage(),
    )

    record_model_event(model, input_messages, output)


def record_tool_event(
    tool_call_id: str,
    function_name: str,
    arguments: dict[str, Any],
    result: str,
    is_error: bool = False,
    message_id: str | None = None,
) -> None:
    """Record a ToolEvent in the Inspect AI transcript.

    Creates and records a ToolEvent for tool executions, making them
    visible in the Inspect AI log viewer.

    Args:
        tool_call_id: Unique ID of the tool call
        function_name: Name of the tool function
        arguments: Arguments passed to the tool
        result: Result string from the tool execution
        is_error: Whether the execution resulted in an error
        message_id: Optional ID of the associated ChatMessageTool
    """
    error_obj = None
    if is_error:
        error_obj = ToolCallError(type="unknown", message=result)

    event = ToolEvent(
        id=tool_call_id,
        function=function_name,
        arguments=arguments,
        result=result,
        error=error_obj,
        completed=datetime.now(timezone.utc),
        message_id=message_id,
    )
    transcript()._event(event)


def record_tool_event_from_call(
    tool_call: ToolCall,
    result: str | dict[str, Any],
    is_error: bool = False,
    message_id: str | None = None,
) -> None:
    """Record a ToolEvent from a ToolCall object (convenience wrapper).

    Args:
        tool_call: The ToolCall that was executed
        result: The result of the tool execution
        is_error: Whether the execution resulted in an error
        message_id: Optional ID of the associated ChatMessageTool
    """
    content = result if isinstance(result, str) else str(result)
    record_tool_event(
        tool_call_id=tool_call.id,
        function_name=tool_call.function,
        arguments=tool_call.arguments,
        result=content,
        is_error=is_error,
        message_id=message_id,
    )


__all__ = [
    # Sandbox utilities
    "get_saber_sandbox",
    "get_mcp_client_from_sandbox",
    "get_saber_mcp_url_and_headers",
    # System message
    "build_system_message",
    # Transcript recording
    "record_model_event",
    "record_model_event_from_content",
    "record_tool_event",
    "record_tool_event_from_call",
]
