"""Copilot model conversion functions.

Pure functions for converting between inspect_ai types and the Copilot SDK
types. These have no SDK dependency at import time — SDK objects are accessed
via ``getattr``/``hasattr`` to avoid hard imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolCall, ToolInfo


@dataclass(frozen=True)
class CopilotToolDef:
    """Lightweight tool definition for the Copilot SDK.

    Mirrors copilot.tools.Tool schema without requiring SDK import.
    Converted to SDK Tool at session creation time.
    """

    name: str
    description: str
    parameters: dict[str, object]


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Serialize inspect_ai ChatMessage list to a single prompt string.

    Format::

        [system]
        {content}

        [user]
        {content}

        [assistant]
        {content}

        [tool: {function_name}]
        {content}

    Args:
        messages: List of chat messages to serialize.

    Returns:
        Formatted prompt string. Empty string if messages is empty.
    """
    if not messages:
        return ""

    parts: list[str] = []
    for msg in messages:
        prefix = _role_prefix(msg)
        text = msg.text

        section = f"{prefix}\n{text}"

        # Append tool_call annotations for assistant messages
        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            call_lines: list[str] = []
            for tc in msg.tool_calls:
                args_json = json.dumps(tc.arguments, separators=(",", ":"))
                call_lines.append(f"[tool_call: {tc.function}({args_json})]")
            section = section + "\n" + "\n".join(call_lines)

        parts.append(section)

    return "\n\n".join(parts)


def _role_prefix(msg: ChatMessage) -> str:
    """Build the bracket-prefixed role tag for a message.

    Args:
        msg: A chat message.

    Returns:
        Role prefix string, e.g. ``[system]`` or ``[tool: func_name]``.
    """
    if isinstance(msg, ChatMessageTool) and msg.function:
        return f"[tool: {msg.function}]"
    return f"[{msg.role}]"


def _tool_info_to_sdk_tool(tool_info: ToolInfo) -> CopilotToolDef:
    """Convert an inspect_ai ToolInfo to a CopilotToolDef.

    Args:
        tool_info: The inspect_ai tool specification.

    Returns:
        A frozen ``CopilotToolDef`` with name, description, and JSON-Schema
        parameters.
    """
    return CopilotToolDef(
        name=tool_info.name,
        description=tool_info.description,
        parameters=tool_info.parameters.model_dump(exclude_none=True),
    )


def _sdk_response_to_model_output(
    response_data: object | None,
    model_name: str,
    usage: ModelUsage | None,
) -> ModelOutput:
    """Convert a Copilot SDK response to an inspect_ai ModelOutput.

    Accesses SDK response attributes via ``getattr``/``hasattr`` to avoid
    importing the Copilot SDK at module level.

    Args:
        response_data: An ``AssistantMessageData`` from the Copilot SDK,
            or ``None`` if no response was returned.
        model_name: Model identifier for the output.
        usage: Optional token usage information.

    Returns:
        A ``ModelOutput`` suitable for inspect_ai evaluation.
    """
    if response_data is None:
        return ModelOutput(
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content="", source="generate"),
                    stop_reason="unknown",
                ),
            ],
            usage=usage,
        )

    content: str = getattr(response_data, "content", "") or ""
    tool_requests: list[object] = getattr(response_data, "tool_requests", None) or []

    tool_calls: list[ToolCall] | None = None
    stop_reason: str = "stop"

    if tool_requests:
        stop_reason = "tool_calls"
        tool_calls = [_convert_tool_request(req) for req in tool_requests]

    return ModelOutput(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=content,
                    tool_calls=tool_calls,
                    source="generate",
                ),
                stop_reason=stop_reason,  # type: ignore[arg-type]
            ),
        ],
        usage=usage,
    )


def _convert_tool_request(req: object) -> ToolCall:
    """Convert a single SDK tool request to an inspect_ai ToolCall.

    Args:
        req: A tool request object with ``tool_call_id``, ``name``, and
            ``arguments`` attributes.

    Returns:
        An inspect_ai ``ToolCall``.
    """
    call_id: str = getattr(req, "tool_call_id", "") or ""
    function: str = getattr(req, "name", "") or ""
    raw_arguments = getattr(req, "arguments", {})

    arguments: dict[str, object]
    if isinstance(raw_arguments, str):
        arguments = json.loads(raw_arguments)
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        arguments = {}

    return ToolCall(
        id=call_id,
        function=function,
        arguments=arguments,
    )


def _extract_usage(events: list[object]) -> ModelUsage | None:
    """Extract token usage from Copilot SDK session events.

    Iterates over events looking for those whose ``.data`` has
    ``input_tokens`` / ``output_tokens`` attributes and sums them.

    Args:
        events: List of SDK ``SessionEvent`` objects.

    Returns:
        A ``ModelUsage`` with aggregated token counts, or ``None`` if no
        usage events were found.
    """
    input_tokens = 0
    output_tokens = 0
    found = False

    for event in events:
        data = getattr(event, "data", None)
        if data is not None and hasattr(data, "input_tokens"):
            found = True
            input_tokens += int(getattr(data, "input_tokens", 0) or 0)
            output_tokens += int(getattr(data, "output_tokens", 0) or 0)

    if not found:
        return None

    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
