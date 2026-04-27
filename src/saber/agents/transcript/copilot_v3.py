"""Parser for copilot-log schemaVersion 3 → inspect_ai ChatMessage list.

The v3 format uses a flat ``timeline`` of sequence-ordered entries rather
than nested turn objects.  Each entry has a ``kind`` that determines how
to map it to inspect_ai messaging primitives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage
    from inspect_ai.tool import ToolCall

from saber.agents.transcript.models import CopilotSession, TimelineEntry

logger = logging.getLogger(__name__)


def parse_copilot_v3(session: CopilotSession) -> list[ChatMessage]:
    """Parse a copilot-log v3 session into inspect_ai ChatMessage list.

    Walks the timeline in sequence order, converting system/user/assistant
    messages to the corresponding ChatMessage types.  Consecutive
    ``tool_request`` entries are collapsed into a single
    ``ChatMessageAssistant`` (with accumulated ``tool_calls``), and each
    ``tool_execution_complete`` becomes a ``ChatMessageTool``.

    Args:
        session: Parsed CopilotSession from a copilot-log JSON file.

    Returns:
        Flat list of ChatMessage objects in chronological order.
    """
    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageTool,
        ChatMessageUser,
    )
    from inspect_ai.tool import ToolCall as _TC

    sorted_entries = sorted(session.timeline, key=lambda e: e.sequence)

    messages: list[ChatMessage] = []
    # Accumulate consecutive tool_request entries into one assistant message
    pending_tool_calls: list[ToolCall] = []
    # Maps toolCallId → toolName for resolving function names in tool results
    tool_call_names: dict[str, str] = {}

    def _flush_pending_tool_calls() -> None:
        nonlocal pending_tool_calls
        if pending_tool_calls:
            messages.append(
                ChatMessageAssistant(content="", tool_calls=pending_tool_calls)
            )
            pending_tool_calls = []

    for entry in sorted_entries:
        if entry.kind == "system_message":
            _flush_pending_tool_calls()
            messages.append(ChatMessageSystem(content=entry.content or ""))

        elif entry.kind == "user_message":
            _flush_pending_tool_calls()
            messages.append(ChatMessageUser(content=entry.content or ""))

        elif entry.kind == "assistant_message":
            _flush_pending_tool_calls()
            if entry.content:
                messages.append(ChatMessageAssistant(content=entry.content))

        elif entry.kind == "tool_request":
            tc = _parse_tool_call(entry, _TC)
            if tc is not None:
                pending_tool_calls.append(tc)
                if entry.toolCallId and entry.toolName:
                    tool_call_names[entry.toolCallId] = entry.toolName

        elif entry.kind == "tool_execution_complete":
            # Flush any pending tool_request batch *before* the first result
            _flush_pending_tool_calls()
            if entry.toolCallId:
                result_text = ""
                if entry.result:
                    result_text = entry.result.get("content", "")
                fn_name = tool_call_names.get(entry.toolCallId, "unknown")
                messages.append(
                    ChatMessageTool(
                        content=result_text,
                        tool_call_id=entry.toolCallId,
                        function=fn_name,
                    )
                )

        # tool_execution_start is informational — skip

    _flush_pending_tool_calls()
    return messages


def _parse_tool_call(
    entry: TimelineEntry,
    tool_call_cls: type[ToolCall],
) -> ToolCall | None:
    """Convert a timeline tool_request entry to an inspect_ai ToolCall."""
    if not entry.toolCallId or not entry.toolName:
        logger.debug("Skipping tool_request with missing id/name: seq=%d", entry.sequence)
        return None

    arguments: dict[str, object] = {}
    if entry.arguments is not None:
        arguments = entry.arguments

    return tool_call_cls(
        id=entry.toolCallId, function=entry.toolName, arguments=arguments
    )
