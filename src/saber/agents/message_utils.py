"""Message-level utilities for the agent solver layer.

Helpers that operate on ``inspect_ai`` ``ChatMessage`` lists and are
shared across **all** agent types (not just bridge agents).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage

from saber.logging import get_logger

logger = get_logger(__name__)

# Shared message injected when the tool call limit is reached.
TOOL_CALL_LIMIT_MESSAGE = (
    "IMPORTANT: You have reached the tool call limit. You cannot use "
    "any more tools. Please provide your final answer immediately as "
    "plain text in your next response."
)


def count_tool_calls(messages: list[ChatMessage]) -> int:
    """Count total tool calls across all assistant messages.

    Args:
        messages: Conversation message list.

    Returns:
        Total number of tool calls found.
    """
    from inspect_ai.model import ChatMessageAssistant

    return sum(len(m.tool_calls) for m in messages if isinstance(m, ChatMessageAssistant) and m.tool_calls)


def patch_orphaned_tool_calls(messages: list[ChatMessage]) -> None:
    """Inject dummy ``ChatMessageTool`` results for orphaned tool calls.

    The OpenAI API requires every ``tool_call`` in an assistant message
    to have a corresponding ``role: "tool"`` result with a matching
    ``tool_call_id`` before any subsequent non-tool message.  When the
    tool-call limit is reached, the solver injects a user message (or
    strips tools) after an assistant turn whose tool calls have not yet
    been answered.  Without placeholder results the API rejects the
    request with a 400 error.

    This helper walks the *entire* message list, collects all
    ``tool_call`` IDs from assistant messages and all resolved IDs from
    tool-result messages, then appends a placeholder result for each
    orphaned call.

    The dummy results are appended at the **end** of the list so that
    the caller can subsequently append a user message after them.

    Args:
        messages: The mutable conversation list.  Modified in place.
    """
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool

    # Collect all tool_call IDs from assistant messages
    all_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            for tc in msg.tool_calls:
                all_call_ids.add(tc.id)

    # Collect IDs that already have a tool-result message
    resolved_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ChatMessageTool) and msg.tool_call_id:
            resolved_ids.add(msg.tool_call_id)

    orphaned = all_call_ids - resolved_ids
    if not orphaned:
        return

    logger.debug(
        "Injecting %d dummy tool results for orphaned call IDs: %s",
        len(orphaned),
        orphaned,
    )
    for call_id in sorted(orphaned):
        messages.append(
            ChatMessageTool(
                content="[Tool call limit reached \u2014 not executed]",
                tool_call_id=call_id,
            )
        )
