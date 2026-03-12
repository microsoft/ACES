"""Graceful tool-call-limit agent wrapper for the react loop.

Wraps model generation with proactive tool-call counting.  When the
cumulative tool-call count reaches the configured limit, tools are
stripped and a limit notification is injected, allowing one final
text-only generation.  The react loop's natural ``elif not tool_calls:
break`` then exits cleanly.

Design mirror: ``create_tool_call_limit_filter()`` in ``bridge_utils.py``
performs the same role for bridge agents as a ``GenerateFilter``.

IMPORTANT: This module must NOT use ``from __future__ import annotations``.
``inspect_ai``'s ``_agent_generate()`` calls ``parse_tool_info(model)``
which uses ``get_type_hints()`` to resolve parameter annotations.
Stringified annotations from ``__future__`` would fail to resolve
``AgentState`` and ``Tool`` unless they exist in ``__globals__``.
"""

from inspect_ai.agent import Agent, AgentState  # Must be runtime import
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.tool import Tool  # Must be runtime import

from saber.agents.message_utils import (
    TOOL_CALL_LIMIT_MESSAGE,
    count_tool_calls,
    patch_orphaned_tool_calls,
)
from saber.logging import get_logger

logger = get_logger(__name__)


def create_react_limit_agent(tool_call_limit: int) -> Agent:
    """Create an Agent that wraps model generation with tool-call-limit awareness.

    The returned callable has the Agent protocol signature:
    ``(state: AgentState, tools: list[Tool]) -> AgentState``

    On each call:
    - Counts tool calls already in ``state.messages``
    - If count >= tool_call_limit: patches orphans, injects limit msg,
      generates with tools=[] -> react loop breaks naturally
    - Otherwise: normal generation with full tool set

    Args:
        tool_call_limit: Max tool calls before graceful stop.

    Returns:
        An Agent callable for ``react(model=...)``.
    """

    async def generate(state: AgentState, tools: list[Tool]) -> AgentState:
        current_count = count_tool_calls(state.messages)

        if current_count >= tool_call_limit:
            logger.info(
                "Tool-call limit reached (%d/%d). Generating final answer without tools.",
                current_count,
                tool_call_limit,
            )
            patch_orphaned_tool_calls(state.messages)
            state.messages.append(ChatMessageUser(content=TOOL_CALL_LIMIT_MESSAGE))
            output = await get_model().generate(state.messages, tools=[])
        else:
            output = await get_model().generate(state.messages, tools=tools)

        state.output = output
        state.messages.append(output.message)
        return state

    return generate
