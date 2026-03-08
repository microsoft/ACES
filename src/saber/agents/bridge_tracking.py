"""Bridge generation tracking filter for subagent classification."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from saber.agents.bridge_tracking_models import (
    DELEGATION_TOOL_NAMES,
    BridgeSessionSummary,
    CallType,
    GenerationMetadata,
)

try:
    from inspect_ai.log._transcript import transcript
except Exception:  # pragma: no cover
    transcript = None

if TYPE_CHECKING:
    from collections.abc import Callable

    from inspect_ai.model import ChatMessage

logger = logging.getLogger(__name__)

_INFO_SOURCE = "saber.bridge_tracking"

_DELEGATION_TAIL_SIZE = 5


def _has_active_delegation(messages: list[ChatMessage]) -> bool:
    """Check if trailing messages indicate an active delegation.

    Scans the last few messages for tool result messages whose
    function name is in DELEGATION_TOOL_NAMES, and only triggers
    if no user/assistant messages appear *after* the delegation
    result (indicating the main thread has already resumed).

    This detects the Copilot SDK pattern where subagent model
    calls include the full parent history plus the task tool call
    and result.
    """
    tail = messages[-_DELEGATION_TAIL_SIZE:] if messages else []
    for i, m in enumerate(tail):
        role = getattr(m, "role", None)
        func = getattr(m, "function", None)
        if isinstance(role, str) and role == "tool" and isinstance(func, str) and func in DELEGATION_TOOL_NAMES:
            # Only active if no user/assistant messages follow
            remaining = tail[i + 1 :]
            if not any(getattr(r, "role", None) in ("assistant", "user") for r in remaining):
                return True
    return False


def create_tracking_filter() -> tuple[object, Callable[[], BridgeSessionSummary]]:
    """Create a GenerateFilter that classifies and records bridge generations.

    Uses a dual heuristic:
    - H1 (message-count drop): main-thread counts grow monotonically;
      subagent calls use shorter histories (Claude Code pattern).
    - H2 (delegation-tool scan): trailing messages contain a delegation
      tool result with no subsequent user/assistant messages
      (Copilot SDK pattern).

    Returns:
        Tuple of (filter, get_summary). filter is a GenerateFilter (async callable).
        get_summary returns a BridgeSessionSummary snapshot.
    """
    _generation_index: int = 0
    _high_water_mark: int = 0
    _main_count: int = 0
    _subagent_count: int = 0
    _total_tool_calls: int = 0
    _models_seen: set[str] = set()

    async def _filter(
        model: object,
        messages: list[ChatMessage],
        tools: list[object],
        tool_choice: object | None,
        config: object,
    ) -> None:
        nonlocal _generation_index, _high_water_mark
        nonlocal _main_count, _subagent_count, _total_tool_calls, _models_seen

        msg_count = len(messages)
        model_name = getattr(model, "name", str(model))

        # Dual-heuristic classification
        delegation_detected = False
        if msg_count > _high_water_mark:
            # H2: check for delegation tool calls in trailing messages
            if _has_active_delegation(messages):
                call_type = CallType.SUBAGENT
                delegation_detected = True
                _subagent_count += 1
                # Do NOT update _high_water_mark so main resumes correctly
            else:
                call_type = CallType.MAIN
                _high_water_mark = msg_count
                _main_count += 1
        else:
            # H1: message-count drop → subagent
            call_type = CallType.SUBAGENT
            _subagent_count += 1

        # Count tool_calls in assistant messages
        tool_call_count = 0
        for m in messages:
            tc = getattr(m, "tool_calls", None)
            if tc:
                tool_call_count += len(tc)
        _total_tool_calls = max(_total_tool_calls, tool_call_count)

        _models_seen.add(model_name)

        metadata = GenerationMetadata(
            call_type=call_type,
            model_name=model_name,
            message_count=msg_count,
            generation_index=_generation_index,
            tool_count=len(tools) if tools else 0,
            delegation_detected=delegation_detected,
        )

        _generation_index += 1

        # Record as InfoEvent in transcript
        try:
            if transcript is not None:
                transcript().info(
                    metadata.model_dump(mode="json"),
                    source=_INFO_SOURCE,
                )
        except Exception:
            logger.debug("Could not record tracking InfoEvent (no transcript context)")

        logger.debug(
            "Bridge generation #%d: %s (model=%s, msgs=%d, tools=%d)",
            metadata.generation_index,
            metadata.call_type.value,
            metadata.model_name,
            metadata.message_count,
            metadata.tool_count,
        )

        return None  # Observation only

    def get_summary() -> BridgeSessionSummary:
        """Return a snapshot of the current bridge session summary."""
        return BridgeSessionSummary(
            total_generations=_generation_index,
            main_generations=_main_count,
            subagent_generations=_subagent_count,
            total_tool_calls=_total_tool_calls,
            models_used=tuple(sorted(_models_seen)),
        )

    return _filter, get_summary
