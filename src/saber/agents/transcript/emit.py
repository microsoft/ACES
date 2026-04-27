"""Emit inspect_ai transcript events from parsed ChatMessage lists.

Detached-mode agents execute inside containers, producing no transcript
events.  After messages are reconstructed from copilot-log files, this
module re-emits them as ``ModelEvent`` / ``ToolEvent`` entries so the
inspect_ai log viewer can render the conversation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage

logger = logging.getLogger(__name__)


def emit_transcript_events(
    messages: list[ChatMessage],
    model_name: str = "detached",
) -> None:
    """Emit ModelEvent + ToolEvent to the active transcript.

    Groups *messages* into model turns (input → assistant response) and
    emits one ``ModelEvent`` per turn plus one ``ToolEvent`` for each
    tool call in the assistant response.

    Args:
        messages: Ordered ChatMessage list from copilot-log parsing.
        model_name: Label shown in the transcript viewer.
    """
    from inspect_ai.event import ModelEvent, ToolEvent
    from inspect_ai.log._transcript import transcript
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ModelOutput,
    )
    from inspect_ai.model._generate_config import GenerateConfig

    if not messages:
        return

    # Build a map of tool_call_id → tool result content for ToolEvent emission
    tool_results: dict[str, str] = {}
    for msg in messages:
        if msg.role == "tool":
            tcid = getattr(msg, "tool_call_id", None)
            if tcid:
                content = msg.content if isinstance(msg.content, str) else ""
                tool_results[tcid] = content

    input_acc: list[ChatMessage] = []
    model_events = 0
    tool_events = 0

    for msg in messages:
        role = msg.role

        if role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None) or []
            stop = "tool_calls" if tool_calls else "stop"
            content = msg.content if isinstance(msg.content, str) else ""

            output_msg = ChatMessageAssistant(
                content=content,
                tool_calls=tool_calls if tool_calls else None,
                model=model_name,
            )
            choice = ChatCompletionChoice(message=output_msg, stop_reason=stop)
            output = ModelOutput(model=model_name, choices=[choice])

            event = ModelEvent(
                model=model_name,
                input=list(input_acc),
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
                output=output,
            )
            try:
                # _event is a private API of inspect_ai's Transcript; no public
                # emit function exists as of inspect_ai 0.3.x.  If this breaks
                # after an upgrade, check inspect_ai.log._transcript for changes.
                transcript()._event(event)
                model_events += 1
            except Exception:
                logger.debug("Failed to emit ModelEvent", exc_info=True)

            # Emit a ToolEvent for each tool_call with its result
            for tc in tool_calls:
                result = tool_results.get(tc.id, "")
                te = ToolEvent(
                    id=tc.id,
                    function=tc.function,
                    arguments=tc.arguments,
                    result=result,
                )
                try:
                    transcript()._event(te)
                    tool_events += 1
                except Exception:
                    logger.debug("Failed to emit ToolEvent", exc_info=True)

            # Reset — next turn starts after the tool results
            input_acc = []

        elif role == "tool":
            # Add to input for the next model turn
            input_acc.append(msg)

        else:
            # system or user — accumulate as input
            input_acc.append(msg)

    logger.info(
        "Emitted %d model events and %d tool events from %d messages",
        model_events,
        tool_events,
        len(messages),
    )
