"""Trajectory extraction — convert inspect_ai messages to ToolStep objects.

The single public function :func:`extract_tool_steps` walks the flat message
list produced by ``react()`` and returns an ordered list of
:class:`~saber.scoring.context.ToolStep` instances.
"""

from __future__ import annotations

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ContentReasoning,
)

from saber.scoring.context import ToolStep


def _extract_reasoning(content: str | list[object]) -> str | None:
    """Pull ContentReasoning parts from assistant content, if any."""
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for item in content:
        if isinstance(item, ContentReasoning):
            parts.append(item.reasoning)
    return "\n".join(parts) if parts else None


def extract_tool_steps(
    messages: list[ChatMessage] | tuple[ChatMessage, ...],
) -> list[ToolStep]:
    """Convert inspect_ai messages to ToolStep objects.

    Walks the flat message list produced by ``react()``:

        [System, User, Assistant, Tool, Tool, Assistant, Tool, ...]

    For each (Assistant → Tool) pair, produces a :class:`ToolStep` that
    captures tool name, input, output, error status, the assistant's visible
    text (``ContentText``) as ``assistant_message``, and any
    ``ContentReasoning`` parts as ``reasoning``.

    Uses ``tool_call_id`` to match assistant ``tool_calls`` with tool
    responses.

    Args:
        messages: Flat list or tuple of :class:`ChatMessage` objects.

    Returns:
        Ordered list of :class:`ToolStep` instances, numbered starting at 1.
    """
    steps: list[ToolStep] = []
    pending: dict[str, int] = {}  # tool_call_id → index in steps
    step_num = 0
    last_assistant_text: str | None = None
    last_reasoning: str | None = None

    for msg in messages:
        if isinstance(msg, ChatMessageAssistant):
            last_assistant_text = msg.text or None
            last_reasoning = _extract_reasoning(msg.content)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    step_num += 1
                    step = ToolStep(
                        step_number=step_num,
                        tool_name=tc.function,
                        tool_input=tc.arguments or {},
                        output="",
                        assistant_message=last_assistant_text,
                        reasoning=last_reasoning,
                    )
                    steps.append(step)
                    pending[tc.id] = len(steps) - 1

        elif isinstance(msg, ChatMessageTool):
            call_id: str = msg.tool_call_id
            if call_id in pending:
                idx = pending.pop(call_id)
                old = steps[idx]
                error = msg.error
                steps[idx] = ToolStep(
                    step_number=old.step_number,
                    tool_name=old.tool_name,
                    tool_input=old.tool_input,
                    output=msg.text,
                    is_error=error is not None,
                    error_type=error.type if error else None,
                    assistant_message=old.assistant_message,
                    reasoning=old.reasoning,
                )

    return steps
