"""Tests for saber.agents.message_utils."""

from __future__ import annotations

import pytest


class TestPatchOrphanedToolCalls:
    """patch_orphaned_tool_calls injects dummy results for orphaned calls."""

    def test_no_op_when_no_tool_calls(self) -> None:
        """Does nothing when messages have no tool calls."""
        from inspect_ai.model import ChatMessageUser

        from saber.agents.message_utils import patch_orphaned_tool_calls

        messages: list[object] = [ChatMessageUser(content="hello")]
        patch_orphaned_tool_calls(messages)
        assert len(messages) == 1

    def test_no_op_when_all_resolved(self) -> None:
        """Does nothing when every tool call already has a result."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
        from inspect_ai.tool import ToolCall

        from saber.agents.message_utils import patch_orphaned_tool_calls

        messages: list[object] = [
            ChatMessageAssistant(
                content="ok",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={}, type="function"),
                ],
            ),
            ChatMessageTool(content="output", tool_call_id="c1"),
        ]
        patch_orphaned_tool_calls(messages)
        assert len(messages) == 2

    def test_injects_for_all_orphaned(self) -> None:
        """Injects a dummy result for each orphaned tool call."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
        from inspect_ai.tool import ToolCall

        from saber.agents.message_utils import patch_orphaned_tool_calls

        messages: list[object] = [
            ChatMessageAssistant(
                content="running",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={}, type="function"),
                    ToolCall(id="c2", function="bash", arguments={}, type="function"),
                    ToolCall(id="c3", function="report", arguments={}, type="function"),
                ],
            ),
        ]
        patch_orphaned_tool_calls(messages)

        tool_msgs = [m for m in messages if isinstance(m, ChatMessageTool)]
        assert len(tool_msgs) == 3
        injected_ids = {m.tool_call_id for m in tool_msgs}
        assert injected_ids == {"c1", "c2", "c3"}

    def test_skips_already_resolved(self) -> None:
        """Only injects for calls that lack results."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
        from inspect_ai.tool import ToolCall

        from saber.agents.message_utils import patch_orphaned_tool_calls

        messages: list[object] = [
            ChatMessageAssistant(
                content="running",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={}, type="function"),
                    ToolCall(id="c2", function="bash", arguments={}, type="function"),
                ],
            ),
            ChatMessageTool(content="real output", tool_call_id="c1"),
        ]
        patch_orphaned_tool_calls(messages)

        tool_msgs = [m for m in messages if isinstance(m, ChatMessageTool)]
        assert len(tool_msgs) == 2
        # Original preserved
        original = [m for m in tool_msgs if m.tool_call_id == "c1"]
        assert original[0].content == "real output"
        # Injected for c2
        injected = [m for m in tool_msgs if m.tool_call_id == "c2"]
        assert len(injected) == 1

    def test_handles_multiple_assistant_messages(self) -> None:
        """Patches orphans across multiple assistant messages."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
        from inspect_ai.tool import ToolCall

        from saber.agents.message_utils import patch_orphaned_tool_calls

        messages: list[object] = [
            ChatMessageAssistant(
                content="step 1",
                tool_calls=[
                    ToolCall(id="a1", function="bash", arguments={}, type="function"),
                ],
            ),
            ChatMessageTool(content="ok", tool_call_id="a1"),
            ChatMessageAssistant(
                content="step 2",
                tool_calls=[
                    ToolCall(id="b1", function="bash", arguments={}, type="function"),
                    ToolCall(id="b2", function="bash", arguments={}, type="function"),
                ],
            ),
            # b1 and b2 are orphaned
        ]
        patch_orphaned_tool_calls(messages)

        tool_msgs = [m for m in messages if isinstance(m, ChatMessageTool)]
        assert len(tool_msgs) == 3
        ids = {m.tool_call_id for m in tool_msgs}
        assert ids == {"a1", "b1", "b2"}

    def test_idempotent(self) -> None:
        """Calling twice does not duplicate dummy results."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
        from inspect_ai.tool import ToolCall

        from saber.agents.message_utils import patch_orphaned_tool_calls

        messages: list[object] = [
            ChatMessageAssistant(
                content="ok",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={}, type="function"),
                ],
            ),
        ]
        patch_orphaned_tool_calls(messages)
        patch_orphaned_tool_calls(messages)

        tool_msgs = [m for m in messages if isinstance(m, ChatMessageTool)]
        assert len(tool_msgs) == 1

    def test_deterministic_order(self) -> None:
        """Injected results are sorted by call ID for reproducibility."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
        from inspect_ai.tool import ToolCall

        from saber.agents.message_utils import patch_orphaned_tool_calls

        messages: list[object] = [
            ChatMessageAssistant(
                content="ok",
                tool_calls=[
                    ToolCall(id="z_call", function="bash", arguments={}, type="function"),
                    ToolCall(id="a_call", function="bash", arguments={}, type="function"),
                    ToolCall(id="m_call", function="bash", arguments={}, type="function"),
                ],
            ),
        ]
        patch_orphaned_tool_calls(messages)

        tool_msgs = [m for m in messages if isinstance(m, ChatMessageTool)]
        ids = [m.tool_call_id for m in tool_msgs]
        assert ids == ["a_call", "m_call", "z_call"]
