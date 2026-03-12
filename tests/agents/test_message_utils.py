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


class TestCountToolCalls:
    """count_tool_calls counts tool calls across assistant messages."""

    def test_empty_list_returns_zero(self) -> None:
        """Empty message list returns 0."""
        from saber.agents.message_utils import count_tool_calls

        assert count_tool_calls([]) == 0

    def test_only_user_messages_returns_zero(self) -> None:
        """User-only messages return 0 tool calls."""
        from inspect_ai.model import ChatMessageUser

        from saber.agents.message_utils import count_tool_calls

        messages = [ChatMessageUser(content="hello"), ChatMessageUser(content="world")]
        assert count_tool_calls(messages) == 0

    def test_one_assistant_with_tool_calls(self) -> None:
        """Single assistant with 3 tool calls returns 3."""
        from inspect_ai.model import ChatMessageAssistant
        from inspect_ai.tool import ToolCall

        from saber.agents.message_utils import count_tool_calls

        messages = [
            ChatMessageAssistant(
                content="running",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={}, type="function"),
                    ToolCall(id="c2", function="bash", arguments={}, type="function"),
                    ToolCall(id="c3", function="report", arguments={}, type="function"),
                ],
            ),
        ]
        assert count_tool_calls(messages) == 3

    def test_multiple_assistants_sums_all(self) -> None:
        """Multiple assistant messages sum all tool calls."""
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
        from inspect_ai.tool import ToolCall

        from saber.agents.message_utils import count_tool_calls

        messages = [
            ChatMessageAssistant(
                content="step 1",
                tool_calls=[
                    ToolCall(id="a1", function="bash", arguments={}, type="function"),
                ],
            ),
            ChatMessageUser(content="ok"),
            ChatMessageAssistant(
                content="step 2",
                tool_calls=[
                    ToolCall(id="b1", function="bash", arguments={}, type="function"),
                    ToolCall(id="b2", function="bash", arguments={}, type="function"),
                ],
            ),
        ]
        assert count_tool_calls(messages) == 3

    def test_ignores_assistants_without_tool_calls(self) -> None:
        """Assistants without tool_calls are ignored."""
        from inspect_ai.model import ChatMessageAssistant
        from inspect_ai.tool import ToolCall

        from saber.agents.message_utils import count_tool_calls

        messages = [
            ChatMessageAssistant(content="thinking..."),
            ChatMessageAssistant(
                content="now acting",
                tool_calls=[
                    ToolCall(id="a1", function="bash", arguments={}, type="function"),
                ],
            ),
        ]
        assert count_tool_calls(messages) == 1


class TestToolCallLimitMessageConstant:
    """TOOL_CALL_LIMIT_MESSAGE has required content."""

    def test_contains_tool_call_limit(self) -> None:
        """Message mentions 'tool call limit'."""
        from saber.agents.message_utils import TOOL_CALL_LIMIT_MESSAGE

        assert "tool call limit" in TOOL_CALL_LIMIT_MESSAGE.lower()

    def test_contains_final_answer(self) -> None:
        """Message mentions 'final answer'."""
        from saber.agents.message_utils import TOOL_CALL_LIMIT_MESSAGE

        assert "final answer" in TOOL_CALL_LIMIT_MESSAGE.lower()
