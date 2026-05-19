# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for extract_tool_steps trajectory extraction."""

from __future__ import annotations

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall, ToolCallError

from saber.scoring.trajectory import extract_tool_steps


class TestEmptyMessages:
    """Empty message list returns empty list."""

    def test_empty_list(self) -> None:
        assert extract_tool_steps([]) == []

    def test_empty_tuple(self) -> None:
        assert extract_tool_steps(()) == []


class TestNoToolCalls:
    """System + User only — no tool calls → empty list."""

    def test_system_user_only(self) -> None:
        messages = [
            ChatMessageSystem(content="You are helpful."),
            ChatMessageUser(content="Hello"),
        ]
        assert extract_tool_steps(messages) == []


class TestSingleToolCall:
    """Assistant with one tool_call → Tool response → one ToolStep."""

    def test_single_call(self) -> None:
        messages = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="Run ls"),
            ChatMessageAssistant(
                content=[ContentText(text="Let me check.")],
                tool_calls=[
                    ToolCall(id="call_1", function="bash", arguments={"cmd": "ls"}, type="function"),
                ],
            ),
            ChatMessageTool(content="file1.txt\nfile2.txt", tool_call_id="call_1"),
        ]
        steps = extract_tool_steps(messages)
        assert len(steps) == 1
        step = steps[0]
        assert step.step_number == 1
        assert step.tool_name == "bash"
        assert step.tool_input == {"cmd": "ls"}
        assert step.output == "file1.txt\nfile2.txt"
        assert step.is_error is False
        assert step.error_type is None
        assert step.assistant_message == "Let me check."


class TestMultipleToolCallsInOneMessage:
    """Two tool_calls in one assistant message → two Tool responses."""

    def test_two_calls_one_assistant(self) -> None:
        messages = [
            ChatMessageAssistant(
                content=[ContentText(text="Running both.")],
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={"cmd": "ls"}, type="function"),
                    ToolCall(id="c2", function="python", arguments={"code": "1+1"}, type="function"),
                ],
            ),
            ChatMessageTool(content="file.txt", tool_call_id="c1"),
            ChatMessageTool(content="2", tool_call_id="c2"),
        ]
        steps = extract_tool_steps(messages)
        assert len(steps) == 2
        assert steps[0].tool_name == "bash"
        assert steps[0].output == "file.txt"
        assert steps[0].step_number == 1
        assert steps[1].tool_name == "python"
        assert steps[1].output == "2"
        assert steps[1].step_number == 2
        # Both should get the same assistant_message
        assert steps[0].assistant_message == "Running both."
        assert steps[1].assistant_message == "Running both."


class TestMultipleAssistantTurns:
    """Two separate Assistant → Tool rounds → two ToolSteps."""

    def test_two_rounds(self) -> None:
        messages = [
            ChatMessageAssistant(
                content=[ContentText(text="First turn.")],
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={"cmd": "ls"}, type="function"),
                ],
            ),
            ChatMessageTool(content="output1", tool_call_id="c1"),
            ChatMessageAssistant(
                content=[ContentText(text="Second turn.")],
                tool_calls=[
                    ToolCall(id="c2", function="bash", arguments={"cmd": "pwd"}, type="function"),
                ],
            ),
            ChatMessageTool(content="output2", tool_call_id="c2"),
        ]
        steps = extract_tool_steps(messages)
        assert len(steps) == 2
        assert steps[0].assistant_message == "First turn."
        assert steps[0].output == "output1"
        assert steps[1].assistant_message == "Second turn."
        assert steps[1].output == "output2"


class TestToolError:
    """Tool response with error → ToolStep.is_error=True."""

    def test_error_response(self) -> None:
        messages = [
            ChatMessageAssistant(
                content="trying",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={"cmd": "rm /"}, type="function"),
                ],
            ),
            ChatMessageTool(
                content="Permission denied",
                tool_call_id="c1",
                error=ToolCallError(type="permission", message="Access denied"),
            ),
        ]
        steps = extract_tool_steps(messages)
        assert len(steps) == 1
        assert steps[0].is_error is True
        assert steps[0].error_type == "permission"
        assert steps[0].output == "Permission denied"


class TestAssistantTextAttribution:
    """Assistant visible text captured in ToolStep.assistant_message."""

    def test_text_attribution(self) -> None:
        messages = [
            ChatMessageAssistant(
                content=[ContentText(text="Here is my plan.")],
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={"cmd": "echo hi"}, type="function"),
                ],
            ),
            ChatMessageTool(content="hi", tool_call_id="c1"),
        ]
        steps = extract_tool_steps(messages)
        assert steps[0].assistant_message == "Here is my plan."

    def test_no_text_content(self) -> None:
        """Assistant with no ContentText → assistant_message is empty or None."""
        messages = [
            ChatMessageAssistant(
                content=[],
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={"cmd": "ls"}, type="function"),
                ],
            ),
            ChatMessageTool(content="files", tool_call_id="c1"),
        ]
        steps = extract_tool_steps(messages)
        # msg.text returns "" for empty content; `"" or None` → None
        assert steps[0].assistant_message is None


class TestReasoningExtraction:
    """ContentReasoning parts captured in ToolStep.reasoning."""

    def test_reasoning_captured(self) -> None:
        messages = [
            ChatMessageAssistant(
                content=[
                    ContentReasoning(reasoning="I should list files first."),
                    ContentText(text="Let me check."),
                    ContentReasoning(reasoning="Then analyze them."),
                ],
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={"cmd": "ls"}, type="function"),
                ],
            ),
            ChatMessageTool(content="file.txt", tool_call_id="c1"),
        ]
        steps = extract_tool_steps(messages)
        assert steps[0].reasoning is not None
        assert "I should list files first." in steps[0].reasoning
        assert "Then analyze them." in steps[0].reasoning

    def test_no_reasoning(self) -> None:
        messages = [
            ChatMessageAssistant(
                content=[ContentText(text="Just text.")],
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={"cmd": "ls"}, type="function"),
                ],
            ),
            ChatMessageTool(content="output", tool_call_id="c1"),
        ]
        steps = extract_tool_steps(messages)
        assert steps[0].reasoning is None


class TestUnmatchedToolResponse:
    """Tool response without matching tool_call_id → ignored."""

    def test_unmatched_ignored(self) -> None:
        messages = [
            ChatMessageAssistant(
                content="text",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={"cmd": "ls"}, type="function"),
                ],
            ),
            ChatMessageTool(content="output", tool_call_id="c1"),
            # Unmatched tool response (no corresponding tool_call)
            ChatMessageTool(content="orphan", tool_call_id="c_unknown"),
        ]
        steps = extract_tool_steps(messages)
        assert len(steps) == 1
        assert steps[0].output == "output"


class TestStepNumbering:
    """Sequential step_number starting at 1."""

    def test_sequential_numbering(self) -> None:
        messages = [
            ChatMessageAssistant(
                content="first",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={"cmd": "ls"}, type="function"),
                ],
            ),
            ChatMessageTool(content="out1", tool_call_id="c1"),
            ChatMessageAssistant(
                content="second",
                tool_calls=[
                    ToolCall(id="c2", function="bash", arguments={"cmd": "pwd"}, type="function"),
                    ToolCall(id="c3", function="python", arguments={"code": "1"}, type="function"),
                ],
            ),
            ChatMessageTool(content="out2", tool_call_id="c2"),
            ChatMessageTool(content="out3", tool_call_id="c3"),
        ]
        steps = extract_tool_steps(messages)
        assert len(steps) == 3
        assert steps[0].step_number == 1
        assert steps[1].step_number == 2
        assert steps[2].step_number == 3


class TestPendingToolCallWithoutResponse:
    """Tool call with no matching Tool response -> step with empty output."""

    def test_pending_call_has_empty_output(self) -> None:
        messages = [
            ChatMessageAssistant(
                content="Running command",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function="bash",
                        arguments={"cmd": "ls"},
                        type="function",
                    ),
                ],
            ),
            # No ChatMessageTool for c1
        ]
        steps = extract_tool_steps(messages)
        assert len(steps) == 1
        assert steps[0].tool_name == "bash"
        assert steps[0].output == ""
        assert steps[0].is_error is False
