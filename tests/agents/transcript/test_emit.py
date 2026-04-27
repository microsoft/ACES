"""Tests for saber.agents.transcript.emit."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall

from saber.agents.transcript.emit import emit_transcript_events


def _make_msgs() -> list:
    """Build a minimal system→user→assistant(tool_calls)→tool→assistant sequence."""
    return [
        ChatMessageSystem(content="You are a helper."),
        ChatMessageUser(content="Hello"),
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="call_1", function="my_tool", arguments={"x": 1})],
        ),
        ChatMessageTool(content="result_1", tool_call_id="call_1", function="my_tool"),
        ChatMessageAssistant(content="Done!"),
    ]


class TestEmitTranscriptEvents:
    """Tests for emit_transcript_events."""

    @patch("inspect_ai.log._transcript.transcript")
    def test_emits_model_and_tool_events(self, mock_transcript_fn: MagicMock) -> None:
        mock_t = MagicMock()
        mock_transcript_fn.return_value = mock_t

        emit_transcript_events(_make_msgs(), model_name="test/detached")

        emitted = [call.args[0] for call in mock_t._event.call_args_list]
        from inspect_ai.event import ModelEvent, ToolEvent

        model_events = [e for e in emitted if isinstance(e, ModelEvent)]
        tool_events = [e for e in emitted if isinstance(e, ToolEvent)]

        assert len(model_events) == 2
        assert len(tool_events) == 1
        assert tool_events[0].function == "my_tool"
        assert tool_events[0].result == "result_1"
        assert tool_events[0].id == "call_1"

    @patch("inspect_ai.log._transcript.transcript")
    def test_empty_messages_noop(self, mock_transcript_fn: MagicMock) -> None:
        emit_transcript_events([], model_name="test")
        mock_transcript_fn.assert_not_called()

    @patch("inspect_ai.log._transcript.transcript")
    def test_model_event_input_accumulates(self, mock_transcript_fn: MagicMock) -> None:
        """System + user messages are accumulated as input for the next model turn."""
        mock_t = MagicMock()
        mock_transcript_fn.return_value = mock_t

        sys_msg = ChatMessageSystem(content="sys")
        user_msg = ChatMessageUser(content="usr")
        asst_msg = ChatMessageAssistant(content="hi")
        emit_transcript_events([sys_msg, user_msg, asst_msg])

        emitted = [call.args[0] for call in mock_t._event.call_args_list]
        from inspect_ai.event import ModelEvent

        model_events = [e for e in emitted if isinstance(e, ModelEvent)]
        assert len(model_events) == 1
        assert model_events[0].input == [sys_msg, user_msg]

    @patch("inspect_ai.log._transcript.transcript")
    def test_event_emission_failure_is_swallowed(
        self, mock_transcript_fn: MagicMock
    ) -> None:
        """If transcript()._event raises, we log but don't crash."""
        mock_t = MagicMock()
        mock_t._event.side_effect = RuntimeError("no transcript context")
        mock_transcript_fn.return_value = mock_t

        # Should not raise
        emit_transcript_events([ChatMessageAssistant(content="hello")])
