"""Tests for copilot-log v3 → inspect_ai ChatMessage parser (timeline-based)."""

from __future__ import annotations

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)

from saber.agents.transcript.copilot_v3 import parse_copilot_v3
from saber.agents.transcript.models import (
    CopilotSession,
    CopilotSessionMeta,
    TimelineEntry,
)


def _meta(**overrides: object) -> CopilotSessionMeta:
    defaults = {
        "agentName": "test-agent",
        "model": "gpt-5.4",
        "startedAt": "2026-01-01T00:00:00Z",
        "endedAt": "2026-01-01T00:01:00Z",
        "durationMs": 60000,
        "turnCount": 1,
    }
    return CopilotSessionMeta(**{**defaults, **overrides})  # type: ignore[arg-type]


def _session(entries: list[TimelineEntry]) -> CopilotSession:
    return CopilotSession(session=_meta(), timeline=entries)


def _entry(seq: int, kind: str, **kwargs: object) -> TimelineEntry:
    return TimelineEntry(
        sequence=seq,
        timestamp=f"2026-01-01T00:00:{seq:02d}Z",
        kind=kind,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


class TestParseCopilotV3:
    def test_empty_session(self) -> None:
        """Empty timeline → empty messages."""
        result = parse_copilot_v3(_session([]))
        assert result == []

    def test_system_user_assistant(self) -> None:
        """System, user, assistant timeline entries map to corresponding ChatMessages."""
        entries = [
            _entry(1, "system_message", content="You are a helper."),
            _entry(2, "user_message", content="Hello"),
            _entry(3, "assistant_message", content="Hi there"),
        ]
        result = parse_copilot_v3(_session(entries))

        assert len(result) == 3
        assert isinstance(result[0], ChatMessageSystem)
        assert result[0].content == "You are a helper."
        assert isinstance(result[1], ChatMessageUser)
        assert result[1].content == "Hello"
        assert isinstance(result[2], ChatMessageAssistant)
        assert result[2].content == "Hi there"

    def test_tool_call_round_trip(self) -> None:
        """tool_request → tool_execution_complete maps correctly."""
        entries = [
            _entry(1, "user_message", content="List files"),
            _entry(
                2,
                "tool_request",
                toolCallId="call_1",
                toolName="run_bash",
                arguments={"cmd": "ls"},
            ),
            _entry(3, "tool_execution_start", toolCallId="call_1"),
            _entry(
                4,
                "tool_execution_complete",
                toolCallId="call_1",
                success=True,
                result={"content": "file1.txt\nfile2.txt"},
            ),
            _entry(5, "assistant_message", content="Found two files."),
        ]
        result = parse_copilot_v3(_session(entries))

        assert len(result) == 4
        # User
        assert isinstance(result[0], ChatMessageUser)
        assert result[0].content == "List files"
        # Assistant with tool call (from flushed tool_request batch)
        assert isinstance(result[1], ChatMessageAssistant)
        assert result[1].tool_calls is not None
        assert len(result[1].tool_calls) == 1
        assert result[1].tool_calls[0].id == "call_1"
        assert result[1].tool_calls[0].function == "run_bash"
        assert result[1].tool_calls[0].arguments == {"cmd": "ls"}
        # Tool result
        assert isinstance(result[2], ChatMessageTool)
        assert result[2].content == "file1.txt\nfile2.txt"
        assert result[2].tool_call_id == "call_1"
        assert result[2].function == "run_bash"
        # Final assistant
        assert isinstance(result[3], ChatMessageAssistant)
        assert result[3].content == "Found two files."

    def test_consecutive_tool_requests_collapsed(self) -> None:
        """Multiple consecutive tool_request entries collapse into one assistant message."""
        entries = [
            _entry(1, "user_message", content="Do two things"),
            _entry(
                2,
                "tool_request",
                toolCallId="c1",
                toolName="tool_a",
                arguments={"a": 1},
            ),
            _entry(
                3,
                "tool_request",
                toolCallId="c2",
                toolName="tool_b",
                arguments={"b": 2},
            ),
            _entry(4, "tool_execution_start", toolCallId="c1"),
            _entry(5, "tool_execution_start", toolCallId="c2"),
            _entry(
                6,
                "tool_execution_complete",
                toolCallId="c1",
                success=True,
                result={"content": "res_a"},
            ),
            _entry(
                7,
                "tool_execution_complete",
                toolCallId="c2",
                success=True,
                result={"content": "res_b"},
            ),
        ]
        result = parse_copilot_v3(_session(entries))

        # User + 1 collapsed assistant + 2 tool results
        assert len(result) == 4
        assert isinstance(result[1], ChatMessageAssistant)
        assert result[1].tool_calls is not None
        assert len(result[1].tool_calls) == 2
        assert result[1].tool_calls[0].id == "c1"
        assert result[1].tool_calls[1].id == "c2"
        assert isinstance(result[2], ChatMessageTool)
        assert result[2].tool_call_id == "c1"
        assert isinstance(result[3], ChatMessageTool)
        assert result[3].tool_call_id == "c2"

    def test_entries_sorted_by_sequence(self) -> None:
        """Entries are sorted by sequence, not input order."""
        entries = [
            _entry(3, "assistant_message", content="resp"),
            _entry(1, "system_message", content="sys"),
            _entry(2, "user_message", content="hi"),
        ]
        result = parse_copilot_v3(_session(entries))

        assert len(result) == 3
        assert isinstance(result[0], ChatMessageSystem)
        assert isinstance(result[1], ChatMessageUser)
        assert isinstance(result[2], ChatMessageAssistant)

    def test_empty_assistant_message_skipped(self) -> None:
        """assistant_message with no content is skipped."""
        entries = [
            _entry(1, "user_message", content="hi"),
            _entry(2, "assistant_message", content=""),  # empty
            _entry(3, "assistant_message", content=None),  # None
        ]
        result = parse_copilot_v3(_session(entries))

        # Only the user message; empty/None assistant messages are skipped
        assert len(result) == 1
        assert isinstance(result[0], ChatMessageUser)

    def test_tool_execution_start_ignored(self) -> None:
        """tool_execution_start events are purely informational, skipped."""
        entries = [
            _entry(1, "tool_request", toolCallId="c1", toolName="t", arguments={}),
            _entry(2, "tool_execution_start", toolCallId="c1"),
            _entry(
                3,
                "tool_execution_complete",
                toolCallId="c1",
                success=True,
                result={"content": "done"},
            ),
        ]
        result = parse_copilot_v3(_session(entries))

        # assistant with tool_call + tool result
        assert len(result) == 2
        assert isinstance(result[0], ChatMessageAssistant)
        assert isinstance(result[1], ChatMessageTool)

    def test_tool_request_missing_id_skipped(self) -> None:
        """tool_request without toolCallId is skipped."""
        entries = [
            _entry(1, "tool_request", toolName="t", arguments={}),  # no toolCallId
        ]
        result = parse_copilot_v3(_session(entries))
        assert result == []

    def test_unmatched_tool_result(self) -> None:
        """tool_execution_complete with no matching request uses 'unknown' function."""
        entries = [
            _entry(
                1,
                "tool_execution_complete",
                toolCallId="orphan",
                success=True,
                result={"content": "result"},
            ),
        ]
        result = parse_copilot_v3(_session(entries))

        assert len(result) == 1
        assert isinstance(result[0], ChatMessageTool)
        assert result[0].tool_call_id == "orphan"
        assert result[0].function == "unknown"

    def test_dict_arguments_passed_through(self) -> None:
        """Arguments dict is passed directly to ToolCall."""
        entries = [
            _entry(
                1,
                "tool_request",
                toolCallId="c1",
                toolName="t",
                arguments={"key": "value"},
            ),
        ]
        result = parse_copilot_v3(_session(entries))
        assert len(result) == 1
        assert isinstance(result[0], ChatMessageAssistant)
        assert result[0].tool_calls is not None
        assert result[0].tool_calls[0].arguments == {"key": "value"}
