"""Tests for copilot transcript models (timeline-based v3 format)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


class TestTimelineEntry:
    def test_message_entry(self) -> None:
        entry = TimelineEntry(
            sequence=1,
            timestamp="2026-01-01T00:00:00Z",
            kind="user_message",
            content="Hello",
        )
        assert entry.kind == "user_message"
        assert entry.content == "Hello"

    def test_tool_request_entry(self) -> None:
        entry = TimelineEntry(
            sequence=2,
            timestamp="2026-01-01T00:00:01Z",
            kind="tool_request",
            toolCallId="call_1",
            toolName="run_bash",
            arguments={"cmd": "ls"},
        )
        assert entry.toolCallId == "call_1"
        assert entry.toolName == "run_bash"
        assert entry.arguments == {"cmd": "ls"}

    def test_tool_execution_complete(self) -> None:
        entry = TimelineEntry(
            sequence=3,
            timestamp="2026-01-01T00:00:02Z",
            kind="tool_execution_complete",
            toolCallId="call_1",
            success=True,
            result={"content": "file1.txt", "detailedContent": "file1.txt"},
        )
        assert entry.success is True
        assert entry.result is not None
        assert entry.result["content"] == "file1.txt"

    def test_frozen(self) -> None:
        entry = TimelineEntry(
            sequence=1,
            timestamp="2026-01-01T00:00:00Z",
            kind="system_message",
            content="You are a helper.",
        )
        with pytest.raises(ValidationError):
            entry.content = "changed"  # type: ignore[misc]

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValidationError):
            TimelineEntry(
                sequence=1,
                timestamp="2026-01-01T00:00:00Z",
                kind="invalid_kind",  # type: ignore[arg-type]
            )


class TestCopilotSessionMeta:
    def test_construction(self) -> None:
        meta = _meta()
        assert meta.agentName == "test-agent"
        assert meta.model == "gpt-5.4"

    def test_optional_fields_default(self) -> None:
        meta = _meta()
        assert meta.route == ""
        assert meta.endpointLabel == ""

    def test_frozen(self) -> None:
        meta = _meta()
        with pytest.raises(ValidationError):
            meta.agentName = "changed"  # type: ignore[misc]


class TestCopilotSession:
    def test_construction(self) -> None:
        session = CopilotSession(
            schemaVersion=3,
            stage="prepare",
            session=_meta(),
            timeline=[
                TimelineEntry(
                    sequence=1,
                    timestamp="2026-01-01T00:00:00Z",
                    kind="user_message",
                    content="hi",
                )
            ],
        )
        assert session.stage == "prepare"
        assert len(session.timeline) == 1

    def test_defaults(self) -> None:
        session = CopilotSession(session=_meta())
        assert session.schemaVersion == 3
        assert session.stage == ""
        assert session.timeline == []

    def test_missing_session_raises(self) -> None:
        with pytest.raises(ValidationError):
            CopilotSession()  # type: ignore[call-arg]
