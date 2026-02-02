"""Tests for Copilot events timestamp handling."""

from __future__ import annotations

from datetime import datetime, timedelta

from inspect_ai.event._model import ModelEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.log._transcript import transcript

from saber.inspect_ai.agents.registry.copilot.events import (
    AssistantMessageData,
    AssistantMessageEvent,
    ToolExecutionCompleteData,
    ToolExecutionCompleteEvent,
    ToolExecutionStartData,
    ToolExecutionStartEvent,
    ToolRequest,
    UserMessageData,
    UserMessageEvent,
    events_to_model_events,
    events_to_tool_events,
)


def test_events_to_tool_events_uses_jsonl_timestamps() -> None:
    trans = transcript()
    trans._events = []

    start_ts = "2026-02-02T12:00:00Z"
    end_ts = "2026-02-02T12:00:01Z"

    events = [
        ToolExecutionStartEvent(
            type="tool.execution_start",
            timestamp=start_ts,
            data=ToolExecutionStartData(
                toolCallId="call_1",
                toolName="list_files",
                arguments={"path": "/tmp"},
            ),
        ),
        ToolExecutionCompleteEvent(
            type="tool.execution_complete",
            timestamp=end_ts,
            data=ToolExecutionCompleteData(
                toolCallId="call_1",
                success=True,
                result={"content": "ok"},
            ),
        ),
    ]

    tool_events = events_to_tool_events(events)

    assert len(tool_events) == 1
    assert isinstance(tool_events[0], ToolEvent)

    expected_start = datetime.fromisoformat("2026-02-02T12:00:00+00:00")
    expected_end = datetime.fromisoformat("2026-02-02T12:00:01+00:00")

    assert tool_events[0].timestamp == expected_start
    assert tool_events[0].completed == expected_end


def test_events_to_model_events_uses_assistant_timestamp() -> None:
    trans = transcript()
    trans._events = []

    assistant_ts = "2026-02-02T12:00:05Z"

    events = [
        UserMessageEvent(
            type="user.message",
            timestamp="2026-02-02T12:00:00Z",
            data=UserMessageData(content="hello"),
        ),
        AssistantMessageEvent(
            type="assistant.message",
            timestamp=assistant_ts,
            data=AssistantMessageData(content="hi"),
        ),
    ]

    events_to_model_events(events=events, model_name="copilot", system_content="")

    model_events = [event for event in trans._events if isinstance(event, ModelEvent)]
    assert len(model_events) == 1

    expected_ts = datetime.fromisoformat("2026-02-02T12:00:05+00:00")
    assert model_events[0].timestamp == expected_ts


def test_events_to_model_events_fallbacks_when_timestamp_missing() -> None:
    trans = transcript()
    trans._events = []

    tool_request = ToolRequest(
        toolCallId="call_2",
        name="run",
        arguments={"command": "echo hello"},
    )

    events = [
        UserMessageEvent(
            type="user.message",
            timestamp="2026-02-02T12:00:00Z",
            data=UserMessageData(content="go"),
        ),
        AssistantMessageEvent(
            type="assistant.message",
            timestamp="2026-02-02T12:00:00Z",
            data=AssistantMessageData(content="ack"),
        ),
        AssistantMessageEvent(
            type="assistant.message",
            timestamp=None,
            data=AssistantMessageData(content="", toolRequests=[tool_request]),
        ),
        ToolExecutionStartEvent(
            type="tool.execution_start",
            timestamp="2026-02-02T12:00:01Z",
            data=ToolExecutionStartData(
                toolCallId="call_2",
                toolName="run",
                arguments={"command": "echo hello"},
            ),
        ),
        ToolExecutionCompleteEvent(
            type="tool.execution_complete",
            timestamp="2026-02-02T12:00:02Z",
            data=ToolExecutionCompleteData(
                toolCallId="call_2",
                success=True,
                result={"content": "hello"},
            ),
        ),
    ]

    events_to_model_events(events=events, model_name="copilot", system_content="")

    model_events = [event for event in trans._events if isinstance(event, ModelEvent)]
    assert len(model_events) == 2

    expected_first = datetime.fromisoformat("2026-02-02T12:00:00+00:00")
    assert model_events[0].timestamp == expected_first

    expected_fallback = datetime.fromisoformat("2026-02-02T12:00:01+00:00") - timedelta(milliseconds=1)
    assert model_events[1].timestamp == expected_fallback


def test_events_to_model_events_aligns_with_missing_tool_timestamps() -> None:
    trans = transcript()
    trans._events = []

    first_tool = ToolRequest(
        toolCallId="call_1",
        name="run",
        arguments={"command": "echo first"},
    )
    second_tool = ToolRequest(
        toolCallId="call_2",
        name="run",
        arguments={"command": "echo second"},
    )

    events = [
        UserMessageEvent(
            type="user.message",
            timestamp="2026-02-02T12:00:00Z",
            data=UserMessageData(content="go"),
        ),
        AssistantMessageEvent(
            type="assistant.message",
            timestamp=None,
            data=AssistantMessageData(content="", toolRequests=[first_tool]),
        ),
        ToolExecutionStartEvent(
            type="tool.execution_start",
            timestamp=None,
            data=ToolExecutionStartData(
                toolCallId="call_1",
                toolName="run",
                arguments={"command": "echo first"},
            ),
        ),
        ToolExecutionCompleteEvent(
            type="tool.execution_complete",
            timestamp=None,
            data=ToolExecutionCompleteData(
                toolCallId="call_1",
                success=True,
                result={"content": "first"},
            ),
        ),
        AssistantMessageEvent(
            type="assistant.message",
            timestamp=None,
            data=AssistantMessageData(content="", toolRequests=[second_tool]),
        ),
        ToolExecutionStartEvent(
            type="tool.execution_start",
            timestamp="2026-02-02T12:00:10Z",
            data=ToolExecutionStartData(
                toolCallId="call_2",
                toolName="run",
                arguments={"command": "echo second"},
            ),
        ),
        ToolExecutionCompleteEvent(
            type="tool.execution_complete",
            timestamp="2026-02-02T12:00:11Z",
            data=ToolExecutionCompleteData(
                toolCallId="call_2",
                success=True,
                result={"content": "second"},
            ),
        ),
    ]

    events_to_model_events(events=events, model_name="copilot", system_content="")

    model_events = [event for event in trans._events if isinstance(event, ModelEvent)]
    assert len(model_events) == 2

    expected_second = datetime.fromisoformat("2026-02-02T12:00:10+00:00") - timedelta(milliseconds=1)
    assert model_events[1].timestamp == expected_second
    assert model_events[0].timestamp < model_events[1].timestamp


def test_events_to_tool_events_uses_tool_request_metadata() -> None:
    trans = transcript()
    trans._events = []

    tool_request = ToolRequest(
        toolCallId="call_skill",
        name="skill",
        arguments={"skill": "romulus-test"},
    )

    events = [
        AssistantMessageEvent(
            type="assistant.message",
            timestamp="2026-02-02T12:00:00Z",
            data=AssistantMessageData(content="", toolRequests=[tool_request]),
        ),
        ToolExecutionCompleteEvent(
            type="tool.execution_complete",
            timestamp="2026-02-02T12:00:01Z",
            data=ToolExecutionCompleteData(
                toolCallId="call_skill",
                success=True,
                result={"content": "Skill loaded"},
            ),
        ),
    ]

    tool_events = events_to_tool_events(events)

    assert len(tool_events) == 1
    assert tool_events[0].function == "skill"
    assert tool_events[0].arguments == {"skill": "romulus-test"}


def test_events_to_model_events_offsets_when_timestamp_matches_tool() -> None:
    trans = transcript()
    trans._events = []

    tool_request = ToolRequest(
        toolCallId="call_same",
        name="bash",
        arguments={"command": "echo hi"},
    )

    events = [
        AssistantMessageEvent(
            type="assistant.message",
            timestamp="2026-02-02T12:00:00Z",
            data=AssistantMessageData(content="", toolRequests=[tool_request]),
        ),
        ToolExecutionStartEvent(
            type="tool.execution_start",
            timestamp="2026-02-02T12:00:00Z",
            data=ToolExecutionStartData(
                toolCallId="call_same",
                toolName="bash",
                arguments={"command": "echo hi"},
            ),
        ),
    ]

    events_to_model_events(events=events, model_name="copilot", system_content="")

    model_events = [event for event in trans._events if isinstance(event, ModelEvent)]
    assert len(model_events) == 1

    expected_tool_ts = datetime.fromisoformat("2026-02-02T12:00:00+00:00")
    assert model_events[0].timestamp == expected_tool_ts - timedelta(milliseconds=1)
