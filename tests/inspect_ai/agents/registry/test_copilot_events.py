"""Tests for Copilot events.jsonl parsing.

Tests follow TDD - written before implementation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage


class TestCopilotEventModels:
    """Tests for Pydantic event models."""

    def test_parse_session_start_event(self) -> None:
        """Test parsing a session.start event from JSONL."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "session.start",
            "timestamp": "2026-01-28T10:00:00Z",
            "data": {"sessionId": "test-session-123"},
        }
        event = parse_event(event_json)

        assert event.type == "session.start"
        assert event.data.sessionId == "test-session-123"

    def test_parse_user_message_event(self) -> None:
        """Test parsing a user.message event."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "user.message",
            "timestamp": "2026-01-28T10:00:01Z",
            "data": {"content": "Hello, please help me with this task."},
        }
        event = parse_event(event_json)

        assert event.type == "user.message"
        assert event.data.content == "Hello, please help me with this task."

    def test_parse_assistant_message_event_with_tool_requests(self) -> None:
        """Test parsing assistant.message event with tool calls."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "assistant.message",
            "timestamp": "2026-01-28T10:00:02Z",
            "data": {
                "messageId": "msg-456",
                "content": "I'll run a command to check.",
                "toolRequests": [
                    {
                        "toolCallId": "call-789",
                        "name": "bash",
                        "arguments": {"command": "ls -la"},
                        "type": "function",
                    }
                ],
            },
        }
        event = parse_event(event_json)

        assert event.type == "assistant.message"
        assert event.data.messageId == "msg-456"
        assert event.data.content == "I'll run a command to check."
        assert len(event.data.toolRequests) == 1
        assert event.data.toolRequests[0].name == "bash"
        assert event.data.toolRequests[0].arguments == {"command": "ls -la"}

    def test_parse_assistant_message_event_without_tool_requests(self) -> None:
        """Test parsing assistant.message event without tool calls."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "assistant.message",
            "timestamp": "2026-01-28T10:00:02Z",
            "data": {
                "messageId": "msg-456",
                "content": "Task completed successfully.",
            },
        }
        event = parse_event(event_json)

        assert event.type == "assistant.message"
        assert event.data.content == "Task completed successfully."
        assert event.data.toolRequests == []

    def test_parse_tool_execution_complete_event(self) -> None:
        """Test parsing tool.execution_complete event."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "tool.execution_complete",
            "timestamp": "2026-01-28T10:00:03Z",
            "data": {
                "toolCallId": "call-789",
                "success": True,
                "result": {"content": "file1.txt\nfile2.txt", "detailedContent": "..."},
            },
        }
        event = parse_event(event_json)

        assert event.type == "tool.execution_complete"
        assert event.data.toolCallId == "call-789"
        assert event.data.success is True
        assert event.data.result["content"] == "file1.txt\nfile2.txt"

    def test_parse_session_idle_event(self) -> None:
        """Test parsing session.idle event."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "session.idle",
            "timestamp": "2026-01-28T10:00:04Z",
        }
        event = parse_event(event_json)

        assert event.type == "session.idle"

    def test_parse_assistant_turn_start_event(self) -> None:
        """Test parsing assistant.turn_start event."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "assistant.turn_start",
            "timestamp": "2026-01-28T10:00:05Z",
        }
        event = parse_event(event_json)

        assert event.type == "assistant.turn_start"

    def test_parse_assistant_turn_end_event(self) -> None:
        """Test parsing assistant.turn_end event."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "assistant.turn_end",
            "timestamp": "2026-01-28T10:00:06Z",
            "data": {"summary": "Completed analysis"},
        }
        event = parse_event(event_json)

        assert event.type == "assistant.turn_end"
        assert event.data is not None
        assert event.data.summary == "Completed analysis"

    def test_parse_session_error_event(self) -> None:
        """Test parsing a session.error event."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "session.error",
            "timestamp": "2026-01-28T10:00:00Z",
            "data": {"message": "Connection failed"},
        }
        event = parse_event(event_json)

        assert event.type == "session.error"
        assert event.data is not None
        assert event.data["message"] == "Connection failed"

    def test_parse_unknown_event_type(self) -> None:
        """Test that unknown event types are handled gracefully."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "unknown.event",
            "timestamp": "2026-01-28T10:00:07Z",
            "data": {"foo": "bar"},
        }
        event = parse_event(event_json)

        # Unknown events should be parsed as UnknownEvent
        assert event.type == "unknown.event"

    def test_parse_tool_execution_start_event(self) -> None:
        """Test parsing tool.execution_start event."""
        from saber.inspect_ai.agents.registry.copilot.events import parse_event

        event_json = {
            "type": "tool.execution_start",
            "timestamp": "2026-01-28T10:00:08Z",
            "data": {"toolCallId": "call-789", "name": "bash"},
        }
        event = parse_event(event_json)

        assert event.type == "tool.execution_start"
        assert event.data.toolCallId == "call-789"
        assert event.data.name == "bash"


class TestSessionEventLog:
    """Tests for SessionEventLog class."""

    def test_read_events_from_file(self, tmp_path: Path) -> None:
        """Test reading events from a JSONL file."""
        from saber.inspect_ai.agents.registry.copilot.events import SessionEventLog

        # Create a mock events.jsonl file
        events_dir = tmp_path / "session-123"
        events_dir.mkdir()
        events_file = events_dir / "events.jsonl"

        events_data = [
            {"type": "session.start", "timestamp": "2026-01-28T10:00:00Z", "data": {"sessionId": "session-123"}},
            {"type": "user.message", "timestamp": "2026-01-28T10:00:01Z", "data": {"content": "Hello"}},
            {"type": "session.idle", "timestamp": "2026-01-28T10:00:02Z"},
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events_data))

        # Read events
        log = SessionEventLog(session_id="session-123", base_path=tmp_path)
        events = log.read_events()

        assert len(events) == 3
        assert events[0].type == "session.start"
        assert events[1].type == "user.message"
        assert events[2].type == "session.idle"

    def test_read_events_missing_file(self, tmp_path: Path) -> None:
        """Test handling of missing events.jsonl file."""
        from saber.inspect_ai.agents.registry.copilot.events import SessionEventLog

        log = SessionEventLog(session_id="nonexistent-session", base_path=tmp_path)
        events = log.read_events()

        # Should return empty list for missing file
        assert events == []

    def test_read_events_corrupted_line(self, tmp_path: Path) -> None:
        """Test handling of corrupted JSONL lines."""
        from saber.inspect_ai.agents.registry.copilot.events import SessionEventLog

        events_dir = tmp_path / "session-456"
        events_dir.mkdir()
        events_file = events_dir / "events.jsonl"

        # Mix of valid and invalid lines
        events_file.write_text(
            '{"type": "session.start", "timestamp": "2026-01-28T10:00:00Z", "data": {"sessionId": "session-456"}}\n'
            "not valid json\n"
            '{"type": "session.idle", "timestamp": "2026-01-28T10:00:02Z"}\n'
        )

        log = SessionEventLog(session_id="session-456", base_path=tmp_path)
        events = log.read_events()

        # Should skip corrupted lines
        assert len(events) == 2
        assert events[0].type == "session.start"
        assert events[1].type == "session.idle"

    def test_find_submission(self, tmp_path: Path) -> None:
        """Test finding submission answer from events."""
        from saber.inspect_ai.agents.registry.copilot.events import SessionEventLog

        events_dir = tmp_path / "session-789"
        events_dir.mkdir()
        events_file = events_dir / "events.jsonl"

        events_data = [
            {"type": "session.start", "timestamp": "2026-01-28T10:00:00Z", "data": {"sessionId": "session-789"}},
            {"type": "user.message", "timestamp": "2026-01-28T10:00:01Z", "data": {"content": "Task prompt"}},
            {
                "type": "assistant.message",
                "timestamp": "2026-01-28T10:00:02Z",
                "data": {
                    "messageId": "msg-1",
                    "content": "I found the answer.",
                    "toolRequests": [
                        {
                            "toolCallId": "call-submit",
                            "name": "submit",
                            "arguments": {"answer": "The answer is 42"},
                            "type": "function",
                        }
                    ],
                },
            },
            {"type": "session.idle", "timestamp": "2026-01-28T10:00:03Z"},
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events_data))

        log = SessionEventLog(session_id="session-789", base_path=tmp_path)
        answer = log.find_submission()

        assert answer == "The answer is 42"

    def test_find_submission_no_submit(self, tmp_path: Path) -> None:
        """Test find_submission returns None when no submit call exists."""
        from saber.inspect_ai.agents.registry.copilot.events import SessionEventLog

        events_dir = tmp_path / "session-no-submit"
        events_dir.mkdir()
        events_file = events_dir / "events.jsonl"

        events_data = [
            {"type": "session.start", "timestamp": "2026-01-28T10:00:00Z", "data": {"sessionId": "session-no-submit"}},
            {"type": "user.message", "timestamp": "2026-01-28T10:00:01Z", "data": {"content": "Task prompt"}},
            {"type": "session.idle", "timestamp": "2026-01-28T10:00:02Z"},
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events_data))

        log = SessionEventLog(session_id="session-no-submit", base_path=tmp_path)
        answer = log.find_submission()

        assert answer is None


class TestChatMessageConversion:
    """Tests for converting events to ChatMessage format."""

    def test_events_to_chat_messages_basic(self, tmp_path: Path) -> None:
        """Test basic conversion of events to chat messages."""
        from saber.inspect_ai.agents.registry.copilot.events import (
            SessionEventLog,
            events_to_chat_messages,
        )

        events_dir = tmp_path / "session-conv"
        events_dir.mkdir()
        events_file = events_dir / "events.jsonl"

        events_data = [
            {"type": "session.start", "timestamp": "2026-01-28T10:00:00Z", "data": {"sessionId": "session-conv"}},
            {"type": "user.message", "timestamp": "2026-01-28T10:00:01Z", "data": {"content": "Hello agent"}},
            {
                "type": "assistant.message",
                "timestamp": "2026-01-28T10:00:02Z",
                "data": {"messageId": "msg-1", "content": "Hello! How can I help?"},
            },
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events_data))

        log = SessionEventLog(session_id="session-conv", base_path=tmp_path)
        events = log.read_events()
        messages = events_to_chat_messages(events, system_content="You are a helpful assistant.")

        # System, User, Assistant
        assert len(messages) == 3

        # Check types
        from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

        assert isinstance(messages[0], ChatMessageSystem)
        assert isinstance(messages[1], ChatMessageUser)
        assert isinstance(messages[2], ChatMessageAssistant)

        assert messages[0].content == "You are a helpful assistant."
        assert messages[1].content == "Hello agent"
        assert messages[2].content == "Hello! How can I help?"

    def test_events_to_chat_messages_with_tool_calls(self, tmp_path: Path) -> None:
        """Test conversion with tool calls and results."""
        from saber.inspect_ai.agents.registry.copilot.events import (
            SessionEventLog,
            events_to_chat_messages,
        )

        events_dir = tmp_path / "session-tools"
        events_dir.mkdir()
        events_file = events_dir / "events.jsonl"

        events_data = [
            {"type": "session.start", "timestamp": "2026-01-28T10:00:00Z", "data": {"sessionId": "session-tools"}},
            {"type": "user.message", "timestamp": "2026-01-28T10:00:01Z", "data": {"content": "List files"}},
            {
                "type": "assistant.message",
                "timestamp": "2026-01-28T10:00:02Z",
                "data": {
                    "messageId": "msg-1",
                    "content": "I'll list the files.",
                    "toolRequests": [
                        {
                            "toolCallId": "call-1",
                            "name": "bash",
                            "arguments": {"command": "ls"},
                            "type": "function",
                        }
                    ],
                },
            },
            {
                "type": "tool.execution_complete",
                "timestamp": "2026-01-28T10:00:03Z",
                "data": {"toolCallId": "call-1", "success": True, "result": {"content": "file1.txt\nfile2.txt"}},
            },
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events_data))

        log = SessionEventLog(session_id="session-tools", base_path=tmp_path)
        events = log.read_events()
        messages = events_to_chat_messages(events, system_content="System")

        # System, User, Assistant (with tool_calls), Tool result
        assert len(messages) == 4

        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool

        # Assistant message should have tool_calls
        assistant_msg = messages[2]
        assert isinstance(assistant_msg, ChatMessageAssistant)
        assert assistant_msg.tool_calls is not None
        assert len(assistant_msg.tool_calls) == 1
        assert assistant_msg.tool_calls[0].function == "bash"
        assert assistant_msg.tool_calls[0].id == "call-1"

        # Tool result message
        tool_msg = messages[3]
        assert isinstance(tool_msg, ChatMessageTool)
        assert tool_msg.tool_call_id == "call-1"
        assert tool_msg.content == "file1.txt\nfile2.txt"

    def test_events_to_chat_messages_multiple_turns(self, tmp_path: Path) -> None:
        """Test conversion with multiple conversation turns."""
        from saber.inspect_ai.agents.registry.copilot.events import (
            SessionEventLog,
            events_to_chat_messages,
        )

        events_dir = tmp_path / "session-multi"
        events_dir.mkdir()
        events_file = events_dir / "events.jsonl"

        events_data = [
            {"type": "session.start", "timestamp": "2026-01-28T10:00:00Z", "data": {"sessionId": "session-multi"}},
            {"type": "user.message", "timestamp": "2026-01-28T10:00:01Z", "data": {"content": "First prompt"}},
            {
                "type": "assistant.message",
                "timestamp": "2026-01-28T10:00:02Z",
                "data": {"messageId": "msg-1", "content": "First response"},
            },
            {"type": "user.message", "timestamp": "2026-01-28T10:00:03Z", "data": {"content": "Continue"}},
            {
                "type": "assistant.message",
                "timestamp": "2026-01-28T10:00:04Z",
                "data": {"messageId": "msg-2", "content": "Second response"},
            },
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events_data))

        log = SessionEventLog(session_id="session-multi", base_path=tmp_path)
        events = log.read_events()
        messages = events_to_chat_messages(events, system_content="System")

        # System, User1, Assistant1, User2, Assistant2
        assert len(messages) == 5

        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        assert isinstance(messages[1], ChatMessageUser)
        assert messages[1].content == "First prompt"
        assert isinstance(messages[2], ChatMessageAssistant)
        assert messages[2].content == "First response"
        assert isinstance(messages[3], ChatMessageUser)
        assert messages[3].content == "Continue"
        assert isinstance(messages[4], ChatMessageAssistant)
        assert messages[4].content == "Second response"

    def test_events_to_chat_messages_empty_content(self, tmp_path: Path) -> None:
        """Test handling of events with empty content."""
        from saber.inspect_ai.agents.registry.copilot.events import (
            SessionEventLog,
            events_to_chat_messages,
        )

        events_dir = tmp_path / "session-empty"
        events_dir.mkdir()
        events_file = events_dir / "events.jsonl"

        events_data = [
            {"type": "session.start", "timestamp": "2026-01-28T10:00:00Z", "data": {"sessionId": "session-empty"}},
            {"type": "user.message", "timestamp": "2026-01-28T10:00:01Z", "data": {"content": "Question"}},
            {
                "type": "assistant.message",
                "timestamp": "2026-01-28T10:00:02Z",
                "data": {
                    "messageId": "msg-1",
                    "content": "",  # Empty content but with tool call
                    "toolRequests": [
                        {"toolCallId": "call-1", "name": "bash", "arguments": {"command": "pwd"}, "type": "function"}
                    ],
                },
            },
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events_data))

        log = SessionEventLog(session_id="session-empty", base_path=tmp_path)
        events = log.read_events()
        messages = events_to_chat_messages(events, system_content="System")

        # Should still include the assistant message with tool calls even if content is empty
        from inspect_ai.model import ChatMessageAssistant

        assistant_msgs = [m for m in messages if isinstance(m, ChatMessageAssistant)]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].tool_calls is not None
