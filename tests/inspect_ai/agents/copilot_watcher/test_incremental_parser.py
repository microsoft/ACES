"""Tests for IncrementalEventParser.

TDD tests for the incremental events.jsonl parser that reads new events
without re-parsing the entire file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from saber.inspect_ai.agents.registry.copilot.events import (
    AssistantMessageEvent,
    SessionStartEvent,
    UserMessageEvent,
)

from .conftest import (
    SAMPLE_ASSISTANT_MESSAGE,
    SAMPLE_SESSION_IDLE,
    SAMPLE_SESSION_START,
    SAMPLE_USER_MESSAGE,
)


class TestParserState:
    """Tests for ParserState immutable model."""

    def test_parser_state_is_frozen(self) -> None:
        """ParserState should be immutable."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import ParserState

        state = ParserState(byte_offset=0, partial_line="", events_parsed=0)

        with pytest.raises(ValidationError):  # Pydantic ValidationError for frozen
            state.byte_offset = 10  # type: ignore[misc]

    def test_parser_state_fields(self) -> None:
        """ParserState should have correct fields."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import ParserState

        state = ParserState(byte_offset=100, partial_line="partial", events_parsed=5)

        assert state.byte_offset == 100
        assert state.partial_line == "partial"
        assert state.events_parsed == 5


class TestIncrementalEventParserInitialization:
    """Tests for IncrementalEventParser initialization."""

    def test_init_with_path(self, events_file_path: Path) -> None:
        """Parser should initialize with a file path."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(events_file_path)

        assert parser.events_path == events_file_path

    def test_initial_state_is_zeroed(self, events_file_path: Path) -> None:
        """Initial state should have zero offset and empty buffer."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(events_file_path)
        state = parser.state

        assert state.byte_offset == 0
        assert state.partial_line == ""
        assert state.events_parsed == 0

    def test_state_property_returns_immutable_snapshot(self, events_file_path: Path) -> None:
        """State property should return an immutable snapshot."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(events_file_path)
        state = parser.state

        # State should be frozen
        with pytest.raises(ValidationError):
            state.byte_offset = 999  # type: ignore[misc]


class TestIncrementalEventParserParsing:
    """Tests for parsing events incrementally."""

    def test_parse_nonexistent_file_returns_empty(self, events_file_path: Path) -> None:
        """Parsing a non-existent file should return empty list."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(events_file_path)
        events = parser.parse_new_events()

        assert events == []
        assert parser.state.byte_offset == 0

    def test_parse_empty_file_returns_empty(self, empty_events_file: Path) -> None:
        """Parsing an empty file should return empty list."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(empty_events_file)
        events = parser.parse_new_events()

        assert events == []

    def test_parse_initial_events(self, sample_events_jsonl: Path) -> None:
        """First parse should return all existing events."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(sample_events_jsonl)
        events = parser.parse_new_events()

        assert len(events) == 2
        assert isinstance(events[0], SessionStartEvent)
        assert isinstance(events[1], UserMessageEvent)

    def test_parse_updates_byte_offset(self, sample_events_jsonl: Path) -> None:
        """Parsing should update the byte offset."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(sample_events_jsonl)
        parser.parse_new_events()

        # Byte offset should be at end of file
        file_size = sample_events_jsonl.stat().st_size
        assert parser.state.byte_offset == file_size

    def test_parse_updates_events_parsed_count(self, sample_events_jsonl: Path) -> None:
        """Parsing should update the events_parsed count."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(sample_events_jsonl)
        parser.parse_new_events()

        assert parser.state.events_parsed == 2

    def test_second_parse_returns_empty_if_no_changes(self, sample_events_jsonl: Path) -> None:
        """Second parse with no new data should return empty list."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(sample_events_jsonl)
        parser.parse_new_events()

        # Second parse should return nothing
        events = parser.parse_new_events()
        assert events == []

    def test_parse_new_appended_events(self, sample_events_jsonl: Path) -> None:
        """Parser should only return newly appended events."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(sample_events_jsonl)

        # First parse gets initial events
        initial_events = parser.parse_new_events()
        assert len(initial_events) == 2

        # Append new event
        with open(sample_events_jsonl, "a") as f:
            f.write(f"{SAMPLE_ASSISTANT_MESSAGE}\n")

        # Second parse should only get the new event
        new_events = parser.parse_new_events()
        assert len(new_events) == 1
        assert isinstance(new_events[0], AssistantMessageEvent)

    def test_parse_multiple_new_events(self, sample_events_jsonl: Path) -> None:
        """Parser should handle multiple new events in one append."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(sample_events_jsonl)
        parser.parse_new_events()

        # Append multiple events
        with open(sample_events_jsonl, "a") as f:
            f.write(f"{SAMPLE_ASSISTANT_MESSAGE}\n")
            f.write(f"{SAMPLE_SESSION_IDLE}\n")

        new_events = parser.parse_new_events()
        assert len(new_events) == 2
        assert parser.state.events_parsed == 4


class TestPartialLineHandling:
    """Tests for partial line buffering at EOF."""

    def test_partial_line_is_buffered(self, tmp_path: Path) -> None:
        """Incomplete line at EOF should be buffered, not parsed."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        events_file = tmp_path / "events.jsonl"
        # Write complete line + partial line (no newline)
        events_file.write_text(f"{SAMPLE_SESSION_START}\n" + '{"type": "user')

        parser = IncrementalEventParser(events_file)
        events = parser.parse_new_events()

        # Should only parse the complete line
        assert len(events) == 1
        assert isinstance(events[0], SessionStartEvent)
        # Partial line should be buffered
        assert parser.state.partial_line == '{"type": "user'

    def test_partial_line_completed_on_next_parse(self, tmp_path: Path) -> None:
        """Buffered partial line should be completed when rest arrives."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        events_file = tmp_path / "events.jsonl"
        # Write partial JSON
        events_file.write_text('{"type": "session.start", "data": {"sessionId":')

        parser = IncrementalEventParser(events_file)
        events = parser.parse_new_events()

        # Nothing complete yet
        assert len(events) == 0
        assert '"sessionId":' in parser.state.partial_line

        # Complete the JSON
        with open(events_file, "a") as f:
            f.write(' "test-123"}}\n')

        events = parser.parse_new_events()

        # Now should parse complete event
        assert len(events) == 1
        assert isinstance(events[0], SessionStartEvent)
        assert parser.state.partial_line == ""


class TestCorruptedLineHandling:
    """Tests for handling corrupted/invalid JSON lines."""

    def test_skip_corrupted_json_line(self, tmp_path: Path) -> None:
        """Corrupted JSON lines should be skipped, not crash."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        events_file = tmp_path / "events.jsonl"
        events_file.write_text(
            f"{SAMPLE_SESSION_START}\n"
            "this is not valid json\n"
            f"{SAMPLE_USER_MESSAGE}\n"
        )

        parser = IncrementalEventParser(events_file)
        events = parser.parse_new_events()

        # Should get 2 valid events, skip the corrupted one
        assert len(events) == 2
        assert isinstance(events[0], SessionStartEvent)
        assert isinstance(events[1], UserMessageEvent)

    def test_skip_empty_lines(self, tmp_path: Path) -> None:
        """Empty lines should be skipped."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        events_file = tmp_path / "events.jsonl"
        events_file.write_text(
            f"{SAMPLE_SESSION_START}\n"
            "\n"
            "   \n"
            f"{SAMPLE_USER_MESSAGE}\n"
        )

        parser = IncrementalEventParser(events_file)
        events = parser.parse_new_events()

        assert len(events) == 2


class TestParserReset:
    """Tests for parser reset functionality."""

    def test_reset_clears_state(self, sample_events_jsonl: Path) -> None:
        """Reset should clear all state."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(sample_events_jsonl)
        parser.parse_new_events()

        # State should be non-zero
        assert parser.state.byte_offset > 0
        assert parser.state.events_parsed > 0

        parser.reset()

        # State should be zeroed
        assert parser.state.byte_offset == 0
        assert parser.state.partial_line == ""
        assert parser.state.events_parsed == 0

    def test_parse_after_reset_re_reads_all(self, sample_events_jsonl: Path) -> None:
        """Parse after reset should re-read all events."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(sample_events_jsonl)

        first_events = parser.parse_new_events()
        assert len(first_events) == 2

        parser.reset()

        second_events = parser.parse_new_events()
        assert len(second_events) == 2


class TestFileTruncationDetection:
    """Tests for detecting file truncation/rotation."""

    def test_detect_file_truncation(self, sample_events_jsonl: Path) -> None:
        """Parser should detect when file is truncated and reset."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

        parser = IncrementalEventParser(sample_events_jsonl)
        parser.parse_new_events()

        old_offset = parser.state.byte_offset

        # Truncate file (simulating rotation)
        sample_events_jsonl.write_text(f"{SAMPLE_SESSION_START}\n")

        # File is now smaller than our offset
        events = parser.parse_new_events()

        # Parser should have auto-reset and re-read
        assert parser.state.byte_offset < old_offset
        assert len(events) == 1
