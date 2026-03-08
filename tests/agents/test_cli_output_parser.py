"""Tests for CLI output parsing utilities."""

import json

import pytest
from pydantic import ValidationError

from saber.agents.cli_output_parser import (
    ParsedCLIOutput,
    parse_claude_code_stream_json,
    parse_copilot_stderr,
)


class TestParsedCLIOutput:
    """ParsedCLIOutput model tests."""

    def test_defaults(self) -> None:
        output = ParsedCLIOutput(agent_type="claude_code")
        assert output.agent_type == "claude_code"
        assert output.final_response is None
        assert output.error_messages == ()
        assert output.tool_uses == 0
        assert output.stream_events == 0

    def test_frozen(self) -> None:
        output = ParsedCLIOutput(agent_type="copilot")
        with pytest.raises(ValidationError):
            output.agent_type = "other"


class TestParseClaudeCodeStreamJson:
    """parse_claude_code_stream_json() tests."""

    def test_empty_stdout(self) -> None:
        result = parse_claude_code_stream_json("")
        assert result.agent_type == "claude_code"
        assert result.final_response is None
        assert result.tool_uses == 0
        assert result.stream_events == 0

    def test_single_text_event(self) -> None:
        """A 'result' type event with 'result' text field → final_response."""
        line = json.dumps({"type": "result", "result": "Hello world"})
        result = parse_claude_code_stream_json(line)
        assert result.final_response == "Hello world"
        assert result.stream_events == 1

    def test_tool_use_events_counted(self) -> None:
        """Events with type='tool_use' are counted."""
        lines = "\n".join(
            [
                json.dumps({"type": "tool_use", "name": "bash"}),
                json.dumps({"type": "tool_use", "name": "read_file"}),
                json.dumps({"type": "text", "text": "done"}),
            ]
        )
        result = parse_claude_code_stream_json(lines)
        assert result.tool_uses == 2
        assert result.stream_events == 3

    def test_malformed_json_lines_skipped(self) -> None:
        """Non-JSON lines are silently skipped."""
        lines = "not json\n" + json.dumps({"type": "result", "result": "ok"}) + "\nalso not json"
        result = parse_claude_code_stream_json(lines)
        assert result.final_response == "ok"
        assert result.stream_events == 1  # only valid JSON lines counted

    def test_error_events_captured(self) -> None:
        """Events with type='error' have their message captured."""
        line = json.dumps({"type": "error", "error": {"message": "something failed"}})
        result = parse_claude_code_stream_json(line)
        assert "something failed" in result.error_messages

    def test_multiple_text_events_last_wins(self) -> None:
        """Multiple result events → last one's text is final_response."""
        lines = "\n".join(
            [
                json.dumps({"type": "result", "result": "first"}),
                json.dumps({"type": "result", "result": "second"}),
            ]
        )
        result = parse_claude_code_stream_json(lines)
        assert result.final_response == "second"


class TestParseCopilotStderr:
    """parse_copilot_stderr() tests."""

    def test_empty_stderr(self) -> None:
        result = parse_copilot_stderr("")
        assert result.agent_type == "copilot"
        assert result.final_response is None
        assert result.error_messages == ()

    def test_copilot_response_extracted(self) -> None:
        """Lines with COPILOT_RESPONSE: prefix are captured."""
        stderr = "COPILOT_RESPONSE: Hello from copilot\nother noise"
        result = parse_copilot_stderr(stderr)
        assert result.final_response == "Hello from copilot"

    def test_error_lines_captured(self) -> None:
        """Lines with ERROR: prefix are captured in error_messages."""
        stderr = "INFO: starting\nERROR: something broke\nERROR: another issue"
        result = parse_copilot_stderr(stderr)
        assert len(result.error_messages) == 2
        assert "something broke" in result.error_messages[0]

    def test_non_prefixed_lines_ignored(self) -> None:
        """Random stderr lines not captured."""
        stderr = "debug info line 1\nwarning: blah\nsome output"
        result = parse_copilot_stderr(stderr)
        assert result.final_response is None
        assert result.error_messages == ()
