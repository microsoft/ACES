# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""CLI output parsing for bridge-based agent solvers.

Extracts structured data from agent CLI stdout/stderr for observability.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ParsedCLIOutput(BaseModel):
    """Structured data extracted from agent CLI output."""

    model_config = ConfigDict(frozen=True)

    agent_type: str
    """'claude_code' or 'copilot'."""

    final_response: str | None = None
    """Final text response from the agent, if found."""

    error_messages: tuple[str, ...] = Field(default_factory=tuple)
    """Error messages extracted from output."""

    tool_uses: int = 0
    """Number of tool-use events detected in stream output."""

    stream_events: int = 0
    """Total number of stream events parsed (Claude Code stream-json)."""


def parse_claude_code_stream_json(stdout: str) -> ParsedCLIOutput:
    """Parse Claude Code's --output-format stream-json --verbose output.

    Each line of stdout is a JSON object with a 'type' field.
    Malformed lines are silently skipped.

    Args:
        stdout: Raw stdout from the Claude Code CLI.

    Returns:
        Parsed output with extracted events, tool uses, and final response.
    """
    final_response: str | None = None
    error_messages: list[str] = []
    tool_uses = 0
    stream_events = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(event, dict):
            continue

        stream_events += 1
        event_type = event.get("type", "")

        if event_type == "tool_use":
            tool_uses += 1
        elif event_type == "result":
            result_text = event.get("result")
            if isinstance(result_text, str):
                final_response = result_text
        elif event_type == "error":
            err = event.get("error", {})
            if isinstance(err, dict):
                msg = err.get("message", "")
                if msg:
                    error_messages.append(str(msg))
            elif isinstance(err, str):
                error_messages.append(err)

    return ParsedCLIOutput(
        agent_type="claude_code",
        final_response=final_response,
        error_messages=tuple(error_messages),
        tool_uses=tool_uses,
        stream_events=stream_events,
    )


def parse_copilot_stderr(stderr: str) -> ParsedCLIOutput:
    """Parse Copilot SDK stderr for responses and error lines.

    Looks for:
    - ``COPILOT_RESPONSE: <text>`` — the agent's final response
    - ``ERROR: <text>`` — error messages

    Args:
        stderr: Raw stderr from the Copilot runner process.

    Returns:
        Parsed output with extracted response and error messages.
    """
    final_response: str | None = None
    error_messages: list[str] = []

    for line in stderr.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("COPILOT_RESPONSE:"):
            final_response = line[len("COPILOT_RESPONSE:") :].strip()
        elif line.startswith("ERROR:"):
            error_messages.append(line[len("ERROR:") :].strip())

    return ParsedCLIOutput(
        agent_type="copilot",
        final_response=final_response,
        error_messages=tuple(error_messages),
    )
