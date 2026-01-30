"""Shared fixtures for Copilot agent tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser


# =============================================================================
# Sample Events Data
# =============================================================================

SAMPLE_SESSION_START = '{"type": "session.start", "data": {"sessionId": "test-123"}}'
SAMPLE_USER_MESSAGE = '{"type": "user.message", "data": {"content": "Hello, world!"}}'
SAMPLE_ASSISTANT_MESSAGE = (
    '{"type": "assistant.message", "data": {"messageId": "msg-1", '
    '"content": "I can help with that.", "toolRequests": []}}'
)
SAMPLE_ASSISTANT_WITH_TOOL = (
    '{"type": "assistant.message", "data": {"messageId": "msg-2", '
    '"content": "", "toolRequests": [{"toolCallId": "call-1", "name": "bash", '
    '"arguments": {"command": "ls -la"}, "type": "function"}]}}'
)
SAMPLE_TOOL_COMPLETE = (
    '{"type": "tool.execution_complete", "data": {"toolCallId": "call-1", '
    '"success": true, "result": {"content": "file1.txt\\nfile2.txt"}}}'
)
SAMPLE_SESSION_IDLE = '{"type": "session.idle"}'


# =============================================================================
# File Fixtures
# =============================================================================


@pytest.fixture
def sample_events_jsonl(tmp_path: Path) -> Path:
    """Create a sample events.jsonl file for testing."""
    events_file = tmp_path / "events.jsonl"
    events_file.write_text(
        f"{SAMPLE_SESSION_START}\n"
        f"{SAMPLE_USER_MESSAGE}\n"
    )
    return events_file


@pytest.fixture
def empty_events_file(tmp_path: Path) -> Path:
    """Create an empty events.jsonl file."""
    events_file = tmp_path / "events.jsonl"
    events_file.touch()
    return events_file


@pytest.fixture
def events_file_path(tmp_path: Path) -> Path:
    """Return path to a non-existent events.jsonl file."""
    return tmp_path / "events.jsonl"


@pytest.fixture
def events_parser(sample_events_jsonl: Path) -> IncrementalEventParser:
    """Create an IncrementalEventParser for the sample file."""
    from saber.inspect_ai.agents.registry.copilot.events_watcher import IncrementalEventParser

    return IncrementalEventParser(sample_events_jsonl)


# =============================================================================
# Multi-Event Fixtures
# =============================================================================


@pytest.fixture
def full_conversation_events(tmp_path: Path) -> Path:
    """Create events.jsonl with a full conversation."""
    events_file = tmp_path / "events.jsonl"
    events_file.write_text(
        f"{SAMPLE_SESSION_START}\n"
        f"{SAMPLE_USER_MESSAGE}\n"
        f"{SAMPLE_ASSISTANT_WITH_TOOL}\n"
        f"{SAMPLE_TOOL_COMPLETE}\n"
        f"{SAMPLE_ASSISTANT_MESSAGE}\n"
        f"{SAMPLE_SESSION_IDLE}\n"
    )
    return events_file
