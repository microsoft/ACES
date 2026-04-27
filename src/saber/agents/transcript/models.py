"""Pydantic models for copilot-log schemaVersion 3 format.

Hyenas writes one JSON file per agent conversation (e.g. repo-mapper,
function-auditor, etc.) under a ``copilot-log/`` directory.  Each file
uses the **timeline** structure (not ``turns``), containing a flat,
sequence-ordered list of events: system/user/assistant messages,
tool_request, tool_execution_start, and tool_execution_complete.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# ---------- timeline entries ----------

TimelineKind = Literal[
    "system_message",
    "user_message",
    "assistant_message",
    "tool_request",
    "tool_execution_start",
    "tool_execution_complete",
]


class TimelineEntry(BaseModel):
    """A single entry in the copilot-log timeline."""

    model_config = ConfigDict(frozen=True)

    sequence: int
    timestamp: str
    kind: TimelineKind

    # Present on message kinds
    content: str | None = None

    # Present on tool_request / tool_execution_*
    toolCallId: str | None = None
    toolName: str | None = None
    arguments: dict[str, object] | None = None

    # Present on tool_execution_complete
    success: bool | None = None
    result: dict[str, str] | None = None


# ---------- session metadata ----------


class CopilotSessionMeta(BaseModel):
    """Session metadata from the copilot-log ``session`` object."""

    model_config = ConfigDict(frozen=True)

    agentName: str
    model: str
    startedAt: str
    endedAt: str
    durationMs: int
    turnCount: int
    route: str = ""
    endpointLabel: str = ""


# ---------- top-level session ----------


class CopilotSession(BaseModel):
    """A full copilot-log JSON file (one agent conversation)."""

    model_config = ConfigDict(frozen=True)

    schemaVersion: int = 3
    stage: str = ""
    session: CopilotSessionMeta
    timeline: list[TimelineEntry] = []
