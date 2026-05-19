# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Pydantic models for bridge generation tracking."""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field

DELEGATION_TOOL_NAMES: frozenset[str] = frozenset({"task", "read_agent"})
# Tool function names that indicate subagent delegation (Copilot SDK pattern).


class CallType(enum.StrEnum):
    """Classification of a model generation call through the bridge."""

    MAIN = "main"
    SUBAGENT = "subagent"


class GenerationMetadata(BaseModel):
    """Per-generation metadata recorded as an InfoEvent."""

    model_config = ConfigDict(frozen=True)

    call_type: CallType
    """Whether this was a main-thread or subagent call."""

    model_name: str
    """The model name reported by the bridge."""

    message_count: int
    """Number of messages in the conversation at generation time."""

    generation_index: int
    """Zero-based index of this generation within the bridge session."""

    tool_count: int
    """Number of tools available for this generation."""

    delegation_detected: bool = False
    """Whether this generation was classified via delegation-tool heuristic (H2)."""

    input_tokens: int | None = None
    """Input token count from model usage, if available."""

    output_tokens: int | None = None
    """Output token count from model usage, if available."""


class BridgeSessionSummary(BaseModel):
    """Summary of a complete bridge session."""

    model_config = ConfigDict(frozen=True)

    total_generations: int = 0
    """Total number of model generation calls (main + subagent)."""

    main_generations: int = 0
    """Number of generation calls from the main agent thread."""

    subagent_generations: int = 0
    """Number of generation calls from subagent threads."""

    total_tool_calls: int = 0
    """High-water mark of tool call count seen in any generation's message history."""

    models_used: tuple[str, ...] = Field(default_factory=tuple)
    """Distinct model names observed during the session."""
