# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Scoring context models for SABER evaluation.

Provides ``ToolStep`` and ``ScoringContext`` — the immutable value objects
passed to every :class:`SaberScoringStrategy`.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessage
from pydantic import BaseModel, ConfigDict, Field

from saber.config.models import ScorerConfig


class ToolStep(BaseModel):
    """A single tool-call step extracted from the conversation."""

    model_config = ConfigDict(frozen=True)

    step_number: int
    tool_name: str
    tool_input: dict[str, object]
    output: str
    is_error: bool = False
    error_type: str | None = None
    assistant_message: str | None = None
    reasoning: str | None = None


class EpisodeContext(BaseModel):
    """Episode context for judge templates — wraps tool steps with helper methods."""

    model_config = ConfigDict(frozen=True)

    steps: tuple[ToolStep, ...]

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def get_step_count(self) -> int:
        """Callable form for Jinja2 templates: {{ episode.get_step_count() }}."""
        return self.step_count


class CheckpointObjective(BaseModel):
    """A checkpoint objective for judge template rendering."""

    model_config = ConfigDict(frozen=True)

    id: str
    objective: str
    title: str = ""
    description: str = ""
    hints: tuple[str, ...] = ()


class TaskMetadata(BaseModel):
    """Task metadata exposed to judge templates via ``{{ task.field }}``.

    Commonly accessed fields are declared explicitly.
    Additional metadata fields are accessible via ``model_extra`` (``extra='allow'``).
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    task_id: str = ""
    title: str = ""
    description: str = ""


class JudgeTemplateContext(BaseModel):
    """Typed context for LLM judge Jinja2 templates.

    Used by both single-scorer and batch-scorer pipelines.
    All fields are available in Jinja2 templates via attribute access.
    """

    model_config = ConfigDict(frozen=True)

    # Always present
    task_id: str
    domain: str
    question: str
    episode: EpisodeContext
    task: TaskMetadata

    # Submission scoring (set in both single-scorer and batch pipelines)
    submission: str | None = None

    # Description / expected answer for the current scorer
    description: str = ""

    # Subtask/checkpoint fields (None when scoring submission only)
    objective: str | None = None
    checkpoint_id: str | None = None
    checkpoints: tuple[CheckpointObjective, ...] = ()

    # Batch chunking fields (only for batch pipeline)
    chunk_index: int | None = None
    total_chunks: int | None = None


class ScoringContext(BaseModel):
    """Immutable context passed to every scoring strategy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, populate_by_name=True)

    submission: str
    tool_steps: tuple[ToolStep, ...]
    messages: tuple[ChatMessage, ...]
    target_text: str = Field(alias="target")  # renamed; accepts both 'target' and 'target_text'
    task_id: str
    domain: str
    metadata: dict[str, object]
    # ── Scoring system ──────────────────────────────────────────────
    scorer: ScorerConfig

    @property
    def target(self) -> str:
        """alias for ``target_text``."""
        return self.target_text
