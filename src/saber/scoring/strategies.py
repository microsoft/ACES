# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Scoring strategies for SABER evaluation.

Provides the :class:`SaberScoringStrategy` protocol and built-in
strategies: :class:`StaticStrategy`, :class:`NoneStrategy`,
:class:`LLMJudgeStrategy`, :class:`ToolCallStrategy`,
:class:`ToolCallCountStrategy`, and :class:`StaticJaccardStrategy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from inspect_ai.scorer import Score

from saber.config.models import ScorerTarget

if TYPE_CHECKING:
    from saber.config.models import (
        DomainCriteria,
        LLMJudgeCriteria,
        StaticCriteria,
        ToolCallCriteria,
    )
    from saber.scoring.context import JudgeTemplateContext, ScoringContext
    from saber.scoring.templates import TemplateRenderer


class SaberScoringStrategy(Protocol):
    """Unified protocol for all scoring strategies.

    Every strategy — whether evaluating a final submission or
    a checkpoint — implements this single method.
    """

    async def score(
        self,
        ctx: ScoringContext,
        renderer: TemplateRenderer | None,
    ) -> Score: ...


class NoneStrategy:
    """No-op scorer — returns 0.0. Used for tasks with checkpoint-only scoring."""

    async def score(self, ctx: ScoringContext, renderer: TemplateRenderer | None) -> Score:
        return Score(
            value=0.0,
            answer=_answer_text(ctx),
            explanation="No evaluation configured",
        )


@dataclass(frozen=True)
class _ScoringParams:
    """Resolved scoring parameters from a ScorerConfig."""

    criteria: StaticCriteria | LLMJudgeCriteria | ToolCallCriteria | DomainCriteria
    max_score: float
    is_submission: bool


def _resolve_params(ctx: ScoringContext) -> _ScoringParams:
    """Extract scoring parameters from ctx.scorer."""
    return _ScoringParams(
        criteria=ctx.scorer.criteria,
        max_score=ctx.scorer.max_score,
        is_submission=ctx.scorer.target == ScorerTarget.SUBMISSION,
    )


def _answer_text(ctx: ScoringContext) -> str:
    """Return the appropriate answer text for a scorer.

    Submission-targeted scorers show the agent's final answer.
    Trajectory-targeted scorers (checkpoints) show their description
    so each checkpoint is distinguishable in the eval viewer.
    """
    if ctx.scorer.target == ScorerTarget.SUBMISSION:
        submission = ctx.submission
        if not isinstance(submission, str):
            raise TypeError(f"Expected submission to be a string, got {type(submission).__name__}")
        return submission
    parts: list[str] = []
    if ctx.scorer.title:
        parts.append(ctx.scorer.title)
    if ctx.scorer.description:
        parts.append(ctx.scorer.description)
    return " — ".join(parts) if parts else ctx.scorer.scorer_name


class StaticStrategy:
    """Exact/substring match against expected answers.

    When scoring a submission: checks final answer text.
    When scoring a checkpoint (trajectory target): scans tool output steps.
    """

    async def score(self, ctx: ScoringContext, renderer: TemplateRenderer | None) -> Score:
        from saber.config.models import StaticCriteria

        params = _resolve_params(ctx)
        criteria = params.criteria
        max_score = params.max_score
        is_submission = params.is_submission

        if not isinstance(criteria, StaticCriteria):
            return Score(
                value=0.0,
                answer=_answer_text(ctx),
                explanation=f"StaticStrategy requires StaticCriteria, got {type(criteria).__name__}",
            )

        if is_submission:
            # Submission scoring — check final answer
            text_to_search = ctx.submission.strip().lower()
        else:
            # Subtask/trajectory scoring — check all tool outputs
            text_to_search = " ".join(step.output.lower() for step in ctx.tool_steps)

        for answer in criteria.expected_answers:
            normalised = answer.strip().lower()
            if not normalised:
                continue
            if normalised in text_to_search:
                return Score(
                    value=max_score,
                    answer=_answer_text(ctx),
                    explanation=f"Match found: '{answer}'",
                )

        return Score(
            value=0.0,
            answer=_answer_text(ctx),
            explanation=f"No match for {criteria.expected_answers}",
        )


# ── Helper ──────────────────────────────────────────────────────────


def _build_judge_context(ctx: ScoringContext, criteria: LLMJudgeCriteria) -> JudgeTemplateContext:
    """Build the typed template context for LLM judge rendering."""
    from saber.scoring.context import (
        CheckpointObjective,
        EpisodeContext,
        JudgeTemplateContext,
        TaskMetadata,
    )

    objective = ctx.scorer.description
    checkpoint_id = ctx.scorer.scorer_name
    checkpoints = (
        CheckpointObjective(
            id=ctx.scorer.scorer_name,
            objective=ctx.scorer.description or "",
            title=ctx.scorer.title or "",
            description=ctx.scorer.description or "",
            hints=tuple(ctx.scorer.hints) if ctx.scorer.hints else (),
        ),
    )

    return JudgeTemplateContext(
        submission=ctx.submission,
        description=ctx.scorer.description or "",
        question=str(ctx.metadata.get("description", "")),
        task=TaskMetadata(**ctx.metadata),
        task_id=ctx.task_id,
        domain=ctx.domain,
        episode=EpisodeContext(steps=ctx.tool_steps),
        objective=objective,
        checkpoint_id=checkpoint_id,
        checkpoints=checkpoints,
    )


# ── LLMJudgeStrategy ───────────────────────────────────────────────


class LLMJudgeStrategy:
    """LLM-as-judge scoring with Jinja2 template rendering.

    Supports three response formats via ``LLMJudgeCriteria.response_format``:

    - **BINARY**: CORRECT/INCORRECT → max_score or 0.
    - **CONTINUOUS**: ``{"score": 0.0-1.0}`` → scaled to max_score.
    - **STEP_EVALUATIONS**: ``{"step_evaluations": [...]}`` → proportional.

    When scoring a submission: renders submission judge templates.
    When scoring a checkpoint: renders checkpoint judge templates with trajectory.
    """

    async def score(self, ctx: ScoringContext, renderer: TemplateRenderer | None) -> Score:
        from saber.config.models import LLMJudgeCriteria, LLMJudgeResponseFormat
        from saber.scoring.parsing import parse_judge_response
        from saber.scoring.templates import TemplateRenderer

        params = _resolve_params(ctx)
        criteria = params.criteria
        max_score = params.max_score

        if not isinstance(criteria, LLMJudgeCriteria):
            return Score(
                value=0.0,
                answer=_answer_text(ctx),
                explanation=(f"LLMJudgeStrategy requires LLMJudgeCriteria, got {type(criteria).__name__}"),
            )

        if not isinstance(renderer, TemplateRenderer):
            return Score(
                value=0.0,
                answer=_answer_text(ctx),
                explanation="LLMJudgeStrategy requires a TemplateRenderer",
            )

        # Build template context
        template_context = _build_judge_context(ctx, criteria)

        # Render judge prompts
        from saber.config.models import to_template_vars

        system_prompt = renderer.render(criteria.system_template, to_template_vars(template_context))
        user_prompt = renderer.render(criteria.user_template, to_template_vars(template_context))

        # Call judge model
        from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model

        judge = get_model(criteria.model)
        result = await judge.generate(
            [
                ChatMessageSystem(content=system_prompt),
                ChatMessageUser(content=user_prompt),
            ]
        )

        # Parse response
        # For checkpoint scoring with default BINARY format, use checkpoint-aware
        # parsing that handles both [step: checkpoint_id] and CORRECT/INCORRECT.
        # For explicit formats (STEP_EVALUATIONS, CONTINUOUS) or submission
        # scoring, use the standard format-based parser.
        is_checkpoint = not params.is_submission
        checkpoint_id = ctx.scorer.scorer_name

        if is_checkpoint and criteria.response_format is LLMJudgeResponseFormat.BINARY:
            from saber.scoring.parsing import parse_checkpoint_score

            score_val = parse_checkpoint_score(result.completion, checkpoint_id, max_score)
        else:
            score_val = parse_judge_response(result.completion, criteria.response_format, max_score)

        return Score(
            value=score_val,
            answer=_answer_text(ctx),
            explanation=result.completion,
            metadata={
                "judge_model": criteria.model,
                "judge_response": result.completion,
            },
        )


# ── Tool-name normalisation ─────────────────────────────────────────


def _normalize_tool_name(name: str) -> str:
    """Normalize a tool name for case- and prefix-insensitive comparison.

    Handles two MCP bridge naming conventions:

    1. **Claude Code** (double-underscore): ``mcp__<server>__<tool>`` → ``<tool>``
    2. **Copilot SDK** (hyphen): ``<server_label>-<tool>`` → ``<tool>``

    The Copilot SDK convention is only applied when the prefix matches a
    known bridged-tools server name (currently ``saber_tools``) to avoid
    false positives with tools that legitimately contain hyphens.

    If stripping the prefix would produce an empty string, the original
    lowercased name is returned.
    """
    lowered = name.lower()

    # Convention 1: mcp__<server>__<tool_name>  (Claude Code)
    parts = lowered.split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        stripped = "__".join(parts[2:])
        return stripped if stripped else lowered

    # Convention 2: <server_label>-<tool_name>  (Copilot SDK)
    # Only strip known server prefixes to avoid mangling tool names
    # that legitimately contain hyphens (e.g. "my-tool").
    _KNOWN_MCP_PREFIXES = ("saber_tools-",)
    for prefix in _KNOWN_MCP_PREFIXES:
        if lowered.startswith(prefix):
            stripped = lowered[len(prefix) :]
            return stripped if stripped else lowered

    return lowered


# ── ToolCallStrategy ───────────────────────────────────────────────


class ToolCallStrategy:
    """Binary pass/fail based on whether agent called a specific tool."""

    async def score(self, ctx: ScoringContext, renderer: TemplateRenderer | None) -> Score:
        from saber.config.models import ToolCallCriteria

        params = _resolve_params(ctx)
        is_submission = params.is_submission
        criteria = params.criteria
        max_score = params.max_score

        if is_submission:
            return Score(
                value=0.0,
                answer=_answer_text(ctx),
                explanation="N/A for submission",
            )

        if not isinstance(criteria, ToolCallCriteria):
            return Score(
                value=0.0,
                answer=_answer_text(ctx),
                explanation=(f"ToolCallStrategy requires ToolCallCriteria, got {type(criteria).__name__}"),
            )

        expected_tools = {_normalize_tool_name(t) for t in (criteria.expected_tools or [criteria.tool_name])}
        called_tools = {_normalize_tool_name(s.tool_name) for s in ctx.tool_steps}

        matched = called_tools & expected_tools
        if matched:
            return Score(
                value=max_score,
                answer=_answer_text(ctx),
                explanation=f"Called expected tool: {matched}",
            )
        return Score(
            value=0.0,
            answer=_answer_text(ctx),
            explanation=f"Expected {expected_tools}, called {sorted(called_tools)}",
        )


# ── ToolCallCountStrategy ──────────────────────────────────────────


class ToolCallCountStrategy:
    """Binary pass if tool executed >= threshold times."""

    async def score(self, ctx: ScoringContext, renderer: TemplateRenderer | None) -> Score:
        from saber.config.models import ToolCallCriteria

        params = _resolve_params(ctx)
        is_submission = params.is_submission
        criteria = params.criteria
        max_score = params.max_score

        if is_submission:
            return Score(
                value=0.0,
                answer=_answer_text(ctx),
                explanation="N/A for submission",
            )

        if not isinstance(criteria, ToolCallCriteria):
            return Score(
                value=0.0,
                answer=_answer_text(ctx),
                explanation=(f"ToolCallCountStrategy requires ToolCallCriteria, got {type(criteria).__name__}"),
            )

        tool_name = _normalize_tool_name(criteria.tool_name)
        min_count = criteria.min_executions

        count = sum(1 for s in ctx.tool_steps if _normalize_tool_name(s.tool_name) == tool_name)

        if count >= min_count:
            return Score(
                value=max_score,
                answer=_answer_text(ctx),
                explanation=f"Called '{tool_name}' {count}x (min: {min_count})",
            )
        return Score(
            value=0.0,
            answer=_answer_text(ctx),
            explanation=f"Called '{tool_name}' {count}x, need {min_count}",
        )


# ── StaticJaccardStrategy ──────────────────────────────────────────


class StaticJaccardStrategy:
    """Jaccard similarity between expected answer tokens and agent output tokens."""

    async def score(self, ctx: ScoringContext, renderer: TemplateRenderer | None) -> Score:
        from saber.config.models import StaticCriteria

        params = _resolve_params(ctx)
        criteria = params.criteria
        max_score = params.max_score
        is_submission = params.is_submission

        if not isinstance(criteria, StaticCriteria):
            return Score(
                value=0.0,
                answer=_answer_text(ctx),
                explanation=(f"StaticJaccardStrategy requires StaticCriteria, got {type(criteria).__name__}"),
            )

        # Word-split each expected answer so multi-word answers use the same
        # tokenisation as the actual text (which is also word-split below).
        expected: set[str] = set()
        for a in criteria.expected_answers:
            for token in a.strip().lower().split():
                if token:
                    expected.add(token)

        if is_submission:
            actual_text = ctx.submission.strip().lower()
        else:
            actual_text = " ".join(step.output.lower() for step in ctx.tool_steps)

        actual_tokens = set(actual_text.split())

        if not expected or not actual_tokens:
            return Score(
                value=0.0,
                answer=_answer_text(ctx),
                explanation="Empty expected or actual set — jaccard = 0.0",
            )

        intersection = expected & actual_tokens
        union = expected | actual_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        return Score(
            value=jaccard * max_score,
            answer=_answer_text(ctx),
            explanation=(
                f"Jaccard similarity: {jaccard:.3f} (matched {len(intersection)}/{len(expected)} expected tokens)"
            ),
        )
