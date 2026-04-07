"""Batched LLM judge scoring for multiple scorer checkpoints.

Evaluates multiple checkpoint ``ScoringContext`` objects in fewer LLM calls
by grouping checkpoints by criteria and chunking the trajectory.
"""

from __future__ import annotations

from collections import defaultdict

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score

from saber.config.models import LLMJudgeCriteria, to_template_vars
from saber.scoring.context import (
    CheckpointObjective,
    EpisodeContext,
    JudgeTemplateContext,
    ScoringContext,
    TaskMetadata,
    ToolStep,
)
from saber.scoring.parsing import parse_llm_step_evaluations, parse_llm_not_completed_explanations
from saber.scoring.strategies import _answer_text
from saber.scoring.templates import TemplateRenderer


def _chunk_steps(steps: tuple[ToolStep, ...], chunk_size: int | None) -> list[tuple[ToolStep, ...]]:
    """Split steps into chunks of *chunk_size*. Returns all in one chunk if None/0."""
    if not chunk_size or chunk_size <= 0:
        return [steps] if steps else [()]
    return [tuple(steps[i : i + chunk_size]) for i in range(0, len(steps), chunk_size)]


def _criteria_key(criteria: LLMJudgeCriteria) -> tuple[str, str, str]:
    """Return a hashable grouping key for *criteria* (model, system, user templates)."""
    return (criteria.model, criteria.system_template, criteria.user_template)


async def _score_criteria_group(
    *,
    contexts: list[ScoringContext],
    criteria: LLMJudgeCriteria,
    renderer: TemplateRenderer,
) -> dict[str, Score]:
    """Score a single criteria-group of contexts sharing the same LLMJudgeCriteria.

    All *contexts* must have identical ``tool_steps`` (same trajectory).
    Steps are chunked by ``criteria.steps_per_message``; one LLM call per
    chunk evaluates all checkpoints in the group simultaneously.

    Returns:
        Mapping of scorer_name → Score for every context in the group.

    Raises:
        ValueError: If tool_steps differ across contexts in the group.
    """
    # ── Validate trajectory consistency within the group ────────────
    first_steps = contexts[0].tool_steps
    for i, ctx in enumerate(contexts[1:], start=1):
        if ctx.tool_steps != first_steps:
            raise ValueError(f"Context {i} has different tool_steps; all contexts must share the same trajectory")

    # ── Chunk steps ─────────────────────────────────────────────────
    chunks = _chunk_steps(first_steps, criteria.steps_per_message)

    # ── Initialize results to 0.0 ───────────────────────────────────
    results: dict[str, Score] = {}
    scorer_map: dict[str, ScoringContext] = {}
    for ctx in contexts:
        sid = ctx.scorer.scorer_name
        scorer_map[sid] = ctx
        results[sid] = Score(
            value=0.0,
            answer=_answer_text(ctx),
            explanation="Not completed",
        )

    completed_ids: set[str] = set()

    # ── Process each chunk ──────────────────────────────────────────
    judge = get_model(criteria.model)

    for chunk_idx, chunk in enumerate(chunks):
        # Build remaining checkpoint objectives (not yet completed)
        remaining_checkpoints: tuple[CheckpointObjective, ...] = tuple(
            CheckpointObjective(
                id=ctx.scorer.scorer_name,
                objective=ctx.scorer.description or "",
                title=ctx.scorer.title or "",
                description=ctx.scorer.description or "",
                hints=tuple(ctx.scorer.hints) if ctx.scorer.hints else (),
            )
            for ctx in contexts
            if ctx.scorer.scorer_name not in completed_ids
        )

        if not remaining_checkpoints:
            break

        # Build template context
        template_context = JudgeTemplateContext(
            task=TaskMetadata(**contexts[0].metadata),
            question=str(contexts[0].metadata.get("description", "")),
            task_id=contexts[0].task_id,
            domain=contexts[0].domain,
            episode=EpisodeContext(steps=chunk),
            checkpoints=remaining_checkpoints,
            chunk_index=chunk_idx,
            total_chunks=len(chunks),
            submission=contexts[0].submission or None,
            description=contexts[0].scorer.description or "",
        )

        # Render prompts
        system_prompt = renderer.render(criteria.system_template, to_template_vars(template_context))
        user_prompt = renderer.render(criteria.user_template, to_template_vars(template_context))

        # Call judge model
        result = await judge.generate(
            [
                ChatMessageSystem(content=system_prompt),
                ChatMessageUser(content=user_prompt),
            ]
        )

        # Parse response for completed checkpoint IDs
        newly_completed = parse_llm_step_evaluations(result.completion)

        # Parse NOT_COMPLETED explanations for uncompleted checkpoints
        not_completed_explanations = parse_llm_not_completed_explanations(result.completion)
        for cp_id, explanation in not_completed_explanations.items():
            if cp_id in scorer_map and cp_id not in completed_ids:
                ctx = scorer_map[cp_id]
                results[cp_id] = Score(
                    value=0.0,
                    answer=_answer_text(ctx),
                    explanation=explanation,
                    metadata={
                        "judge_model": criteria.model,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                    },
                )

        # Update results for newly completed checkpoints
        for cp_id in newly_completed:
            if cp_id in scorer_map and cp_id not in completed_ids:
                completed_ids.add(cp_id)
                ctx = scorer_map[cp_id]
                results[cp_id] = Score(
                    value=ctx.scorer.max_score,
                    answer=_answer_text(ctx),
                    explanation=result.completion,
                    metadata={
                        "judge_model": criteria.model,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                    },
                )

    return results


async def score_checkpoints_llm_batch(
    *,
    contexts: list[ScoringContext],
    renderer: TemplateRenderer,
) -> dict[str, Score]:
    """Score multiple checkpoint ScoringContexts in batched LLM calls.

    Contexts are grouped by LLMJudgeCriteria (model + templates).
    Each group is processed independently. Steps are chunked by
    steps_per_message; one LLM call per chunk evaluates all checkpoints
    in the group simultaneously.

    Args:
        contexts: List of ScoringContext, one per checkpoint. All must have
                  LLMJudgeCriteria. Contexts with different criteria are
                  grouped and processed separately.
        renderer: Jinja2 template renderer.

    Returns:
        Mapping of scorer_name → Score.

    Raises:
        ValueError: If any context has criteria that are not LLMJudgeCriteria,
                    or if contexts within the same criteria group have
                    different tool_steps.
    """
    if not contexts:
        return {}

    # ── Validate all contexts have LLMJudgeCriteria ─────────────────
    for ctx in contexts:
        if not isinstance(ctx.scorer.criteria, LLMJudgeCriteria):
            raise ValueError(f"All contexts must use LLMJudgeCriteria, got {type(ctx.scorer.criteria).__name__}")

    # ── Group contexts by criteria key ──────────────────────────────
    groups: dict[tuple[str, str, str], list[ScoringContext]] = defaultdict(list)
    for ctx in contexts:
        criteria = ctx.scorer.criteria
        assert isinstance(criteria, LLMJudgeCriteria)  # validated above
        groups[_criteria_key(criteria)].append(ctx)

    # ── Process each group and merge results ────────────────────────
    results: dict[str, Score] = {}
    for group_contexts in groups.values():
        group_criteria = group_contexts[0].scorer.criteria
        assert isinstance(group_criteria, LLMJudgeCriteria)
        group_results = await _score_criteria_group(
            contexts=group_contexts,
            criteria=group_criteria,
            renderer=renderer,
        )
        results.update(group_results)

    return results
