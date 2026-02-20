"""Pure score aggregation functions for SABER scoring.

Provides a strategy-based approach to combining submission and subtask scores
into a single normalized total score. This module contains no side effects —
all functions are pure and deterministic.

Strategies:
    - AVERAGE: Mean of independently normalized scores (default, original behavior)
    - WEIGHTED_SUM: Raw sum normalized by total max possible
    - MAX: Best normalized component score
"""

from dataclasses import dataclass

from ...models.constants import ScoreAggregationStrategy


@dataclass(frozen=True)
class AggregatedScoreResult:
    """Result of score aggregation — immutable value object.

    Attributes:
        normalized_total_score: Final score in [0, 1] range.
        normalized_submission_score: Submission score normalized by its max.
        normalized_subtask_score: Subtask score normalized by its max.
        strategy: The aggregation strategy that produced this result.
        max_possible: Raw max possible (sum of max scores for active components).
    """

    normalized_total_score: float
    normalized_submission_score: float
    normalized_subtask_score: float
    strategy: ScoreAggregationStrategy
    max_possible: float


def _safe_normalize(score: float, max_score: float) -> float:
    """Normalize a score by its maximum, returning 0.0 if max is zero.

    Args:
        score: Raw score value.
        max_score: Maximum possible score.

    Returns:
        Normalized score in [0, 1] range, or 0.0 if max_score <= 0.
    """
    if max_score <= 0.0:
        return 0.0
    return score / max_score


def compute_aggregated_score(
    strategy: ScoreAggregationStrategy,
    submission_score: float,
    submission_max_score: float,
    subtask_score: float,
    subtask_max_score: float,
    has_submission: bool,
    has_subtasks: bool,
) -> AggregatedScoreResult:
    """Compute the aggregated score from submission and subtask components.

    When only one component is active, all strategies produce identical results
    (just that component's normalized score). Differences emerge only when both
    submission and subtask components are present.

    Args:
        strategy: Which aggregation formula to apply.
        submission_score: Raw submission score.
        submission_max_score: Maximum possible submission score.
        subtask_score: Raw subtask score (already weighted by subtask weights).
        subtask_max_score: Maximum possible subtask score (already weighted).
        has_submission: Whether submission evaluation is active.
        has_subtasks: Whether subtask evaluation is active.

    Returns:
        AggregatedScoreResult with normalized scores and metadata.
    """
    # Normalize each component independently
    norm_sub = _safe_normalize(submission_score, submission_max_score) if has_submission else 0.0
    norm_step = _safe_normalize(subtask_score, subtask_max_score) if has_subtasks else 0.0

    # Raw max possible for metadata
    max_possible = 0.0
    if has_submission:
        max_possible += submission_max_score
    if has_subtasks:
        max_possible += subtask_max_score

    # Compute total based on strategy
    if not has_submission and not has_subtasks:
        normalized_total = 0.0
    elif has_submission and not has_subtasks:
        # Single component — all strategies produce the same result
        normalized_total = norm_sub
    elif not has_submission and has_subtasks:
        # Single component — all strategies produce the same result
        normalized_total = norm_step
    else:
        # Both components present — strategy matters
        normalized_total = _compute_dual_component(
            strategy=strategy,
            norm_sub=norm_sub,
            norm_step=norm_step,
            raw_sub=submission_score,
            raw_step=subtask_score,
            max_sub=submission_max_score,
            max_step=subtask_max_score,
        )

    return AggregatedScoreResult(
        normalized_total_score=normalized_total,
        normalized_submission_score=norm_sub,
        normalized_subtask_score=norm_step,
        strategy=strategy,
        max_possible=max_possible,
    )


def _compute_dual_component(
    strategy: ScoreAggregationStrategy,
    norm_sub: float,
    norm_step: float,
    raw_sub: float,
    raw_step: float,
    max_sub: float,
    max_step: float,
) -> float:
    """Compute normalized total when both components are active.

    Args:
        strategy: Aggregation strategy.
        norm_sub: Normalized submission score.
        norm_step: Normalized subtask score.
        raw_sub: Raw submission score.
        raw_step: Raw subtask score.
        max_sub: Max submission score.
        max_step: Max subtask score.

    Returns:
        Normalized total score in [0, 1] range.
    """
    if strategy == ScoreAggregationStrategy.AVERAGE:
        return (norm_sub + norm_step) / 2.0

    if strategy == ScoreAggregationStrategy.WEIGHTED_SUM:
        total_max = max_sub + max_step
        if total_max <= 0.0:
            return 0.0
        return (raw_sub + raw_step) / total_max

    if strategy == ScoreAggregationStrategy.MAX:
        return max(norm_sub, norm_step)

    # Should be unreachable with a valid enum, but satisfy exhaustiveness
    raise ValueError(f"Unknown score aggregation strategy: {strategy}")
