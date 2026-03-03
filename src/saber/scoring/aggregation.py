"""Score aggregation and checkpoint summary utilities.

Combines scorer results using configurable strategies,
and builds human-readable checkpoint evaluation summaries.
"""

from __future__ import annotations

from saber.config.models import AggregationConfig, ScoreAggregation, ScorerConfig


def compute_aggregated_score_from_groups(
    *,
    group_scores: list[float],
    strategy: ScoreAggregation,
) -> float:
    """Combine group-level normalized scores using the specified strategy.

    Each score in *group_scores* must already be normalized to [0, 1].

    Args:
        group_scores: Pre-normalized [0,1] score per group.
        strategy: Aggregation strategy (MAX or AVERAGE).

    Returns:
        Combined score in [0, 1].

    Raises:
        ValueError: If strategy is unknown.
    """
    if not group_scores:
        return 0.0

    if strategy is ScoreAggregation.MAX:
        return max(group_scores)
    if strategy is ScoreAggregation.AVERAGE:
        return sum(group_scores) / len(group_scores)

    msg = f"Unknown aggregation strategy: {strategy!r}"
    raise ValueError(msg)


def aggregate_scorer_results(
    *,
    scorer_configs: tuple[ScorerConfig, ...],
    raw_scores: dict[str, float],
    agg_config: AggregationConfig | None,
) -> tuple[float, list[dict[str, float | str]]]:
    """Compute normalized aggregate and scorer details from raw scores.

    This is the single source of truth for the clamping → weighting →
    normalization pipeline shared by both the inline overall scorer and
    the per-task aggregate scorer.

    Args:
        scorer_configs: The scorers to aggregate.
        raw_scores: Mapping of scorer_name → raw score value.
        agg_config: Optional structured aggregation configuration.

    Returns:
        Tuple of (normalized_score, scorer_details_list).
    """
    scorer_map = {sc.scorer_name: sc for sc in scorer_configs}

    # --- Normalize ---
    if agg_config is None:
        total_weighted = 0.0
        total_max_weighted = 0.0
        for sc in scorer_configs:
            raw = raw_scores.get(sc.scorer_name, 0.0)
            clamped = min(raw, sc.max_score)
            total_weighted += clamped * sc.weight
            total_max_weighted += sc.max_score * sc.weight
        normalized = total_weighted / total_max_weighted if total_max_weighted > 0 else 0.0
    else:
        group_norms: list[float] = []
        for entry in agg_config.scores:
            if isinstance(entry, str):
                sc = scorer_map[entry]
                raw = raw_scores.get(sc.scorer_name, 0.0)
                group_norms.append(min(raw, sc.max_score) / sc.max_score if sc.max_score > 0 else 0.0)
            else:
                g_total = 0.0
                g_max = 0.0
                for name in entry:
                    sc = scorer_map[name]
                    raw = raw_scores.get(sc.scorer_name, 0.0)
                    clamped = min(raw, sc.max_score)
                    g_total += clamped * sc.weight
                    g_max += sc.max_score * sc.weight
                group_norms.append(g_total / g_max if g_max > 0 else 0.0)
        normalized = compute_aggregated_score_from_groups(
            group_scores=group_norms,
            strategy=agg_config.strategy,
        )

    # --- Build scorer details ---
    scorer_details: list[dict[str, float | str]] = []
    for sc in scorer_configs:
        raw = raw_scores.get(sc.scorer_name, 0.0)
        clamped = min(raw, sc.max_score)
        scorer_details.append(
            {
                "name": sc.scorer_name,
                "description": sc.description,
                "score": raw,
                "weighted": clamped * sc.weight,
                "max_score": sc.max_score,
                "weight": sc.weight,
            }
        )

    return normalized, scorer_details


def build_scorer_summary(
    scorer_details: list[dict[str, float | str]],
) -> str:
    """Build human-readable scorer evaluation summary.

    Args:
        scorer_details: List of dicts with keys ``name``, ``description``,
            ``score``, ``weighted``, ``max_score``, ``weight``.

    Returns:
        Multi-line summary string.
    """
    lines: list[str] = ["=== Scorer Evaluation Summary ==="]

    passed = 0
    total = len(scorer_details)

    for detail in scorer_details:
        score = float(detail["score"])
        max_score = float(detail["max_score"])
        weighted = float(detail["weighted"])
        name = str(detail["name"])
        description = str(detail.get("description", ""))

        is_passed = score >= max_score
        if is_passed:
            passed += 1

        mark = "\u2713" if is_passed else "\u2717"

        if description:
            label = f"{name} \u2014 {description}"
        else:
            label = name

        lines.append(f"  {mark} {label}: {score:.2f}/{max_score:.1f} (weighted: {weighted:.2f})")

    if total > 0:
        lines.append(f"  Total: {passed}/{total} checkpoints passed")

    return "\n".join(lines)
