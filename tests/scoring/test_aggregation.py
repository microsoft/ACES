# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for saber.scoring.aggregation – written before implementation (TDD)."""

from __future__ import annotations

import pytest

from saber.config.models import AggregationConfig, ScoreAggregation, ScorerConfig
from saber.scoring.aggregation import (
    aggregate_scorer_results,
    build_scorer_summary,
    compute_aggregated_score_from_groups,
)


def _sc(name: str, *, max_score: float = 1.0, weight: float = 1.0) -> ScorerConfig:
    """Build a minimal ScorerConfig for testing."""
    return ScorerConfig(scorer_name=name, strategy="static", max_score=max_score, weight=weight)


# ── compute_aggregated_score_from_groups ────────────────────────────


class TestComputeAggregatedScoreFromGroupsMax:
    """Max strategy picks highest group score."""

    def test_picks_maximum(self) -> None:
        result = compute_aggregated_score_from_groups(
            group_scores=[0.3, 0.8, 0.5],
            strategy=ScoreAggregation.MAX,
        )
        assert result == pytest.approx(0.8)

    def test_single_group(self) -> None:
        result = compute_aggregated_score_from_groups(
            group_scores=[0.6],
            strategy=ScoreAggregation.MAX,
        )
        assert result == pytest.approx(0.6)

    def test_all_zero(self) -> None:
        result = compute_aggregated_score_from_groups(
            group_scores=[0.0, 0.0],
            strategy=ScoreAggregation.MAX,
        )
        assert result == pytest.approx(0.0)

    def test_all_perfect(self) -> None:
        result = compute_aggregated_score_from_groups(
            group_scores=[1.0, 1.0, 1.0],
            strategy=ScoreAggregation.MAX,
        )
        assert result == pytest.approx(1.0)


class TestComputeAggregatedScoreFromGroupsAverage:
    """Average strategy takes mean."""

    def test_average_of_three(self) -> None:
        result = compute_aggregated_score_from_groups(
            group_scores=[0.3, 0.6, 0.9],
            strategy=ScoreAggregation.AVERAGE,
        )
        assert result == pytest.approx(0.6)

    def test_single_group(self) -> None:
        result = compute_aggregated_score_from_groups(
            group_scores=[0.7],
            strategy=ScoreAggregation.AVERAGE,
        )
        assert result == pytest.approx(0.7)


class TestComputeAggregatedScoreFromGroupsEmpty:
    """Empty group_scores returns 0.0."""

    def test_empty_returns_zero(self) -> None:
        for strategy in ScoreAggregation:
            result = compute_aggregated_score_from_groups(
                group_scores=[],
                strategy=strategy,
            )
            assert result == pytest.approx(0.0), f"Failed for {strategy}"


# ── build_scorer_summary ────────────────────────────────────────────


class TestBuildScorerSummary:
    """Tests for scorer-based evaluation summary."""

    def test_empty_details(self) -> None:
        result = build_scorer_summary([])
        assert "=== Scorer Evaluation Summary ===" in result
        assert "checkpoints passed" not in result

    def test_all_passed(self) -> None:
        details = [
            {
                "name": "submission",
                "description": "",
                "score": 1.0,
                "weighted": 1.0,
                "max_score": 1.0,
                "weight": 1.0,
            },
            {
                "name": "checkpoint_1",
                "description": "Found the flag",
                "score": 1.0,
                "weighted": 0.5,
                "max_score": 1.0,
                "weight": 0.5,
            },
        ]
        result = build_scorer_summary(details)
        assert result.count("\u2713") == 2
        assert result.count("\u2717") == 0
        assert "2/2" in result

    def test_mixed_pass_fail(self) -> None:
        details = [
            {
                "name": "submission",
                "description": "",
                "score": 1.0,
                "weighted": 1.0,
                "max_score": 1.0,
                "weight": 1.0,
            },
            {
                "name": "cp_1",
                "description": "Find flag",
                "score": 0.0,
                "weighted": 0.0,
                "max_score": 1.0,
                "weight": 0.5,
            },
        ]
        result = build_scorer_summary(details)
        assert "\u2713" in result
        assert "\u2717" in result
        assert "1/2" in result

    def test_description_shown_as_label(self) -> None:
        details = [
            {
                "name": "cp_1",
                "description": "Find the flag",
                "score": 1.0,
                "weighted": 1.0,
                "max_score": 1.0,
                "weight": 1.0,
            },
        ]
        result = build_scorer_summary(details)
        assert "cp_1" in result
        assert "Find the flag" in result

    def test_no_description_shows_name_only(self) -> None:
        details = [
            {"name": "submission", "description": "", "score": 0.5, "weighted": 0.5, "max_score": 1.0, "weight": 1.0},
        ]
        result = build_scorer_summary(details)
        assert "submission" in result
        assert " \u2014 " not in result.split("submission")[1].split("\n")[0]


# ── aggregate_scorer_results ────────────────────────────────────────


class TestAggregateResultsFlatWeightedSum:
    """No agg_config → flat weighted sum."""

    def test_single_scorer_perfect(self) -> None:
        normalized, details = aggregate_scorer_results(
            scorer_configs=(_sc("a"),),
            raw_scores={"a": 1.0},
            agg_config=None,
        )
        assert normalized == pytest.approx(1.0)
        assert len(details) == 1
        assert details[0]["name"] == "a"
        assert details[0]["score"] == 1.0

    def test_single_scorer_zero(self) -> None:
        normalized, _ = aggregate_scorer_results(
            scorer_configs=(_sc("a"),),
            raw_scores={"a": 0.0},
            agg_config=None,
        )
        assert normalized == pytest.approx(0.0)

    def test_two_equal_weight_scorers(self) -> None:
        normalized, details = aggregate_scorer_results(
            scorer_configs=(_sc("a"), _sc("b")),
            raw_scores={"a": 1.0, "b": 0.0},
            agg_config=None,
        )
        assert normalized == pytest.approx(0.5)
        assert len(details) == 2

    def test_different_weights(self) -> None:
        # a: max=1 weight=0.8,  b: max=1 weight=0.2
        normalized, _ = aggregate_scorer_results(
            scorer_configs=(_sc("a", weight=0.8), _sc("b", weight=0.2)),
            raw_scores={"a": 1.0, "b": 0.0},
            agg_config=None,
        )
        # total_weighted = 1.0*0.8 + 0*0.2 = 0.8
        # total_max_weighted = 1*0.8 + 1*0.2 = 1.0
        assert normalized == pytest.approx(0.8)

    def test_clamping_above_max(self) -> None:
        # raw=2.0 but max_score=1.0 → clamped to 1.0
        normalized, details = aggregate_scorer_results(
            scorer_configs=(_sc("a", max_score=1.0),),
            raw_scores={"a": 2.0},
            agg_config=None,
        )
        assert normalized == pytest.approx(1.0)
        # raw is preserved in details
        assert details[0]["score"] == 2.0
        assert details[0]["weighted"] == pytest.approx(1.0)

    def test_missing_scorer_defaults_to_zero(self) -> None:
        normalized, details = aggregate_scorer_results(
            scorer_configs=(_sc("a"), _sc("b")),
            raw_scores={"a": 1.0},
            agg_config=None,
        )
        assert normalized == pytest.approx(0.5)
        assert details[1]["score"] == 0.0

    def test_max_score_greater_than_one(self) -> None:
        normalized, _ = aggregate_scorer_results(
            scorer_configs=(_sc("a", max_score=10.0),),
            raw_scores={"a": 5.0},
            agg_config=None,
        )
        assert normalized == pytest.approx(0.5)


class TestAggregateResultsGrouped:
    """With agg_config → grouped aggregation."""

    def test_single_group_single_scorer(self) -> None:
        agg = AggregationConfig(strategy=ScoreAggregation.AVERAGE, scores=["a"])
        normalized, details = aggregate_scorer_results(
            scorer_configs=(_sc("a"),),
            raw_scores={"a": 0.5},
            agg_config=agg,
        )
        assert normalized == pytest.approx(0.5)
        assert len(details) == 1

    def test_two_individual_groups_max(self) -> None:
        agg = AggregationConfig(strategy=ScoreAggregation.MAX, scores=["a", "b"])
        normalized, _ = aggregate_scorer_results(
            scorer_configs=(_sc("a"), _sc("b")),
            raw_scores={"a": 0.3, "b": 0.8},
            agg_config=agg,
        )
        assert normalized == pytest.approx(0.8)

    def test_grouped_list_entries(self) -> None:
        # One group containing both scorers → weighted sum within group
        agg = AggregationConfig(strategy=ScoreAggregation.AVERAGE, scores=[["a", "b"]])
        normalized, _ = aggregate_scorer_results(
            scorer_configs=(_sc("a", weight=0.6), _sc("b", weight=0.4)),
            raw_scores={"a": 1.0, "b": 0.0},
            agg_config=agg,
        )
        # g_total = 1*0.6 + 0*0.4 = 0.6, g_max = 1*0.6 + 1*0.4 = 1.0
        assert normalized == pytest.approx(0.6)

    def test_mixed_individual_and_grouped(self) -> None:
        agg = AggregationConfig(
            strategy=ScoreAggregation.AVERAGE,
            scores=["a", ["b", "c"]],
        )
        normalized, _ = aggregate_scorer_results(
            scorer_configs=(_sc("a"), _sc("b"), _sc("c")),
            raw_scores={"a": 1.0, "b": 1.0, "c": 0.0},
            agg_config=agg,
        )
        # group1 (a): 1.0/1.0 = 1.0
        # group2 (b+c): (1*1+0*1)/(1*1+1*1) = 0.5
        # average of [1.0, 0.5] = 0.75
        assert normalized == pytest.approx(0.75)


class TestAggregateResultsScorerDetails:
    """Verify scorer_details structure."""

    def test_details_contain_all_fields(self) -> None:
        _, details = aggregate_scorer_results(
            scorer_configs=(_sc("x", max_score=2.0, weight=0.5),),
            raw_scores={"x": 1.5},
            agg_config=None,
        )
        d = details[0]
        assert d["name"] == "x"
        assert d["score"] == 1.5
        assert d["max_score"] == 2.0
        assert d["weight"] == 0.5
        assert d["weighted"] == pytest.approx(1.5 * 0.5)

    def test_details_order_matches_scorer_configs(self) -> None:
        _, details = aggregate_scorer_results(
            scorer_configs=(_sc("z"), _sc("a"), _sc("m")),
            raw_scores={"z": 0.1, "a": 0.2, "m": 0.3},
            agg_config=None,
        )
        assert [d["name"] for d in details] == ["z", "a", "m"]

    def test_empty_scorers(self) -> None:
        normalized, details = aggregate_scorer_results(
            scorer_configs=(),
            raw_scores={},
            agg_config=None,
        )
        assert normalized == pytest.approx(0.0)
        assert details == []
