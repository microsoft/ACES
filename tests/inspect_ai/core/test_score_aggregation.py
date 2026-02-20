"""Tests for score aggregation strategies.

Tests the pure compute_aggregated_score() function with all three strategies:
- average: (norm_sub + norm_step) / 2
- weighted_sum: (raw_sub + raw_step) / (max_sub + max_step)
- max: max(norm_sub, norm_step)

Covers:
- All 3 strategies x {both, submission-only, subtask-only, neither}
- Asymmetric max scores
- Zero max scores
- Perfect scores
- Enum validation
"""

import pytest

from saber.inspect_ai.core.score_aggregation import (
    AggregatedScoreResult,
    compute_aggregated_score,
)
from saber.models.constants import DEFAULT_SCORE_AGGREGATION, ScoreAggregationStrategy

# ============================================================================
# Enum tests
# ============================================================================


class TestScoreAggregationStrategy:
    """Tests for ScoreAggregationStrategy enum."""

    def test_values(self) -> None:
        assert ScoreAggregationStrategy.AVERAGE.value == "average"
        assert ScoreAggregationStrategy.WEIGHTED_SUM.value == "weighted_sum"
        assert ScoreAggregationStrategy.MAX.value == "max"

    def test_str(self) -> None:
        assert str(ScoreAggregationStrategy.AVERAGE) == "average"
        assert str(ScoreAggregationStrategy.WEIGHTED_SUM) == "weighted_sum"
        assert str(ScoreAggregationStrategy.MAX) == "max"

    def test_from_string(self) -> None:
        assert ScoreAggregationStrategy("average") == ScoreAggregationStrategy.AVERAGE
        assert ScoreAggregationStrategy("weighted_sum") == ScoreAggregationStrategy.WEIGHTED_SUM
        assert ScoreAggregationStrategy("max") == ScoreAggregationStrategy.MAX

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            ScoreAggregationStrategy("invalid")

    def test_default_is_average(self) -> None:
        assert DEFAULT_SCORE_AGGREGATION == ScoreAggregationStrategy.AVERAGE


# ============================================================================
# AggregatedScoreResult tests
# ============================================================================


class TestAggregatedScoreResult:
    """Tests for the frozen dataclass result."""

    def test_frozen(self) -> None:
        result = AggregatedScoreResult(
            normalized_total_score=0.5,
            normalized_submission_score=0.5,
            normalized_subtask_score=0.5,
            strategy=ScoreAggregationStrategy.AVERAGE,
            max_possible=2.0,
        )
        with pytest.raises(AttributeError):
            result.normalized_total_score = 1.0  # type: ignore[misc]

    def test_fields(self) -> None:
        result = AggregatedScoreResult(
            normalized_total_score=0.75,
            normalized_submission_score=0.5,
            normalized_subtask_score=1.0,
            strategy=ScoreAggregationStrategy.MAX,
            max_possible=4.0,
        )
        assert result.normalized_total_score == 0.75
        assert result.normalized_submission_score == 0.5
        assert result.normalized_subtask_score == 1.0
        assert result.strategy == ScoreAggregationStrategy.MAX
        assert result.max_possible == 4.0


# ============================================================================
# AVERAGE strategy tests
# ============================================================================


class TestAverageStrategy:
    """Tests for the AVERAGE aggregation strategy (default, current behavior)."""

    def test_both_components_equal_max(self) -> None:
        """Both components with equal max scores => simple average."""
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.AVERAGE,
            submission_score=0.5,
            submission_max_score=1.0,
            subtask_score=1.5,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        # norm_sub = 0.5/1.0 = 0.5, norm_step = 1.5/3.0 = 0.5
        # average = (0.5 + 0.5) / 2 = 0.5
        assert result.normalized_submission_score == pytest.approx(0.5)
        assert result.normalized_subtask_score == pytest.approx(0.5)
        assert result.normalized_total_score == pytest.approx(0.5)
        assert result.strategy == ScoreAggregationStrategy.AVERAGE

    def test_both_components_asymmetric(self) -> None:
        """Asymmetric max scores: sub_max=1, step_max=3."""
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.AVERAGE,
            submission_score=1.0,
            submission_max_score=1.0,
            subtask_score=1.5,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        # norm_sub = 1.0/1.0 = 1.0, norm_step = 1.5/3.0 = 0.5
        # average = (1.0 + 0.5) / 2 = 0.75
        assert result.normalized_submission_score == pytest.approx(1.0)
        assert result.normalized_subtask_score == pytest.approx(0.5)
        assert result.normalized_total_score == pytest.approx(0.75)

    def test_submission_only(self) -> None:
        """Only submission, no subtasks."""
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.AVERAGE,
            submission_score=0.7,
            submission_max_score=1.0,
            subtask_score=0.0,
            subtask_max_score=0.0,
            has_submission=True,
            has_subtasks=False,
        )
        # Only one component => total = norm_sub = 0.7
        assert result.normalized_submission_score == pytest.approx(0.7)
        assert result.normalized_subtask_score == pytest.approx(0.0)
        assert result.normalized_total_score == pytest.approx(0.7)

    def test_subtask_only(self) -> None:
        """Only subtasks, no submission."""
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.AVERAGE,
            submission_score=0.0,
            submission_max_score=0.0,
            subtask_score=2.0,
            subtask_max_score=4.0,
            has_submission=False,
            has_subtasks=True,
        )
        # Only one component => total = norm_step = 0.5
        assert result.normalized_submission_score == pytest.approx(0.0)
        assert result.normalized_subtask_score == pytest.approx(0.5)
        assert result.normalized_total_score == pytest.approx(0.5)

    def test_neither_component(self) -> None:
        """No components => 0."""
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.AVERAGE,
            submission_score=0.0,
            submission_max_score=0.0,
            subtask_score=0.0,
            subtask_max_score=0.0,
            has_submission=False,
            has_subtasks=False,
        )
        assert result.normalized_total_score == pytest.approx(0.0)

    def test_perfect_scores(self) -> None:
        """Both components at max."""
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.AVERAGE,
            submission_score=1.0,
            submission_max_score=1.0,
            subtask_score=3.0,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        assert result.normalized_total_score == pytest.approx(1.0)


# ============================================================================
# WEIGHTED_SUM strategy tests
# ============================================================================


class TestWeightedSumStrategy:
    """Tests for WEIGHTED_SUM aggregation (raw_sum / total_max)."""

    def test_both_components_equal_max(self) -> None:
        """Equal max => same as average when scores are proportionally equal."""
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.WEIGHTED_SUM,
            submission_score=0.5,
            submission_max_score=1.0,
            subtask_score=1.5,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        # weighted_sum = (0.5 + 1.5) / (1.0 + 3.0) = 2.0 / 4.0 = 0.5
        assert result.normalized_total_score == pytest.approx(0.5)
        assert result.max_possible == pytest.approx(4.0)

    def test_both_components_asymmetric(self) -> None:
        """Asymmetric: sub_max=1, step_max=3. Step contributes more weight."""
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.WEIGHTED_SUM,
            submission_score=1.0,
            submission_max_score=1.0,
            subtask_score=1.5,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        # weighted_sum = (1.0 + 1.5) / (1.0 + 3.0) = 2.5 / 4.0 = 0.625
        assert result.normalized_total_score == pytest.approx(0.625)

    def test_weighted_sum_differs_from_average(self) -> None:
        """Show difference between average and weighted_sum with asymmetric max."""
        avg_result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.AVERAGE,
            submission_score=1.0,
            submission_max_score=1.0,
            subtask_score=0.0,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        ws_result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.WEIGHTED_SUM,
            submission_score=1.0,
            submission_max_score=1.0,
            subtask_score=0.0,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        # average: (1.0 + 0.0) / 2 = 0.5
        # weighted_sum: (1.0 + 0.0) / (1.0 + 3.0) = 0.25
        assert avg_result.normalized_total_score == pytest.approx(0.5)
        assert ws_result.normalized_total_score == pytest.approx(0.25)

    def test_submission_only(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.WEIGHTED_SUM,
            submission_score=0.7,
            submission_max_score=1.0,
            subtask_score=0.0,
            subtask_max_score=0.0,
            has_submission=True,
            has_subtasks=False,
        )
        # Only sub: 0.7 / 1.0 = 0.7
        assert result.normalized_total_score == pytest.approx(0.7)

    def test_subtask_only(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.WEIGHTED_SUM,
            submission_score=0.0,
            submission_max_score=0.0,
            subtask_score=2.0,
            subtask_max_score=4.0,
            has_submission=False,
            has_subtasks=True,
        )
        # Only step: 2.0 / 4.0 = 0.5
        assert result.normalized_total_score == pytest.approx(0.5)

    def test_neither_component(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.WEIGHTED_SUM,
            submission_score=0.0,
            submission_max_score=0.0,
            subtask_score=0.0,
            subtask_max_score=0.0,
            has_submission=False,
            has_subtasks=False,
        )
        assert result.normalized_total_score == pytest.approx(0.0)

    def test_perfect_scores(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.WEIGHTED_SUM,
            submission_score=1.0,
            submission_max_score=1.0,
            subtask_score=3.0,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        assert result.normalized_total_score == pytest.approx(1.0)


# ============================================================================
# MAX strategy tests
# ============================================================================


class TestMaxStrategy:
    """Tests for MAX aggregation (best normalized component)."""

    def test_both_components_equal(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.MAX,
            submission_score=0.5,
            submission_max_score=1.0,
            subtask_score=1.5,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        # norm_sub = 0.5, norm_step = 0.5 => max = 0.5
        assert result.normalized_total_score == pytest.approx(0.5)

    def test_both_components_submission_higher(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.MAX,
            submission_score=1.0,
            submission_max_score=1.0,
            subtask_score=1.0,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        # norm_sub = 1.0, norm_step = 0.333 => max = 1.0
        assert result.normalized_total_score == pytest.approx(1.0)

    def test_both_components_subtask_higher(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.MAX,
            submission_score=0.2,
            submission_max_score=1.0,
            subtask_score=2.4,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        # norm_sub = 0.2, norm_step = 0.8 => max = 0.8
        assert result.normalized_total_score == pytest.approx(0.8)

    def test_submission_only(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.MAX,
            submission_score=0.7,
            submission_max_score=1.0,
            subtask_score=0.0,
            subtask_max_score=0.0,
            has_submission=True,
            has_subtasks=False,
        )
        assert result.normalized_total_score == pytest.approx(0.7)

    def test_subtask_only(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.MAX,
            submission_score=0.0,
            submission_max_score=0.0,
            subtask_score=2.0,
            subtask_max_score=4.0,
            has_submission=False,
            has_subtasks=True,
        )
        assert result.normalized_total_score == pytest.approx(0.5)

    def test_neither_component(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.MAX,
            submission_score=0.0,
            submission_max_score=0.0,
            subtask_score=0.0,
            subtask_max_score=0.0,
            has_submission=False,
            has_subtasks=False,
        )
        assert result.normalized_total_score == pytest.approx(0.0)

    def test_perfect_scores(self) -> None:
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.MAX,
            submission_score=1.0,
            submission_max_score=1.0,
            subtask_score=3.0,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        assert result.normalized_total_score == pytest.approx(1.0)


# ============================================================================
# Cross-strategy comparison tests
# ============================================================================


class TestCrossStrategyComparisons:
    """Tests that show the differences between strategies."""

    def test_single_component_all_strategies_equal(self) -> None:
        """When only one component exists, all strategies produce identical results."""
        for strategy in ScoreAggregationStrategy:
            result = compute_aggregated_score(
                strategy=strategy,
                submission_score=0.7,
                submission_max_score=1.0,
                subtask_score=0.0,
                subtask_max_score=0.0,
                has_submission=True,
                has_subtasks=False,
            )
            assert result.normalized_total_score == pytest.approx(0.7), (
                f"Strategy {strategy} should produce 0.7 for submission-only"
            )

    def test_single_component_subtask_only_all_equal(self) -> None:
        """When only subtasks exist, all strategies produce identical results."""
        for strategy in ScoreAggregationStrategy:
            result = compute_aggregated_score(
                strategy=strategy,
                submission_score=0.0,
                submission_max_score=0.0,
                subtask_score=2.0,
                subtask_max_score=4.0,
                has_submission=False,
                has_subtasks=True,
            )
            assert result.normalized_total_score == pytest.approx(0.5), (
                f"Strategy {strategy} should produce 0.5 for subtask-only"
            )

    def test_zero_max_scores_all_produce_zero(self) -> None:
        """Zero max scores => all strategies produce 0."""
        for strategy in ScoreAggregationStrategy:
            result = compute_aggregated_score(
                strategy=strategy,
                submission_score=0.0,
                submission_max_score=0.0,
                subtask_score=0.0,
                subtask_max_score=0.0,
                has_submission=False,
                has_subtasks=False,
            )
            assert result.normalized_total_score == pytest.approx(0.0)

    def test_asymmetric_max_shows_strategy_differences(self) -> None:
        """With asymmetric max scores, average and weighted_sum diverge."""
        kwargs = {
            "submission_score": 1.0,
            "submission_max_score": 1.0,
            "subtask_score": 0.0,
            "subtask_max_score": 9.0,  # Large asymmetry
            "has_submission": True,
            "has_subtasks": True,
        }
        avg = compute_aggregated_score(strategy=ScoreAggregationStrategy.AVERAGE, **kwargs)
        ws = compute_aggregated_score(strategy=ScoreAggregationStrategy.WEIGHTED_SUM, **kwargs)
        mx = compute_aggregated_score(strategy=ScoreAggregationStrategy.MAX, **kwargs)

        # average: (1.0 + 0.0) / 2 = 0.5
        assert avg.normalized_total_score == pytest.approx(0.5)
        # weighted_sum: (1.0 + 0.0) / (1.0 + 9.0) = 0.1
        assert ws.normalized_total_score == pytest.approx(0.1)
        # max: max(1.0, 0.0) = 1.0
        assert mx.normalized_total_score == pytest.approx(1.0)

    def test_max_possible_metadata(self) -> None:
        """max_possible should reflect raw maximum across submission + subtask."""
        result = compute_aggregated_score(
            strategy=ScoreAggregationStrategy.AVERAGE,
            submission_score=0.5,
            submission_max_score=1.0,
            subtask_score=1.0,
            subtask_max_score=3.0,
            has_submission=True,
            has_subtasks=True,
        )
        assert result.max_possible == pytest.approx(4.0)
