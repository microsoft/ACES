"""
Static evaluator implementation for exact string matching.
"""

from typing import Optional

from ...base import Episode
from ...benchmarks.task import Task
from ..constants import EVAL_STRATEGY_STATIC
from ..exceptions import EvaluationValidationError
from ..models import EpisodeEvaluationData, EvaluationConfig, EvaluationResult
from .base import BaseEvaluator


class StaticEvaluator(BaseEvaluator):
    """Static evaluator for exact string matching."""

    async def evaluate(
        self,
        episode_data: EpisodeEvaluationData,
        config: EvaluationConfig,
        task: Task,
        episode: Optional[Episode] = None,
    ) -> EvaluationResult:
        """
        Evaluate episode using static exact matching.

        Args:
            episode_data: EpisodeEvaluationData containing episode information
            config: Evaluation configuration from task
            task: Task being evaluated

        Returns:
            EvaluationResult with evaluation outcome

        Raises:
            EvaluationValidationError: If configuration is invalid
        """
        submission = episode_data.submission.strip()
        expected_answers = config.criteria.get("expected_answers", [])

        # Fail fast validation
        if not expected_answers:
            raise EvaluationValidationError("Static evaluation requires expected_answers in criteria")

        if not isinstance(expected_answers, list):
            raise EvaluationValidationError("expected_answers must be a list")

        # Enforce uniqueness to avoid accidental weighting skew
        if len(set(expected_answers)) != len(expected_answers):
            raise EvaluationValidationError(
                "expected_answers must contain unique values (duplicates create ambiguous scoring)"
            )

        # Perform exact matching (case-sensitive)
        matches = [ans for ans in expected_answers if ans.strip() == submission]
        total = len(expected_answers)
        raw_score = len(matches) / total

        # Get max score from config
        max_score = float(config.scoring.get("max_score", 1.0))
        score = raw_score * max_score

        # Determine success: for single answer tasks, must be exact match (1.0)
        # for multiple answer tasks, any match counts as success
        if total == 1:
            success = raw_score >= 1.0
        else:
            success = raw_score > 0.0

        # Invariant checks (defensive but fail-fast if logic changes later)
        assert 0.0 <= raw_score <= 1.0, f"raw_score out of bounds: {raw_score}"
        assert score <= max_score + 1e-9, "score exceeds max_score"

        return EvaluationResult.from_episode_data(
            episode_data=episode_data,
            strategy=EVAL_STRATEGY_STATIC,
            raw_score=raw_score,
            max_score=max_score,
            score=score,
            success=success,
            details={
                "matched_answers": matches,
                "total_expected": total,
                "submission": submission,
                "expected_answers": expected_answers,
            },
        )
