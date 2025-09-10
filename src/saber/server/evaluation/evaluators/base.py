"""
Base evaluator class for evaluation system.
"""

from abc import ABC, abstractmethod

from ...benchmarks.task import Task
from ..models import EpisodeEvaluationData, EvaluationConfig, EvaluationResult


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    @abstractmethod
    async def evaluate(
        self, episode_data: EpisodeEvaluationData, config: EvaluationConfig, task: Task
    ) -> EvaluationResult:
        """
        Evaluate episode data against criteria.

        Args:
            episode_data: EpisodeEvaluationData containing episode information
            config: Evaluation configuration from task
            task: Task being evaluated

        Returns:
            EvaluationResult with evaluation outcome

        Raises:
            EvaluationError: If evaluation fails
        """
        pass
