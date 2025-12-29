"""Scoring method registry for SABER evaluation system.

This module provides a factory pattern for registering and retrieving scoring methods.
Domains can register their own custom scoring strategies, and the scorer will dispatch
to the appropriate method based on the evaluation strategy.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ....logging_config import LogCategory, get_saber_logger
from ....models.rest.evaluation import (
    EpisodeStepsResponse,
    EpisodeSubmissionResponse,
    StepEvaluation,
    SubmissionEvaluationCriteriaResponse,
    SubtaskEvaluationCriteriaResponse,
)

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


@dataclass
class ScorerMetadata:
    """Metadata about a scoring function.

    Attributes:
        uses_llm: Whether the scorer uses LLM calls (requires state deepcopy for thread safety)
        requires_submission: Whether the scorer needs submission_data parameter
    """

    uses_llm: bool = False
    requires_submission: bool = False


# Type aliases for scoring functions
SubmissionScorerFunc = Callable[
    [
        EpisodeSubmissionResponse,  # submission_data
        SubmissionEvaluationCriteriaResponse,  # criteria
        Any,  # session_manager
        Any,  # state (TaskState)
    ],
    Awaitable[tuple[float, str]],  # Returns (score, explanation)
]

SubtaskScorerFunc = Callable[
    [
        EpisodeStepsResponse,  # steps_data
        SubtaskEvaluationCriteriaResponse,  # criteria
        Any,  # task_context
        Any,  # session_manager
        Any,  # state (TaskState)
        EpisodeSubmissionResponse,  # submission_data (optional for some strategies)
    ],
    Awaitable[tuple[float, list[StepEvaluation]]],  # Returns (score, step_evaluations)
]


class ScoringRegistry:
    """Registry for submission and subtask scoring methods.

    This registry allows domains to register custom scoring strategies that can be
    invoked by name. Standard scoring methods are registered by default.
    """

    # Class-level storage for registered scorers
    _submission_scorers: dict[str, SubmissionScorerFunc] = {}
    _subtask_scorers: dict[str, SubtaskScorerFunc] = {}
    _subtask_metadata: dict[str, ScorerMetadata] = {}  # Metadata for subtask scorers

    @classmethod
    def register_submission_scorer(cls, strategy: str, scorer_func: SubmissionScorerFunc) -> None:
        """Register a submission scoring function.

        Args:
            strategy: Strategy name (e.g., "static", "llm_judge", "trajectory_analysis")
            scorer_func: Async function that scores submissions
        """
        if strategy in cls._submission_scorers:
            logger.warning(
                f"Overwriting existing submission scorer for strategy '{strategy}'",
                extra={"strategy": strategy, "event": "scorer_overwrite"},
            )

        cls._submission_scorers[strategy] = scorer_func
        logger.info(
            f"Registered submission scorer for strategy '{strategy}'",
            extra={"strategy": strategy, "event": "scorer_registered"},
        )

    @classmethod
    def register_subtask_scorer(
        cls, strategy: str, scorer_func: SubtaskScorerFunc, metadata: ScorerMetadata | None = None
    ) -> None:
        """Register a subtask scoring function.

        Args:
            strategy: Strategy name (e.g., "static", "llm_judge", "tool_call")
            scorer_func: Async function that scores subtasks
            metadata: Optional metadata about the scorer (uses_llm, requires_submission)
        """
        if strategy in cls._subtask_scorers:
            logger.warning(
                f"Overwriting existing subtask scorer for strategy '{strategy}'",
                extra={"strategy": strategy, "event": "scorer_overwrite"},
            )

        cls._subtask_scorers[strategy] = scorer_func
        cls._subtask_metadata[strategy] = metadata or ScorerMetadata()
        uses_llm = cls._subtask_metadata[strategy].uses_llm
        logger.info(
            f"Registered subtask scorer for strategy '{strategy}' (uses_llm={uses_llm})",
            extra={
                "strategy": strategy,
                "event": "scorer_registered",
                "uses_llm": cls._subtask_metadata[strategy].uses_llm,
            },
        )

    @classmethod
    def get_submission_scorer(cls, strategy: str) -> SubmissionScorerFunc:
        """Get a registered submission scorer by strategy name.

        Args:
            strategy: Strategy name

        Returns:
            Scoring function

        Raises:
            KeyError: If strategy is not registered
        """
        if strategy not in cls._submission_scorers:
            available = ", ".join(cls._submission_scorers.keys())
            raise KeyError(f"Unknown submission evaluation strategy: '{strategy}'. Available strategies: {available}")
        return cls._submission_scorers[strategy]

    @classmethod
    def get_subtask_scorer(cls, strategy: str) -> SubtaskScorerFunc:
        """Get a registered subtask scorer by strategy name.

        Args:
            strategy: Strategy name

        Returns:
            Scoring function

        Raises:
            KeyError: If strategy is not registered
        """
        if strategy not in cls._subtask_scorers:
            available = ", ".join(cls._subtask_scorers.keys())
            raise KeyError(f"Unknown subtask evaluation strategy: '{strategy}'. Available strategies: {available}")
        return cls._subtask_scorers[strategy]

    @classmethod
    def get_subtask_scorer_metadata(cls, strategy: str) -> ScorerMetadata:
        """Get metadata for a registered subtask scorer.

        Args:
            strategy: Strategy name

        Returns:
            Scorer metadata
        """
        return cls._subtask_metadata.get(strategy, ScorerMetadata())

    @classmethod
    def list_submission_strategies(cls) -> list[str]:
        """List all registered submission scoring strategies."""
        return list(cls._submission_scorers.keys())

    @classmethod
    def list_subtask_strategies(cls) -> list[str]:
        """List all registered subtask scoring strategies."""
        return list(cls._subtask_scorers.keys())


# Convenience functions for registration
def register_submission_scorer(strategy: str, scorer_func: SubmissionScorerFunc) -> None:
    """Register a submission scoring function.

    Args:
        strategy: Strategy name
        scorer_func: Async scoring function
    """
    ScoringRegistry.register_submission_scorer(strategy, scorer_func)


def register_subtask_scorer(
    strategy: str, scorer_func: SubtaskScorerFunc, metadata: ScorerMetadata | None = None
) -> None:
    """Register a subtask scoring function.

    Args:
        strategy: Strategy name
        scorer_func: Async scoring function
        metadata: Optional metadata about the scorer
    """
    ScoringRegistry.register_subtask_scorer(strategy, scorer_func, metadata)


def get_submission_scorer(strategy: str) -> SubmissionScorerFunc:
    """Get a registered submission scorer.

    Args:
        strategy: Strategy name

    Returns:
        Scoring function
    """
    return ScoringRegistry.get_submission_scorer(strategy)


def get_subtask_scorer(strategy: str) -> SubtaskScorerFunc:
    """Get a registered subtask scorer.

    Args:
        strategy: Strategy name

    Returns:
        Scoring function
    """
    return ScoringRegistry.get_subtask_scorer(strategy)


def get_subtask_scorer_metadata(strategy: str) -> ScorerMetadata:
    """Get metadata for a registered subtask scorer.

    Args:
        strategy: Strategy name

    Returns:
        Scorer metadata
    """
    return ScoringRegistry.get_subtask_scorer_metadata(strategy)
