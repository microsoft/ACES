"""
SABER Task Scorer for inspect_ai

Scorer implementation for SABER tasks within inspect_ai framework.
This provides basic client-side scoring while maintaining compatibility
with SABER's server-side success criteria evaluation.

Following SABER's philosophy:
- Fail fast when scoring prerequisites are not met
- Clean interface between inspect_ai scoring and SABER task evaluation
- Explicit handling of server-side vs client-side scoring
- No silent scoring failures that mask evaluation issues
"""

import logging
from typing import Any, Dict, List, Optional

from inspect_ai.scorer import Score, Scorer, Target, metric, scorer
from inspect_ai.scorer._metric import Metric, SampleScore, ValueToFloat, value_to_float
from inspect_ai.solver import TaskState

logger = logging.getLogger(__name__)


@metric  # type: ignore[misc]
def saber_success(to_float: ValueToFloat = value_to_float()) -> Metric:
    """
    Metric for SABER task success rate.

    Args:
        to_float: Function for mapping Value to float for computing metrics

    Returns:
        Success rate metric function
    """

    def metric_fn(scores: List[SampleScore]) -> float:
        total = 0.0
        for item in scores:
            total += to_float(item.score.value)
        return total / float(len(scores))

    return metric_fn


@metric  # type: ignore[misc]
def saber_completion(to_float: ValueToFloat = value_to_float()) -> Metric:
    """
    Metric for SABER task completion rate.

    Args:
        to_float: Function for mapping Value to float for computing metrics

    Returns:
        Completion rate metric function
    """

    def metric_fn(scores: List[SampleScore]) -> float:
        total = 0.0
        for item in scores:
            # Extract completion score from metadata if available
            completion_score = (
                item.score.metadata.get("completion_score", item.score.value)
                if item.score.metadata
                else item.score.value
            )
            total += to_float(completion_score)
        return total / float(len(scores))

    return metric_fn


@metric  # type: ignore[misc]
def saber_server_score(to_float: ValueToFloat = value_to_float()) -> Metric:
    """
    Metric for SABER server-side evaluation score.

    Args:
        to_float: Function for mapping Value to float for computing metrics

    Returns:
        Server evaluation metric function
    """

    def metric_fn(scores: List[SampleScore]) -> float:
        total = 0.0
        for item in scores:
            # Extract server score from metadata if available
            server_score = (
                item.score.metadata.get("server_score", item.score.value) if item.score.metadata else item.score.value
            )
            total += to_float(server_score)
        return total / float(len(scores))

    return metric_fn


@scorer(metrics=[saber_success(), saber_completion()])  # type: ignore[misc]
def saber_task_scorer() -> Scorer:
    """
    SABER task scorer for inspect_ai framework.

    This scorer provides basic client-side evaluation for SABER tasks.
    The real evaluation logic resides on the SABER server, but this
    scorer provides immediate feedback and compatibility with inspect_ai's
    scoring framework.

    Returns:
        Scorer function that evaluates SABER task completion
    """

    async def score(state: TaskState, target: Target) -> Score:
        """
        Score SABER task completion based on TaskState and target.

        STUBBED IMPLEMENTATION: Real evaluation happens server-side.
        This is a placeholder that returns a basic score for inspect_ai compatibility.

        Args:
            state: Current TaskState after agent execution
            target: Expected target/success criteria

        Returns:
            Score object with basic evaluation results
        """

        # STUB: Basic scoring based on whether agent executed
        has_messages = len(state.messages) > 0
        completion_score = 0.5 if has_messages else 0.0
        success_score = 0.3 if has_messages else 0.0

        explanation = (
            f"STUB: Basic client-side score ({len(state.messages)} messages). Real evaluation happens server-side."
        )

        logger.debug(
            f"SABER task scored (STUB): completion={completion_score}, success={success_score} - {explanation}"
        )

        # Return overall score (average for simplicity)
        overall_score = (completion_score + success_score) / 2.0

        return Score(
            value=overall_score,
            explanation=explanation,
            metadata={
                "message_count": len(state.messages),
                "completion_score": completion_score,
                "success_score": success_score,
                "scorer_type": "saber_client_side_stub",
            },
        )

    return score


def _extract_scoring_info_from_state(state: TaskState) -> Dict[str, Any]:
    """
    STUB: Extract scoring-relevant information from TaskState.

    Real implementation will analyze container execution results and success indicators.
    Currently returns minimal placeholder data.

    Args:
        state: TaskState to extract information from

    Returns:
        Dictionary with basic scoring information
    """

    return {"message_count": len(state.messages), "stub": True}


def _calculate_completion_score(state: TaskState, target: Target) -> float:
    """
    STUB: Calculate basic completion score.

    Real implementation will provide sophisticated completion analysis.
    Currently returns simple message-based score.

    Args:
        state: TaskState after execution
        target: Target criteria

    Returns:
        Basic completion score between 0.0 and 1.0
    """

    return 0.5 if len(state.messages) > 0 else 0.0


def _calculate_success_score(state: TaskState, scoring_info: Dict[str, Any]) -> float:
    """
    STUB: Calculate success score based on explicit success indicators.

    Real implementation will analyze container results and success criteria.
    Currently returns placeholder score.

    Args:
        state: TaskState after execution
        scoring_info: Extracted scoring information

    Returns:
        Basic success score between 0.0 and 1.0
    """

    return 0.3  # Placeholder success score


def _create_score_explanation(completion_score: float, success_score: float, scoring_info: Dict[str, Any]) -> str:
    """
    STUB: Create human-readable explanation of the scoring decision.

    Real implementation will provide detailed scoring rationale.
    Currently returns basic explanation.

    Args:
        completion_score: Basic completion score
        success_score: Success-based score
        scoring_info: Extracted scoring information

    Returns:
        Simple string explanation of the score
    """

    return f"STUB: {scoring_info['message_count']} messages processed. Real evaluation happens server-side."


@scorer(metrics=[saber_server_score()])  # type: ignore[misc]
def saber_server_scorer() -> Scorer:
    """
    Future enhancement: Scorer that retrieves evaluation results from SABER server.

    This scorer would connect to the SABER server to get the authoritative
    evaluation results based on server-side success criteria evaluation.

    Currently returns a placeholder implementation.
    """

    async def score(state: TaskState, target: Target) -> Score:
        """
        STUB: Retrieve score from SABER server.

        Real implementation will connect to SABER server for authoritative evaluation.
        Currently falls back to client-side stub scoring.

        Args:
            state: TaskState after execution
            target: Target criteria

        Returns:
            Score from server evaluation (currently stubbed)
        """

        # STUB: Server-side scoring not implemented yet
        logger.debug("Server-side scoring not yet implemented, using client-side stub")

        # Use basic stub scoring
        has_messages = len(state.messages) > 0
        server_score_value = 0.4 if has_messages else 0.0

        explanation = (
            f"STUB: Server-side scoring ({len(state.messages)} messages). Real server evaluation not implemented."
        )

        return Score(
            value=server_score_value,
            explanation=explanation,
            metadata={
                "message_count": len(state.messages),
                "server_score": server_score_value,
                "scorer_type": "saber_server_side_stub",
            },
        )

    return score


async def _get_server_side_score(state: TaskState) -> Optional[float]:
    """
    STUB: Retrieve actual task score from SABER server.

    Real implementation will connect to SABER server for authoritative evaluation.
    Currently always returns None to indicate server scoring unavailable.

    Args:
        state: TaskState with sample metadata containing server info

    Returns:
        None (server-side scoring not implemented)
    """

    # STUB: Always return None - server-side scoring not implemented
    logger.debug("Server-side score retrieval is stubbed out")
    return None


class SABERTaskScorer:
    """
    Factory class for creating SABER task scorer instances.

    Provides convenience methods for different types of SABER scoring:
    - Client-side basic scoring
    - Server-side authoritative scoring (future)
    - Combined scoring approaches
    """

    @staticmethod
    def create_client_scorer() -> Scorer:
        """
        Create client-side SABER task scorer.

        Returns:
            Scorer that performs basic client-side evaluation
        """
        return saber_task_scorer()

    @staticmethod
    def create_server_scorer() -> Scorer:
        """
        Create server-side SABER task scorer.

        Returns:
            Scorer that retrieves evaluation from SABER server
        """
        return saber_server_scorer()

    @staticmethod
    def create_default_scorer() -> Scorer:
        """
        Create default SABER task scorer.

        Currently returns client-side scorer. In the future, this could
        be enhanced to try server-side scoring with client-side fallback.

        Returns:
            Default SABER task scorer
        """
        return saber_task_scorer()
