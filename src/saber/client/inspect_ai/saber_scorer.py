"""
SABER Task Scorer for inspect_ai

Scorer implementation for SABER tasks within inspect_ai framework.
This provides both client-side and server-side scoring capabilities
while maintaining compatibility with SABER's server-side success criteria evaluation.

Following SABER's philosophy:
- Fail fast when scoring prerequisites are not met
- Clean interface between inspect_ai scoring and SABER task evaluation
- Explicit handling of server-side vs client-side scoring
- No silent scoring failures that mask evaluation issues
"""

import logging
from typing import List, Optional

from inspect_ai.scorer import Score, Scorer, Target, metric, scorer
from inspect_ai.scorer._metric import Metric, SampleScore, ValueToFloat, value_to_float
from inspect_ai.solver import TaskState
from inspect_ai.util import store

from ...models.rest.evaluation import EvaluationResultResponse
from ..client_session import ClientSessionManager
from ..exceptions import EvaluationNotFoundError, InvalidEvaluationRequestError, SessionEvaluationError

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

    This scorer retrieves evaluation results from the SABER server,
    providing authoritative scoring based on server-side success criteria.
    Follows SABER's fail-fast principles.

    Returns:
        Scorer function that retrieves server evaluation results
    """
    # Return the server scorer implementation
    return saber_server_scorer()


@scorer(metrics=[saber_server_score()])  # type: ignore[misc]
def saber_server_scorer() -> Scorer:
    """
    Scorer that retrieves evaluation results from SABER server.

    This scorer connects to the SABER server to get the authoritative
    evaluation results based on server-side success criteria evaluation.
    Follows fail-fast principles - if evaluation retrieval fails, the scorer fails.

    The scorer expects session_id and episode_id to be available in the TaskState
    metadata, typically set by the SABER agent during episode execution.

    Returns:
        Scorer function that retrieves server evaluation results
    """

    async def score(state: TaskState, target: Target) -> Score:
        """
        Retrieve score from SABER server evaluation.

        Args:
            state: TaskState after execution containing SABER context
            target: Target criteria (unused for server-side scoring)

        Returns:
            Score from server evaluation

        Raises:
            RuntimeError: If SABER context is missing or evaluation retrieval fails
        """
        try:
            # Extract SABER context from inspect_ai store
            task_store = store()
            session_manager = task_store.get("saber_session_manager")
            session_id = task_store.get("saber_session_id")

            # Get episode_id from task store (set by SABER agent)
            current_episode = task_store.get("saber_current_episode")
            episode_id = current_episode.episode_id if current_episode else None

            # Validate SABER context
            if not session_manager:
                raise RuntimeError(
                    "SABER session manager not found in context. " "This scorer requires SABER agent integration."
                )

            if not session_id:
                raise RuntimeError(
                    "SABER session ID not found in context. " "This scorer requires an active SABER session."
                )

            if not episode_id:
                raise RuntimeError(
                    "SABER episode ID not found in context. " "This scorer requires an active SABER episode."
                )

            logger.info(f"Retrieving server evaluation for session {session_id}, episode {episode_id}")

            # Retrieve evaluation from server
            evaluation = await session_manager.get_episode_evaluation(session_id, episode_id)

            # Convert server evaluation to inspect_ai Score
            return _convert_evaluation_to_score(evaluation)

        except (EvaluationNotFoundError, SessionEvaluationError, InvalidEvaluationRequestError) as e:
            # SABER fail-fast principle: evaluation errors should fail the scorer
            logger.error(f"Server evaluation retrieval failed: {e}")
            raise RuntimeError(f"Server evaluation failed: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error in server scorer: {e}")
            raise RuntimeError(f"Server scorer error: {e}") from e

    return score


def _convert_evaluation_to_score(evaluation: EvaluationResultResponse) -> Score:
    """
    Convert server evaluation result to inspect_ai Score.

    Args:
        evaluation: Server evaluation result

    Returns:
        inspect_ai Score object with server evaluation data
    """
    # Calculate scaled score (server provides this directly)
    scaled_score = evaluation.score

    # Extract submission and analysis from details
    submission = evaluation.details.get("submission", "")
    analysis = evaluation.details.get("analysis", "")

    # Create explanation combining strategy and analysis
    explanation_parts = [
        f"Server evaluation using {evaluation.strategy} strategy",
        f"Raw score: {evaluation.raw_score}/{evaluation.max_score}",
        f"Success: {evaluation.success}",
    ]

    if analysis:
        explanation_parts.append(f"Analysis: {analysis}")

    explanation = ". ".join(explanation_parts)

    # Build comprehensive metadata
    metadata = {
        "episode_id": evaluation.episode_id,
        "task_id": evaluation.task_id,
        "strategy": evaluation.strategy,
        "raw_score": evaluation.raw_score,
        "max_score": evaluation.max_score,
        "server_score": scaled_score,
        "success": evaluation.success,
        "timestamp": evaluation.timestamp.isoformat(),
        "scorer_type": "saber_server_side",
        "evaluation_details": evaluation.details,
    }

    return Score(
        value=scaled_score,
        answer=submission,
        explanation=explanation,
        metadata=metadata,
    )


async def _get_server_side_score(state: TaskState) -> Optional[float]:
    """
    Retrieve actual task score from SABER server.

    This function attempts to retrieve the server-side evaluation score
    from the SABER server. It's used by other scorers that need just the
    numeric score value rather than the full Score object.

    Args:
        state: TaskState with SABER context in inspect_ai store

    Returns:
        Server evaluation score (0.0-1.0) or None if retrieval fails
    """
    try:
        # Extract SABER context from inspect_ai store
        task_store = store()
        session_manager = task_store.get("saber_session_manager")
        session_id = task_store.get("saber_session_id")
        current_episode = task_store.get("saber_current_episode")
        episode_id = current_episode.episode_id if current_episode else None

        # Validate context
        if not session_manager or not session_id or not episode_id:
            logger.debug("SABER context incomplete, cannot retrieve server score")
            return None

        # Retrieve evaluation from server
        evaluation = await session_manager.get_episode_evaluation(session_id, episode_id)

        logger.debug(f"Retrieved server score: {evaluation.score} for episode {episode_id}")
        return float(evaluation.score)

    except Exception as e:
        logger.debug(f"Server score retrieval failed: {e}")
        return None


class SABERTaskScorer:
    """
    Factory class for creating SABER task scorer instances.

    Provides convenience methods for different types of SABER scoring:
    - Client-side basic scoring
    - Server-side authoritative scoring
    - Combined scoring approaches
    - Evaluation retrieval utilities
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

        Now returns server-side scorer for authoritative evaluation.
        Falls back to client-side scoring if server evaluation fails.

        Returns:
            Default SABER task scorer
        """
        return saber_server_scorer()

    @staticmethod
    async def get_server_evaluation(
        session_manager: ClientSessionManager, session_id: str, episode_id: str
    ) -> EvaluationResultResponse:
        """
        Retrieve server evaluation for specific episode.

        Args:
            session_manager: Active session manager
            session_id: Session ID
            episode_id: Episode ID

        Returns:
            Server evaluation result

        Raises:
            EvaluationRetrievalError: If evaluation retrieval fails
        """
        return await session_manager.get_episode_evaluation(session_id, episode_id)

    @staticmethod
    async def get_session_evaluations(
        session_manager: ClientSessionManager, session_id: str, task_id: Optional[str] = None
    ) -> List[EvaluationResultResponse]:
        """
        Retrieve all evaluations for session.

        Args:
            session_manager: Active session manager
            session_id: Session ID
            task_id: Optional task filter

        Returns:
            List of evaluation results

        Raises:
            EvaluationRetrievalError: If evaluation retrieval fails
        """
        return await session_manager.get_session_evaluations(session_id, task_id)

    @staticmethod
    async def convert_server_evaluation_to_score(evaluation: EvaluationResultResponse) -> Score:
        """
        Convert server evaluation to inspect_ai Score.

        Args:
            evaluation: Server evaluation result

        Returns:
            inspect_ai Score object
        """
        return _convert_evaluation_to_score(evaluation)

    @staticmethod
    async def score_completed_episodes(
        session_manager: ClientSessionManager, session_id: str, task_id: Optional[str] = None
    ) -> List[Score]:
        """
        Score all completed episodes in session using server evaluations.

        Args:
            session_manager: Active session manager
            session_id: Session ID
            task_id: Optional task filter

        Returns:
            List of inspect_ai Score objects

        Raises:
            EvaluationRetrievalError: If evaluation retrieval fails
        """
        evaluations = await session_manager.get_session_evaluations(session_id, task_id)
        return [_convert_evaluation_to_score(eval_result) for eval_result in evaluations]
