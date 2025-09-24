"""
SessionEvaluationService - Business logic layer for evaluation retrieval.

Logging Category: EVALUATION

Provides session-scoped evaluation retrieval operations with fail-fast behavior.
No backwards compatibility or defensive programming fallbacks.
"""

from typing import Any, Dict, List, Optional

from saber.logging_config import LogCategory, get_saber_logger

from .exceptions import EvaluationNotFoundError, InvalidEvaluationRequestError, SessionEvaluationError
from .models import EvaluationResult
from .store import EvaluationStore

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


class SessionEvaluationService:
    """Service layer for session-scoped evaluation retrieval operations."""

    def __init__(self, store: EvaluationStore):
        """
        Initialize the service with an evaluation store.

        Args:
            store: EvaluationStore implementation for data access
        """
        self.store = store
        logger.info(
            "Evaluation session service initialized",
            extra={
                "event": "session_evaluation_service_initialized",
                "store_type": type(store).__name__,
            },
        )

    async def get_evaluation(self, session_id: str, episode_id: str) -> EvaluationResult:
        """
        Get single evaluation result by episode.

        Args:
            session_id: Session ID containing the episode
            episode_id: Episode ID to retrieve evaluation for

        Returns:
            EvaluationResult object

        Raises:
            EvaluationNotFoundError: If evaluation not found
            InvalidEvaluationRequestError: If parameters are invalid
            SessionEvaluationError: If session access fails
        """
        logger.debug(
            "Retrieving evaluation",
            extra={
                "event": "session_evaluation_retrieval_requested",
                "session_id": session_id,
                "episode_id": episode_id,
            },
        )

        try:
            result = await self.store.get(session_id, episode_id)
            logger.info(
                "Evaluation retrieved",
                extra={
                    "event": "session_evaluation_retrieved",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "score": result.score,
                    "max_score": result.max_score,
                    "strategy": result.strategy,
                    "success": result.success,
                },
            )
            return result
        except (EvaluationNotFoundError, InvalidEvaluationRequestError):
            # Re-raise these specific errors as-is (fail-fast)
            raise
        except Exception as e:
            # Wrap unexpected errors in SessionEvaluationError
            raise SessionEvaluationError(
                f"Failed to retrieve evaluation for episode {episode_id} in session {session_id}: {e}"
            ) from e

    async def list_session_evaluations(self, session_id: str, task_id: Optional[str] = None) -> List[EvaluationResult]:
        """
        List all evaluations for a session, optionally filtered by task.

        Args:
            session_id: Session ID to list evaluations for
            task_id: Optional task ID filter

        Returns:
            List of EvaluationResult objects, sorted by timestamp

        Raises:
            InvalidEvaluationRequestError: If parameters are invalid
            SessionEvaluationError: If session access fails
        """
        logger.debug(
            "Listing session evaluations",
            extra={
                "event": "session_evaluations_listing_requested",
                "session_id": session_id,
                "task_id": task_id,
            },
        )

        try:
            results = await self.store.list_by_session(session_id, task_id)
            logger.info(
                "Session evaluations listed",
                extra={
                    "event": "session_evaluations_listed",
                    "session_id": session_id,
                    "task_id": task_id,
                    "evaluation_count": len(results),
                },
            )
            return results
        except InvalidEvaluationRequestError:
            # Re-raise validation errors as-is (fail-fast)
            raise
        except Exception as e:
            # Wrap unexpected errors in SessionEvaluationError
            raise SessionEvaluationError(f"Failed to list evaluations for session {session_id}: {e}") from e

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get aggregate evaluation summary for session.

        Args:
            session_id: Session ID to summarize

        Returns:
            Dictionary with session summary statistics

        Raises:
            InvalidEvaluationRequestError: If parameters are invalid
            SessionEvaluationError: If session access fails
        """
        logger.debug(
            "Session summary requested",
            extra={
                "event": "session_evaluation_summary_requested",
                "session_id": session_id,
            },
        )

        try:
            # Get all evaluations for the session
            evaluations: List[EvaluationResult] = await self.store.list_by_session(session_id)

            if not evaluations:
                return {
                    "session_id": session_id,
                    "total_episodes": 0,
                    "successful_episodes": 0,
                    "average_score": 0.0,
                    "task_summaries": {},
                }

            # Calculate aggregate statistics
            total_episodes = len(evaluations)
            successful_episodes = sum(1 for eval_result in evaluations if eval_result.success)
            total_score: float = sum(eval_result.score for eval_result in evaluations)
            average_score = total_score / total_episodes if total_episodes > 0 else 0.0

            # Group by task for task-level summaries
            task_summaries: Dict[str, Dict[str, Any]] = {}
            for eval_result in evaluations:
                task_id = eval_result.task_id
                if task_id not in task_summaries:
                    task_summaries[task_id] = {
                        "total_episodes": 0,
                        "successful_episodes": 0,
                        "total_score": 0.0,
                        "average_score": 0.0,
                        "strategy": eval_result.strategy,
                    }

                task_summary = task_summaries[task_id]
                task_summary["total_episodes"] += 1
                task_summary["total_score"] += eval_result.score
                if eval_result.success:
                    task_summary["successful_episodes"] += 1

            # Calculate task-level averages
            for task_summary in task_summaries.values():
                if task_summary["total_episodes"] > 0:
                    task_summary["average_score"] = task_summary["total_score"] / task_summary["total_episodes"]

            summary = {
                "session_id": session_id,
                "total_episodes": total_episodes,
                "successful_episodes": successful_episodes,
                "average_score": round(average_score, 3),
                "task_summaries": task_summaries,
            }

            logger.info(
                "Session summary generated",
                extra={
                    "event": "session_evaluation_summary_generated",
                    "session_id": session_id,
                    "total_episodes": total_episodes,
                    "successful_episodes": successful_episodes,
                    "average_score": summary["average_score"],
                    "task_count": len(task_summaries),
                },
            )
            return summary

        except InvalidEvaluationRequestError:
            # Re-raise validation errors as-is (fail-fast)
            raise
        except Exception as e:
            # Wrap unexpected errors in SessionEvaluationError
            raise SessionEvaluationError(f"Failed to generate summary for session {session_id}: {e}") from e

    async def validate_session_access(self, session_id: str) -> bool:
        """
        Validate that session exists and has evaluation data.

        Args:
            session_id: Session ID to validate

        Returns:
            True if session exists and has data

        Raises:
            InvalidEvaluationRequestError: If session_id is invalid
        """
        if not session_id or not session_id.strip():
            raise InvalidEvaluationRequestError("session_id cannot be empty")

        try:
            return await self.store.session_exists(session_id)
        except Exception as e:
            raise SessionEvaluationError(f"Failed to validate session {session_id}: {e}") from e
