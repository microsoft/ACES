"""
Enhanced EvaluationManager implementation for SABER domain server.

Logging Category: EVALUATION

This implementation provides fail-fast evaluation capabilities with no backwards compatibility.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from saber.logging_config import LogCategory, get_saber_logger

from ..base import Episode
from ..benchmarks.task import Task
from .constants import EVAL_STRATEGY_LLM_JUDGE, EVAL_STRATEGY_STATIC, SUPPORTED_STRATEGIES
from .evaluators import BaseEvaluator, LLMEvaluator, StaticEvaluator
from .exceptions import (
    EvaluationConfigError,
    EvaluatorNotFoundError,
    IncompleteEpisodeError,
    InvalidEvaluationRequestError,
    InvalidEvaluationStrategyError,
    MissingSubmissionError,
)
from .models import EpisodeEvaluationData, EvaluationConfig, EvaluationResult
from .store import EvaluationStore, JsonFileEvaluationStore

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


class EvaluationManager:
    """
    Enhanced EvaluationManager for evaluating agent performance.

    Follows fail-fast principles with no backwards compatibility or fallbacks.
    """

    def __init__(
        self,
        config_dir: Optional[str] = None,
        store: Optional[Union[EvaluationStore, str, Path]] = None,
    ) -> None:
        """
        Initialize the EvaluationManager with evaluators.

        Args:
            config_dir: Optional configuration directory for evaluators
        """
        self.evaluators: Dict[str, BaseEvaluator] = {
            EVAL_STRATEGY_STATIC: StaticEvaluator(),
            EVAL_STRATEGY_LLM_JUDGE: LLMEvaluator(config_dir=config_dir),
        }
        self.evaluation_configs: Dict[str, EvaluationConfig] = {}
        self.config_dir = config_dir

        # Initialize store with proper typing
        if store is None:
            self.store: EvaluationStore = JsonFileEvaluationStore()
        elif isinstance(store, (str, Path)):
            # Treat as base directory path for JSON store
            self.store = JsonFileEvaluationStore(base_dir=str(store))
        elif isinstance(store, EvaluationStore):
            self.store = store
        else:
            raise TypeError(
                f"Unsupported store type: {type(store)}. Provide EvaluationStore, path string, or Path object."
            )

        logger.info(
            "Evaluation manager initialized",
            extra={
                "event": "evaluation_manager_initialized",
                "evaluators": sorted(self.evaluators.keys()),
                "config_dir": config_dir,
                "store_type": type(self.store).__name__,
            },
        )

    def configure_for_task(self, task: Task) -> None:
        """
        Configure evaluation for a specific task. Fails fast on invalid config.

        Args:
            task: Task to configure evaluation for

        Raises:
            EvaluationConfigError: If task lacks evaluation_config or config is invalid
            InvalidEvaluationStrategyError: If strategy is not supported
        """
        if not hasattr(task, "evaluation_config") or not task.evaluation_config:
            raise EvaluationConfigError(
                f"Task {task.task_id} missing required evaluation_config. "
                "All tasks MUST have evaluation configuration."
            )

        try:
            config = EvaluationConfig.from_dict(task.evaluation_config)
        except Exception as e:
            raise EvaluationConfigError(f"Invalid evaluation_config for task {task.task_id}: {e}") from e

        if config.strategy not in SUPPORTED_STRATEGIES:
            raise InvalidEvaluationStrategyError(
                f"Unsupported evaluation strategy '{config.strategy}' for task {task.task_id}. "
                f"Supported strategies: {SUPPORTED_STRATEGIES}"
            )

        if config.strategy not in self.evaluators:
            raise EvaluatorNotFoundError(f"No evaluator available for strategy: {config.strategy}")

        # Validate strategy-specific configuration
        self._validate_strategy_config(config, task.task_id)

        self.evaluation_configs[task.task_id] = config
        logger.info(
            "Task evaluation configured",
            extra={
                "event": "task_evaluation_configured",
                "task_id": task.task_id,
                "strategy": config.strategy,
                "criteria_keys": sorted(config.criteria.keys()),
                "scoring_keys": sorted(config.scoring.keys()),
            },
        )

    def _validate_strategy_config(self, config: EvaluationConfig, task_id: str) -> None:
        """
        Validate strategy-specific configuration. Fails fast on invalid config.

        Args:
            config: Evaluation configuration to validate
            task_id: Task ID for error reporting

        Raises:
            EvaluationConfigError: If configuration is invalid
        """
        if config.strategy == EVAL_STRATEGY_STATIC:
            expected_answers = config.criteria.get("expected_answers")
            if not expected_answers or not isinstance(expected_answers, list):
                raise EvaluationConfigError(
                    f"Task {task_id}: static strategy requires 'expected_answers' as a list in criteria"
                )

        # Validate scoring configuration
        max_score = config.scoring.get("max_score", 1.0)
        if not isinstance(max_score, (int, float)) or max_score <= 0:
            raise EvaluationConfigError(f"Task {task_id}: max_score must be a positive number, got: {max_score}")

    async def evaluate_episode(self, episode: Episode, task: Task) -> EvaluationResult:
        """
        Evaluate a completed episode. Fails fast on any issues.

        Args:
            episode: Completed episode to evaluate
            task: Task that was being executed

        Returns:
            EvaluationResult with evaluation outcome

        Raises:
            IncompleteEpisodeError: If episode is not complete
            EvaluationConfigError: If task not configured for evaluation
            MissingSubmissionError: If episode lacks required submission
        """
        # Fail fast validation
        if not episode.is_complete:
            raise IncompleteEpisodeError(f"Cannot evaluate incomplete episode {episode.episode_id}")

        if task.task_id not in self.evaluation_configs:
            raise EvaluationConfigError(f"Task {task.task_id} not configured for evaluation")

        if not hasattr(episode, "submission") or not episode.submission:
            raise MissingSubmissionError(f"Episode {episode.episode_id} missing required submission for evaluation")

        config = self.evaluation_configs[task.task_id]
        evaluator = self.evaluators[config.strategy]

        # Create strongly typed evaluation data directly
        episode_data = EpisodeEvaluationData(
            episode_id=episode.episode_id,
            task_id=episode.task_id,
            submission=episode.submission,
            executed_commands=episode.get_executed_commands(),
            completion_reason=episode.completion_reason,
            step_count=len(episode.steps),
            # Add rich EvalSubmission data if available
            model=(
                episode.eval_submission.model
                if hasattr(episode, "eval_submission") and episode.eval_submission
                else None
            ),
            choices=(
                episode.eval_submission.choices
                if hasattr(episode, "eval_submission") and episode.eval_submission
                else []
            ),
            tokens=(
                episode.eval_submission.tokens
                if hasattr(episode, "eval_submission") and episode.eval_submission
                else {}
            ),
            execution_time=(
                episode.eval_submission.time
                if hasattr(episode, "eval_submission") and episode.eval_submission
                else None
            ),
        )

        # Log enhanced evaluation data if available
        if hasattr(episode, "eval_submission") and episode.eval_submission:
            eval_submission = episode.eval_submission
            logger.debug(
                "Episode evaluation submission metadata",
                extra={
                    "event": "episode_evaluation_submission_metadata",
                    "episode_id": episode.episode_id,
                    "task_id": episode.task_id,
                    "model": eval_submission.model,
                    "total_tokens": eval_submission.tokens.get("total_tokens", 0),
                    "evaluation_time": eval_submission.time,
                },
            )

        logger.info(
            "Episode evaluation started",
            extra={
                "event": "episode_evaluation_started",
                "episode_id": episode.episode_id,
                "task_id": task.task_id,
                "strategy": config.strategy,
                "step_count": len(episode.steps),
            },
        )

        # Pass episode object to LLM evaluator for enhanced judge prompt context
        if config.strategy == EVAL_STRATEGY_LLM_JUDGE:
            result = await evaluator.evaluate(episode_data, config, task, episode)
        else:
            result = await evaluator.evaluate(episode_data, config, task, episode)
        # Persist result (fail fast on any write issues)
        if config.strategy == EVAL_STRATEGY_LLM_JUDGE:
            golden_answer = config.criteria.get("golden_answer")
        else:
            golden_answer = None
        try:
            await self.store.save(result, session_id=episode.session_id, golden_answer=golden_answer)
        except Exception:
            # Re-raise to enforce atomic contract (no silent persistence failures)
            raise
        logger.info(
            "Episode evaluation completed",
            extra={
                "event": "episode_evaluation_complete",
                "episode_id": episode.episode_id,
                "task_id": task.task_id,
                "strategy": config.strategy,
                "score": result.score,
                "max_score": result.max_score,
                "raw_score": result.raw_score,
                "success": result.success,
            },
        )

        return result

    async def override_evaluation_result(
        self,
        session_id: str,
        episode_id: str,
        evaluation_data: EpisodeEvaluationData,
        strategy: str,
        raw_score: float,
        max_score: float,
        score: float,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Override existing evaluation result with externally provided evaluation data.

        Args:
            session_id: Session ID for context
            episode_id: Episode ID being overridden (must match evaluation_data.episode_id)
            evaluation_data: EpisodeEvaluationData containing episode information
            strategy: Evaluation strategy used for this result
            raw_score: Raw evaluation score
            max_score: Maximum possible score
            score: Normalized score (0.0 to max_score)
            success: Whether the evaluation was successful
            details: Optional additional evaluation details

        Returns:
            EvaluationResult with the overridden evaluation data

        Raises:
            InvalidEvaluationRequestError: If inputs are invalid
            EvaluationConfigError: If evaluation data is inconsistent
        """
        # Fail-fast validation of inputs
        if not session_id or not session_id.strip():
            raise InvalidEvaluationRequestError("session_id cannot be empty")

        if not episode_id or not episode_id.strip():
            raise InvalidEvaluationRequestError("episode_id cannot be empty")

        if episode_id != evaluation_data.episode_id:
            raise InvalidEvaluationRequestError(
                f"Episode ID mismatch: URL parameter '{episode_id}' does not match "
                f"evaluation data episode_id '{evaluation_data.episode_id}'"
            )

        if not strategy or not strategy.strip():
            raise InvalidEvaluationRequestError("strategy cannot be empty")

        if max_score <= 0:
            raise InvalidEvaluationRequestError("max_score must be greater than 0")

        if raw_score < 0:
            raise InvalidEvaluationRequestError("raw_score cannot be negative")

        if score < 0 or score > max_score:
            raise InvalidEvaluationRequestError(f"score must be between 0 and {max_score}")

        logger.info(
            "Episode evaluation override requested",
            extra={
                "event": "episode_evaluation_override_requested",
                "episode_id": episode_id,
                "session_id": session_id,
                "task_id": evaluation_data.task_id,
                "strategy": strategy,
                "raw_score": raw_score,
                "max_score": max_score,
                "score": score,
                "success": success,
            },
        )

        # Create EvaluationResult from the provided data
        result = EvaluationResult.from_episode_data(
            evaluation_data,
            strategy=strategy,
            raw_score=raw_score,
            max_score=max_score,
            score=score,
            success=success,
            details=details or {},
        )

        # Persist the override result (fail fast on any write issues)
        try:
            await self.store.save(result, session_id=session_id, golden_answer=None)
        except Exception as e:
            # Re-raise to enforce atomic contract (no silent persistence failures)
            raise RuntimeError(f"Failed to persist override evaluation result: {e}") from e

        logger.info(
            "Episode evaluation override completed",
            extra={
                "event": "episode_evaluation_override_complete",
                "episode_id": episode_id,
                "session_id": session_id,
                "task_id": evaluation_data.task_id,
                "strategy": strategy,
                "score": result.score,
                "max_score": result.max_score,
                "raw_score": result.raw_score,
                "success": result.success,
                "override": True,
            },
        )

        return result

    # Legacy logging methods for compatibility with existing SessionManager
    async def log_session_start(self, session_id: str, client_id: str) -> None:
        """Log session start event."""
        logger.info(
            "Evaluation session started",
            extra={
                "event": "evaluation_session_started",
                "session_id": session_id,
                "client_id": client_id,
            },
        )

    async def log_session_end(self, session_id: str) -> None:
        """Log session end event."""
        logger.info(
            "Evaluation session ended",
            extra={
                "event": "evaluation_session_ended",
                "session_id": session_id,
            },
        )

    async def log_episode_start(self, session_id: str, episode_id: str, task_id: str) -> None:
        """Log episode start event (legacy compatibility)."""
        logger.info(
            "Evaluation episode started",
            extra={
                "event": "episode_start",
                "session_id": session_id,
                "episode_id": episode_id,
                "task_id": task_id,
            },
        )

    async def log_episode_end(self, session_id: str, completion_reason: str) -> None:
        """Log episode end event."""
        logger.info(
            "Evaluation episode ended",
            extra={
                "event": "evaluation_episode_ended",
                "session_id": session_id,
                "completion_reason": completion_reason,
            },
        )

    async def log_action(self, session_id: str, episode_id: str, action: Any, result: Any) -> None:
        """Log action execution event."""
        logger.debug(
            "Evaluation action recorded",
            extra={
                "event": "evaluation_action_recorded",
                "session_id": session_id,
                "episode_id": episode_id,
                "action_tool": getattr(action, "tool_name", None),
                "result_type": type(result).__name__ if result is not None else None,
            },
        )

    async def get_trajectory(self, session_id: str) -> List[Any]:
        """Get trajectory for a session (legacy stub)."""
        return []
