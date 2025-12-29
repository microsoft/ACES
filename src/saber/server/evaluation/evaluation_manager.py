"""
Enhanced EvaluationManager implementation for SABER domain server.

Logging Category: EVALUATION

CLIENT-SIDE EVALUATION MIGRATION: Server is now dumb storage only.
- Removed: evaluate_episode() - evaluation happens on client
- Removed: Evaluator instantiation - no server-side evaluation logic
- Kept: Store operations for saving/loading evaluation results
- Kept: Configuration validation for task setup
"""

from pathlib import Path
from typing import Any

from saber.logging_config import LogCategory, get_saber_logger

from ..benchmarks.task import Task
from .constants import EVAL_STRATEGY_STATIC, SUPPORTED_STRATEGIES
from .exceptions import EvaluationConfigError, InvalidEvaluationRequestError
from .models import EpisodeEvaluationData, EvaluationConfig, EvaluationResult
from .store import EvaluationStore, JsonFileEvaluationStore

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


class EvaluationManager:
    """
    EvaluationManager for storing and validating evaluation results.

    CLIENT-SIDE EVALUATION: This class no longer performs evaluation.
    It only:
    - Validates task configuration
    - Stores evaluation results from clients
    - Retrieves evaluation results
    """

    def __init__(
        self,
        config_dir: str | None = None,
        store: EvaluationStore | str | Path | None = None,
    ) -> None:
        """
        Initialize the EvaluationManager.

        Args:
            config_dir: Optional configuration directory (legacy, not used for client-side eval)
            store: Optional evaluation store or path
        """
        # Remove evaluator instantiation - evaluation happens client-side now
        self.evaluation_configs: dict[str, EvaluationConfig] = {}
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
            "Evaluation manager initialized (client-side evaluation mode)",
            extra={
                "event": "evaluation_manager_initialized",
                "config_dir": config_dir,
                "store_type": type(self.store).__name__,
                "mode": "client_side_evaluation",
            },
        )

    def configure_for_task(self, task: Task) -> None:
        """
        Configure evaluation for a specific task. Fails fast on invalid config.

        Args:
            task: Task to configure evaluation for

        Raises:
            EvaluationConfigError: If task lacks submission_evaluation_config or config is invalid

        Note:
            Strategy validation happens client-side via the scorer registry.
            The server accepts any strategy string - if no scorer is registered
            for it, the client will fail fast when attempting to score.
        """
        # NEW FORMAT: Check for submission_evaluation_config
        if not hasattr(task, "submission_evaluation_config") or not task.submission_evaluation_config:
            raise EvaluationConfigError(
                f"Task {task.task_id} missing required submission_evaluation_config. "
                "All tasks MUST have submission evaluation configuration."
            )

        try:
            config = EvaluationConfig.from_dict(task.submission_evaluation_config)
        except Exception as e:
            raise EvaluationConfigError(f"Invalid submission_evaluation_config for task {task.task_id}: {e}") from e

        # Log non-standard strategies for visibility (validation happens client-side)
        if config.strategy not in SUPPORTED_STRATEGIES:
            logger.info(
                "Task uses custom evaluation strategy (validated client-side)",
                extra={
                    "event": "custom_strategy_configured",
                    "task_id": task.task_id,
                    "strategy": config.strategy,
                },
            )

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

    # ============================================================================
    # CLIENT-SIDE EVALUATION: evaluate_episode() REMOVED
    # Evaluation now happens on the client. Server only stores results.
    # ============================================================================

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
        details: dict[str, Any] | None = None,
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

    async def get_trajectory(self, session_id: str) -> list[Any]:
        """Get trajectory for a session (legacy stub)."""
        return []
