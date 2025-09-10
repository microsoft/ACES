"""
Enhanced EvaluationManager implementation for SABER domain server.

This implementation provides fail-fast evaluation capabilities with no backwards compatibility.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..base import Episode
from ..benchmarks.task import Task
from .constants import EVAL_STRATEGY_LLM_JUDGE, EVAL_STRATEGY_STATIC, SUPPORTED_STRATEGIES
from .evaluators import BaseEvaluator, LLMEvaluator, StaticEvaluator
from .exceptions import (
    EvaluationConfigError,
    EvaluatorNotFoundError,
    IncompleteEpisodeError,
    InvalidEvaluationStrategyError,
    MissingSubmissionError,
)
from .models import EvaluationConfig, EvaluationResult
from .store import EvaluationStore, JsonFileEvaluationStore

logger = logging.getLogger(__name__)


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

        logger.info("EvaluationManager initialized with evaluators: %s", list(self.evaluators.keys()))

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
        logger.info(f"Configured evaluation for task {task.task_id} with strategy: {config.strategy}")

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

        # Prepare episode data for evaluation
        episode_data = {
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "submission": episode.submission,
            "executed_commands": episode.get_executed_commands(),
            "completion_reason": episode.completion_reason,
            "step_count": len(episode.steps),
        }

        logger.info(f"Evaluating episode {episode.episode_id} with strategy: {config.strategy}")
        result = await evaluator.evaluate(episode_data, config, task)
        # Persist result (fail fast on any write issues)
        if config.strategy == EVAL_STRATEGY_LLM_JUDGE:
            golden_answer = config.criteria.get("golden_answer")
        else:
            golden_answer = None
        try:
            await self.store.save(
                result, submission=episode.submission, session_id=episode.session_id, golden_answer=golden_answer
            )
        except Exception:
            # Re-raise to enforce atomic contract (no silent persistence failures)
            raise
        logger.info(
            "episode_evaluation_complete",
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

    # Legacy logging methods for compatibility with existing SessionManager
    async def log_session_start(self, session_id: str, client_id: str) -> None:
        """Log session start event."""
        logger.info(f"Session started: {session_id} for client {client_id}")

    async def log_session_end(self, session_id: str) -> None:
        """Log session end event."""
        logger.info(f"Session ended: {session_id}")

    async def log_episode_start(self, session_id: str, episode_id: str, task_id: str) -> None:
        """Log episode start event (legacy compatibility)."""
        logger.info(
            "episode_start",
            extra={
                "event": "episode_start",
                "session_id": session_id,
                "episode_id": episode_id,
                "task_id": task_id,
            },
        )

    async def log_episode_end(self, session_id: str, completion_reason: str) -> None:
        """Log episode end event."""
        logger.info(f"Episode ended in session {session_id}: {completion_reason}")

    async def log_action(self, session_id: str, episode_id: str, action: Any, result: Any) -> None:
        """Log action execution event."""
        logger.info(f"Action logged for episode {episode_id} in session {session_id}: {action.tool_name}")

    async def get_trajectory(self, session_id: str) -> list:
        """Get trajectory for a session (legacy stub)."""
        return []
