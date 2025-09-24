"""
LLM-based evaluator implementation using OpenAI for binary assessment.

Logging Category: EVALUATION

This evaluator uses an LLM to compare agent submissions against golden answers
and provides binary scoring (0 or 1) based on correctness assessment.
"""

import asyncio
import os
from typing import Any, Optional, TypedDict

from openai import OpenAI

from saber.logging_config import LogCategory, get_saber_logger

from ...base import Episode
from ...benchmarks.task import Task
from ..constants import EVAL_STRATEGY_LLM_JUDGE
from ..exceptions import EvaluationError
from ..models import EpisodeEvaluationData, EvaluationConfig, EvaluationResult
from .base import BaseEvaluator

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


class _LLMJudgeResponse(TypedDict):
    """Typed schema for the expected LLM judge response."""

    analysis: str
    is_correct: bool


class LLMEvaluator(BaseEvaluator):
    """
    LLM-based evaluator that uses OpenAI to assess agent submissions.

    This evaluator provides binary scoring (0 or 1) based on LLM judgment
    comparing the submission against a golden answer.
    """

    def __init__(self, config_dir: Optional[str] = None, timeout_seconds: int = 15):
        """
        Initialize the LLM evaluator.

        Args:
            config_dir: Optional configuration directory (unused for now)
            timeout_seconds: Timeout for LLM API calls (default: 15 seconds)
        """
        self.config_dir = config_dir
        self.timeout_seconds = timeout_seconds
        self.llm_client: Optional[OpenAI] = None
        logger.info(
            "LLM evaluator initialized",
            extra={
                "event": "llm_evaluator_initialized",
                "timeout_seconds": timeout_seconds,
                "config_dir": config_dir,
            },
        )

    async def evaluate(
        self,
        episode_data: EpisodeEvaluationData,
        config: EvaluationConfig,
        task: Task,
        episode: Optional["Episode"] = None,
    ) -> EvaluationResult:
        """
        Stubbed LLM evaluator - returns default success result.

        TODO: Implement actual LLM evaluation logic.

        Args:
            episode_data: EpisodeEvaluationData containing episode information including submission
            config: Evaluation configuration containing golden answer and model
            task: Task being evaluated
            episode: Optional full episode object for enhanced judge prompt context

        Returns:
            Stubbed EvaluationResult with default success (score=1.0)
        """
        logger.info(
            "Stubbed LLM evaluation executed",
            extra={
                "event": "llm_evaluation_stubbed",
                "episode_id": episode_data.episode_id,
                "task_id": episode_data.task_id,
                "strategy": EVAL_STRATEGY_LLM_JUDGE,
            },
        )

        max_score = float(config.scoring.get("max_score", 1.0))

        return EvaluationResult.from_episode_data(
            episode_data=episode_data,
            strategy=EVAL_STRATEGY_LLM_JUDGE,
            raw_score=1.0,  # Stubbed success
            max_score=max_score,
            score=max_score,  # Full score
            success=True,  # Stubbed success
            details={
                "stubbed": True,
                "message": "LLM evaluator is stubbed - returning default success",
                "submission": episode_data.submission,
            },
        )

    async def _call_llm_json(self, judge_payload: Any) -> str:
        """
        Make an LLM API call with timeout and return the response text.

        Args:
            judge_payload: JudgePromptPayload with complete messages and model configuration

        Returns:
            Raw response text from the LLM

        Raises:
            asyncio.TimeoutError: If call times out
            Exception: If API call fails
        """

        def _sync_call() -> str:
            """Execute the blocking client call in a thread to honor async timeout."""
            if self.llm_client is None:
                raise EvaluationError("LLM client not initialized")
            try:
                response = self.llm_client.chat.completions.create(**judge_payload.to_dict())
                if not response.choices:
                    raise EvaluationError("LLM response contained no choices")
                content = response.choices[0].message.content
                if content is None:
                    raise EvaluationError("LLM response message content is empty")
                return str(content)  # Explicit str conversion to satisfy mypy
            except Exception as e:  # Log & re-raise; upstream wraps context
                logger.error(
                    "LLM API call failed",
                    extra={
                        "event": "llm_api_call_failed",
                        "model": judge_payload.model,
                        "error": str(e),
                    },
                )
                raise

        return await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=self.timeout_seconds)

    def _initialize_llm_client(self) -> None:
        """
        Initialize the OpenAI client.

        Raises:
            EvaluationError: If OPENAI_API_KEY environment variable is not set
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "OpenAI API key missing",
                extra={
                    "event": "llm_api_key_missing",
                },
            )
            raise EvaluationError("OPENAI_API_KEY environment variable not set. LLM evaluation cannot proceed.")

        try:
            self.llm_client = OpenAI(api_key=api_key)
            logger.info(
                "OpenAI client initialized",
                extra={
                    "event": "llm_openai_client_initialized",
                },
            )
        except Exception as e:
            logger.error(
                "Failed to initialize OpenAI client",
                extra={
                    "event": "llm_openai_client_initialization_failed",
                    "error": str(e),
                },
            )
            raise EvaluationError(f"Failed to initialize OpenAI client: {e}") from e
