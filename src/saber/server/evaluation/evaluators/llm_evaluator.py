"""
LLM-based evaluator implementation using OpenAI for binary assessment.

This evaluator uses an LLM to compare agent submissions against golden answers
and provides binary scoring (0 or 1) based on correctness assessment.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional, TypedDict

from openai import OpenAI

from ...benchmarks.task import Task
from ..constants import EVAL_STRATEGY_LLM_JUDGE
from ..exceptions import EvaluationConfigError, EvaluationError
from ..models import EpisodeEvaluationData, EvaluationConfig, EvaluationResult
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


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
        logger.info("LLMEvaluator initialized with %d second timeout", timeout_seconds)

    async def evaluate(
        self, episode_data: EpisodeEvaluationData, config: EvaluationConfig, task: Task
    ) -> EvaluationResult:
        """
        Evaluate episode submission using LLM judgment.

        Args:
            episode_data: EpisodeEvaluationData containing episode information including submission
            config: Evaluation configuration containing golden answer and model
            task: Task being evaluated

        Returns:
            EvaluationResult with binary score (0 or 1)

        Raises:
            EvaluationError: If LLM evaluation fails
            EvaluationConfigError: If configuration is invalid
        """
        submission = episode_data.submission.strip()
        golden_answer = config.criteria.get("golden_answer", "").strip()
        model = config.criteria.get("model")

        # Validate required configuration first (before initializing client)
        if not golden_answer:
            raise EvaluationConfigError("LLM evaluation requires golden_answer in criteria")
        if not model:
            raise EvaluationConfigError("LLM evaluation requires model in criteria")

        # Initialize LLM client if not already done
        if not self.llm_client:
            self._initialize_llm_client()

        episode_id = episode_data.episode_id
        task_id = episode_data.task_id

        # Validate required fields
        if not episode_id:
            raise EvaluationError("episode_id is required in episode_data")
        if not task_id:
            raise EvaluationError("task_id is required in episode_data")
        logger.info(
            "llm_evaluation_start",
            extra={
                "event": "llm_evaluation_start",
                "strategy": EVAL_STRATEGY_LLM_JUDGE,
                "episode_id": episode_id,
                "task_id": task_id,
                "model": model,
                "timeout_seconds": self.timeout_seconds,
            },
        )

        # Prepare prompt payload
        prompt_payload = {
            "question": task.description,
            "golden_answer": golden_answer,
            "submission": submission,
            "instructions": (
                "Determine if the submission contains the essential information from the golden answer. "
                "Respond strictly as JSON with fields: analysis (string), is_correct (boolean)."
            ),
        }

        start = time.perf_counter()
        try:
            response_text = await self._call_llm_json(model, prompt_payload)
        except asyncio.TimeoutError as e:
            raise EvaluationError(
                f"LLM evaluation timed out after {self.timeout_seconds} seconds for episode {episode_id}"
            ) from e
        except Exception as e:  # Re-wrap with contextual message
            raise EvaluationError(f"LLM evaluation failed for episode {episode_id}: {e}") from e
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Parse & validate JSON strictly
        try:
            data_raw = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise EvaluationError(
                "Malformed LLM judge response: invalid JSON. Expected keys 'analysis', 'is_correct'. "
                f"Got: {response_text}"
            ) from e

        if not isinstance(data_raw, dict):
            raise EvaluationError("Malformed LLM judge response: top-level JSON must be an object")

        if "analysis" not in data_raw or "is_correct" not in data_raw:
            raise EvaluationError("Malformed LLM judge response: missing required keys 'analysis' and/or 'is_correct'")

        analysis_val = data_raw["analysis"]
        is_correct_val = data_raw["is_correct"]
        if not isinstance(analysis_val, str):
            raise EvaluationError("Malformed LLM judge response: 'analysis' must be a string")
        if not isinstance(is_correct_val, bool):
            raise EvaluationError("Malformed LLM judge response: 'is_correct' must be a boolean")

        analysis = analysis_val.strip()
        if not analysis:
            analysis = "(empty analysis)"  # keep explicit marker
        # Enforce a soft upper bound to avoid runaway verbosity
        if len(analysis) > 5000:
            analysis = analysis[:5000] + "...<truncated>"

        is_correct = is_correct_val

        max_score = float(config.scoring.get("max_score", 1.0))
        raw_score = 1.0 if is_correct else 0.0
        score = raw_score * max_score

        logger.info(
            "llm_evaluation_complete",
            extra={
                "event": "llm_evaluation_complete",
                "strategy": EVAL_STRATEGY_LLM_JUDGE,
                "episode_id": episode_id,
                "task_id": task_id,
                "model": model,
                "is_correct": is_correct,
                "raw_score": raw_score,
                "score": score,
                "max_score": max_score,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

        return EvaluationResult.from_episode_data(
            episode_data=episode_data,
            strategy=EVAL_STRATEGY_LLM_JUDGE,
            raw_score=raw_score,
            max_score=max_score,
            score=score,
            success=is_correct,
            details={
                "golden_answer": golden_answer,
                "submission": submission,
                "analysis": analysis,
                "model": model,
                "llm_response": response_text,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

    async def _call_llm_json(self, model: str, payload: Dict[str, Any]) -> str:
        """
        Make an LLM API call with timeout and return the response text.

        Args:
            model: OpenAI model name
            payload: Payload to send to the LLM

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
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a deterministic JSON judge. Respond only with valid JSON. "
                                'Return strictly: {"analysis": <string>, "is_correct": <true|false>}'
                            ),
                        },
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    temperature=0.0,
                    max_tokens=500,  # conservative limit
                )
                if not response.choices:
                    raise EvaluationError("LLM response contained no choices")
                content = response.choices[0].message.content
                if content is None:
                    raise EvaluationError("LLM response message content is empty")
                return str(content)  # Explicit str conversion to satisfy mypy
            except Exception as e:  # Log & re-raise; upstream wraps context
                logger.error("llm_api_call_failed", extra={"error": str(e), "model": model})
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
            raise EvaluationError("OPENAI_API_KEY environment variable not set. LLM evaluation cannot proceed.")

        try:
            self.llm_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            raise EvaluationError(f"Failed to initialize OpenAI client: {e}") from e
