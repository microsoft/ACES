"""
Evaluation result persistence layer.

Phase 1 introduces the contract and fail-fast behavior. Phase 2/3 can
swap implementation (DB, object store, etc.) without changing callers.

Failure to persist MUST raise immediately – no silent degradation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .exceptions import EvaluationNotFoundError, InvalidEvaluationRequestError, SessionEvaluationError
from .models import EvaluationResult


class EvaluationStore:
    """Abstract base for evaluation persistence (duck-typed)."""

    async def save(
        self,
        result: EvaluationResult,
        session_id: str,
        golden_answer: Optional[str] = None,
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def get(self, session_id: str, episode_id: str) -> EvaluationResult:
        """Retrieve evaluation result by session and episode."""
        raise NotImplementedError

    async def list_by_session(self, session_id: str, task_id: Optional[str] = None) -> List[EvaluationResult]:
        """List evaluation results for session, optionally filtered by task."""
        raise NotImplementedError

    async def session_exists(self, session_id: str) -> bool:
        """Check if session has any evaluation data."""
        raise NotImplementedError


class JsonFileEvaluationStore(EvaluationStore):
    """Persist each evaluation as a JSON artifact under a directory.

    Layout:
        <base_dir>/<session_id>/<task_id>/<episode_id>.json

    This is intentionally simple and human‑inspectable. A future
    implementation can introduce a DB-backed store; callers should not care.
    """

    def __init__(self, base_dir: str = "data/evaluations") -> None:
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save(
        self,
        result: EvaluationResult,
        session_id: str,
        golden_answer: Optional[str] = None,
    ) -> None:
        session_dir = self.base_path / session_id
        task_dir = session_dir / result.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = task_dir / f"{result.episode_id}.json"
        payload = {
            # Core evaluation results
            "episode_id": result.episode_id,
            "task_id": result.task_id,
            "strategy": result.strategy,
            "raw_score": result.raw_score,
            "max_score": result.max_score,
            "score": result.score,
            "success": result.success,
            "timestamp": result.timestamp.isoformat(),
            "details": result.details,
            # Enhanced submission metadata
            "submission": result.submission,
            "executed_commands": result.executed_commands,
            "completion_reason": result.completion_reason,
            "step_count": result.step_count,
            # Model and execution metadata
            "model": result.model,
            "choices": result.choices,
            "tokens": result.tokens,
            "execution_time": result.execution_time,
            # Legacy support (remove in future versions)
            "golden_answer": golden_answer,
            "schema_version": 2,  # Increment version for enhanced format
            "tool": "saber",
        }
        try:
            with artifact_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except Exception as e:  # Fail fast
            raise RuntimeError(f"Failed to persist evaluation result to {artifact_path}: {e}") from e  # noqa: E501

    async def get(self, session_id: str, episode_id: str) -> EvaluationResult:
        """
        Retrieve evaluation result by session and episode.

        Args:
            session_id: Session ID to search in
            episode_id: Episode ID to retrieve

        Returns:
            EvaluationResult object

        Raises:
            EvaluationNotFoundError: If evaluation not found
            InvalidEvaluationRequestError: If parameters are invalid
        """
        # Validate inputs fail-fast
        if not session_id or not session_id.strip():
            raise InvalidEvaluationRequestError("session_id cannot be empty")
        if not episode_id or not episode_id.strip():
            raise InvalidEvaluationRequestError("episode_id cannot be empty")

        session_dir = self.base_path / session_id
        if not session_dir.exists():
            raise EvaluationNotFoundError(f"No evaluation data found for session: {session_id}")

        # Search for episode across all tasks in the session
        for task_dir in session_dir.iterdir():
            if task_dir.is_dir():
                artifact_path = task_dir / f"{episode_id}.json"
                if artifact_path.exists():
                    try:
                        with artifact_path.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                        return self._json_to_evaluation_result(data)
                    except Exception as e:
                        raise EvaluationNotFoundError(
                            f"Failed to load evaluation result from {artifact_path}: {e}"
                        ) from e

        raise EvaluationNotFoundError(f"Evaluation not found for episode: {episode_id} in session: {session_id}")

    async def list_by_session(self, session_id: str, task_id: Optional[str] = None) -> List[EvaluationResult]:
        """
        List evaluation results for session, optionally filtered by task.

        Args:
            session_id: Session ID to search in
            task_id: Optional task ID filter

        Returns:
            List of EvaluationResult objects

        Raises:
            SessionEvaluationError: If session access fails
            InvalidEvaluationRequestError: If parameters are invalid
        """
        # Validate inputs fail-fast
        if not session_id or not session_id.strip():
            raise InvalidEvaluationRequestError("session_id cannot be empty")

        session_dir = self.base_path / session_id
        if not session_dir.exists():
            return []  # Empty session is valid - return empty list

        results = []
        try:
            # If task_id is specified, search only that task directory
            if task_id:
                task_dir = session_dir / task_id
                if task_dir.exists() and task_dir.is_dir():
                    results.extend(self._load_evaluations_from_dir(task_dir))
            else:
                # Search all task directories
                for task_dir in session_dir.iterdir():
                    if task_dir.is_dir():
                        results.extend(self._load_evaluations_from_dir(task_dir))

            return sorted(results, key=lambda r: r.timestamp)

        except Exception as e:
            raise SessionEvaluationError(f"Failed to list evaluations for session {session_id}: {e}") from e

    async def session_exists(self, session_id: str) -> bool:
        """
        Check if session has any evaluation data.

        Args:
            session_id: Session ID to check

        Returns:
            True if session has evaluation data, False otherwise
        """
        if not session_id or not session_id.strip():
            return False

        session_dir = self.base_path / session_id
        return session_dir.exists() and any(session_dir.iterdir())

    def _load_evaluations_from_dir(self, task_dir: Path) -> List[EvaluationResult]:
        """Load all evaluations from a task directory."""
        results = []
        for artifact_path in task_dir.glob("*.json"):
            try:
                with artifact_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append(self._json_to_evaluation_result(data))
            except Exception as e:
                # Log and skip corrupted files - don't fail entire listing
                # This follows fail-fast for the specific file but allows listing to continue
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Skipping corrupted evaluation file {artifact_path}: {e}")
        return results

    def _json_to_evaluation_result(self, data: dict) -> EvaluationResult:
        """Convert JSON data to EvaluationResult object."""
        from datetime import datetime

        # Validate required fields fail-fast
        required_fields = ["episode_id", "task_id", "strategy", "raw_score", "max_score", "score", "success"]
        for field in required_fields:
            if field not in data:
                raise EvaluationNotFoundError(f"Invalid evaluation data: missing required field '{field}'")

        # Parse timestamp
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError as e:
                raise EvaluationNotFoundError(f"Invalid timestamp format: {timestamp_str}") from e
        else:
            from datetime import timezone

            timestamp = datetime.now(timezone.utc)

        return EvaluationResult(
            # Core evaluation results
            episode_id=data["episode_id"],
            task_id=data["task_id"],
            strategy=data["strategy"],
            raw_score=data["raw_score"],
            max_score=data["max_score"],
            score=data["score"],
            success=data["success"],
            timestamp=timestamp,
            details=data.get("details", {}),
            # Enhanced submission metadata (with backward compatibility)
            submission=data.get("submission", ""),  # Required but provide fallback for old data
            executed_commands=data.get("executed_commands", []),
            completion_reason=data.get("completion_reason"),
            step_count=data.get("step_count", 0),
            # Model and execution metadata (with backward compatibility)
            model=data.get("model"),
            choices=data.get("choices", []),
            tokens=data.get("tokens", {}),
            execution_time=data.get("execution_time"),
        )
