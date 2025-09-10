"""
Pydantic models for evaluation system.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluationConfig(BaseModel):
    """Configuration for episode evaluation."""

    strategy: str = Field(..., description="Evaluation strategy: static or llm_judge")
    criteria: Dict[str, Any] = Field(..., description="Strategy-specific evaluation criteria")
    scoring: Dict[str, Any] = Field(default_factory=dict, description="Scoring configuration")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        """Create config from task YAML data."""
        return cls(**data)


class EvaluationResult(BaseModel):
    """Result of episode evaluation with enhanced metadata."""

    # Core evaluation results
    episode_id: str
    task_id: str
    strategy: str
    raw_score: float = Field(..., ge=0.0)
    max_score: float = Field(..., gt=0.0)
    score: float = Field(..., ge=0.0)
    success: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = Field(default_factory=dict)

    # Enhanced submission metadata (from EpisodeEvaluationData)
    submission: str = Field(..., description="Agent submission that was evaluated")
    executed_commands: List[str] = Field(default_factory=list, description="Commands executed during episode")
    completion_reason: Optional[str] = Field(None, description="Reason episode completed")
    step_count: int = Field(..., ge=0, description="Number of steps in episode")

    # Model and execution metadata
    model: Optional[str] = Field(None, description="Model name used for the submission")
    choices: List[Dict[str, Any]] = Field(default_factory=list, description="Model response choices")
    tokens: Dict[str, Any] = Field(default_factory=dict, description="Token usage information")
    execution_time: Optional[float] = Field(None, ge=0.0, description="Model execution time in seconds")

    @classmethod
    def from_episode_data(
        cls,
        episode_data: "EpisodeEvaluationData",
        strategy: str,
        raw_score: float,
        max_score: float,
        score: float,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> "EvaluationResult":
        """Create EvaluationResult from EpisodeEvaluationData with evaluation results."""
        return cls(
            episode_id=episode_data.episode_id,
            task_id=episode_data.task_id,
            strategy=strategy,
            raw_score=raw_score,
            max_score=max_score,
            score=score,
            success=success,
            details=details or {},
            submission=episode_data.submission,
            executed_commands=episode_data.executed_commands,
            completion_reason=episode_data.completion_reason,
            step_count=episode_data.step_count,
            model=episode_data.model,
            choices=episode_data.choices,
            tokens=episode_data.tokens,
            execution_time=episode_data.execution_time,
        )


class EpisodeEvaluationData(BaseModel):
    """Strongly typed data for episode evaluation."""

    # Core episode information
    episode_id: str = Field(..., description="Episode identifier")
    task_id: str = Field(..., description="Task identifier")
    submission: str = Field(..., description="Agent submission for evaluation")
    executed_commands: List[str] = Field(default_factory=list, description="Commands executed during episode")
    completion_reason: Optional[str] = Field(None, description="Reason episode completed")
    step_count: int = Field(..., ge=0, description="Number of steps in episode")

    # Enhanced ModelOutput data (from EvalSubmission)
    model: Optional[str] = Field(None, description="Model name used")
    choices: List[Dict[str, Any]] = Field(default_factory=list, description="Model response choices")
    tokens: Dict[str, Any] = Field(default_factory=dict, description="Token usage information")
    execution_time: Optional[float] = Field(None, ge=0.0, description="Model execution time in seconds")

    @classmethod
    def from_episode_dict(cls, episode_data: Dict[str, Any]) -> "EpisodeEvaluationData":
        """Create from episode dictionary data."""
        return cls(**episode_data)
