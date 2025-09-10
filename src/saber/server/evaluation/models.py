"""
Pydantic models for evaluation system.
"""

from datetime import datetime, timezone
from typing import Any, Dict

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
    """Result of episode evaluation."""

    episode_id: str
    task_id: str
    strategy: str
    raw_score: float = Field(..., ge=0.0)
    max_score: float = Field(..., gt=0.0)
    score: float = Field(..., ge=0.0)
    success: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = Field(default_factory=dict)
