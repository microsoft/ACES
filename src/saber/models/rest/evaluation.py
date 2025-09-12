"""
SABER Evaluation REST API Models.

Response models for evaluation retrieval endpoints.
Follows fail-fast principles with no backwards compatibility.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluationResultResponse(BaseModel):
    """Evaluation result data for API responses."""

    episode_id: str
    task_id: str
    strategy: str
    raw_score: float = Field(..., ge=0.0)
    max_score: float = Field(..., gt=0.0)
    score: float = Field(..., ge=0.0)
    success: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResponse(BaseModel):
    """Single evaluation result response."""

    evaluation_result: EvaluationResultResponse = Field(description="The evaluation result")
    session_id: str = Field(description="Session ID for the evaluation")


class EvaluationListResponse(BaseModel):
    """List of evaluation results response."""

    evaluations: List[EvaluationResultResponse] = Field(description="List of evaluation results")
    total_count: int = Field(description="Total number of evaluations returned")
    session_id: str = Field(description="Session ID for the evaluations")
    task_filter: Optional[str] = Field(None, description="Task ID filter applied, if any")


class EvaluationSummaryResponse(BaseModel):
    """Session evaluation summary response."""

    session_id: str = Field(description="Session ID")
    total_episodes: int = Field(description="Total number of episodes evaluated")
    successful_episodes: int = Field(description="Number of successful episodes")
    average_score: float = Field(description="Average score across all episodes")
    task_summaries: Dict[str, Dict[str, Any]] = Field(description="Per-task summary statistics")


class EvaluationErrorResponse(BaseModel):
    """Error response for evaluation endpoints."""

    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    session_id: Optional[str] = Field(None, description="Session ID if available")
    episode_id: Optional[str] = Field(None, description="Episode ID if available")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class TaskEvaluationContext(BaseModel):
    """Task context needed for evaluation."""

    task_id: str = Field(description="Task identifier")
    title: str = Field(description="Task title")
    description: str = Field(description="Task description (used as 'question' in LLM evaluation)")
    domain: str = Field(description="Security domain")


class JudgeMessages(BaseModel):
    """Pre-rendered judge messages for LLM evaluation."""

    system_message: str = Field(description="Fully rendered system prompt")
    user_message: str = Field(description="Fully rendered user prompt")
    model: str = Field(description="Model to use for evaluation")


class EvaluationCriteriaResponse(BaseModel):
    """Complete evaluation criteria package for client-side evaluation."""

    # Core identifiers
    session_id: str = Field(description="Session identifier")
    episode_id: str = Field(description="Episode identifier")
    task_id: str = Field(description="Task identifier")

    # Authoritative submission to evaluate
    submission: str = Field(description="Exact submission content to evaluate (from server)")

    # Task context for evaluation
    task_context: TaskEvaluationContext = Field(description="Task information for evaluation")

    # Evaluation configuration (strategy, criteria, scoring)
    evaluation_config: Dict[str, Any] = Field(description="Complete evaluation configuration")

    # Pre-rendered judge messages (for LLM evaluation only)
    judge_messages: Optional[JudgeMessages] = Field(None, description="Pre-rendered messages for LLM evaluation")
