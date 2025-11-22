"""
SABER Evaluation REST API Models.

Response models for evaluation retrieval endpoints.
Follows fail-fast principles with no backwards compatibility.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

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
    # Add subtasks for step-level evaluation
    subtasks: List[Dict[str, Any]] = Field(default_factory=list, description="Subtask definitions for step evaluation")


class JudgeMessages(BaseModel):
    """Pre-rendered judge messages for LLM evaluation."""

    system_message: str = Field(description="Fully rendered system prompt")
    user_message: Union[str, List[str]] = Field(
        description="Fully rendered user prompt(s) - can be single string or list for chunked evaluation"
    )
    model: str = Field(description="Model to use for evaluation")


class EvaluationCriteriaResponse(BaseModel):
    """Complete evaluation criteria package for client-side evaluation."""

    # Core identifiers
    session_id: str = Field(description="Session identifier")
    episode_id: str = Field(description="Episode identifier")
    task_id: str = Field(description="Task identifier")

    # Authoritative submission to evaluate (None for incomplete episodes)
    submission: Optional[str] = Field(
        None, description="Exact submission content to evaluate (from server, None for incomplete episodes)"
    )

    # Task context for evaluation
    task_context: TaskEvaluationContext = Field(description="Task information for evaluation")

    # Evaluation configuration (strategy, criteria, scoring)
    evaluation_config: Dict[str, Any] = Field(description="Complete evaluation configuration")

    # Pre-rendered judge messages (for LLM evaluation only)
    judge_messages: Optional[JudgeMessages] = Field(None, description="Pre-rendered messages for LLM evaluation")


class StepEvaluation(BaseModel):
    """Step-level evaluation result mapping step to completed objective."""

    step_number: int = Field(..., ge=0, description="Episode step number (0-indexed)")
    objective_id: str = Field(..., description="Subtask ID or task ID that was completed")
    objective_type: str = Field(..., description="Type: 'subtask' or 'task'")
    completed: bool = Field(
        default=True, description="Whether objective was completed (always True for parsed results)"
    )


class StepEvaluationResult(BaseModel):
    """Enhanced evaluation result with step-level analysis."""

    # Standard evaluation fields
    episode_id: str = Field(description="Episode identifier")
    task_id: str = Field(description="Task identifier")
    submission: str = Field(description="Agent's final submission")
    score: float = Field(..., ge=0.0, description="Overall evaluation score")
    max_score: float = Field(..., gt=0.0, description="Maximum possible score")
    success: bool = Field(description="Whether main task was completed")

    # Step-level evaluation results
    step_evaluations: List[StepEvaluation] = Field(
        default_factory=list, description="Step-by-step objective completion analysis"
    )
    task_completed_at_step: Optional[int] = Field(
        None, description="Step number where main task was completed (if any)"
    )
    subtasks_completed: List[str] = Field(default_factory=list, description="List of subtask IDs that were completed")

    # Evaluation metadata
    strategy: str = Field(description="Evaluation strategy used")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional evaluation details")


class EvaluationOverrideRequest(BaseModel):
    """Request model for overriding evaluation results."""

    evaluation_data: Dict[str, Any] = Field(description="Episode evaluation data as EpisodeEvaluationData dict")
    strategy: str = Field(..., min_length=1, description="Evaluation strategy used for this result")
    raw_score: float = Field(..., ge=0.0, description="Raw evaluation score")
    max_score: float = Field(..., gt=0.0, description="Maximum possible score")
    score: float = Field(..., ge=0.0, description="Normalized score (0.0 to max_score)")
    success: bool = Field(description="Whether the evaluation was successful")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional evaluation details")

    def __init__(self, **data: Any) -> None:
        """Initialize with validation of score bounds."""
        super().__init__(**data)
        if self.score > self.max_score:
            raise ValueError(f"score ({self.score}) cannot exceed max_score ({self.max_score})")


class EvaluationOverrideResponse(BaseModel):
    """Response model for override evaluation endpoint."""

    message: str = Field(description="Success message")
    evaluation_result: EvaluationResultResponse = Field(description="The overridden evaluation result")
    session_id: str = Field(description="Session ID")
    episode_id: str = Field(description="Episode ID")


class EvaluationFileUploadResponse(BaseModel):
    """Response model for evaluation file upload endpoint."""

    message: str = Field(description="Success message")
    session_id: str = Field(description="Session ID")
    filename: str = Field(description="Uploaded filename")
    file_size: int = Field(description="File size in bytes")
    upload_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Upload timestamp"
    )


# ============================================================================
# NEW CLIENT-SIDE EVALUATION MODELS (Breaking Change Migration)
# ============================================================================


class EpisodeStepData(BaseModel):
    """Single episode step data for client-side evaluation."""

    step_number: int = Field(..., ge=0, description="Step number (0-indexed)")
    tool_name: str = Field(description="Name of the tool executed")
    tool_input: Dict[str, Any] = Field(description="Input parameters to the tool")
    tool_output: str = Field(description="Tool execution output")
    timestamp: datetime = Field(description="When the step was executed")
    assistant_message: Optional[str] = Field(None, description="Assistant message before tool call")
    reasoning: Optional[str] = Field(None, description="Assistant reasoning (if available)")


class EpisodeSubmissionResponse(BaseModel):
    """Episode submission data - client fetches this to evaluate."""

    session_id: str = Field(description="Session identifier")
    episode_id: str = Field(description="Episode identifier")
    task_id: str = Field(description="Task identifier")
    submission: str = Field(description="Agent's final submission content")
    model: Optional[str] = Field(None, description="Model used for the episode")
    tokens: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    execution_time: Optional[float] = Field(None, description="Episode execution time in seconds")


class EpisodeStepsResponse(BaseModel):
    """Episode step history - client fetches this for step-level evaluation."""

    session_id: str = Field(description="Session identifier")
    episode_id: str = Field(description="Episode identifier")
    task_id: str = Field(description="Task identifier")
    steps: List[EpisodeStepData] = Field(description="List of episode steps")
    total_steps: int = Field(description="Total number of steps")


class SubmissionEvaluationCriteriaResponse(BaseModel):
    """Submission evaluation criteria - contains everything needed for client-side evaluation."""

    session_id: str = Field(description="Session identifier")
    episode_id: str = Field(description="Episode identifier")
    task_id: str = Field(description="Task identifier")
    strategy: str = Field(
        description="Submission evaluation strategy (see SubmissionEvaluationStrategy enum): 'static' or 'llm_judge'"
    )
    criteria: Dict[str, Any] = Field(
        description="Criteria dict with template CONTENT, golden_answer, model - everything needed for evaluation"
    )
    scoring: Dict[str, float] = Field(description="Scoring configuration (e.g., max_score)")
    task_context: TaskEvaluationContext = Field(description="Task context for evaluation")


class SubtaskEvaluationCriteriaResponse(BaseModel):
    """Step evaluation criteria - contains everything needed for client-side evaluation."""

    session_id: str = Field(description="Session identifier")
    episode_id: str = Field(description="Episode identifier")
    task_id: str = Field(description="Task identifier")
    subtask_id: str = Field(description="Subtask identifier")
    strategy: str = Field(
        description="Step evaluation strategy (see StepEvaluationStrategy enum): 'static', 'llm_judge', or 'tool_call'"
    )
    criteria: Dict[str, Any] = Field(
        description="Criteria dict with template CONTENT, model, steps_per_message - everything needed for evaluation"
    )
    max_score: float = Field(description="Subtask maximum score for scoring")
    weight: float = Field(description="Subtask weight for scoring")
    objective: str = Field(description="Subtask objective description")
    title: str = Field(description="Subtask title")
    description: str = Field(description="Subtask description")
    # subtasks: List[Dict[str, Any]] = Field(description="Subtask definitions with max_score")
    task_context: TaskEvaluationContext = Field(description="Task context for evaluation")


class TemplateContentResponse(BaseModel):
    """Raw template content - client fetches and renders templates."""

    template_path: str = Field(description="Relative template path (e.g., 'judge/submission/system.md')")
    content: str = Field(description="Raw Jinja2 template content (unrendered)")


class EvaluationResultSubmission(BaseModel):
    """Client submits evaluation result after performing client-side evaluation."""

    strategy: str = Field(description="Evaluation strategy used")
    raw_score: float = Field(..., ge=0.0, description="Raw evaluation score")
    max_score: float = Field(..., gt=0.0, description="Maximum possible score")
    score: float = Field(..., ge=0.0, description="Final score")
    success: bool = Field(description="Whether evaluation was successful")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Evaluation details (submission_score, step_score, etc.)"
    )
