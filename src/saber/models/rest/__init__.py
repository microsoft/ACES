"""
SABER REST API Models - HTTP REST endpoint request/response models.

These models define the HTTP REST API contract between SABER clients and servers.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepResponse(BaseModel):
    """Response from server step execution."""

    success: bool = Field(description="Command succeeded")
    output: str = Field(description="Command output to show agent")
    error: Optional[str] = Field(None, description="Error message if failed")
    done: bool = Field(description="Episode complete")
    task_completed: bool = Field(default=False, description="Current task completed")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional server info")


class SessionCreateResponse(BaseModel):
    """Response model for session creation."""

    session_id: str = Field(description="Unique session identifier")
    message: str = Field(description="Success message")


class SessionTerminateResponse(BaseModel):
    """Response model for session termination."""

    message: str = Field(description="Termination confirmation message")


class EpisodeContext(BaseModel):
    """Typed episode context data."""

    session_id: str = Field(description="Session ID this episode belongs to")
    task_timeout: Optional[int] = Field(None, description="Task timeout in seconds")
    max_steps: Optional[int] = Field(None, description="Maximum allowed steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")


class EpisodeTaskResponse(BaseModel):
    """Response model for episode task information."""

    task_id: str = Field(description="Task identifier")
    title: str = Field(description="Task title")
    description: str = Field(description="Task description")
    episode_id: str = Field(description="Episode identifier")
    episode_context: EpisodeContext = Field(description="Episode context information")


class PolicyResponse(BaseModel):
    """Response model for policy information."""

    prompt: str = Field(description="Policy prompt/guidelines")
    domain: Optional[str] = Field(None, description="Security domain name")


class EpisodeCreateResponse(BaseModel):
    """Response model for episode creation."""

    episode_id: str = Field(description="Unique episode identifier")
    task_id: str = Field(description="Task identifier for this episode")
    session_id: str = Field(description="Session identifier")
    state: str = Field(description="Initial episode state")
    message: str = Field(description="Success message")
    episode_context: Optional[EpisodeContext] = Field(None, description="Episode context with limits and metadata")


class EpisodeListResponse(BaseModel):
    """Response model for listing episodes."""

    session_id: str = Field(description="Session identifier")
    active_episodes: List[str] = Field(description="List of active episode IDs")
    episode_history: List[str] = Field(description="List of completed episode IDs")
    episode_counts: Dict[str, int] = Field(description="Episode count statistics")
    task_queue: List[str] = Field(description="Queued tasks for orchestration")


class ActiveEpisodeInfo(BaseModel):
    """Information about an active episode."""

    episode_id: str = Field(description="Episode identifier")
    task_id: str = Field(description="Task identifier")
    state: str = Field(description="Current episode state")
    step_count: int = Field(description="Number of steps taken")
    start_time: str = Field(description="Episode start time (ISO format)")
    duration: Optional[float] = Field(None, description="Episode duration in seconds")


class ActiveEpisodesResponse(BaseModel):
    """Response model for listing active episodes."""

    session_id: str = Field(description="Session identifier")
    active_episodes: List[ActiveEpisodeInfo] = Field(description="List of active episode details")
    count: int = Field(description="Number of active episodes")


class EpisodeDetailResponse(BaseModel):
    """Response model for detailed episode information."""

    episode_id: str = Field(description="Episode identifier")
    task_id: str = Field(description="Task identifier")
    session_id: str = Field(description="Session identifier")
    state: str = Field(description="Current episode state")
    step_count: int = Field(description="Number of steps taken")
    start_time: str = Field(description="Episode start time (ISO format)")
    end_time: Optional[str] = Field(None, description="Episode end time (ISO format)")
    duration: Optional[float] = Field(None, description="Episode duration in seconds")
    completion_reason: Optional[str] = Field(None, description="Reason for completion")
    context: Dict[str, Any] = Field(description="Episode context data")
    metadata: Dict[str, Any] = Field(description="Episode metadata")


class ActionExecutionResponse(BaseModel):
    """Response model for action execution."""

    success: bool = Field(description="Whether the action succeeded")
    data: Any = Field(description="Action result data")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class TaskOrchestrationResponse(BaseModel):
    """Response model for task orchestration."""

    session_id: str = Field(description="Session identifier")
    queued_tasks: List[str] = Field(description="List of queued task IDs")
    message: str = Field(description="Success message")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(description="Health status")
    domain: str = Field(description="Domain name")


class SessionSummary(BaseModel):
    """Summary information about a session."""

    session_id: str = Field(description="Session identifier")
    client_id: str = Field(description="Client identifier")
    created_at: str = Field(description="Session creation time (ISO format)")
    last_activity: str = Field(description="Last activity timestamp (ISO format)")
    is_active: bool = Field(description="Whether session is active")
    active_episodes: int = Field(description="Number of active episodes")
    total_episodes: int = Field(description="Total episodes in session")


class SessionListResponse(BaseModel):
    """Response model for listing sessions."""

    sessions: List[SessionSummary] = Field(description="List of session summaries")
    total_count: int = Field(description="Total number of sessions")
    active_count: int = Field(description="Number of active sessions")


class SessionStatsResponse(BaseModel):
    """Response model for session statistics."""

    total_sessions: int = Field(description="Total number of sessions")
    active_sessions: int = Field(description="Number of active sessions")
    total_episodes: int = Field(description="Total number of episodes")
    active_episodes: int = Field(description="Number of active episodes")
    average_session_duration: Optional[float] = Field(None, description="Average session duration")
    oldest_session_age: Optional[float] = Field(None, description="Age of oldest session in hours")


class EpisodeEndResponse(BaseModel):
    """Response model for episode termination.

    Phase 1 guarantee: evaluation_result is ALWAYS present (never None)
    and contains the serialized EvaluationResult (dict form) produced
    during fail-fast evaluation. This is a breaking change vs legacy.
    """

    episode_ended: bool = Field(description="Whether the episode has ended")
    episode_id: str = Field(description="ID of the episode that ended")
    success: bool = Field(description="Whether the episode completed successfully")
    reason: str = Field(description="Reason for episode termination")
    previous_task_id: Optional[str] = Field(None, description="Task ID of the completed episode")
    active_episodes_remaining: int = Field(description="Number of active episodes remaining in session")
    evaluation_result: Dict[str, Any] = Field(description="Episode evaluation result (non-null)")


# Legacy compatibility - keeping SessionInfo for backward compatibility
class SessionInfo(BaseModel):
    """Session information - legacy model, use SessionCreateResponse instead."""

    session_id: str
    message: str


# Import evaluation models
