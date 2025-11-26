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
    initial_context: Optional[Dict[str, Any]] = Field(
        None, description="Task initial context with target services and configuration"
    )


class PolicyResponse(BaseModel):
    """Response model for policy information."""

    prompt: str = Field(description="Policy prompt/guidelines")
    domain: Optional[str] = Field(None, description="Security domain name")


class EpisodeCreateResponse(BaseModel):
    """Response model for episode creation (async)."""

    episode_id: str = Field(description="Unique episode identifier")
    task_id: str = Field(description="Task identifier for this episode")
    session_id: str = Field(description="Session identifier")
    state: str = Field(description="Initial episode state (typically 'creating')")
    message: str = Field(description="Status message")
    episode_context: Optional[EpisodeContext] = Field(None, description="Episode context with limits and metadata")
    attached_to_episode_id: Optional[str] = Field(
        None, description="Episode ID this episode is attached to due to dependencies"
    )


class EpisodeStatusResponse(BaseModel):
    """Response model for episode status/readiness polling."""

    episode_id: str = Field(description="Episode identifier")
    task_id: str = Field(description="Task identifier")
    session_id: str = Field(description="Session identifier")
    state: str = Field(description="Current episode state")
    is_ready: bool = Field(description="Whether episode is ready for execution")
    message: str = Field(description="Status message")
    creation_error: Optional[str] = Field(None, description="Error message if creation failed")
    episode_context: Optional[EpisodeContext] = Field(None, description="Episode context (available when ready)")
    attached_to_episode_id: Optional[str] = Field(None, description="Attached episode ID if applicable")


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
    domain_slug: Optional[str] = Field(default=None, description="Domain slug from manifest")
    schema_version: Optional[str] = Field(default=None, description="Manifest schema version")
    capabilities: Optional[Dict[str, Any]] = Field(default=None, description="Domain capabilities from manifest")
    config_checksum: Optional[str] = Field(default=None, description="Configuration directory checksum")
    manifest_path: Optional[str] = Field(default=None, description="Path to domain manifest")
    build_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Build metadata")


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

    CLIENT-SIDE EVALUATION: evaluation_result is now optional and will be None
    when the episode ends. Clients should submit evaluation results separately
    via POST /api/v1/session/{session_id}/episodes/{episode_id}/evaluation.
    """

    episode_ended: bool = Field(description="Whether the episode has ended")
    episode_id: str = Field(description="ID of the episode that ended")
    success: bool = Field(description="Whether the episode completed successfully (based on termination reason)")
    reason: str = Field(description="Reason for episode termination")
    previous_task_id: Optional[str] = Field(None, description="Task ID of the completed episode")
    active_episodes_remaining: int = Field(description="Number of active episodes remaining in session")
    evaluation_result: Optional[Dict[str, Any]] = Field(
        None, description="Episode evaluation result (None for client-side evaluation)"
    )


# Legacy compatibility - keeping SessionInfo for backward compatibility
class SessionInfo(BaseModel):
    """Session information - legacy model, use SessionCreateResponse instead."""

    session_id: str
    message: str


# Transcript synchronization models
class ToolCall(BaseModel):
    """Tool call structure in chat messages."""

    id: str = Field(description="Tool call identifier")
    function: str = Field(description="Tool/function name")
    arguments: Dict[str, Any] = Field(description="Tool arguments")


class ChatMessage(BaseModel):
    """Structured chat message format (compatible with Inspect AI).

    This schema validates conversation messages to ensure consistency
    and prevent malformed data from breaking downstream consumers.
    """

    role: str = Field(description="Message role (user, assistant, system, tool)")
    content: str = Field(default="", description="Message text content")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls (for assistant messages)")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID (for tool response messages)")
    name: Optional[str] = Field(None, description="Tool name (for tool response messages)")


class TranscriptPushRequest(BaseModel):
    """Request model for pushing transcript to server."""

    messages: List[ChatMessage] = Field(description="List of conversation messages")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Transcript metadata (step_number, timestamp, source)"
    )


class TranscriptPushResponse(BaseModel):
    """Response model for transcript push operation."""

    success: bool = Field(description="Whether transcript was stored successfully")
    episode_id: str = Field(description="Episode identifier")
    message_count: int = Field(description="Number of messages stored")
    stored_at: str = Field(description="Timestamp when transcript was stored (ISO format)")


class TranscriptGetResponse(BaseModel):
    """Response model for transcript retrieval."""

    episode_id: str = Field(description="Episode identifier")
    messages: List[ChatMessage] = Field(description="List of conversation messages")
    message_count: int = Field(description="Number of messages in transcript")
    last_updated: Optional[str] = Field(None, description="Last update timestamp (ISO format)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Transcript metadata if available")


# Message injection models (Phase 4: Red Team message injection)
class MessageInjectRequest(BaseModel):
    """Request model for red team message injection."""

    role: str = Field(description="Message role (typically 'user' for red team injections)")
    content: str = Field(description="Message content to inject")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata (injected_by, reason, etc.)")


class MessageInjectResponse(BaseModel):
    """Response model for message injection operation."""

    success: bool = Field(description="Whether message was queued successfully")
    episode_id: str = Field(description="Episode identifier")
    injection_id: str = Field(description="Unique injection identifier")
    injected_at: str = Field(description="Timestamp when message was queued (ISO format)")


class PendingMessagesResponse(BaseModel):
    """Response model for retrieving pending injections."""

    episode_id: str = Field(description="Episode identifier")
    messages: List[ChatMessage] = Field(description="List of pending messages to inject")
    pending_count: int = Field(description="Number of pending messages")
    retrieved_at: str = Field(description="Timestamp when messages were retrieved (ISO format)")


# Import evaluation models
