"""
SABER Models - Centralized model exports.

Provides convenient imports for all SABER models organized by category.
"""

# Administrative models
from .admin import (
    ActiveCleanupInfo,
    ActiveCleanupsResponse,
    CleanupHistoryEntry,
    CleanupHistoryResponse,
    SessionCleanupHistoryResponse,
)

# Benchmark task models
from .benchmark_task import (
    BenchmarkTask,
    DependencyGraph,
    OrchestratedTask,
    OrchestrationStrategy,
    SingleEpisodeTask,
    SubTaskDefinition,
    TaskExecutionMode,
)

# Constants and enumerations
from .constants import (
    EVAL_STRATEGY_LLM_JUDGE,
    EVAL_STRATEGY_STATIC,
    EVAL_STRATEGY_TOOL_CALL,
    VALID_EVAL_STRATEGIES,
    VALID_STEP_EVAL_STRATEGIES,
    VALID_SUBMISSION_EVAL_STRATEGIES,
    EvaluationStrategy,
    MessageRole,
    MetadataKeys,
    StepEvaluationStrategy,
    SubmissionEvaluationStrategy,
)

# Core domain models (includes execution constants)
from .core import (
    BenchmarkInfo,
    CleanupReason,
    ClientIdentifiers,
    EvalSubmission,
    ExecutionMode,
    PolicyInfo,
    TaskInfo,
    TaskInitMode,
)

# HTTP headers
from .headers import HTTPHeaders

# MCP protocol models
from .mcp import (
    MCPConnectionError,
    MCPErrorResponse,
    MCPExecutionError,
    MCPInputSchema,
    MCPPropertySchema,
    MCPTimeoutError,
    MCPToolCallRequest,
    MCPToolCallResponse,
    MCPToolListResponse,
    MCPToolNotFoundError,
    MCPToolSchema,
    OrchestrationEnvironment,
    RequestHeaders,
    SessionContext,
)

# REST API models (includes endpoints and config)
from .rest import (
    ActionExecutionResponse,
    ActiveEpisodeInfo,
    ActiveEpisodesResponse,
    APIEndpoints,
    EpisodeContext,
    EpisodeCreateResponse,
    EpisodeDetailResponse,
    EpisodeEndResponse,
    EpisodeListResponse,
    EpisodeStatusResponse,
    EpisodeTaskResponse,
    HealthResponse,
    MessageInjectRequest,
    MessageInjectResponse,
    PendingMessagesResponse,
    PolicyResponse,
    SessionCreateResponse,
    SessionInfo,
    SessionListResponse,
    SessionStatsResponse,
    SessionSummary,
    SessionTerminateResponse,
    StepResponse,
    TaskOrchestrationResponse,
    TranscriptCountResponse,
    TranscriptGetResponse,
    TranscriptPushRequest,
    TranscriptPushResponse,
    TranscriptSyncConfig,
)
from .rest.websocket_constants import WebSocketCloseCode, WebSocketDefaults

__all__ = [
    # Benchmark task models
    "BenchmarkTask",
    "SingleEpisodeTask",
    "OrchestratedTask",
    "OrchestrationStrategy",
    "SubTaskDefinition",
    "TaskExecutionMode",
    "DependencyGraph",
    # Constants and enumerations
    "EvaluationStrategy",
    "SubmissionEvaluationStrategy",
    "StepEvaluationStrategy",
    "MessageRole",
    "EVAL_STRATEGY_STATIC",
    "EVAL_STRATEGY_LLM_JUDGE",
    "EVAL_STRATEGY_TOOL_CALL",
    "VALID_EVAL_STRATEGIES",
    "VALID_SUBMISSION_EVAL_STRATEGIES",
    "VALID_STEP_EVAL_STRATEGIES",
    # Core domain models
    "BenchmarkInfo",
    "PolicyInfo",
    "TaskInfo",
    "EvalSubmission",
    # REST API models
    "StepResponse",
    "SessionCreateResponse",
    "SessionTerminateResponse",
    "EpisodeContext",
    "EpisodeTaskResponse",
    "PolicyResponse",
    "EpisodeCreateResponse",
    "EpisodeStatusResponse",
    "EpisodeListResponse",
    "ActiveEpisodeInfo",
    "ActiveEpisodesResponse",
    "EpisodeDetailResponse",
    "ActionExecutionResponse",
    "TaskOrchestrationResponse",
    "HealthResponse",
    "SessionSummary",
    "SessionListResponse",
    "SessionStatsResponse",
    "EpisodeEndResponse",
    "SessionInfo",
    # Transcript synchronization models
    "TranscriptPushRequest",
    "TranscriptPushResponse",
    "TranscriptGetResponse",
    "TranscriptCountResponse",
    # Message injection models
    "MessageInjectRequest",
    "MessageInjectResponse",
    "PendingMessagesResponse",
    # MCP protocol models
    "MCPToolSchema",
    "MCPInputSchema",
    "MCPPropertySchema",
    "MCPToolListResponse",
    "MCPToolCallResponse",
    "MCPErrorResponse",
    "MCPToolCallRequest",
    "SessionContext",
    "MCPConnectionError",
    "MCPToolNotFoundError",
    "MCPExecutionError",
    "MCPTimeoutError",
    # Administrative models
    "CleanupHistoryEntry",
    "CleanupHistoryResponse",
    "SessionCleanupHistoryResponse",
    "ActiveCleanupInfo",
    "ActiveCleanupsResponse",
    # HTTP headers
    "HTTPHeaders",
    # Orchestration models
    "OrchestrationEnvironment",
    "RequestHeaders",
    # Constants
    "MetadataKeys",
    "TranscriptSyncConfig",
    "APIEndpoints",
    "ExecutionMode",
    "TaskInitMode",
    "ClientIdentifiers",
    "CleanupReason",
    "WebSocketCloseCode",
    "WebSocketDefaults",
]

# Rebuild models with forward references after all imports are complete
BenchmarkInfo.model_rebuild()
