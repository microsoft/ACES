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
    MetadataKeys,
    StepEvaluationStrategy,
    SubmissionEvaluationStrategy,
)
# Core domain models
from .core import BenchmarkInfo, EvalSubmission, PolicyInfo, TaskInfo

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

# REST API models
from .rest import (
    ActionExecutionResponse,
    ActiveEpisodeInfo,
    ActiveEpisodesResponse,
    EpisodeContext,
    EpisodeCreateResponse,
    EpisodeDetailResponse,
    EpisodeEndResponse,
    EpisodeListResponse,
    EpisodeStatusResponse,
    EpisodeTaskResponse,
    HealthResponse,
    PolicyResponse,
    SessionCreateResponse,
    SessionInfo,
    SessionListResponse,
    SessionStatsResponse,
    SessionSummary,
    SessionTerminateResponse,
    StepResponse,
    TaskOrchestrationResponse,
)

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
]

# Rebuild models with forward references after all imports are complete
BenchmarkInfo.model_rebuild()
