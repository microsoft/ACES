"""Constants for SABER models and metadata.

This module provides centralized constants and enumerations that are used across
multiple components of the SABER system to ensure consistency and avoid magic strings.
"""

from enum import Enum


class SubmissionEvaluationStrategy(str, Enum):
    """
    Enumeration of supported evaluation strategies for main task submissions.

    These strategies are available for evaluating final task submissions:
    - STATIC: Pattern matching against expected answers
    - LLM_JUDGE: LLM-based evaluation using judge templates
    """

    STATIC = "static"
    """Static evaluation using pattern matching against expected answers"""

    LLM_JUDGE = "llm_judge"
    """LLM-based evaluation using judge templates and models"""

    def __str__(self) -> str:
        """Return the enum value as string for logging and serialization."""
        return self.value


class StepEvaluationStrategy(str, Enum):
    """
    Enumeration of supported evaluation strategies for subtask step evaluation.

    These strategies are available for evaluating individual steps/subtasks:
    - STATIC: Pattern matching against expected outputs in step results
    - LLM_JUDGE: LLM-based evaluation using judge templates
    - TOOL_CALL: Evaluation based on tool call analysis and execution patterns
    """

    STATIC = "static"
    """Static evaluation using pattern matching against expected step outputs"""

    LLM_JUDGE = "llm_judge"
    """LLM-based evaluation using judge templates and models"""

    TOOL_CALL = "tool_call"
    """Evaluation based on tool call analysis and execution (subtasks only)"""

    def __str__(self) -> str:
        """Return the enum value as string for logging and serialization."""
        return self.value


# Legacy unified enum for backwards compatibility
class EvaluationStrategy(str, Enum):
    """
    Legacy unified evaluation strategy enum.

    DEPRECATED: Use SubmissionEvaluationStrategy or StepEvaluationStrategy instead.
    Kept for backwards compatibility only.
    """

    STATIC = "static"
    LLM_JUDGE = "llm_judge"
    TOOL_CALL = "tool_call"

    def __str__(self) -> str:
        return self.value


# Backwards compatibility - provide the old constant names
EVAL_STRATEGY_STATIC = SubmissionEvaluationStrategy.STATIC
EVAL_STRATEGY_LLM_JUDGE = SubmissionEvaluationStrategy.LLM_JUDGE
EVAL_STRATEGY_TOOL_CALL = StepEvaluationStrategy.TOOL_CALL

# Valid strategies for different evaluation types
VALID_SUBMISSION_EVAL_STRATEGIES = [strategy.value for strategy in SubmissionEvaluationStrategy]
VALID_STEP_EVAL_STRATEGIES = [strategy.value for strategy in StepEvaluationStrategy]
VALID_EVAL_STRATEGIES = VALID_STEP_EVAL_STRATEGIES  # Legacy compatibility


class MetadataKeys(str, Enum):
    """Standard metadata keys used in sample metadata dictionaries.

    These keys are used when converting benchmark tasks to Inspect AI samples
    and when deserializing tasks from sample metadata.
    """

    # Core identifiers
    TASK_ID = "task_id"
    SAMPLE_ID = "sample_id"
    SESSION_ID = "session_id"
    EPISODE_ID = "episode_id"

    # Task execution metadata
    EXECUTION_MODE = "execution_mode"
    BENCHMARK_TASK = "benchmark_task"
    TASK_TYPE = "task_type"

    # Orchestration metadata
    ORCHESTRATION_ID = "orchestration_id"
    ORCHESTRATION_STRATEGY = "orchestration_strategy"
    SUB_TASK_ROLE = "sub_task_role"
    DEPENDS_ON_ROLE = "depends_on_role"
    ORDER = "order"

    # SABER-specific metadata (for eval-retry)
    SABER_SESSION_ID = "saber_session_id"
    SABER_EPISODE_ID = "saber_episode_id"
    SABER_DOMAIN_SLUG = "saber_domain_slug"

    # Task configuration
    ATTEMPT = "attempt"
    TOTAL_ATTEMPTS = "total_attempts"
    TOOL_CALL_LIMIT = "tool_call_limit"
    INSTRUCTION_PROMPT = "instruction_prompt"
    ASSISTANT_PROMPT = "assistant_prompt"
    SUBMIT_PROMPT = "submit_prompt"

    # Score metadata (used in score.metadata dictionaries)
    SUBMISSION_SCORE = "submission_score"
    SUBTASK_SCORE = "subtask_score"
    WEIGHTED_SUBTASK_SCORE = "weighted_subtask_score"
    STEP_EVALUATIONS = "step_evaluations"
    SUBTASK_SCORES = "subtask_scores"

    # Transcript synchronization context keys
    CLIENT_TRANSCRIPT = "_client_transcript"
    TRANSCRIPT_UPDATED_AT = "_transcript_updated_at"
    TRANSCRIPT_METADATA = "_transcript_metadata"

    # Message injection context keys (Phase 4: Red team message injection)
    PENDING_INJECTIONS = "_pending_injections"
    INJECTION_HISTORY = "_injection_history"

    # Transcript metadata keys
    TRANSCRIPT_STEP_NUMBER = "step_number"
    TRANSCRIPT_TIMESTAMP = "timestamp"
    TRANSCRIPT_SOURCE = "source"


# Transcript synchronization configuration constants
class TranscriptSyncConfig:
    """Configuration constants for transcript synchronization between client and server."""

    # Payload size limits
    MAX_PAYLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

    # Network timeouts
    REQUEST_TIMEOUT_SECONDS = 5.0

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAYS_SECONDS = [0.5, 1.0, 2.0]

    # Default transcript source identifier
    DEFAULT_SOURCE = "inspect_ai"


# API endpoint paths
class APIEndpoints:
    """REST API endpoint path constants."""

    # Session endpoints
    SESSION = "/api/v1/session"
    SESSION_BY_ID = "/api/v1/session/{session_id}"

    # Episode endpoints
    EPISODES = "/api/v1/session/{session_id}/episodes"
    EPISODE_BY_ID = "/api/v1/session/{session_id}/episodes/{episode_id}"
    EPISODE_STATUS = "/api/v1/session/{session_id}/episodes/{episode_id}/status"
    EPISODE_TRANSCRIPT = "/api/v1/session/{session_id}/episodes/{episode_id}/transcript"
    EPISODE_SUBMISSION = "/api/v1/session/{session_id}/episodes/{episode_id}/submission"
    EPISODE_MESSAGES_INJECT = "/api/v1/session/{session_id}/episodes/{episode_id}/messages/inject"

    # Health endpoint
    HEALTH = "/api/v1/health"


# Execution mode constants
class ExecutionMode:
    """Execution mode identifiers for task execution."""

    SINGLE = "single"
    ORCHESTRATED = "orchestrated"
    ORCHESTRATED_SUB_TASK = "orchestrated_sub_task"


# Task initialization mode constants
class TaskInitMode:
    """Task initialization mode identifiers."""

    OWNERSHIP_TRANSFER = "ownership_transfer"
    FRESH_START = "fresh_start"
    EVAL_RETRY_REUSE = "eval_retry_reuse"
    EVAL_RETRY_COMPLETED_SAMPLE = "eval_retry_completed_sample"


# Episode termination reason constants
class EpisodeTerminationReason:
    """Episode termination reason identifiers."""

    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    TIMEOUT = "timeout"


# Client identifier constants
class ClientIdentifiers:
    """Client identifier prefixes and names."""

    INSPECT_AI_PREFIX = "inspect_ai_"
    INSPECT_AI_SANDBOX = "inspect_ai_sandbox"


# Cleanup reason constants
class CleanupReason:
    """Cleanup operation reason identifiers."""

    MANUAL_CLEANUP = "manual_cleanup"
    TASK_COMPLETE = "task_complete"
    TASK_INTERRUPTED = "task_interrupted"


__all__ = [
    "EvaluationStrategy",
    "SubmissionEvaluationStrategy",
    "StepEvaluationStrategy",
    "EVAL_STRATEGY_STATIC",
    "EVAL_STRATEGY_LLM_JUDGE",
    "EVAL_STRATEGY_TOOL_CALL",
    "VALID_SUBMISSION_EVAL_STRATEGIES",
    "VALID_STEP_EVAL_STRATEGIES",
    "VALID_EVAL_STRATEGIES",
    "MetadataKeys",
    "TranscriptSyncConfig",
    "APIEndpoints",
    "ExecutionMode",
    "TaskInitMode",
    "EpisodeTerminationReason",
    "ClientIdentifiers",
    "CleanupReason",
]
