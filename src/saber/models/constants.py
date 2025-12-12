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
    AUTO_CONTINUE_ENABLED = "auto_continue_enabled"
    CONTINUE_PROMPT = "continue_prompt"
    LAST_AUTO_CONTINUE_VERSION = "last_auto_continue_version"
    ALLOW_CLIENT_USER_MESSAGES = "allow_client_user_messages"

    # Score metadata (used in score.metadata dictionaries)
    SUBMISSION_SCORE = "submission_score"
    SUBTASK_SCORE = "subtask_score"
    WEIGHTED_SUBTASK_SCORE = "weighted_subtask_score"
    STEP_EVALUATIONS = "step_evaluations"
    SUBTASK_SCORES = "subtask_scores"

    # Transcript synchronization context keys
    CLIENT_TRANSCRIPT = "_client_transcript"
    INITIAL_TRANSCRIPT = "_initial_transcript"  # Original transcript (system->user) for restart operation
    TRANSCRIPT_METADATA = "_transcript_metadata"

    # WebSocket-based transcript coordination (Phase 2)
    TRANSCRIPT_VERSION = "_transcript_version"  # Monotonic sequence number (0, 1, 2, 3...)
    TRANSCRIPT_LAST_OPERATION = "_transcript_last_operation"  # Last operation type: append, rewrite, insert, rewind
    TRANSCRIPT_CHECKSUM = "_transcript_checksum"  # SHA256 checksum of transcript

    # Prompt formatting
    PROMPT_SECTION_DELIMITER = "\n\n---\n\n"  # Delimiter between prompt sections in system message

    # Stuck state monitoring (Phase 4)
    CURRENT_TRANSCRIPT_STATE = "current_transcript_state"  # Current state for comparison
    STUCK_STATE_THRESHOLD = "stuck_state_threshold"  # Custom threshold per episode (seconds)
    EPISODE_STUCK = "episode_stuck"  # Flag indicating episode is stuck
    STUCK_SINCE = "stuck_since"  # Timestamp when stuck was detected

    # Blocking transcript solver - Timestamp-driven coordination (Legacy - being replaced)
    TRANSCRIPT_LAST_PUSHED_AT = "_transcript_last_pushed_at"  # ISO timestamp of last transcript push
    TRANSCRIPT_LAST_MODIFIED_AT = "_transcript_last_modified_at"  # ISO timestamp of last red team modification
    TRANSCRIPT_MODIFICATION_COUNT = "_transcript_modification_count"  # Monotonic counter (1, 2, 3...)

    # Transcript metadata keys
    TRANSCRIPT_STEP_NUMBER = "step_number"
    TRANSCRIPT_TIMESTAMP = "timestamp"
    TRANSCRIPT_SOURCE = "source"

    # Message injection metadata
    PENDING_INJECTIONS = "_pending_injections"  # List of messages pending injection
    INJECTION_HISTORY = "_injection_history"  # History of injected messages

    # Orchestration cross-episode metadata
    ORCHESTRATION_TARGET_EPISODES = "_orchestration_target_episodes"
    ORCHESTRATION_ROLE = "_orchestration_role"


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
]
