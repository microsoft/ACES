"""Core execution and lifecycle constants.

This module defines constants related to task execution modes, initialization,
episode termination, client identification, and cleanup operations.
"""


class ExecutionMode:
    """Execution mode identifiers for task execution."""

    SINGLE = "single"
    ORCHESTRATED = "orchestrated"
    ORCHESTRATED_SUB_TASK = "orchestrated_sub_task"


class TaskInitMode:
    """Task initialization mode identifiers."""

    OWNERSHIP_TRANSFER = "ownership_transfer"
    FRESH_START = "fresh_start"
    EVAL_RETRY_REUSE = "eval_retry_reuse"
    EVAL_RETRY_COMPLETED_SAMPLE = "eval_retry_completed_sample"


class ClientIdentifiers:
    """Client identifier prefixes and names."""

    INSPECT_AI_PREFIX = "inspect_ai_"
    INSPECT_AI_SANDBOX = "inspect_ai_sandbox"


class CleanupReason:
    """Cleanup operation reason identifiers."""

    MANUAL_CLEANUP = "manual_cleanup"
    TASK_COMPLETE = "task_complete"
    TASK_INTERRUPTED = "task_interrupted"


__all__ = [
    "ExecutionMode",
    "TaskInitMode",
    "ClientIdentifiers",
    "CleanupReason",
]
