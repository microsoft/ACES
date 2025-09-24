"""Episode management constants.

Logging Category: EPISODE (no active instrumentation required in this module).
"""

from enum import Enum


class EpisodeTerminationReason(str, Enum):
    """Enumeration of possible episode termination reasons."""

    # Successful completion reasons
    COMPLETED = "completed"
    AGENT_COMPLETED = "agent_completed"
    SUCCESS = "success"

    # Termination reasons
    TERMINATED = "terminated"
    SESSION_TERMINATED = "session_terminated"
    TIMEOUT = "timeout"
    MAX_STEPS_REACHED = "max_steps_reached"
    ERROR = "error"

    @classmethod
    def get_success_reasons(cls) -> set[str]:
        """Get all termination reasons that indicate success."""
        return {cls.COMPLETED, cls.AGENT_COMPLETED, cls.SUCCESS}

    @classmethod
    def is_success(cls, reason: str) -> bool:
        """Check if a termination reason indicates success."""
        # Check if any success reason is contained in the reason string
        # This handles cases where the reason includes extra formatting like "(30/30)"
        success_reasons = cls.get_success_reasons()
        return any(success_reason in reason.lower() for success_reason in success_reasons)


class EpisodeResponseKeys(str, Enum):
    """Standard keys used in episode response dictionaries."""

    # Episode status
    EPISODE_ENDED = "episode_ended"
    SUCCESS = "success"
    REASON = "reason"

    # Task transition
    PREVIOUS_TASK_ID = "previous_task_id"
    NEXT_TASK_ID = "next_task_id"
    NEXT_EPISODE_ID = "next_episode_id"

    # Benchmark status
    BENCHMARK_CONTINUES = "benchmark_continues"
    BENCHMARK_COMPLETE = "benchmark_complete"

    # Environment changes
    ENVIRONMENT_CHANGED = "environment_changed"
