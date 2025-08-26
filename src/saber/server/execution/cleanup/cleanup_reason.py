"""
Cleanup reason enumeration for standardized container cleanup tracking.
"""

from enum import Enum


class CleanupReason(Enum):
    """
    Standardized reasons for container cleanup operations.

    This enum provides a consistent way to track why containers were cleaned up,
    enabling better debugging and operational insights.
    """

    # Normal completion scenarios
    EPISODE_COMPLETED = "episode_completed"
    SESSION_TERMINATED = "session_terminated"

    # Error scenarios
    EPISODE_FAILED = "episode_failed"
    ERROR_TRIGGERED = "error_triggered"
    HEALTH_CHECK_FAILED = "health_check_failed"

    # Timeout scenarios
    SESSION_TIMEOUT = "session_timeout"

    # Administrative scenarios
    SERVER_SHUTDOWN = "server_shutdown"
    MANUAL_CLEANUP = "manual_cleanup"

    def __str__(self) -> str:
        """Return the enum value as string for logging."""
        return self.value

    @property
    def is_error_scenario(self) -> bool:
        """Check if this cleanup reason indicates an error condition."""
        return self in {
            CleanupReason.EPISODE_FAILED,
            CleanupReason.ERROR_TRIGGERED,
            CleanupReason.HEALTH_CHECK_FAILED,
            CleanupReason.SESSION_TIMEOUT,
        }

    @property
    def is_normal_scenario(self) -> bool:
        """Check if this cleanup reason indicates normal completion."""
        return self in {CleanupReason.EPISODE_COMPLETED, CleanupReason.SESSION_TERMINATED}
