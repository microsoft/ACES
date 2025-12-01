"""REST API configuration constants.

This module defines configuration constants for REST API operations,
including transcript synchronization and network settings.
"""


class TranscriptSyncConfig:
    """Configuration constants for transcript synchronization between client and server."""

    # Payload size limits
    MAX_PAYLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

    # Network timeouts
    REQUEST_TIMEOUT_SECONDS = 30.0  # Increased from 5.0 to 30.0 seconds

    # Retry configuration - exponential backoff with unlimited retries
    INITIAL_RETRY_DELAY_SECONDS = 1.0
    MAX_RETRY_DELAY_SECONDS = 60.0  # Cap backoff at 60 seconds
    BACKOFF_MULTIPLIER = 2.0  # Double the delay each retry

    # Default transcript source identifier
    DEFAULT_SOURCE = "inspect_ai"


__all__ = ["TranscriptSyncConfig"]
