"""REST API configuration constants.

This module defines configuration constants for REST API operations,
including transcript synchronization and network settings.
"""


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


__all__ = ["TranscriptSyncConfig"]
