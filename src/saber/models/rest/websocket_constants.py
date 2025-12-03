"""WebSocket constants for transcript coordination.

This module defines all constants used in WebSocket communication to avoid
magic numbers and ensure consistency across the codebase.
"""


class WebSocketCloseCode:
    """WebSocket close codes per RFC 6455.

    See: https://datatracker.ietf.org/doc/html/rfc6455#section-7.4.1
    """

    # Standard codes (1000-1015)
    NORMAL_CLOSURE = 1000  # Successful operation, connection closed
    ENDPOINT_GOING_AWAY = 1001  # Server going down or browser navigating away
    PROTOCOL_ERROR = 1002  # WebSocket protocol error
    UNSUPPORTED_DATA = 1003  # Received data type that cannot be accepted
    NO_STATUS_RECEIVED = 1005  # No status code was present (reserved)
    ABNORMAL_CLOSURE = 1006  # Connection closed abnormally (reserved)
    INVALID_PAYLOAD = 1007  # Invalid payload data (e.g., non-UTF-8 in text)
    POLICY_VIOLATION = 1008  # Generic policy violation
    MESSAGE_TOO_BIG = 1009  # Message too big to process
    MANDATORY_EXTENSION = 1010  # Client expected server to negotiate extension
    INTERNAL_ERROR = 1011  # Server encountered unexpected condition
    SERVICE_RESTART = 1012  # Server restarting
    TRY_AGAIN_LATER = 1013  # Temporary server condition
    BAD_GATEWAY = 1014  # Server acting as gateway received invalid response
    TLS_HANDSHAKE_FAILED = 1015  # TLS handshake failed (reserved)


class WebSocketDefaults:
    """Default values for WebSocket configuration.

    Centralized defaults make testing and tuning easier. These values
    have been tuned for the SABER use case (long-running AI agent episodes).
    """

    # Connection timeouts
    CONNECTION_TIMEOUT_SECONDS = 10.0  # Initial connection timeout

    # Keepalive settings
    PING_INTERVAL_SECONDS = 30.0  # How often to send ping
    PONG_TIMEOUT_SECONDS = 10.0  # How long to wait for pong response

    # Pull configuration (receiving transcript modifications from server)
    PULL_EVENT_TIMEOUT_SECONDS = 300.0  # 5 minutes - max wait for red team injection
    PULL_SYNC_TIMEOUT_SECONDS = 5.0  # Sync request/response round-trip timeout
    PULL_EVENT_QUEUE_MAX_SIZE = 100  # Max queued events (prevents memory exhaustion)

    # Push configuration (sending new messages to server)
    PUSH_CONFIRMATION_TIMEOUT_SECONDS = 5.0  # Acknowledgment timeout
    PUSH_MAX_RETRY_ATTEMPTS = 3  # Number of retry attempts on failure
    PUSH_RETRY_BACKOFF_MULTIPLIER = 2.0  # Exponential backoff multiplier

    # Reconnection policy
    RECONNECT_INITIAL_DELAY_SECONDS = 1.0  # Initial delay before reconnecting
    RECONNECT_MAX_DELAY_SECONDS = 30.0  # Maximum delay between reconnect attempts
    RECONNECT_BACKOFF_MULTIPLIER = 2.0  # Exponential backoff multiplier
    RECONNECT_MAX_ATTEMPTS = 3  # Maximum reconnection attempts before giving up

    # Cleanup timeouts
    LISTENER_TASK_CANCEL_TIMEOUT_SECONDS = 1.0  # Timeout when cancelling listener task
    WEBSOCKET_CLOSE_TIMEOUT_SECONDS = 2.0  # Timeout when closing WebSocket connection


__all__ = ["WebSocketCloseCode", "WebSocketDefaults"]
