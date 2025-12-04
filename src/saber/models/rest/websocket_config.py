"""WebSocket configuration for transcript coordination.

This module defines configuration parameters for WebSocket-based
transcript synchronization with separate push and pull configurations.
"""

from dataclasses import dataclass, field
from typing import Optional

from .websocket_constants import WebSocketDefaults


@dataclass
class PushConfig:
    """Configuration for pushing messages to server via WebSocket.

    Controls how the client sends new messages and handles confirmations.

    Attributes:
        enabled: Whether to push messages via WebSocket
        confirmation_timeout: Timeout waiting for push_ack from server (seconds)
        retry_enabled: Whether to retry failed push operations
        max_retry_attempts: Maximum number of retry attempts for push
        retry_backoff_multiplier: Multiplier for exponential backoff on retry
    """

    enabled: bool = True
    confirmation_timeout: float = WebSocketDefaults.PUSH_CONFIRMATION_TIMEOUT_SECONDS
    retry_enabled: bool = True
    max_retry_attempts: int = WebSocketDefaults.PUSH_MAX_RETRY_ATTEMPTS
    retry_backoff_multiplier: float = WebSocketDefaults.PUSH_RETRY_BACKOFF_MULTIPLIER

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.confirmation_timeout <= 0:
            raise ValueError(f"confirmation_timeout must be positive, got {self.confirmation_timeout}")
        if self.max_retry_attempts < 0:
            raise ValueError(f"max_retry_attempts must be non-negative, got {self.max_retry_attempts}")
        if self.retry_backoff_multiplier <= 0:
            raise ValueError(f"retry_backoff_multiplier must be positive, got {self.retry_backoff_multiplier}")


@dataclass
class PullConfig:
    """Configuration for pulling transcript modifications via WebSocket.

    Controls how the client waits for and receives server-initiated updates.

    Attributes:
        enabled: Whether to wait for transcript_modified events
        blocking: Whether to block waiting for events (if False, use event_timeout=0 semantics)
        event_timeout: Timeout waiting for transcript modification events (seconds)
        sync_timeout: Timeout for sync_request/sync_response round-trip (seconds)
        event_queue_max_size: Maximum number of queued transcript_modified events
    """

    enabled: bool = True
    blocking: bool = True
    event_timeout: float = WebSocketDefaults.PULL_EVENT_TIMEOUT_SECONDS
    sync_timeout: float = WebSocketDefaults.PULL_SYNC_TIMEOUT_SECONDS
    event_queue_max_size: int = WebSocketDefaults.PULL_EVENT_QUEUE_MAX_SIZE

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.event_timeout <= 0:
            raise ValueError(f"event_timeout must be positive, got {self.event_timeout}")
        if self.sync_timeout <= 0:
            raise ValueError(f"sync_timeout must be positive, got {self.sync_timeout}")
        if self.event_queue_max_size <= 0:
            raise ValueError(f"event_queue_max_size must be positive, got {self.event_queue_max_size}")


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connections and bidirectional coordination.

    Centralizes all WebSocket-related configuration to enable per-domain
    tuning and testing without code changes. Separates push (client→server)
    and pull (server→client) operations for independent control.

    Attributes:
        connection_timeout: Timeout for initial WebSocket connection (seconds)
        ping_interval: Interval for sending WebSocket ping keepalive (seconds)
        pong_timeout: Timeout waiting for pong response (seconds)
        reconnect_enabled: Whether to automatically reconnect on connection loss
        max_reconnect_attempts: Maximum number of reconnection attempts
        initial_reconnect_delay: Initial delay before first reconnect (seconds)
        max_reconnect_delay: Maximum delay between reconnect attempts (seconds)
        reconnect_backoff_multiplier: Multiplier for exponential backoff
        push: Configuration for pushing messages to server
        pull: Configuration for pulling transcript modifications
    """

    # Connection configuration
    connection_timeout: float = WebSocketDefaults.CONNECTION_TIMEOUT_SECONDS

    # Keepalive
    ping_interval: Optional[float] = WebSocketDefaults.PING_INTERVAL_SECONDS
    pong_timeout: float = WebSocketDefaults.PONG_TIMEOUT_SECONDS

    # Reconnection policy
    reconnect_enabled: bool = True
    max_reconnect_attempts: int = WebSocketDefaults.RECONNECT_MAX_ATTEMPTS
    initial_reconnect_delay: float = WebSocketDefaults.RECONNECT_INITIAL_DELAY_SECONDS
    max_reconnect_delay: float = WebSocketDefaults.RECONNECT_MAX_DELAY_SECONDS
    reconnect_backoff_multiplier: float = WebSocketDefaults.RECONNECT_BACKOFF_MULTIPLIER

    # Push and pull configurations
    push: PushConfig = field(default_factory=PushConfig)
    pull: PullConfig = field(default_factory=PullConfig)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.connection_timeout <= 0:
            raise ValueError(f"connection_timeout must be positive, got {self.connection_timeout}")
        if self.initial_reconnect_delay <= 0:
            raise ValueError(f"initial_reconnect_delay must be positive, got {self.initial_reconnect_delay}")
        if self.max_reconnect_delay <= 0:
            raise ValueError(f"max_reconnect_delay must be positive, got {self.max_reconnect_delay}")
        if self.max_reconnect_delay < self.initial_reconnect_delay:
            raise ValueError(
                f"max_reconnect_delay ({self.max_reconnect_delay}) must be >= "
                f"initial_reconnect_delay ({self.initial_reconnect_delay})"
            )
        if self.reconnect_backoff_multiplier <= 0:
            raise ValueError(f"reconnect_backoff_multiplier must be positive, got {self.reconnect_backoff_multiplier}")
        if self.max_reconnect_attempts < 1:
            raise ValueError(f"max_reconnect_attempts must be at least 1, got {self.max_reconnect_attempts}")
        if self.pong_timeout <= 0:
            raise ValueError(f"pong_timeout must be positive, got {self.pong_timeout}")


__all__ = ["WebSocketConfig", "PushConfig", "PullConfig"]
