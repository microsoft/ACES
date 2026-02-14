"""Type-safe WebSocket message models.

This module defines Pydantic models and enums for all WebSocket messages
exchanged between client and server, ensuring type safety and validation.
"""

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

# Enums for type-safe constants


class WebSocketMessageType(str, Enum):
    """WebSocket message types."""

    # Connection lifecycle
    CONNECTED = "connected"

    # Push operations
    PUSH_MESSAGE = "push_message"
    PUSH_ACK = "push_ack"

    # State machine events
    IS_WAITING_ON_USER = "is_waiting_on_user"
    IS_WAITING_ON_ASSISTANT = "is_waiting_on_assistant"
    IS_WAITING_ON_TOOLS = "is_waiting_on_tools"

    # Notifications
    TRANSCRIPT_MODIFIED = "transcript_modified"
    TRANSCRIPT_ERROR = "transcript_error"

    # Health check
    PING = "ping"
    PONG = "pong"


class TranscriptErrorType(str, Enum):
    """Transcript error types."""

    STUCK_STATE = "stuck_state"
    INVALID_TRANSITION = "invalid_transition"
    PUSH_FAILED = "push_failed"


class TranscriptOperation(str, Enum):
    """Transcript operation types.

    Simplified to:
    - INIT: Initial transcript creation
    - APPEND: Add messages at end (default)
    - RESTART: Reset to initial transcript then append
    """

    INIT = "init"  # Initial transcript creation
    APPEND = "append"  # Add messages at end (default)
    RESTART = "restart"  # Reset to initial transcript, then append


# Pydantic models for structured validation


class PushMessageRequestData(BaseModel):
    """Data payload for push_message messages (normal and injection mode)."""

    message: dict[str, Any] = Field(description="Message to push")
    since_sequence: int = Field(default=0, ge=0)

    # Injection fields (optional - red team only)
    target_episode_id: str | None = Field(default=None, description="Target episode for injection")
    strategy: str = Field(default="append", description="Operation strategy: append or restart")


class PushMessageData(BaseModel):
    """Client push message data (both normal and injection)."""

    message: dict[str, Any]
    since_sequence: int = Field(ge=0)

    # Injection fields (red team only)
    target_episode_id: str | None = Field(default=None, description="Target episode for injection")
    strategy: TranscriptOperation | None = Field(default=None, description="Operation strategy")


class PushAckData(BaseModel):
    """Server push acknowledgment data."""

    sequence: int = Field(description="New sequence number after push", ge=0)
    modification_count: int | None = Field(default=None, description="Injection counter", ge=0)
    target_episode_id: str | None = Field(default=None, description="Target episode for injection")


class StateEventData(BaseModel):
    """State machine event data."""

    version: int = Field(ge=0)
    operation: TranscriptOperation = Field(description="Operation that caused this event")
    modification_count: int = Field(ge=0)
    injected_by: str | None = Field(default=None)
    state: str = Field(description="Current transcript state")


class TranscriptErrorData(BaseModel):
    """Transcript error notification data."""

    error: TranscriptErrorType
    state: str | None = Field(default=None)
    duration_seconds: float | None = Field(default=None)
    threshold_seconds: float | None = Field(default=None)
    message: str | None = Field(default=None)


class ConnectedData(BaseModel):
    """Connected message data."""

    episode_id: str
    server_version: str | None = Field(default=None)


class ConnectionMetadata(BaseModel):
    """Connection metadata."""

    episode_id: str
    connected_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# Message wrapper classes


class ConnectedMessage(BaseModel):
    """WebSocket connection confirmation."""

    type: Literal["connected"] = "connected"
    episode_id: str
    timestamp: str


class PongMessage(BaseModel):
    """WebSocket pong response."""

    type: Literal["pong"] = "pong"
    timestamp: str


class PushAckMessage(BaseModel):
    """WebSocket push acknowledgment."""

    type: Literal["push_ack"] = "push_ack"
    data: PushAckData
    id: str
    timestamp: str


class StateEventMessage(BaseModel):
    """WebSocket state event."""

    type: Literal["transcript_modified", "is_waiting_on_user", "is_waiting_on_assistant", "is_waiting_on_tools"]
    data: StateEventData
    id: str
    timestamp: str


class TranscriptErrorMessage(BaseModel):
    """WebSocket error notification."""

    type: Literal["transcript_error"] = "transcript_error"
    data: TranscriptErrorData
    timestamp: str


# Union type for all possible WebSocket messages sent by server
# Uses discriminated union on the 'type' field for efficient parsing
WebSocketServerMessage = Annotated[
    ConnectedMessage | PongMessage | PushAckMessage | StateEventMessage | TranscriptErrorMessage,
    Field(discriminator="type"),
]

# TypeAdapter for parsing raw JSON into the correct message type
WebSocketServerMessageAdapter: TypeAdapter[WebSocketServerMessage] = TypeAdapter(WebSocketServerMessage)


__all__ = [
    # Enums
    "WebSocketMessageType",
    "TranscriptErrorType",
    "TranscriptOperation",
    # Data models (request dataclasses)
    "PushMessageRequestData",
    # Data models (response Pydantic)
    "PushMessageData",
    "PushAckData",
    "StateEventData",
    "TranscriptErrorData",
    "ConnectedData",
    "ConnectionMetadata",
    # Message wrappers
    "ConnectedMessage",
    "PongMessage",
    "PushAckMessage",
    "StateEventMessage",
    "TranscriptErrorMessage",
    # Union types and adapters
    "WebSocketServerMessage",
    "WebSocketServerMessageAdapter",
]
