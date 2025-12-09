"""Type-safe WebSocket message models.

This module defines Pydantic models and enums for all WebSocket messages
exchanged between client and server, ensuring type safety and validation.
"""

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter

# Enums for type-safe constants


class WebSocketMessageType(str, Enum):
    """WebSocket message types."""

    # Connection lifecycle
    CONNECTED = "connected"

    # Transcript synchronization
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"

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


class SyncMode(str, Enum):
    """Transcript sync mode."""

    FULL = "full"  # Full transcript replacement
    DELTA = "delta"  # Incremental delta
    NO_CHANGE = "no_change"  # No changes since requested version


class TranscriptErrorType(str, Enum):
    """Transcript error types."""

    STUCK_STATE = "stuck_state"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    INVALID_TRANSITION = "invalid_transition"


class TranscriptOperation(str, Enum):
    """Transcript operation types."""

    INIT = "init"  # Initial transcript creation
    APPEND = "append"  # Add messages at end (default)
    REWIND = "rewind"  # Remove last N messages, then append
    REWRITE = "rewrite"  # Replace entire transcript
    INSERT = "insert"  # Insert at specific position


# Pydantic models for structured validation


class TranscriptVersion(BaseModel):
    """Transcript version with checksum."""

    sequence: int = Field(description="Sequential version number", ge=0)
    checksum: str = Field(description="SHA256 checksum of transcript")
    message_count: Optional[int] = Field(default=None, description="Total message count", ge=0)
    last_operation: Optional[str] = Field(default=None, description="Last operation")


class SyncRequestData(BaseModel):
    """Client sync request data."""

    since_version: Optional[int] = Field(default=None, description="Last known version", ge=0)
    client_checksum: Optional[str] = Field(default=None, description="Client checksum for verification")


class SyncResponseData(BaseModel):
    """Server sync response data."""

    sync_mode: SyncMode = Field(description="Full or delta sync")
    current_version: TranscriptVersion
    modified: Optional[bool] = Field(default=None, description="Whether transcript was modified")
    full_transcript: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Full transcript (when sync_mode=FULL)"
    )
    delta: Optional[List[Dict[str, Any]]] = Field(default=None, description="Delta messages (when sync_mode=DELTA)")


class PushMessageData(BaseModel):
    """Client push message data (both normal and injection)."""

    message: Dict[str, Any]
    since_version: int = Field(ge=0)
    client_checksum: Optional[str] = Field(default=None)

    # Injection fields (red team only)
    target_episode_id: Optional[str] = Field(default=None, description="Target episode for injection")
    strategy: Optional[TranscriptOperation] = Field(default=None, description="Operation strategy")
    rewind_count: Optional[int] = Field(default=None, ge=0)
    insert_position: Optional[int] = Field(default=None, ge=0)


class PushAckData(BaseModel):
    """Server push acknowledgment data."""

    version: int = Field(description="New version after push", ge=0)
    checksum: str = Field(description="New checksum after push")
    modification_count: Optional[int] = Field(default=None, description="Injection counter", ge=0)
    target_episode_id: Optional[str] = Field(default=None, description="Target episode for injection")


class StateEventData(BaseModel):
    """State machine event data."""

    version: int = Field(ge=0)
    operation: TranscriptOperation = Field(description="Operation that caused this event")
    modification_count: int = Field(ge=0)
    injected_by: Optional[str] = Field(default=None)
    state: str = Field(description="Current transcript state")


class TranscriptErrorData(BaseModel):
    """Transcript error notification data."""

    error: TranscriptErrorType
    state: Optional[str] = Field(default=None)
    duration_seconds: Optional[float] = Field(default=None)
    threshold_seconds: Optional[float] = Field(default=None)
    message: Optional[str] = Field(default=None)


class ConnectedData(BaseModel):
    """Connected message data."""

    episode_id: str
    server_version: Optional[str] = Field(default=None)


class ConnectionMetadata(BaseModel):
    """Connection metadata."""

    episode_id: str
    connected_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


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


class SyncResponseMessage(BaseModel):
    """WebSocket sync response."""

    type: Literal["sync_response"] = "sync_response"
    data: SyncResponseData
    id: str
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
    Union[
        ConnectedMessage,
        PongMessage,
        SyncResponseMessage,
        PushAckMessage,
        StateEventMessage,
        TranscriptErrorMessage,
    ],
    Field(discriminator="type"),
]

# TypeAdapter for parsing raw JSON into the correct message type
WebSocketServerMessageAdapter: TypeAdapter[WebSocketServerMessage] = TypeAdapter(WebSocketServerMessage)


__all__ = [
    # Enums
    "WebSocketMessageType",
    "SyncMode",
    "TranscriptErrorType",
    "TranscriptOperation",
    # Data models
    "TranscriptVersion",
    "SyncRequestData",
    "SyncResponseData",
    "PushMessageData",
    "PushAckData",
    "StateEventData",
    "TranscriptErrorData",
    "ConnectedData",
    "ConnectionMetadata",
    # Message wrappers
    "ConnectedMessage",
    "PongMessage",
    "SyncResponseMessage",
    "PushAckMessage",
    "StateEventMessage",
    "TranscriptErrorMessage",
    # Union types and adapters
    "WebSocketServerMessage",
    "WebSocketServerMessageAdapter",
]
