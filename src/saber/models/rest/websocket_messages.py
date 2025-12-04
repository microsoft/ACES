"""TypedDict definitions for WebSocket message structures.

This module provides typed dictionaries for all WebSocket message types
used in the transcript coordination system. These provide type safety
for message construction and validation.
"""

from typing import Any, Dict, Literal, TypedDict


class TranscriptModifiedData(TypedDict):
    """Data payload for transcript_modified events."""

    version: int
    operation: str
    modification_count: int
    injected_by: str


class TranscriptModifiedMessage(TypedDict):
    """WebSocket message sent when transcript is modified by red team."""

    type: Literal["transcript_modified"]
    data: TranscriptModifiedData
    id: str
    timestamp: str


class ConnectedMessage(TypedDict):
    """WebSocket message sent when connection is established."""

    type: Literal["connected"]
    episode_id: str
    timestamp: str


class PongMessage(TypedDict):
    """WebSocket pong response to ping keepalive."""

    type: Literal["pong"]
    timestamp: str


class SyncResponseVersionData(TypedDict):
    """Version information in sync response."""

    sequence: int
    checksum: str
    message_count: int
    last_operation: str


class SyncResponseData(TypedDict, total=False):
    """Data payload for sync_response messages."""

    current_version: SyncResponseVersionData
    delta: list[Dict[str, Any]]  # Optional: only for delta sync
    full_transcript: list[Dict[str, Any]]  # Optional: only for full sync
    sync_mode: str
    modified: bool


class SyncResponseMessage(TypedDict):
    """WebSocket response to sync_request."""

    type: Literal["sync_response"]
    data: SyncResponseData
    id: str
    timestamp: str


class PushAckData(TypedDict):
    """Data payload for push_ack messages."""

    version: int
    checksum: str


class PushAckMessage(TypedDict):
    """WebSocket acknowledgment of message push."""

    type: Literal["push_ack"]
    data: PushAckData
    id: str
    timestamp: str


class ConnectionMetadata(TypedDict):
    """Metadata stored for each WebSocket connection."""

    episode_id: str
    connected_at: str  # ISO format timestamp
    metadata: Dict[str, Any]  # User-provided metadata (role, session_id, etc.)


# Union type for all possible WebSocket messages sent by server
WebSocketServerMessage = (
    TranscriptModifiedMessage | ConnectedMessage | PongMessage | SyncResponseMessage | PushAckMessage
)


__all__ = [
    "TranscriptModifiedData",
    "TranscriptModifiedMessage",
    "ConnectedMessage",
    "PongMessage",
    "SyncResponseVersionData",
    "SyncResponseData",
    "SyncResponseMessage",
    "PushAckData",
    "PushAckMessage",
    "ConnectionMetadata",
    "WebSocketServerMessage",
]
