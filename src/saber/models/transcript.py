"""Transcript coordination models for WebSocket-based sync.

This module provides data models for the transcript coordination service
implementing differential sync with version tracking and checksum validation.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .rest.websocket_messages import SyncMode


def compute_checksum(messages: list[dict[str, Any]]) -> str:
    """
    Compute stable SHA256 checksum of transcript for rewrite detection.

    Uses full SHA256 hash (64 hex chars = 32 bytes).
    Collision probability with 10K transcripts: ~2.7 × 10^-12 (acceptable).

    Args:
        messages: List of message dictionaries

    Returns:
        SHA256 checksum as 64-character hex string
    """
    stable_json = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(stable_json.encode()).hexdigest()


class SyncStrategy(str, Enum):
    """Coordination strategies for transcript synchronization."""

    IMMEDIATE = "immediate"  # No blocking, return immediately
    WAIT_FOR_CHANGE = "wait_for_change"  # Block until WebSocket event (deprecated - WebSocket handles this)


class TranscriptPushOperation(str, Enum):
    """Operations for modifying transcript when pushing messages.

    Used by both client and server to specify how new messages should be
    integrated into the existing transcript.

    Simplified to two operations:
    - APPEND: Add messages at end (default)
    - RESTART: Reset to initial transcript (system->user->assistant) then append
    """

    APPEND = "append"  # Add messages at end (default)
    RESTART = "restart"  # Reset to initial transcript, then append new message


@dataclass
class TranscriptVersion:
    """
    Version with rewrite detection via checksum.

    Combines monotonic sequence number with SHA256 checksum to detect
    both incremental changes and full rewrites.

    Attributes:
        sequence: Monotonic version counter (0, 1, 2, 3...)
        checksum: Full SHA256 hash (64 hex chars) of entire transcript
        message_count: Total number of messages in transcript
        last_operation: Type of last operation ("append", "restart", etc.)
    """

    sequence: int
    checksum: str
    message_count: int
    last_operation: str

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.sequence < 0:
            raise ValueError(f"Version sequence must be non-negative, got {self.sequence}")
        if not self.checksum or len(self.checksum) != 64:
            raise ValueError(
                f"Checksum must be 64 hex chars (SHA256), got {len(self.checksum) if self.checksum else 0}"
            )
        if self.message_count < 0:
            raise ValueError(f"Message count must be non-negative, got {self.message_count}")


@dataclass
class TranscriptSyncRequest:
    """
    Request for transcript synchronization.

    Supports both push (client sends new messages) and pull (client requests updates).
    Also supports observer mode for cross-episode access with security filtering.

    Attributes:
        episode_id: Target episode identifier
        since_version: Client's current version (for delta sync)
        client_checksum: SHA256 checksum of client's transcript at since_version (for validation)
        messages_to_push: New messages to append (optional)
        strategy: Sync strategy (immediate or wait_for_change)
        operation: Operation type for push ("append" or "restart")
        is_observer: Whether this is an observer request (cross-episode, read-only)
        hide_system_prompt: Hide system/user messages before first assistant (security filter)
        retrieval_mode: For observers: 'full', 'tail', or 'delta'
        tail_count: For observers with tail mode: number of messages to return
    """

    episode_id: str
    since_version: int = 0
    client_checksum: str | None = None
    messages_to_push: list[dict[str, Any]] | None = None
    strategy: str = SyncStrategy.IMMEDIATE.value
    operation: str = "append"
    # Observer mode parameters
    is_observer: bool = False
    hide_system_prompt: bool | None = None  # None = use default (True for observers)
    retrieval_mode: str = "full"  # 'full', 'tail', 'delta'
    tail_count: int = 10

    def __post_init__(self) -> None:
        """Validate fields and set defaults."""
        if self.since_version < 0:
            raise ValueError(f"since_version must be non-negative, got {self.since_version}")
        if self.client_checksum and len(self.client_checksum) != 64:
            raise ValueError(f"client_checksum must be 64 hex chars if provided, got {len(self.client_checksum)}")
        if self.messages_to_push is None:
            self.messages_to_push = []
        # Default hide_system_prompt to True for observers (secure by default)
        if self.hide_system_prompt is None:
            self.hide_system_prompt = self.is_observer
        # Clamp tail_count
        if self.tail_count > 1000:
            self.tail_count = 1000


@dataclass
class TranscriptSyncResponse:
    """
    Response from transcript synchronization.

    Provides either delta (new messages only) or full transcript depending on sync mode.

    Attributes:
        current_version: Server's current version info
        delta: New messages since client's version (None if full sync or no changes)
        full_transcript: Complete transcript (None if delta sync or no changes)
        sync_mode: Mode used (SyncMode.DELTA, SyncMode.FULL, SyncMode.NO_CHANGE)
        modified: Whether transcript was modified
        blocked: Whether request blocked waiting for changes (deprecated)
        wait_time_seconds: Time spent waiting (0 for immediate)
    """

    current_version: TranscriptVersion
    delta: list[dict[str, Any]] | None = None
    full_transcript: list[dict[str, Any]] | None = None
    sync_mode: SyncMode | str = SyncMode.NO_CHANGE
    modified: bool = False
    blocked: bool = False
    wait_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validate sync mode consistency and normalize to enum."""
        # Normalize string to enum if needed
        if isinstance(self.sync_mode, str):
            try:
                object.__setattr__(self, "sync_mode", SyncMode(self.sync_mode))
            except ValueError:
                raise ValueError(
                    f"sync_mode must be one of {[m.value for m in SyncMode]}, got {self.sync_mode}"
                ) from None

        # Validate consistency
        if self.sync_mode == SyncMode.DELTA and self.delta is None:
            raise ValueError("sync_mode=DELTA requires delta to be set")
        if self.sync_mode == SyncMode.FULL and self.full_transcript is None:
            raise ValueError("sync_mode=FULL requires full_transcript to be set")
        if self.sync_mode == SyncMode.NO_CHANGE and (self.delta or self.full_transcript):
            raise ValueError("sync_mode=NO_CHANGE should not have delta or full_transcript")


__all__ = [
    "compute_checksum",
    "SyncStrategy",
    "TranscriptPushOperation",
    "TranscriptVersion",
    "TranscriptSyncRequest",
    "TranscriptSyncResponse",
]
