"""Request models for WebSocket messages.

This module provides strongly-typed dataclass models for WebSocket request payloads,
replacing dictionary-based data access with type-safe attribute access.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class SyncRequestData:
    """Data payload for sync_request messages."""

    since_version: int = 0
    client_checksum: str | None = None

    # Observer/cross-episode fields (for red team accessing blue team transcript)
    target_episode_id: str | None = None
    hide_system_prompt: bool | None = None
    retrieval_mode: str | None = None  # full, delta, tail
    tail_count: int | None = None


@dataclass
class PushMessageRequestData:
    """Data payload for push_message messages (normal and injection mode)."""

    message: dict[str, Any]
    since_version: int = 0
    client_checksum: str | None = None

    # Injection fields (optional - red team only)
    target_episode_id: str | None = None
    strategy: str = "append"  # "append", "rewind", "rewrite", "insert"
    rewind_count: int = 1
    insert_position: int = 0


__all__ = [
    "SyncRequestData",
    "PushMessageRequestData",
]
