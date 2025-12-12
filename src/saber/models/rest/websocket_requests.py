"""Request models for WebSocket messages.

This module provides strongly-typed dataclass models for WebSocket request payloads,
replacing dictionary-based data access with type-safe attribute access.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SyncRequestData:
    """Data payload for sync_request messages."""

    since_version: int = 0
    client_checksum: Optional[str] = None

    # Observer/cross-episode fields (for red team accessing blue team transcript)
    target_episode_id: Optional[str] = None
    hide_system_prompt: Optional[bool] = None
    retrieval_mode: Optional[str] = None  # full, delta, tail
    tail_count: Optional[int] = None


@dataclass
class PushMessageRequestData:
    """Data payload for push_message messages (normal and injection mode)."""

    message: Dict[str, Any]
    since_version: int = 0
    client_checksum: Optional[str] = None

    # Injection fields (optional - red team only)
    target_episode_id: Optional[str] = None
    strategy: str = "append"  # "append", "rewind", "rewrite", "insert"
    rewind_count: int = 1
    insert_position: int = 0


__all__ = [
    "SyncRequestData",
    "PushMessageRequestData",
]
