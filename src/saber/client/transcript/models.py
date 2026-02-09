"""Data models for transcript synchronization.

This module provides generic data classes used by the transcript sync client.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

# Generic message type
M = TypeVar("M")


@dataclass
class SyncResult(Generic[M]):
    """Result of a transcript sync operation.

    Contains the synchronized messages along with metadata about the sync.

    Type Parameter:
        M: The harness-specific message type

    Attributes:
        messages: The synchronized list of messages
        version: Server transcript version after sync
        checksum: Server transcript checksum after sync
        sync_mode: One of "full", "delta", or "no_change"
        modified: Whether the transcript was modified since last sync
    """

    messages: list[M]
    version: int
    checksum: str
    sync_mode: str  # "full", "delta", "no_change"
    modified: bool = False


__all__ = ["SyncResult"]
