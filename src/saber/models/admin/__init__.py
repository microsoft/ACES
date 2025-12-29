"""
SABER Admin Models - Administrative and debugging models.

These models are used for system administration, monitoring, and debugging.
"""

from pydantic import BaseModel, Field


class CleanupHistoryEntry(BaseModel):
    """Single cleanup history entry."""

    session_id: str = Field(description="Session that was cleaned up")
    cleanup_time: str = Field(description="Cleanup timestamp (ISO format)")
    reason: str = Field(description="Cleanup reason")
    episode_count: int = Field(description="Number of episodes cleaned up")


class CleanupHistoryResponse(BaseModel):
    """Response model for cleanup history."""

    cleanup_history: list[CleanupHistoryEntry] = Field(description="List of cleanup entries")
    total_cleanups: int = Field(description="Total number of cleanups performed")


class SessionCleanupHistoryResponse(BaseModel):
    """Response model for session-specific cleanup history."""

    session_id: str = Field(description="Session identifier")
    cleanup_entries: list[CleanupHistoryEntry] = Field(description="Cleanup entries for this session")
    total_cleanups: int = Field(description="Total cleanups for this session")


class ActiveCleanupInfo(BaseModel):
    """Information about an active cleanup operation."""

    session_id: str = Field(description="Session being cleaned up")
    cleanup_type: str = Field(description="Type of cleanup operation")
    start_time: str = Field(description="Cleanup start time (ISO format)")
    progress: str | None = Field(None, description="Cleanup progress information")


class ActiveCleanupsResponse(BaseModel):
    """Response model for active cleanup operations."""

    active_cleanups: list[ActiveCleanupInfo] = Field(description="List of active cleanup operations")
    count: int = Field(description="Number of active cleanups")
