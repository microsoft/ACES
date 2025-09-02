"""
Structured Event Objects for Harness Logging

Provides type-safe event classes that can be serialized to dictionaries for logging.
Each event type has specific fields and validation.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HarnessInitializationEvent:
    """Event for harness initialization."""

    parallelism: int
    client_id: str
    server_url: str
    mcp_url: Optional[str]
    enable_container_logging: bool
    client_log_dir: Optional[Path]
    log_retention_days: int
    task_ids: Optional[List[str]]
    ui_enabled: bool
    ui_backend: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        # Convert Path to string
        if data["client_log_dir"]:
            data["client_log_dir"] = str(data["client_log_dir"])
        return data


@dataclass
class SessionCreationEvent:
    """Event for session creation attempts."""

    session_id: Optional[str]
    success: bool
    error_message: Optional[str] = None
    server_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class TaskResolutionEvent:
    """Event for task resolution."""

    total_tasks: int
    requested_task_ids: Optional[List[str]]
    resolved_task_ids: List[str]
    resolution_mode: str  # "specific" or "all_available"
    missing_task_ids: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class EpisodeQueueEvent:
    """Event for episode queue building."""

    episode_count: int
    episodes: List[Dict[str, Any]]  # [{"task_id": str, "attempt": int}, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class ExecutionStartEvent:
    """Event for execution start."""

    parallelism: int
    episode_count: int
    agent_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class ExecutionCompleteEvent:
    """Event for execution completion."""

    total_episodes: int
    successful_episodes: int
    success_rate: float
    execution_duration_seconds: float
    start_time: str  # ISO format
    end_time: str  # ISO format

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class InfrastructureEvent:
    """Event for infrastructure operations."""

    operation: str  # "initialized", "started", "stopped", etc.
    component: str  # "container_infrastructure", "mcp_sidecar", etc.
    success: bool
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class UIEvent:
    """Event for UI operations."""

    event_type: str  # "session_start", "session_update", "session_complete", etc.
    ui_backend: str
    success: bool
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class CleanupEvent:
    """Event for cleanup operations."""

    cleanup_successful: bool
    cleanup_duration_seconds: float
    cleanup_errors: List[str]
    session_id: Optional[str]
    cleanup_start_time: str  # ISO format
    cleanup_end_time: str  # ISO format

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class ErrorEvent:
    """Event for error conditions."""

    error_type: str
    error_message: str
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    session_id: Optional[str] = None
    execution_phase: Optional[str] = None
    task_id: Optional[str] = None
    episode_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)
