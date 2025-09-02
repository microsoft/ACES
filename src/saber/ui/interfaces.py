#!/usr/bin/env python3
"""
SABER UI Interfaces

Clean interfaces and data types for SABER UI integration.
All inspect-ai dependencies are isolated in adapter implementations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

# Import existing SABER models to maintain consistency
from ..api_models import TaskInfo


class SABERTaskState(Enum):
    """Task execution states."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SERVER_TERMINATED = "server_terminated"


class MCPToolCallStatus(Enum):
    """MCP tool call execution status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class MCPToolCall:
    """Detailed MCP tool call information."""

    tool_name: str
    call_id: str = ""
    status: MCPToolCallStatus = MCPToolCallStatus.RUNNING
    input_args: Dict[str, Any] = field(default_factory=dict)
    output: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_ms: Optional[float] = None


@dataclass
class SABERUISessionInfo:
    """UI-specific session information."""

    session_id: str
    total_tasks: int = 0
    parallel_episodes: int = 1
    ui_tool_detail_level: str = "full"

    # Optional fields for full session details
    client_id: str = "saber-client"
    start_time: Optional[datetime] = None
    server_url: str = "http://localhost:8000"
    mcp_url: str = "http://localhost:8001"
    tasks: List[TaskInfo] = field(default_factory=list)
    total_episodes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SABERUITaskInfo:
    """Enhanced task information for UI."""

    # Core fields used by harness
    task_id: str
    attempt: int
    state: SABERTaskState
    episode_id: Optional[str] = None
    progress_message: Optional[str] = None

    # UI-specific enhancements
    estimated_steps: int = 10
    container_status: Optional[str] = None
    agent_id: Optional[str] = None
    start_time: Optional[datetime] = None


@dataclass
class SABERUIProgressUpdate:
    """Real-time progress update."""

    task_id: str
    episode_id: str
    current_step: int
    total_steps: int
    progress_percentage: float
    current_action: str  # e.g., "Executing tool: file_read"
    container_status: str  # e.g., "running", "completed"

    # MCP tool call tracking
    recent_tool_calls: List[MCPToolCall] = field(default_factory=list)
    active_tool_call: Optional[MCPToolCall] = None

    # Performance metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SABERUITaskResult:
    """Task completion result with UI enhancements."""

    # Core result data
    task_id: str
    episode_id: str
    attempt: int
    success: bool
    final_state: SABERTaskState
    termination_reason: Optional[str] = None
    iterations: int = 0
    error_message: Optional[str] = None
    flag: Optional[str] = None

    # UI-specific data
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # MCP tool call summary
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    tool_call_history: List[MCPToolCall] = field(default_factory=list)

    # Final metrics and logs
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    container_logs: Optional[str] = None


@dataclass
class SABERUISessionSummary:
    """Session completion summary."""

    session_id: str
    total_episodes: int
    successful_episodes: int
    final_success: bool = True
    completion_time: Optional[datetime] = None

    # Extended fields for detailed reporting
    total_tasks: int = 0
    failed_tasks: int = 0
    total_duration_seconds: float = 0.0
    total_tool_calls: int = 0
    tool_call_success_rate: float = 100.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SABERUIAdapter(Protocol):
    """Clean UI adapter interface - no inspect-ai dependencies."""

    async def session_start(self, session_info: SABERUISessionInfo) -> None:
        """Initialize UI for a new session."""
        ...

    async def session_update(self, session_info: SABERUISessionInfo) -> None:
        """Update session information (e.g., task count)."""
        ...

    async def task_start(self, task_info: SABERUITaskInfo) -> None:
        """Start tracking a new task."""
        ...

    async def task_update(self, task_info: SABERUITaskInfo) -> None:
        """Update progress for an active task."""
        ...

    async def tool_call_start(self, tool_call: MCPToolCall) -> None:
        """Report MCP tool call initiation."""
        ...

    async def tool_call_complete(self, tool_call: MCPToolCall) -> None:
        """Report MCP tool call completion."""
        ...

    async def task_complete(self, result: SABERUITaskResult) -> None:
        """Mark task as completed."""
        ...

    async def session_complete(self, summary: SABERUISessionSummary) -> None:
        """Finalize the session and show summary."""
        ...

    async def session_cleanup(self) -> None:
        """Clean up session resources."""
        ...

    async def __aenter__(self) -> "SABERUIAdapter":
        """Async context manager entry."""
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Async context manager exit."""
        ...


@runtime_checkable
class MCPToolMonitor(Protocol):
    """Interface for monitoring MCP tool calls."""

    async def register_session(self, session_id: str, callback: Callable[..., Any]) -> None:
        """Register to monitor tool calls for a session."""
        ...

    async def unregister_session(self, session_id: str) -> None:
        """Stop monitoring a session."""
        ...
