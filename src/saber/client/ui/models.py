"""UI Models - Core data structures for SABER UI components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class UIBackendType(Enum):
    """Supported UI backend types."""

    INSPECT_AI = "inspect_ai"


class MessageType(Enum):
    """Types of messages that can be displayed."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    DEBUG = "debug"


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Per-step execution status for progress updates."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UIMessage:
    """A message to be displayed in the UI."""

    content: str
    message_type: MessageType
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TaskInfo:
    """Information about a task for UI display."""

    task_id: str
    name: str
    status: TaskStatus
    progress: float  # 0.0 to 1.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UIConfig:
    """Configuration for UI backends."""

    backend_type: UIBackendType
    show_progress: bool = True
    show_timestamps: bool = True
    log_level: str = "INFO"
    config: Optional[Dict[str, Any]] = None

    @classmethod
    def for_inspect_ai(cls, mode: str = "plain", **kwargs: Any) -> "UIConfig":
        """Create UIConfig for Inspect-AI backend with specific mode."""
        return cls(backend_type=UIBackendType.INSPECT_AI, config={"mode": mode, **kwargs})


@dataclass
class StepUpdate:
    """Structured step update passed to the UI for progress rendering."""

    step_id: str
    name: str
    description: str
    status: StepStatus
    progress: float  # 0.0 to 1.0
    task_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class UIBackend(ABC):
    """Abstract base class for UI backends."""

    @abstractmethod
    def display_message(self, message: UIMessage) -> None:
        """Display a message in the UI."""
        pass

    @abstractmethod
    def show_progress(self, task: TaskInfo) -> None:
        """Show task progress in the UI."""
        pass

    @abstractmethod
    def update_task_status(self, task: TaskInfo) -> None:
        """Update the status of a task in the UI."""
        pass

    @abstractmethod
    def show_results(self, results: Dict[str, Any]) -> None:
        """Show results in the UI."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up UI resources."""
        pass
