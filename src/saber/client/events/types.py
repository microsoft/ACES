"""
Event Types and Models for SABER Client Event Bus

Defines strongly-typed event models with fail-fast validation.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict


class EventType(str, Enum):
    """Event types for SABER client event bus."""

    # Session lifecycle events
    SESSION_STARTED = "session.started"
    SESSION_COMPLETED = "session.completed"

    # Task lifecycle events
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"

    # Tool execution events (from server SSE)
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"

    # Error events
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class SaberEvent:
    """
    Immutable event container for SABER client event bus.

    Uses slots and frozen for performance and immutability.
    Validates required fields on construction with fail-fast semantics.
    """

    event_type: EventType
    session_id: str
    correlation_id: str  # task_id, call_id, or session_id
    payload: Dict[str, Any]
    timestamp: datetime

    def __post_init__(self) -> None:
        """
        Post-initialization validation with fail-fast semantics.

        Raises:
            ValueError: If any required field is missing or invalid
        """
        # FAIL FAST: Validate required fields
        if not isinstance(self.event_type, EventType):
            raise ValueError(f"event_type must be EventType enum, got: {type(self.event_type)}")

        if not self.session_id or not isinstance(self.session_id, str):
            raise ValueError("session_id must be a non-empty string")

        if not self.correlation_id or not isinstance(self.correlation_id, str):
            raise ValueError("correlation_id must be a non-empty string")

        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dictionary")

        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SaberEvent":
        """
        Create event from dictionary with validation.

        Args:
            data: Event data dictionary

        Returns:
            SaberEvent instance

        Raises:
            ValueError: If data is invalid or missing required fields
        """
        try:
            return cls(
                event_type=EventType(data["event_type"]),
                session_id=data["session_id"],
                correlation_id=data["correlation_id"],
                payload=data["payload"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid event data: {e}")
