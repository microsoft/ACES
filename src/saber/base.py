"""
SABER API Models - Shared data models for client-server communication.

These models define the API contract between SABER clients and servers.
They should be used by both client and server implementations to ensure
consistent data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepResponse(BaseModel):
    """Response from server step execution."""

    success: bool = Field(description="Command succeeded")
    output: str = Field(description="Command output to show agent")
    error: Optional[str] = Field(None, description="Error message if failed")
    done: bool = Field(description="Episode complete")
    task_completed: bool = Field(default=False, description="Current task completed")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional server info")


class TaskInfo(BaseModel):
    """Information about the current task."""

    task_id: str
    title: str
    description: str
    completed: bool = Field(default=False, description="Task completion status")
    episode_id: Optional[str] = None
    state: Optional[str] = None
    step_count: Optional[int] = None
    duration: Optional[float] = None
    subtasks: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detailed subtask information")


class PolicyInfo(BaseModel):
    """Domain policy information."""

    domain: Optional[str] = None
    available_commands: List[str]
    guidelines: str
    constraints: List[str]


class EpisodeInfo(BaseModel):
    """Episode information."""

    episode_id: str
    task_id: str
    message: Optional[str] = None


class SessionInfo(BaseModel):
    """Session information."""

    session_id: str
    message: str


@dataclass
class CommandResult:
    """Result of command execution."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls, data: Any, execution_time: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> "CommandResult":
        """Create a successful command result."""
        return cls(success=True, data=data, execution_time=execution_time, metadata=metadata or {})

    @classmethod
    def error_result(
        cls, error: str, execution_time: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> "CommandResult":
        """Create a failed command result."""
        return cls(success=False, error=error, execution_time=execution_time, metadata=metadata or {})


class Action(BaseModel):
    """Represents a single action taken by an agent during episode execution."""

    tool_name: str = Field(..., description="Name of the tool being executed")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters passed to the tool")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the action was initiated")
    command: Optional[str] = Field(None, description="Extracted command for completion matching")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class Step(BaseModel):
    """Represents a complete action-response cycle within an episode."""

    step_number: int = Field(..., description="Sequential number of this step in the episode")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the step was completed")
    action: Action = Field(..., description="The action that was taken")
    response: Dict[str, Any] = Field(..., description="Tool execution result")
    context_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Context state at this step")
    done: bool = Field(False, description="Whether the episode ended after this step")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
