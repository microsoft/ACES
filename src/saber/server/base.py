"""
SABER API Models - Shared data models for client-server communication.

These models define the API contract between SABER clients and servers.
They should be used by both client and server implementations to ensure
consistent data structures.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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

    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    _original_data: Optional[Dict[str, Any]] = field(default=None, init=False)

    @property
    def success(self) -> bool:
        """Check if command executed successfully."""
        return self.exit_code == 0

    @property
    def data(self) -> Dict[str, Any]:
        """Get command data as dictionary for backward compatibility."""
        # If we have original data from success_result/error_result, prefer that
        if self._original_data is not None:
            return self._original_data

        # Otherwise return the standard format
        return {"exit_code": self.exit_code, "stdout": self.stdout, "stderr": self.stderr}

    @property
    def error(self) -> Optional[str]:
        """Get error message if command failed."""
        return self.stderr if not self.success else None

    @classmethod
    def success_result(
        cls, data: Any, execution_time: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> "CommandResult":
        """Create a successful command result for backward compatibility."""
        if isinstance(data, dict):
            exit_code = data.get("exit_code", 0)
            result = cls(
                exit_code=exit_code,
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
                execution_time=execution_time or 0.0,
                metadata=metadata or {},
            )
            # Store the original data for the data property
            result._original_data = data.copy()
            return result
        else:
            # For non-dict data, preserve the original data as-is for backward compatibility
            result = cls(
                exit_code=0, stdout=str(data), stderr="", execution_time=execution_time or 0.0, metadata=metadata or {}
            )
            # For string/simple data, the data property should return the original value
            result._original_data = data
            return result

    @classmethod
    def error_result(
        cls, error: str, execution_time: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> "CommandResult":
        """Create a failed command result for backward compatibility."""
        error_data = {"exit_code": 1, "stdout": "", "stderr": error, "success": False, "error": error}

        result = cls(
            exit_code=1, stdout="", stderr=error, execution_time=execution_time or 0.0, metadata=metadata or {}
        )
        # Store the original data for the data property
        result._original_data = error_data
        return result


class Action(BaseModel):
    """Represents a single action taken by an agent during episode execution."""

    tool_name: str = Field(..., description="Name of the tool being executed")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters passed to the tool")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the action was initiated")
    arguments: Optional[str] = Field(None, description="Extracted arguments for completion matching")

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


class EpisodeState(Enum):
    """States for RL training episodes."""

    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RESET = "reset"


class Episode(BaseModel):
    """Represents a complete attempt at executing a task (RL episode)."""

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique episode identifier")
    task_id: str = Field(..., description="ID of the task being attempted")
    session_id: str = Field(..., description="ID of the session this episode belongs to")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="When the episode started")
    end_time: Optional[datetime] = Field(None, description="When the episode ended")
    state: EpisodeState = Field(default=EpisodeState.CREATED, description="Current episode state")
    steps: List[Step] = Field(default_factory=list, description="Complete history of all steps taken")
    context: Dict[str, Any] = Field(default_factory=dict, description="Episode context data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional episode metadata")
    completion_reason: Optional[str] = Field(None, description="Reason the episode ended")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    @property
    def is_complete(self) -> bool:
        """Check if the episode is complete."""
        return self.state in [EpisodeState.COMPLETED, EpisodeState.FAILED, EpisodeState.TIMEOUT]

    @property
    def duration(self) -> Optional[float]:
        """Get episode duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def add_step(self, step: Step) -> None:
        """Add a step to the episode history."""
        self.steps.append(step)

    def get_executed_commands(self) -> List[str]:
        """Get all arguments executed during this episode."""
        commands = []
        for step in self.steps:
            if step.action.arguments:
                commands.append(step.action.arguments)
        return commands
