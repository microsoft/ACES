"""Base classes for the task management system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field


class EpisodeState(Enum):
    """States for RL training episodes."""

    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RESET = "reset"


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
    current_subtask: Optional[str] = Field(None, description="Current active subtask ID")
    completed_subtasks: Set[str] = Field(default_factory=set, description="Set of completed subtask IDs")
    in_progress_subtasks: Set[str] = Field(default_factory=set, description="Set of subtasks currently in progress")
    not_visited_subtasks: Set[str] = Field(default_factory=set, description="Set of subtasks not yet started")
    context_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Context state at this step")
    done: bool = Field(False, description="Whether the episode ended after this step")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
