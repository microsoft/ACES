"""Enums and data classes for the task management system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(Enum):
    """Status values for tasks and subtasks."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EpisodeState(Enum):
    """States for RL training episodes."""

    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RESET = "reset"


class Observation(BaseModel):
    """RL observation data structure for episode state."""

    task_id: str = Field(..., description="ID of the current task")
    episode_id: str = Field(..., description="ID of the current episode")
    current_subtask: Optional[str] = Field(None, description="Currently active subtask ID")
    completed_subtasks: List[str] = Field(default_factory=list, description="List of completed subtask IDs")
    in_progress_subtasks: List[str] = Field(default_factory=list, description="List of in-progress subtask IDs")
    not_visited_subtasks: List[str] = Field(default_factory=list, description="List of not-yet-visited subtask IDs")
    total_subtasks: int = Field(..., description="Total number of subtasks in the task")
    completion_percentage: float = Field(..., description="Task completion percentage (0.0-1.0)")
    total_steps: int = Field(..., description="Total number of steps taken in this episode")
    context: Dict[str, Any] = Field(default_factory=dict, description="Episode context data")
    last_action: Optional[Dict[str, Any]] = Field(None, description="Information about the last action taken")
    task_description: str = Field(..., description="Description of the current task")
    current_objective: Optional[str] = Field(None, description="Current objective based on active subtasks")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
