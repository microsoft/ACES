"""Episode implementation for RL-friendly task execution."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .episode_manager import EpisodeState


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
    def current_subtask(self) -> Optional[str]:
        """Get the current active subtask from the latest step."""
        if self.steps:
            return self.steps[-1].current_subtask
        return None

    @property
    def completed_subtasks(self) -> Set[str]:
        """Get completed subtasks from the latest step."""
        if self.steps:
            return self.steps[-1].completed_subtasks
        return set()

    @property
    def in_progress_subtasks(self) -> Set[str]:
        """Get in-progress subtasks from the latest step."""
        if self.steps:
            return self.steps[-1].in_progress_subtasks
        return set()

    @property
    def not_visited_subtasks(self) -> Set[str]:
        """Get not-visited subtasks from the latest step."""
        if self.steps:
            return self.steps[-1].not_visited_subtasks
        return set()

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
        step.step_number = len(self.steps) + 1
        self.steps.append(step)

    def get_executed_commands(self) -> List[str]:
        """Get all commands executed during this episode."""
        commands = []
        for step in self.steps:
            if step.action.command:
                commands.append(step.action.command)
        return commands
