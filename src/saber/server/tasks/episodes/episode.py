"""Episode implementation for RL-friendly task execution."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..base import EpisodeState, Step


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
        """Get all commands executed during this episode."""
        commands = []
        for step in self.steps:
            if step.action.command:
                commands.append(step.action.command)
        return commands
