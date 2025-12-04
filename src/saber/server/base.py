"""
SABER API Models - Shared data models for client-server communication.

These models define the API contract between SABER clients and servers.
They should be used by both client and server implementations to ensure
consistent data structures.
"""

import asyncio
import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import EvalSubmission for Episode model
from ..models.core import EvalSubmission
from .time_source import utc_now as _utc_now


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
    reasoning: Optional[str] = Field(
        None, description="Agent's reasoning before taking this action (from reasoning models)"
    )
    assistant_message: Optional[str] = Field(
        None, description="Agent's full message before taking this action (planning/explanation)"
    )
    timestamp: datetime = Field(default_factory=_utc_now, description="When the action was initiated")


class Step(BaseModel):
    """Represents a complete action-response cycle within an episode."""

    step_number: int = Field(..., description="Sequential number of this step in the episode")
    timestamp: datetime = Field(default_factory=_utc_now, description="When the step was completed")
    action: Action = Field(..., description="The action that was taken")
    response: Dict[str, Any] = Field(..., description="Tool execution result")
    context_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Context state at this step")
    done: bool = Field(False, description="Whether the episode ended after this step")


class EpisodeState(Enum):
    """States for RL training episodes."""

    CREATING = "creating"  # Episode requested but environment not ready
    CREATED = "created"  # Episode created but not started
    READY = "ready"  # Episode environment ready for execution
    ACTIVE = "active"  # Episode actively executing
    COMPLETED = "completed"
    FAILED = "failed"
    FAILED_CREATION = "failed_creation"  # Episode creation/initialization failed
    TIMEOUT = "timeout"
    RESET = "reset"


class Episode(BaseModel):
    """Represents a complete attempt at executing a task (RL episode)."""

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique episode identifier")
    task_id: str = Field(..., description="ID of the task being attempted")
    session_id: str = Field(..., description="ID of the session this episode belongs to")
    start_time: datetime = Field(default_factory=_utc_now, description="When the episode started")
    end_time: Optional[datetime] = Field(None, description="When the episode ended")
    state: EpisodeState = Field(default=EpisodeState.CREATED, description="Current episode state")
    steps: List[Step] = Field(default_factory=list, description="Complete history of all steps taken")
    context: Dict[str, Any] = Field(default_factory=dict, description="Episode context data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional episode metadata")
    completion_reason: Optional[str] = Field(None, description="Reason the episode ended")
    creation_error: Optional[str] = Field(None, description="Error message if episode creation failed")
    max_steps: int = Field(default=10, description="Maximum number of steps allowed for this episode")
    submission: Optional[str] = Field(None, description="Final submission for evaluation")
    eval_submission: Optional[EvalSubmission] = Field(None, description="Rich evaluation submission data")

    # Episode dependency tracking fields
    attached_to_episode_id: Optional[str] = Field(None, description="Episode ID this episode is attached to")
    attached_episode_ids: List[str] = Field(default_factory=list, description="Episode IDs attached to this episode")

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Initialize lock as private attribute after model construction
        object.__setattr__(self, "_context_lock", asyncio.Lock())

    # Type annotation for private attribute (used in update_context_atomic)
    _context_lock: asyncio.Lock

    @property
    def is_complete(self) -> bool:
        """Check if the episode is complete."""
        return self.state in [
            EpisodeState.COMPLETED,
            EpisodeState.FAILED,
            EpisodeState.FAILED_CREATION,
            EpisodeState.TIMEOUT,
        ]

    @property
    def is_ready(self) -> bool:
        """Check if the episode is ready for execution."""
        return self.state in [EpisodeState.READY, EpisodeState.ACTIVE]

    @property
    def duration(self) -> Optional[float]:
        """Get episode duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def add_step(self, step: Step) -> None:
        """Add a step to the episode history."""
        self.steps.append(step)

    async def update_context_atomic(self, updates: Dict[str, Any]) -> None:
        """Atomically update multiple context keys (async-safe).

        This method ensures thread-safe updates to the episode context dictionary
        when multiple concurrent requests may modify the same episode.

        Args:
            updates: Dictionary of key-value pairs to update in context
        """
        async with self._context_lock:
            self.context.update(updates)

    def get_executed_commands(self) -> List[str]:
        """Get all arguments executed during this episode."""
        commands = []
        for step in self.steps:
            if step.action.parameters.get("arguments"):
                commands.append(step.action.parameters["arguments"])
        return commands

    def attach_to_episode(self, target_episode_id: str) -> None:
        """Attach this episode to another episode."""
        self.attached_to_episode_id = target_episode_id

    def add_attached_episode(self, episode_id: str) -> None:
        """Add an episode ID to the list of episodes attached to this one."""
        if episode_id not in self.attached_episode_ids:
            self.attached_episode_ids.append(episode_id)

    def has_attached_episode_with_task(self, task_id: str, episodes_by_id: Dict[str, "Episode"]) -> bool:
        """Check if this episode already has an attached episode with the specified task_id."""
        for attached_episode_id in self.attached_episode_ids:
            attached_episode = episodes_by_id.get(attached_episode_id)
            if attached_episode and attached_episode.task_id == task_id:
                return True
        return False

    def with_step_range(self, start_index: int, end_index: int) -> "Episode":
        """
        Return a shallow copy of this episode with a subset of steps.

        Uses Python's copy.copy() for efficiency - all attributes reference
        the same objects except steps which is sliced.

        Args:
            start_index: Starting index (inclusive, 0-based)
            end_index: Ending index (exclusive, like Python slicing)

        Returns:
            Episode instance with filtered steps

        Example:
            # Get episode with only steps 0-19
            chunk_view = episode.with_step_range(0, 20)
        """
        view = copy.copy(self)  # Shallow copy
        view.steps = self.steps[start_index:end_index]
        return view
