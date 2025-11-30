"""
SABER Core Models - Core business domain models.

These models represent the core business entities and their relationships.
"""

from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import BaseModel, Discriminator, Field

# Import concrete BenchmarkTask types for discriminated union
from ..benchmark_task import OrchestratedTask, SingleEpisodeTask

# Import core execution constants
from .execution import CleanupReason, ClientIdentifiers, ExecutionMode, TaskInitMode


class TaskInfo(BaseModel):
    """Information about a single task in the benchmark."""

    task_id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Human-readable task title")
    description: str = Field(..., description="Task description")
    episode_attempts: int = Field(..., description="Number of episode attempts for this task")
    subtask_count: int = Field(..., description="Number of subtasks in this task")
    max_steps: int = Field(..., description="Maximum steps allowed per episode")

    # NEW: Three distinct prompts (replaces initial_prompt)
    instruction_prompt: str = Field(..., description="Agent instruction prompt")
    assistant_prompt: str = Field(..., description="Agent assistant prompt")
    submit_prompt: str = Field(..., description="Agent submit prompt")


class PolicyInfo(BaseModel):
    """Domain policy information."""

    domain: Optional[str] = None
    available_commands: List[str]
    guidelines: str
    constraints: List[str]


class BenchmarkInfo(BaseModel):
    """
    Complete benchmark information for client-side orchestration.

    This replaces the old server-side task queue management with a clean
    data structure that clients can use for their own orchestration.

    API Version 2.0: Polymorphic BenchmarkTask types only (SingleEpisodeTask, OrchestratedTask).
    Legacy TaskInfo support has been removed.
    """

    api_version: str = Field(default="2.0", description="API version for compatibility checks")
    domain: str = Field(..., description="Security domain name")
    tasks: List[Annotated[Union[SingleEpisodeTask, OrchestratedTask], Discriminator("task_type")]] = Field(
        ..., description="Available tasks in the benchmark (polymorphic BenchmarkTask)"
    )
    total_tasks: int = Field(..., description="Total number of unique tasks")
    total_episodes: int = Field(..., description="Total number of episodes across all tasks")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization."""
        return {
            "api_version": self.api_version,
            "domain": self.domain,
            "tasks": [task.model_dump() for task in self.tasks],
            "total_tasks": self.total_tasks,
            "total_episodes": self.total_episodes,
        }

    def get_task_by_id(self, task_id: str) -> Union[SingleEpisodeTask, OrchestratedTask, None]:
        """Get task info by ID.

        Args:
            task_id: Task ID to search for

        Returns:
            SingleEpisodeTask or OrchestratedTask if found, None otherwise
        """
        for task in self.tasks:
            # Check direct task_id for SingleEpisodeTask
            if hasattr(task, "task_id") and task.task_id == task_id:
                return task
            # For OrchestratedTask, check sub-tasks as well
            if task_id in task.get_task_ids():
                return task
        return None

    def get_task_ids(self) -> List[str]:
        """Get list of all task IDs.

        Returns:
            List of all task IDs (includes sub-task IDs for orchestrated tasks)
        """
        task_ids = []
        for task in self.tasks:
            # All tasks are now BenchmarkTask with polymorphic get_task_ids()
            task_ids.extend(task.get_task_ids())
        return task_ids


class EvalSubmission(BaseModel):
    """Enhanced submission model capturing rich ModelOutput data for evaluation."""

    episode_id: str = Field(..., description="Episode ID for tracking")
    task_id: str = Field(..., description="Task ID for evaluation context")
    model: str = Field(..., description="Model name from ModelOutput.model")
    choices: List[Dict[str, Any]] = Field(default_factory=list, description="Model choices from ModelOutput.choices")
    submission: str = Field(..., description="Completion text from ModelOutput.completion")
    tokens: Dict[str, Any] = Field(default_factory=dict, description="Token usage information from ModelOutput.usage")
    time: float = Field(..., description="Execution time from ModelOutput.time")


# Export all core models
__all__ = [
    "TaskInfo",
    "PolicyInfo",
    "BenchmarkInfo",
    "EvalSubmission",
    "ExecutionMode",
    "TaskInitMode",
    "ClientIdentifiers",
    "CleanupReason",
]
