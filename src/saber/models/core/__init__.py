"""
SABER Core Models - Core business domain models.

These models represent the core business entities and their relationships.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskInfo(BaseModel):
    """Information about a single task in the benchmark."""

    task_id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Human-readable task title")
    description: str = Field(..., description="Task description")
    episode_attempts: int = Field(..., description="Number of episode attempts for this task")
    subtask_count: int = Field(..., description="Number of subtasks in this task")
    max_steps: int = Field(..., description="Maximum steps allowed per episode")
    initial_prompt: str = Field(
        default="", description="Agent initial prompt for this task"
    )  # Made optional with default


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
    """

    domain: str = Field(..., description="Security domain name")
    tasks: List[TaskInfo] = Field(..., description="Available tasks in the benchmark")
    total_tasks: int = Field(..., description="Total number of unique tasks")
    total_episodes: int = Field(..., description="Total number of episodes across all tasks")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization."""
        return {
            "domain": self.domain,
            "tasks": [task.model_dump() for task in self.tasks],
            "total_tasks": self.total_tasks,
            "total_episodes": self.total_episodes,
        }

    def get_task_by_id(self, task_id: str) -> TaskInfo | None:
        """Get task info by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_task_ids(self) -> List[str]:
        """Get list of all task IDs."""
        return [task.task_id for task in self.tasks]
