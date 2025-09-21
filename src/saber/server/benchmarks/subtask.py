"""SubTask implementation for task management system."""

from typing import Optional

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """
    Represents an individual step within a task.

    Simplified to be purely informational - no progression logic,
    entry/exit criteria, or completion conditions.
    """

    subtask_id: str = Field(..., description="Unique identifier for the subtask")
    task_id: str = Field(..., description="ID of the parent task")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Detailed description of the subtask")
    objective: str = Field(..., description="Primary objective to accomplish")
    hint: Optional[str] = Field(None, description="Optional hint to guide without spoiling the challenge")

    def is_dependent_on(self, subtask_id: str) -> bool:
        """
        Check if this subtask depends on another subtask.

        Args:
            subtask_id: The ID of the subtask to check dependency for

        Returns:
            Always False since dependencies are removed
        """
        return False
