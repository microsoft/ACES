"""SubTask implementation for task management system."""

from typing import List, Optional

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """
    Represents an individual step within a task.

    Simplified to be purely informational - no progression logic,
    entry/exit criteria, or completion conditions.

    Scoring: When a subtask appears in STEP_EVALUATIONS and has max_score > 0,
    its score is added to the total. Use max_score = 0.0 for tracking-only subtasks.
    """

    subtask_id: str = Field(..., description="Unique identifier for the subtask")
    task_id: str = Field(..., description="ID of the parent task")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Detailed description of the subtask")
    objective: str = Field(..., description="Primary objective to accomplish")
    hints: Optional[List[str]] = Field(None, description="Optional hints to guide without spoiling the challenge")
    max_score: float = Field(
        0.0,
        description="Maximum score for this subtask. When completed, adds this value to total score.",
        ge=0.0,
    )
    weight: float = Field(
        1.0,
        description="Weight for this subtask in graded scoring. Used to calculate weighted_subtask_score.",
        ge=0.0,
    )

    def is_dependent_on(self, subtask_id: str) -> bool:
        """
        Check if this subtask depends on another subtask.

        Args:
            subtask_id: The ID of the subtask to check dependency for

        Returns:
            Always False since dependencies are removed
        """
        return False
