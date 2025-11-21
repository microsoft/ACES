"""SubTask implementation for task management system."""

from typing import Optional

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
    hint: Optional[str] = Field(None, description="Optional hint to guide without spoiling the challenge")
    subtask_strategy: Optional[str] = Field(
        None, description="Optional evaluation strategy override for this subtask (see StepEvaluationStrategy enum: 'static', 'llm_judge', 'tool_call')"
    )
    subtask_criteria: Optional[dict] = Field(
        None,
        description="Optional evaluation criteria specific to this subtask, used with certain strategies",
    )
    subtask_weight: float = Field(
        1.0,
        description="Weight of this subtask when calculating total score. Default is 1.0.",
        ge=0.0,
    )
    subtask_max_score: float = Field(
        0.0,
        description="Maximum score for this subtask. When completed, adds this value to total score.",
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
