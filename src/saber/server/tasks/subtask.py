"""SubTask implementation for task management system."""

from typing import Any, Dict, List, Set

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """
    Represents an individual step within a domain task.

    Each subtask has specific objectives, required tools, success criteria,
    and can depend on other subtasks for execution order and context.
    """

    subtask_id: str = Field(..., description="Unique identifier for the subtask")
    task_id: str = Field(..., description="ID of the parent task")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Detailed description of the subtask")
    objective: str = Field(..., description="Primary objective to accomplish")
    required_tools: List[str] = Field(
        default_factory=list, description="Tools required for execution"
    )
    success_criteria: List[str] = Field(
        default_factory=list, description="Criteria for successful completion"
    )
    context_dependencies: List[str] = Field(
        default_factory=list,
        description="Context keys needed from other subtasks (e.g., 'static_analysis.file_type')",
    )
    depends_on: List[str] = Field(
        default_factory=list, description="Subtask IDs that must be completed before this one"
    )

    def is_dependent_on(self, subtask_id: str) -> bool:
        """
        Check if this subtask depends on another subtask.

        Args:
            subtask_id: The ID of the subtask to check dependency for

        Returns:
            True if this subtask depends on the given subtask
        """
        return subtask_id in self.depends_on

    def get_context(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract required context for this subtask from the session context.

        Args:
            session_context: The full session context dictionary

        Returns:
            Dictionary containing only the context needed for this subtask
        """
        subtask_context = {}

        for dependency in self.context_dependencies:
            # Handle nested context keys like "static_analysis.file_type"
            keys = dependency.split(".")
            current_value = session_context

            try:
                for key in keys:
                    current_value = current_value[key]
                subtask_context[dependency] = current_value
            except (KeyError, TypeError):
                # Context dependency not available - this might be acceptable
                # depending on the subtask's requirements
                pass

        return subtask_context

    def validate_dependencies(self, completed_subtasks: Set[str]) -> bool:
        """
        Validate that all required subtask dependencies have been completed.

        Args:
            completed_subtasks: Set of subtask IDs that have been completed

        Returns:
            True if all dependencies are met, False otherwise
        """
        return all(dep_id in completed_subtasks for dep_id in self.depends_on)

    def get_missing_dependencies(self, completed_subtasks: Set[str]) -> List[str]:
        """
        Get list of dependencies that are not yet completed.

        Args:
            completed_subtasks: Set of subtask IDs that have been completed

        Returns:
            List of subtask IDs that are required but not completed
        """
        return [dep_id for dep_id in self.depends_on if dep_id not in completed_subtasks]

    def can_execute(self, completed_subtasks: Set[str]) -> bool:
        """
        Check if this subtask can be executed given the current completion state.

        Args:
            completed_subtasks: Set of subtask IDs that have been completed

        Returns:
            True if this subtask can be executed now
        """
        return self.validate_dependencies(completed_subtasks)

    class Config:
        """Pydantic configuration."""

        # Configuration options can be added here as needed
        pass
