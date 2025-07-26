"""SubTask implementation for task management system."""

from typing import List

from pydantic import BaseModel, Field

from ..episodes.episode import Episode


class SubTask(BaseModel):
    """
    Represents an individual step within a task.

    Each subtask acts as an internal checkpoint with specific objectives,
    completion conditions (exact commands), and dependency requirements.
    Note: Refactored for automatic progression - no longer agent-facing.
    """

    subtask_id: str = Field(..., description="Unique identifier for the subtask")
    task_id: str = Field(..., description="ID of the parent task")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Detailed description of the subtask")
    objective: str = Field(..., description="Primary objective to accomplish")
    completion_conditions: List[str] = Field(
        default_factory=list, description="Exact commands that must be executed for completion"
    )
    depends_on: List[str] = Field(
        default_factory=list, description="Subtask IDs that must be completed before this one"
    )

    def check_entry_conditions(self, episode: "Episode") -> bool:
        """
        Check if all parent subtasks are completed (entry conditions).

        Args:
            episode: Current episode state

        Returns:
            True if all dependencies are satisfied and subtask can be started
        """
        return all(dep in episode.completed_subtasks for dep in self.depends_on)

    def check_exit_conditions(self, episode: "Episode") -> bool:
        """
        Check if all required commands were executed (exit conditions).

        Args:
            episode: Current episode state

        Returns:
            True if all completion condition commands were executed
        """
        # Get all commands executed during this episode from steps
        executed_commands = [
            step.action.command for step in episode.steps if hasattr(step.action, "command") and step.action.command
        ]

        # Check if all completion condition commands were executed
        for required_command in self.completion_conditions:
            # Support command templates with variables (e.g., "${sample_path}")
            if not self._command_was_executed(required_command, executed_commands):
                return False

        return True

    def _command_was_executed(self, required_command: str, executed_commands: List[str]) -> bool:
        """
        Check if a required command was executed, supporting template variables.

        Args:
            required_command: Command that should have been executed
            executed_commands: List of commands that were actually executed

        Returns:
            True if the required command was executed
        """
        # Simple implementation - could be enhanced for template matching
        # For now, check if the base command (before first space) was used
        base_command = required_command.split()[0].replace("${", "").replace("}", "")
        return any(base_command in cmd for cmd in executed_commands)

    def is_dependent_on(self, subtask_id: str) -> bool:
        """
        Check if this subtask depends on another subtask.

        Args:
            subtask_id: The ID of the subtask to check dependency for

        Returns:
            True if this subtask depends on the given subtask
        """
        return subtask_id in self.depends_on

    class Config:
        """Pydantic configuration."""

        # Configuration options can be added here as needed
        pass
