"""Exceptions for the task management system."""

from typing import Optional


class TaskManagerException(Exception):
    """Base exception for task manager errors."""

    pass


class TaskNotFoundException(TaskManagerException):
    """Raised when a task ID is not found."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Task not found: {task_id}")


class SubTaskNotFoundException(TaskManagerException):
    """Raised when a subtask ID is not found."""

    def __init__(self, task_id: str, subtask_id: str):
        self.task_id = task_id
        self.subtask_id = subtask_id
        super().__init__(f"SubTask not found: {subtask_id} in task {task_id}")


class InvalidTaskDefinitionException(TaskManagerException):
    """Raised when YAML task definition is invalid."""

    def __init__(self, message: str, file_path: Optional[str] = None):
        self.file_path = file_path
        if file_path:
            super().__init__(f"Invalid task definition in {file_path}: {message}")
        else:
            super().__init__(f"Invalid task definition: {message}")


class EpisodeNotFoundException(TaskManagerException):
    """Raised when an episode is not found."""

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        super().__init__(f"Episode not found: {episode_id}")
