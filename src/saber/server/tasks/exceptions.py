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


class DependencyNotMetException(TaskManagerException):
    """Raised when required subtask dependencies are not completed."""

    def __init__(self, subtask_id: str, missing_dependencies: list[str]):
        self.subtask_id = subtask_id
        self.missing_dependencies = missing_dependencies
        super().__init__(
            f"SubTask {subtask_id} has unmet dependencies: {', '.join(missing_dependencies)}"
        )


class InvalidTaskDefinitionException(TaskManagerException):
    """Raised when YAML task definition is invalid."""

    def __init__(self, message: str, file_path: Optional[str] = None):
        self.file_path = file_path
        if file_path:
            super().__init__(f"Invalid task definition in {file_path}: {message}")
        else:
            super().__init__(f"Invalid task definition: {message}")


class SessionNotFoundException(TaskManagerException):
    """Raised when a task session is not found."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Task session not found: {session_id}")


class SessionStateException(TaskManagerException):
    """Raised when an operation is invalid for the current session state."""

    def __init__(self, session_id: str, current_state: str, required_state: Optional[str] = None):
        self.session_id = session_id
        self.current_state = current_state
        self.required_state = required_state
        if required_state:
            super().__init__(
                f"Session {session_id} is in state {current_state}, required: {required_state}"
            )
        else:
            super().__init__(f"Invalid operation for session {session_id} in state {current_state}")
