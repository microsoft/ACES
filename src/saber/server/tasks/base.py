"""Enums for the task management system."""

from enum import Enum


class TaskStatus(Enum):
    """Status values for tasks and subtasks."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EpisodeState(Enum):
    """States for RL training episodes."""

    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RESET = "reset"


class SessionState(Enum):
    """States for task execution sessions."""

    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
