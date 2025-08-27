"""Constants and enums for benchmark management."""

from enum import Enum


class BenchmarkStatus(str, Enum):
    """Enumeration of benchmark session statuses."""

    READY = "ready"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class BenchmarkResponseKeys(str, Enum):
    """Standard keys used in benchmark response dictionaries."""

    # Session info
    SESSION_ID = "session_id"
    DOMAIN = "domain"

    # Task info
    FIRST_TASK_ID = "first_task_id"
    CURRENT_TASK_ID = "current_task_id"
    CURRENT_EPISODE_ID = "current_episode_id"

    # Task lists
    TASKS = "tasks"
    TOTAL_TASKS = "total_tasks"
    COMPLETED_TASKS = "completed_tasks"
    FAILED_TASKS = "failed_tasks"
    REMAINING_TASKS = "remaining_tasks"

    # Configuration
    BENCHMARK_CONFIG = "benchmark_config"
    STATUS = "status"
