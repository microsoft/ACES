"""BenchmarkManager implementation for benchmark orchestration and task management."""

from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark_config_loader import BenchmarkConfigLoader
from .constants import BenchmarkResponseKeys, BenchmarkStatus
from .exceptions import SubTaskNotFoundException, TaskNotFoundException
from .subtask import SubTask
from .task import Task

logger = getLogger(__name__)


class BenchmarkSession:
    """Represents an active benchmark session with task queue management."""

    def __init__(self, session_id: str, benchmark_config: Dict[str, Any], tasks: List[Dict[str, Any]]):
        self.session_id = session_id
        self.benchmark_config = benchmark_config
        self.tasks = tasks
        self.task_queue = [task["task_id"] for task in tasks]
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self.current_task_id: Optional[str] = None
        self.current_episode_id: Optional[str] = None
        self.status = BenchmarkStatus.READY

    def get_next_task(self) -> Optional[str]:
        """Get the next task ID to execute, or None if benchmark is complete."""
        if self.task_queue:
            self.current_task_id = self.task_queue.pop(0)
            self.status = BenchmarkStatus.ACTIVE
            return self.current_task_id
        else:
            self.status = BenchmarkStatus.COMPLETED
            return None

    def get_first_task_id(self) -> Optional[str]:
        """Get the first task ID without consuming it from the queue."""
        return self.task_queue[0] if self.task_queue else None

    def has_first_task(self) -> bool:
        """Check if there is a first task available."""
        return len(self.task_queue) > 0

    def set_current_episode(self, episode_id: str) -> None:
        """Set the current episode ID for this benchmark session."""
        self.current_episode_id = episode_id

    def get_current_episode_id(self) -> Optional[str]:
        """Get the current episode ID."""
        return self.current_episode_id

    def complete_task(self, task_id: str, success: bool = True) -> Optional[str]:
        """
        Mark a task as completed and get the next task.

        Args:
            task_id: ID of the completed task
            success: Whether the task completed successfully

        Returns:
            Next task ID or None if benchmark is complete
        """
        if success:
            self.completed_tasks.append(task_id)
        else:
            self.failed_tasks.append(task_id)

        self.current_task_id = None
        return self.get_next_task()

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark session to dictionary representation."""
        return {
            BenchmarkResponseKeys.SESSION_ID: self.session_id,
            BenchmarkResponseKeys.BENCHMARK_CONFIG: self.benchmark_config,
            BenchmarkResponseKeys.TASKS: self.tasks,
            BenchmarkResponseKeys.TOTAL_TASKS: len(self.tasks),
            BenchmarkResponseKeys.STATUS: self.status,
            BenchmarkResponseKeys.CURRENT_TASK_ID: self.current_task_id,
            BenchmarkResponseKeys.CURRENT_EPISODE_ID: self.current_episode_id,
            BenchmarkResponseKeys.COMPLETED_TASKS: self.completed_tasks,
            BenchmarkResponseKeys.FAILED_TASKS: self.failed_tasks,
            BenchmarkResponseKeys.REMAINING_TASKS: self.task_queue,
        }

    def to_api_response(self, domain: str) -> Dict[str, Any]:
        """Convert to API response format with domain info."""
        result = self.to_dict()
        result[BenchmarkResponseKeys.DOMAIN] = domain
        if self.current_task_id:
            result[BenchmarkResponseKeys.FIRST_TASK_ID] = self.current_task_id
        return result


class BenchmarkManager:
    """
    Domain benchmark orchestration and task definition manager.

    Enhanced to support benchmark orchestration with multiple episode attempts for pass@k evaluation.
    Manages task definitions and benchmark configurations from YAML files.
    """

    def __init__(self, domain: str, config_dir: str):
        """
        Initialize BenchmarkManager with domain and configuration.

        Args:
            domain: Security domain for benchmarks (e.g., 'webapp_pentest')
            config_dir: Directory path containing task configuration files
        """
        self.domain = domain
        self.config_dir = Path(config_dir)
        self.tasks_file_path = self.config_dir / "tasks.yaml"
        self.tasks: Dict[str, Task] = {}
        self.benchmark_config: Dict[str, Any] = {}
        self.active_benchmarks: Dict[str, BenchmarkSession] = {}

        # Initialize specialized components
        self.config_loader = BenchmarkConfigLoader(domain)

        logger.info(f"Initializing BenchmarkManager for domain '{domain}' with config dir: {config_dir}")
        logger.info(f"Tasks file: {self.tasks_file_path}")

        # Load tasks and benchmark configuration
        self.load_tasks_from_yaml()

    def load_tasks_from_yaml(self) -> None:
        """
        Load and parse YAML task definitions and benchmark configuration using BenchmarkConfigLoader.

        Raises:
            InvalidTaskDefinitionException: If YAML is invalid or malformed
        """
        self.tasks = self.config_loader.load_tasks_from_file(str(self.tasks_file_path))
        self.benchmark_config = self.config_loader.load_benchmark_config()
        logger.info(
            f"BenchmarkManager initialization complete. Loaded {len(self.tasks)} tasks for domain '{self.domain}'"
        )
        logger.info(f"Benchmark config: {self.benchmark_config}")

    def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            Task instance

        Raises:
            TaskNotFoundException: If task is not found
        """
        if task_id not in self.tasks:
            raise TaskNotFoundException(task_id)
        return self.tasks[task_id]

    def get_subtask(self, task_id: str, subtask_id: str) -> SubTask:
        """
        Get a subtask by task ID and subtask ID.

        Args:
            task_id: ID of the parent task
            subtask_id: ID of the subtask

        Returns:
            SubTask instance

        Raises:
            TaskNotFoundException: If task is not found
            SubTaskNotFoundException: If subtask is not found
        """
        task = self.get_task(task_id)
        subtask = task.get_subtask_by_id(subtask_id)

        if subtask is None:
            raise SubTaskNotFoundException(task_id, subtask_id)

        return subtask

    def list_tasks(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available tasks.

        Returns:
            List of task information dictionaries
        """
        return [
            {
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "subtask_count": len(task.subtasks),
            }
            for task in self.tasks.values()
        ]

    def get_benchmark_info(self, benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get benchmark session information with specified configuration.

        Args:
            benchmark_config: Benchmark-specific configuration parameters

        Returns:
            Benchmark session information
        """
        logger.info(f"Getting benchmark info with config: {benchmark_config}")

        # Create benchmark session metadata
        benchmark_session = {
            "domain": self.domain,
            "benchmark_config": benchmark_config,
            "tasks": list(self.tasks.keys()),
            "total_tasks": len(self.tasks),
        }

        logger.info(f"Benchmark session info created for {len(self.tasks)} tasks")
        return benchmark_session

    def get_benchmark_config(self) -> Dict[str, Any]:
        """
        Get the loaded benchmark configuration.

        Returns:
            Domain-level benchmark configuration
        """
        return self.benchmark_config.copy()

    def get_episode_config(self, task_id: str) -> Dict[str, Any]:
        """
        Get episode configuration for a specific task.

        Args:
            task_id: ID of the task to get episode config for

        Returns:
            Episode configuration including max_steps, timeouts, and other settings
        """
        task = self.get_task(task_id)
        episode_config = task.episode_config or {}
        execution_config = task.execution_config or {}

        return {
            "max_steps": episode_config.get("max_steps", 100),
            "timeout_seconds": episode_config.get("timeout_seconds", execution_config.get("timeout", 300)),
            "task_id": task_id,
            "task_title": task.title,
            "task_description": task.description,
            "domain": task.domain,
            "subtask_count": len(task.subtasks),
        }

    def list_benchmark_tasks(self) -> List[Dict[str, Any]]:
        """
        Get a list of all tasks with their benchmark configuration.

        Returns:
            List of task information dictionaries including benchmark settings
        """
        return [
            {
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "subtask_count": len(task.subtasks),
                "episode_attempts": task.get_episode_attempts(),
                "benchmark_config": task.benchmark_config,
            }
            for task in self.tasks.values()
        ]

    def create_benchmark_session(
        self, session_id: str, benchmark_config: Optional[Dict[str, Any]] = None
    ) -> BenchmarkSession:
        """
        Create a new benchmark session with task queue.

        Args:
            session_id: Session ID for the benchmark
            benchmark_config: Optional benchmark configuration parameters

        Returns:
            BenchmarkSession instance
        """
        if benchmark_config is None:
            benchmark_config = {}

        # Get all available tasks for benchmarking
        tasks = self.list_benchmark_tasks()

        # Create and store benchmark session
        benchmark_session = BenchmarkSession(session_id, benchmark_config, tasks)
        self.active_benchmarks[session_id] = benchmark_session

        logger.info(f"Created benchmark session for {session_id} with {len(tasks)} tasks")
        return benchmark_session

    def get_benchmark_session(self, session_id: str) -> Optional[BenchmarkSession]:
        """Get active benchmark session for a session ID."""
        return self.active_benchmarks.get(session_id)

    def start_benchmark(self, session_id: str, benchmark_config: Optional[Dict[str, Any]] = None) -> "BenchmarkSession":
        """
        Start a benchmark session and return the BenchmarkSession object.

        Args:
            session_id: Session ID for the benchmark
            benchmark_config: Optional benchmark configuration parameters

        Returns:
            BenchmarkSession object for direct method access
        """
        benchmark_session = self.create_benchmark_session(session_id, benchmark_config)

        # Get the first task to execute
        first_task_id = benchmark_session.get_next_task()

        logger.info(f"Started benchmark for session {session_id}, first task: {first_task_id}")
        return benchmark_session

    def advance_benchmark(self, session_id: str, completed_task_id: str, success: bool = True) -> Optional[str]:
        """
        Advance benchmark to next task after completing current task.

        Args:
            session_id: Session ID
            completed_task_id: ID of the task that just completed
            success: Whether the task completed successfully

        Returns:
            Next task ID or None if benchmark is complete
        """
        benchmark_session = self.get_benchmark_session(session_id)
        if not benchmark_session:
            logger.warning(f"No active benchmark found for session {session_id}")
            return None

        next_task_id = benchmark_session.complete_task(completed_task_id, success)

        if next_task_id:
            logger.info(f"Advanced benchmark for session {session_id} to task {next_task_id}")
        else:
            logger.info(f"Benchmark completed for session {session_id}")

        return next_task_id

    def end_benchmark(self, session_id: str) -> None:
        """End and cleanup benchmark session."""
        if session_id in self.active_benchmarks:
            del self.active_benchmarks[session_id]
            logger.info(f"Ended benchmark session for {session_id}")

    def is_benchmark_active(self, session_id: str) -> bool:
        """Check if a benchmark is active for the given session."""
        return session_id in self.active_benchmarks
