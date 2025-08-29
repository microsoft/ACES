"""BenchmarkManager implementation for task definition management."""

from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List

from .benchmark_config_loader import BenchmarkConfigLoader
from .benchmark_info import BenchmarkInfo, TaskInfo
from .exceptions import SubTaskNotFoundException, TaskNotFoundException
from .subtask import SubTask
from .task import Task

logger = getLogger(__name__)


class BenchmarkManager:
    """
    Domain task definition manager for client-side orchestration.

    Manages task definitions and configurations from YAML files.
    No longer handles server-side orchestration - that's now client responsibility.
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

    def get_benchmark_info(self) -> BenchmarkInfo:
        """
        Get complete benchmark information for client-side orchestration.

        Returns:
            BenchmarkInfo object with all tasks and their episode attempt configurations
        """
        task_infos = []
        total_episodes = 0

        for task in self.tasks.values():
            episode_attempts = task.get_episode_attempts()
            task_info = TaskInfo(
                task_id=task.task_id,
                title=task.title,
                description=task.description,
                episode_attempts=episode_attempts,
                subtask_count=len(task.subtasks),
            )
            task_infos.append(task_info)
            total_episodes += episode_attempts

        return BenchmarkInfo(
            domain=self.domain, tasks=task_infos, total_tasks=len(task_infos), total_episodes=total_episodes
        )

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
        Get a list of all available tasks (legacy method).

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
        Get a list of all tasks with their benchmark configuration (legacy method).

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
