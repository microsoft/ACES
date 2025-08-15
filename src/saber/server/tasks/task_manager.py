"""TaskManager implementation for task management system."""

from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List

from .exceptions import SubTaskNotFoundException, TaskNotFoundException
from .subtask import SubTask
from .task import Task
from .task_config_loader import TaskConfigLoader

logger = getLogger(__name__)


class TaskManager:
    """
    Domain task definition manager.

    Responsible for loading and providing access to task definitions from YAML configuration.
    """

    def __init__(self, domain: str, config_dir: str):
        """
        Initialize TaskManager with domain and configuration.

        Args:
            domain: Security domain for tasks (e.g., 'malware_classification')
            config_dir: Directory path containing task configuration files
        """
        self.domain = domain
        self.config_dir = Path(config_dir)
        self.tasks_file_path = self.config_dir / "tasks.yaml"
        self.tasks: Dict[str, Task] = {}

        # Initialize specialized components
        self.config_loader = TaskConfigLoader(domain)

        logger.info(f"Initializing TaskManager for domain '{domain}' with config dir: {config_dir}")
        logger.info(f"Tasks file: {self.tasks_file_path}")

        # Load tasks using the config loader
        self.load_tasks_from_yaml()

    def load_tasks_from_yaml(self) -> None:
        """
        Load and parse YAML task definitions using TaskConfigLoader.

        Raises:
            InvalidTaskDefinitionException: If YAML is invalid or malformed
        """
        self.tasks = self.config_loader.load_tasks_from_file(str(self.tasks_file_path))
        logger.info(f"TaskManager initialization complete. Loaded {len(self.tasks)} tasks for domain '{self.domain}'")

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
