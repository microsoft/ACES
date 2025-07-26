"""TaskConfigLoader for loading and parsing YAML tas            if not tasks_path.exists():
    logger.error(f"Tasks file not found: {tasks_path}")
    raise InvalidTaskDefinitionException(f"Tasks file not found: {tasks_path}", str(tasks_path))

with open(tasks_path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)
    logger.debug(f"Successfully loaded YAML data from {tasks_path}")

if not isinstance(data, dict):
    logger.error("YAML root is not a dictionary")
    raise InvalidTaskDefinitionException("YAML root must be a dictionary", str(tasks_path))ns."""

from logging import getLogger
from pathlib import Path
from typing import Any, Dict

import yaml

from ..exceptions import InvalidTaskDefinitionException
from .subtask import SubTask
from .task import Task

logger = getLogger(__name__)


class TaskConfigLoader:
    """
    Handles loading and parsing YAML task definitions into Task objects.

    Responsible for:
    - Loading YAML files
    - Parsing task and subtask definitions
    - Validating task dependencies
    - Creating Task and SubTask objects
    """

    def __init__(self, domain: str):
        """
        Initialize TaskConfigLoader for a specific domain.

        Args:
            domain: The security domain (e.g., 'malware_classification')
        """
        self.domain = domain

    def load_tasks_from_file(self, tasks_file_path: str) -> Dict[str, Task]:
        """
        Load and parse YAML task definitions from file into Task objects.

        Args:
            tasks_file_path: Path to the YAML tasks definition file

        Returns:
            Dictionary mapping task_id to Task objects

        Raises:
            InvalidTaskDefinitionException: If YAML is invalid or malformed
        """
        tasks_path = Path(tasks_file_path)
        logger.info(f"Loading tasks from YAML file: {tasks_path}")

        try:
            if not tasks_path.exists():
                logger.error(f"Tasks file not found: {tasks_path}")
                raise InvalidTaskDefinitionException(f"Tasks file not found: {tasks_path}", str(tasks_path))

            with open(tasks_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
                logger.debug(f"Successfully loaded YAML data from {tasks_path}")

            if not isinstance(data, dict):
                logger.error("YAML root is not a dictionary")
                raise InvalidTaskDefinitionException("YAML root must be a dictionary", str(tasks_path))

            # Validate domain consistency
            yaml_domain = data.get("domain")
            if yaml_domain != self.domain:
                logger.error(f"Domain mismatch: expected '{self.domain}', got '{yaml_domain}'")
                raise InvalidTaskDefinitionException(
                    f"Domain mismatch: expected '{self.domain}', got '{yaml_domain}'",
                    str(tasks_path),
                )

            # Parse tasks
            tasks_data = data.get("tasks", [])
            if not isinstance(tasks_data, list):
                logger.error("Tasks field is not a list")
                raise InvalidTaskDefinitionException("Tasks must be a list", str(tasks_path))

            tasks = {}
            logger.info(f"Found {len(tasks_data)} tasks to load")

            for i, task_data in enumerate(tasks_data):
                logger.debug(f"Parsing task {i+1}/{len(tasks_data)}")
                task = self._parse_task(task_data)
                tasks[task.task_id] = task
                logger.info(f"Successfully loaded task '{task.task_id}' " f"with {len(task.subtasks)} subtasks")

            # Validate all task dependencies
            logger.info("Validating task dependencies")
            self._validate_all_dependencies(tasks)
            logger.info(f"TaskConfigLoader completed. Loaded {len(tasks)} tasks " f"for domain '{self.domain}'")

            return tasks

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise InvalidTaskDefinitionException(f"YAML parsing error: {e}", str(tasks_path))
        except Exception as e:
            if isinstance(e, InvalidTaskDefinitionException):
                raise
            logger.error(f"Unexpected error loading tasks: {e}")
            raise InvalidTaskDefinitionException(f"Error loading tasks: {e}", str(tasks_path))

    def _parse_task(self, task_data: Dict[str, Any]) -> Task:
        """
        Parse a single task from YAML data.

        Args:
            task_data: Dictionary containing task definition

        Returns:
            Task instance
        """
        required_fields = ["task_id", "title", "description"]
        for field in required_fields:
            if field not in task_data:
                logger.error(f"Missing required field '{field}' in task definition")
                raise InvalidTaskDefinitionException(f"Missing required field: {field}")

        task_id = task_data["task_id"]
        title = task_data["title"]
        description = task_data["description"]
        initial_context = task_data.get("initial_context", {})

        logger.debug(f"Parsing task '{task_id}': {title}")

        # Parse subtasks
        subtasks_data = task_data.get("subtasks", [])
        subtasks = []

        logger.debug(f"Task '{task_id}' has {len(subtasks_data)} subtasks")
        for j, subtask_data in enumerate(subtasks_data):
            logger.debug(f"Parsing subtask {j+1}/{len(subtasks_data)} for task '{task_id}'")
            subtask = self._parse_subtask(subtask_data, task_id)
            subtasks.append(subtask)
            logger.debug(f"Successfully parsed subtask '{subtask.subtask_id}'")

        task = Task(
            task_id=task_id,
            domain=self.domain,
            title=title,
            description=description,
            subtasks=subtasks,
            initial_context=initial_context,
        )

        logger.debug(f"Created task '{task_id}' with {len(subtasks)} subtasks")
        return task

    def _parse_subtask(self, subtask_data: Dict[str, Any], task_id: str) -> SubTask:
        """
        Parse a single subtask from YAML data.

        Args:
            subtask_data: Dictionary containing subtask definition
            task_id: ID of the parent task

        Returns:
            SubTask instance
        """
        required_fields = ["subtask_id", "title", "description", "objective"]
        for field in required_fields:
            if field not in subtask_data:
                raise InvalidTaskDefinitionException(f"Missing required subtask field: {field}")

        return SubTask(
            subtask_id=subtask_data["subtask_id"],
            task_id=task_id,
            title=subtask_data["title"],
            description=subtask_data["description"],
            objective=subtask_data["objective"],
            completion_conditions=subtask_data.get("completion_conditions", []),  # New field
            depends_on=subtask_data.get("depends_on", []),
        )

    def _validate_all_dependencies(self, tasks: Dict[str, Task]) -> None:
        """
        Validate dependencies across all tasks.

        Args:
            tasks: Dictionary of tasks to validate

        Raises:
            InvalidTaskDefinitionException: If any dependencies are invalid
        """
        all_errors = []

        for task in tasks.values():
            errors = task.validate_dependencies()
            all_errors.extend(errors)

        if all_errors:
            raise InvalidTaskDefinitionException(f"Dependency validation errors: {'; '.join(all_errors)}")
