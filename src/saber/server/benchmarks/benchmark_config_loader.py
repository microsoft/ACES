"""BenchmarkConfigLoader for loading and parsing YAML task definitions and benchmark configurations."""

from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .exceptions import InvalidTaskDefinitionException
from .subtask import SubTask
from .task import Task

logger = getLogger(__name__)


class BenchmarkConfigLoader:
    """
    Handles loading and parsing YAML task definitions and benchmark configurations into Task objects.
    Enhanced to support benchmark-specific configuration including episode_attempts.
    """

    def __init__(self, domain: str):
        """
        Initialize BenchmarkConfigLoader for a specific domain.

        Args:
            domain: The security domain (e.g., 'webapp_pentest')
        """
        self.domain = domain
        self.allowed_executors: Optional[list[str]] = None
        self.benchmark_config: Dict[str, Any] = {}
        self.yaml_data: Optional[Dict[str, Any]] = None

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
                self.yaml_data = yaml.safe_load(file)
                logger.debug(f"Successfully loaded YAML data from {tasks_path}")

            if not isinstance(self.yaml_data, dict):
                logger.error("YAML root is not a dictionary")
                raise InvalidTaskDefinitionException("YAML root must be a dictionary", str(tasks_path))

            # Validate domain consistency
            yaml_domain = self.yaml_data.get("domain")
            if yaml_domain != self.domain:
                logger.error(f"Domain mismatch: expected '{self.domain}', got '{yaml_domain}'")
                raise InvalidTaskDefinitionException(
                    f"Domain mismatch: expected '{self.domain}', got '{yaml_domain}'",
                    str(tasks_path),
                )

            # Parse benchmark configuration (optional)
            self._parse_benchmark_config()

            # Parse tasks
            tasks_data = self.yaml_data.get("tasks", [])
            if not isinstance(tasks_data, list):
                logger.error("Tasks field is not a list")
                raise InvalidTaskDefinitionException("Tasks must be a list", str(tasks_path))

            # Parse executors configuration (optional)
            executors_data = self.yaml_data.get("executors")
            if executors_data is not None:
                if not isinstance(executors_data, list):
                    logger.error("Executors field is not a list")
                    raise InvalidTaskDefinitionException("Executors must be a list", str(tasks_path))

                self.allowed_executors = executors_data
                logger.info(f"Loaded executor configuration: {self.allowed_executors}")
            else:
                self.allowed_executors = None
                logger.info("No executor configuration found, all executors will be available")

            tasks = {}
            logger.info(f"Found {len(tasks_data)} tasks to load")

            for i, task_data in enumerate(tasks_data):
                logger.debug(f"Parsing task {i+1}/{len(tasks_data)}")
                task = self._parse_task(task_data)
                tasks[task.task_id] = task
                logger.info(f"Successfully loaded task '{task.task_id}' " f"with {len(task.subtasks)} subtasks")

            logger.info(f"BenchmarkConfigLoader completed. Loaded {len(tasks)} tasks " f"for domain '{self.domain}'")

            return tasks

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise InvalidTaskDefinitionException(f"YAML parsing error: {e}", str(tasks_path))
        except Exception as e:
            if isinstance(e, InvalidTaskDefinitionException):
                raise
            logger.error(f"Unexpected error loading tasks: {e}")
            raise InvalidTaskDefinitionException(f"Error loading tasks: {e}", str(tasks_path))

    def get_allowed_executors(self) -> Optional[list[str]]:
        """
        Get the list of allowed executors from the configuration.

        Returns:
            List of allowed executor names, or None if no restriction is configured
        """
        return self.allowed_executors

    def load_benchmark_config(self) -> Dict[str, Any]:
        """
        Get the loaded benchmark configuration.

        Returns:
            Domain-level benchmark configuration
        """
        return self.benchmark_config.copy()

    def _parse_benchmark_config(self) -> None:
        """
        Parse benchmark configuration from YAML data.

        Requires explicit episode_attempts configuration - no defaults provided.

        Raises:
            InvalidTaskDefinitionException: If benchmark_config or episode_attempts is missing
        """
        if self.yaml_data is None:
            raise InvalidTaskDefinitionException("No YAML data loaded")

        benchmark_data = self.yaml_data.get("benchmark_config")

        if benchmark_data is None:
            raise InvalidTaskDefinitionException(
                "Missing required 'benchmark_config' section in YAML. "
                "Benchmark configuration with 'episode_attempts' is required for benchmarking."
            )

        if not isinstance(benchmark_data, dict):
            raise InvalidTaskDefinitionException("benchmark_config must be a dictionary")

        # Require explicit episode_attempts configuration
        if "episode_attempts" not in benchmark_data:
            raise InvalidTaskDefinitionException(
                "Missing required 'episode_attempts' in benchmark_config. "
                "You must explicitly specify the number of episode attempts for benchmarking."
            )

        episode_attempts = benchmark_data["episode_attempts"]
        if not isinstance(episode_attempts, int) or episode_attempts < 1:
            raise InvalidTaskDefinitionException(
                f"episode_attempts must be a positive integer, got: {episode_attempts}"
            )

        # Store benchmark configuration (no defaults)
        self.benchmark_config = benchmark_data.copy()

        logger.info(f"Loaded benchmark configuration: {self.benchmark_config}")

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

        # Get environment string (resolution happens in execution layer)
        environment = task_data.get("environment")

        # Get execution configuration
        execution_config = task_data.get("execution_config", {})

        # Get episode configuration
        episode_config = task_data.get("episode_config", {})

        # Get task-level benchmark configuration (optional, but validated)
        task_benchmark_config = task_data.get("benchmark_config", {})
        if not isinstance(task_benchmark_config, dict):
            raise InvalidTaskDefinitionException(f"Task '{task_id}' benchmark_config must be a dictionary if provided")

        # Validate task-level episode_attempts if provided
        if "episode_attempts" in task_benchmark_config:
            episode_attempts = task_benchmark_config["episode_attempts"]
            if not isinstance(episode_attempts, int) or episode_attempts < 1:
                raise InvalidTaskDefinitionException(
                    f"Task '{task_id}' episode_attempts must be a positive integer, got: {episode_attempts}"
                )

        # Merge domain-level and task-level benchmark config (task-level takes precedence)
        merged_benchmark_config = {**self.benchmark_config, **task_benchmark_config}

        # Ensure final config has valid episode_attempts
        if "episode_attempts" not in merged_benchmark_config or merged_benchmark_config["episode_attempts"] < 1:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}' does not have valid episode_attempts configuration. "
                "Each task must have episode_attempts either from domain-level benchmark_config or task-level override."
            )

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
            environment=environment,
            allowed_executors=self.allowed_executors,
            execution_config=execution_config,
            episode_config=episode_config,
            benchmark_config=merged_benchmark_config,
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
        )
