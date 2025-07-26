"""TaskManager implementation for task management system."""

from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .core.subtask import SubTask
from .core.task import Task
from .episodes.episode import Action, Episode, EpisodeResult
from .episodes.episode_manager import EpisodeManager
from .exceptions import InvalidTaskDefinitionException, SubTaskNotFoundException, TaskNotFoundException

logger = getLogger(__name__)


class TaskResult:
    """Result of a completed task execution."""

    def __init__(
        self,
        session_id: str,
        task_id: str,
        success: bool,
        completion_time: datetime,
        context: Dict[str, Any],
        completed_subtasks: List[str],
    ):
        self.session_id = session_id
        self.task_id = task_id
        self.success = success
        self.completion_time = completion_time
        self.context = context
        self.completed_subtasks = completed_subtasks


class TaskManager:
    """
    Main orchestrator for domain tasks and sessions.

    Responsible for loading YAML task definitions, building task hierarchies,
    managing active sessions, and providing task/subtask navigation.
    """

    def __init__(self, domain: str, tasks_file_path: str):
        """
        Initialize TaskManager for a specific domain.

        Args:
            domain: The security domain (e.g., 'malware_classification')
            tasks_file_path: Path to the YAML tasks definition file
        """
        self.domain = domain
        self.tasks_file_path = Path(tasks_file_path)
        self.tasks: Dict[str, Task] = {}
        self.episode_manager = EpisodeManager()

        logger.info(f"Initializing TaskManager for domain '{domain}' with tasks file: {tasks_file_path}")

        # Load tasks from YAML file
        self.load_tasks_from_yaml()

    def load_tasks_from_yaml(self) -> None:
        """
        Load and parse YAML task definitions into Task objects.

        Raises:
            InvalidTaskDefinitionException: If YAML is invalid or malformed
        """
        logger.info(f"Loading tasks from YAML file: {self.tasks_file_path}")

        try:
            if not self.tasks_file_path.exists():
                logger.error(f"Tasks file not found: {self.tasks_file_path}")
                raise InvalidTaskDefinitionException(
                    f"Tasks file not found: {self.tasks_file_path}", str(self.tasks_file_path)
                )

            with open(self.tasks_file_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
                logger.debug(f"Successfully loaded YAML data from {self.tasks_file_path}")

            if not isinstance(data, dict):
                logger.error("YAML root is not a dictionary")
                raise InvalidTaskDefinitionException("YAML root must be a dictionary", str(self.tasks_file_path))

            # Validate domain consistency
            yaml_domain = data.get("domain")
            if yaml_domain != self.domain:
                logger.error(f"Domain mismatch: expected '{self.domain}', got '{yaml_domain}'")
                raise InvalidTaskDefinitionException(
                    f"Domain mismatch: expected '{self.domain}', got '{yaml_domain}'",
                    str(self.tasks_file_path),
                )

            # Parse tasks
            tasks_data = data.get("tasks", [])
            if not isinstance(tasks_data, list):
                logger.error("Tasks field is not a list")
                raise InvalidTaskDefinitionException("Tasks must be a list", str(self.tasks_file_path))

            self.tasks.clear()
            logger.info(f"Found {len(tasks_data)} tasks to load")

            for i, task_data in enumerate(tasks_data):
                logger.debug(f"Parsing task {i+1}/{len(tasks_data)}")
                task = self._parse_task(task_data)
                self.tasks[task.task_id] = task
                logger.info(f"Successfully loaded task '{task.task_id}' " f"with {len(task.subtasks)} subtasks")

            # Validate all task dependencies
            logger.info("Validating task dependencies")
            self._validate_all_dependencies()
            logger.info(
                f"TaskManager initialization complete. Loaded {len(self.tasks)} tasks " f"for domain '{self.domain}'"
            )

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise InvalidTaskDefinitionException(f"YAML parsing error: {e}", str(self.tasks_file_path))
        except Exception as e:
            if isinstance(e, InvalidTaskDefinitionException):
                raise
            logger.error(f"Unexpected error loading tasks: {e}")
            raise InvalidTaskDefinitionException(f"Error loading tasks: {e}", str(self.tasks_file_path))

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

    def _validate_all_dependencies(self) -> None:
        """
        Validate dependencies across all tasks.

        Raises:
            InvalidTaskDefinitionException: If any dependencies are invalid
        """
        all_errors = []

        for task in self.tasks.values():
            errors = task.validate_dependencies()
            all_errors.extend(errors)

        if all_errors:
            raise InvalidTaskDefinitionException(
                f"Dependency validation errors: {'; '.join(all_errors)}", str(self.tasks_file_path)
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

    def start_episode(self, session_id: str, task_id: str) -> "Episode":
        """
        Start a new episode for a session.

        Args:
            session_id: ID of the session starting the episode
            task_id: ID of the task to execute

        Returns:
            Episode instance

        Raises:
            TaskNotFoundException: If task is not found
        """
        logger.info(f"Starting new episode for session '{session_id}' with task '{task_id}'")

        # Validate task exists
        task = self.get_task(task_id)

        # Start episode with initial context
        episode = self.episode_manager.start_episode(
            session_id=session_id, task_id=task_id, initial_context=task.initial_context.copy()
        )

        logger.info(f"Started episode '{episode.episode_id}' for session '{session_id}' with task '{task_id}'")
        return episode

    def get_current_episode(self, session_id: str) -> Optional[Episode]:
        """
        Get the current active episode for a session.

        Args:
            session_id: ID of the session

        Returns:
            Episode instance if active, None otherwise
        """
        return self.episode_manager.get_current_episode(session_id)

    def record_episode_step(self, session_id: str, action: "Action", response: Dict[str, Any]) -> None:
        """
        Record an episode step after tool execution.

        Args:
            session_id: ID of the session
            action: Action that was taken
            response: Tool execution response
        """
        episode = self.episode_manager.get_current_episode(session_id)
        if episode:
            step = self.episode_manager.create_step(episode, action, response)
            self.episode_manager.update_episode_state(episode, step)

            # Check for automatic subtask progression
            self._check_subtask_progression(episode)

    def _check_subtask_progression(self, episode: Episode) -> None:
        """
        Check and handle automatic subtask progression.

        Args:
            episode: Episode to check for progression
        """
        task = self.get_task(episode.task_id)
        progressed_subtask = task.check_progression_criteria(episode)

        if progressed_subtask:
            logger.info(f"Episode '{episode.episode_id}' progressed subtask '{progressed_subtask.subtask_id}'")
            # Update episode subtask state based on progression
            # This would be implemented based on specific progression logic

    def end_episode(self, session_id: str, reason: str) -> "EpisodeResult":
        """
        End the current episode for a session.

        Args:
            session_id: ID of the session
            reason: Reason for ending the episode

        Returns:
            EpisodeResult with completion information

        Raises:
            SessionNotFoundException: If session has no active episode
        """
        logger.info(f"Ending episode for session '{session_id}': {reason}")
        return self.episode_manager.end_episode(session_id, reason)

    def reset_episode(self, session_id: str) -> Episode:
        """
        Reset the current episode (start a new attempt).

        Args:
            session_id: ID of the session

        Returns:
            New Episode instance

        Raises:
            SessionNotFoundException: If session has no active episode
        """
        logger.info(f"Resetting episode for session '{session_id}'")
        return self.episode_manager.reset_episode(session_id)

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

    def get_episode_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about the current episode for a session.

        Args:
            session_id: ID of the session

        Returns:
            Dictionary with episode information
        """
        episode = self.episode_manager.get_current_episode(session_id)
        if episode:
            return self.episode_manager._get_episode_progress_info(episode)
        return {"error": "No active episode for session"}
