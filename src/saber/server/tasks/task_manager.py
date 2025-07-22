"""TaskManager implementation for task management system."""

from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .domain_task import DomainTask
from .enums import SessionState
from .exceptions import (
    InvalidTaskDefinitionException,
    SessionNotFoundException,
    SubTaskNotFoundException,
    TaskNotFoundException,
)
from .subtask import SubTask
from .task_session import TaskSession

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
        self.tasks: Dict[str, DomainTask] = {}
        self.active_sessions: Dict[str, TaskSession] = {}

        logger.info(f"Initializing TaskManager for domain '{domain}' with tasks file: {tasks_file_path}")

        # Load tasks from YAML file
        self.load_tasks_from_yaml()

    def load_tasks_from_yaml(self) -> None:
        """
        Load and parse YAML task definitions into DomainTask objects.

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
                domain_task = self._parse_domain_task(task_data)
                self.tasks[domain_task.task_id] = domain_task
                logger.info(
                    f"Successfully loaded task '{domain_task.task_id}' " f"with {len(domain_task.subtasks)} subtasks"
                )

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

    def _parse_domain_task(self, task_data: Dict[str, Any]) -> DomainTask:
        """
        Parse a single domain task from YAML data.

        Args:
            task_data: Dictionary containing task definition

        Returns:
            DomainTask instance
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

        logger.debug(f"Parsing domain task '{task_id}': {title}")

        # Parse subtasks
        subtasks_data = task_data.get("subtasks", [])
        subtasks = []

        logger.debug(f"Task '{task_id}' has {len(subtasks_data)} subtasks")
        for j, subtask_data in enumerate(subtasks_data):
            logger.debug(f"Parsing subtask {j+1}/{len(subtasks_data)} for task '{task_id}'")
            subtask = self._parse_subtask(subtask_data, task_id)
            subtasks.append(subtask)
            logger.debug(f"Successfully parsed subtask '{subtask.subtask_id}'")

        domain_task = DomainTask(
            task_id=task_id,
            domain=self.domain,
            title=title,
            description=description,
            subtasks=subtasks,
            initial_context=initial_context,
        )

        logger.debug(f"Created domain task '{task_id}' with {len(subtasks)} subtasks")
        return domain_task

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
            required_tools=subtask_data.get("required_tools", []),
            success_criteria=subtask_data.get("success_criteria", []),
            context_dependencies=subtask_data.get("context_dependencies", []),
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

    def get_task(self, task_id: str) -> DomainTask:
        """
        Get a domain task by ID.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            DomainTask instance

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

    def create_task_session(self, client_id: str, task_id: str) -> TaskSession:
        """
        Create a new task session for a client.

        Args:
            client_id: ID of the client starting the task
            task_id: ID of the task to execute

        Returns:
            TaskSession instance

        Raises:
            TaskNotFoundException: If task is not found
        """
        logger.info(f"Creating task session for client '{client_id}' with task '{task_id}'")

        # Validate task exists
        task = self.get_task(task_id)

        # Create session with initial context
        session = TaskSession(task_id=task_id, client_id=client_id, initial_context=task.initial_context.copy())

        # Store session
        self.active_sessions[session.session_id] = session

        logger.info(f"Created task session '{session.session_id}' for client '{client_id}' with task '{task_id}'")
        logger.debug(f"Active sessions count: {len(self.active_sessions)}")

        return session

    def get_session(self, session_id: str) -> TaskSession:
        """
        Get an active task session.

        Args:
            session_id: ID of the session

        Returns:
            TaskSession instance

        Raises:
            SessionNotFoundException: If session is not found
        """
        if session_id not in self.active_sessions:
            raise SessionNotFoundException(session_id)
        return self.active_sessions[session_id]

    def advance_subtask(self, session_id: str) -> Optional[SubTask]:
        """
        Advance a session to the next available subtask.

        Args:
            session_id: ID of the session to advance

        Returns:
            Next subtask to execute, or None if task is complete

        Raises:
            SessionNotFoundException: If session is not found
        """
        logger.debug(f"Advancing subtask for session '{session_id}'")
        session = self.get_session(session_id)
        next_subtask = session.advance_to_next(self)

        if next_subtask:
            logger.info(f"Session '{session_id}' advanced to subtask '{next_subtask.subtask_id}'")
        else:
            logger.info(f"Session '{session_id}' has completed all subtasks")

        return next_subtask

    def complete_task(self, session_id: str) -> TaskResult:
        """
        Complete a task and return the results.

        Args:
            session_id: ID of the session to complete

        Returns:
            TaskResult with completion information

        Raises:
            SessionNotFoundException: If session is not found
        """
        logger.info(f"Completing task for session '{session_id}'")
        session = self.get_session(session_id)

        # Mark session as completed if not already
        if session.state != SessionState.COMPLETED:
            session.state = SessionState.COMPLETED

        # Create result
        result = TaskResult(
            session_id=session.session_id,
            task_id=session.task_id,
            success=session.state == SessionState.COMPLETED,
            completion_time=datetime.utcnow(),
            context=session.context.copy(),
            completed_subtasks=list(session.completed_subtasks),
        )

        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Removed completed session '{session_id}' from active sessions")
            logger.debug(f"Active sessions count: {len(self.active_sessions)}")

        logger.info(
            f"Task completion result for session '{session_id}': success={result.success}, \
                completed_subtasks={len(result.completed_subtasks)}"
        )
        return result

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

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about a session.

        Args:
            session_id: ID of the session

        Returns:
            Dictionary with session information
        """
        session = self.get_session(session_id)
        return session.get_progress_info(self)
