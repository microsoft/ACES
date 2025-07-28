"""TaskManager implementation for task management system."""

from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..tools.base import ToolResult

from .core.subtask import SubTask
from .core.subtask_progression_engine import SubTaskProgressionEngine
from .core.task import Task
from .core.task_config_loader import TaskConfigLoader
from .episodes.episode import Action, Episode, EpisodeResult, StepResult
from .episodes.episode_manager import EpisodeManager
from .exceptions import EpisodeNotFoundException, SubTaskNotFoundException, TaskNotFoundException

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

    Responsible for coordinating task loading, episode management,
    and RL-style interfaces with specialized components.
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

        # Initialize specialized components
        self.config_loader = TaskConfigLoader(domain)
        self.progression_engine = SubTaskProgressionEngine()
        self.episode_manager = EpisodeManager()

        logger.info(f"Initializing TaskManager for domain '{domain}' with tasks file: {tasks_file_path}")

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

    def start_episode(self, session_id: str, task_id: str) -> Episode:
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

    def step(self, session_id: str, action: Action, tool_result: ToolResult) -> StepResult:
        """
        RL gym-style step function that records episode steps and returns observations.

        Args:
            session_id: ID of the session
            action: Action that was taken
            tool_result: ToolResult from tool execution

        Returns:
            StepResult with observation, done, and info

        Raises:
            EpisodeNotFoundException: If session has no active episode
        """
        episode = self.episode_manager.get_current_episode(session_id)
        if not episode:
            raise EpisodeNotFoundException(f"No active episode found for session '{session_id}'")

        task = self.get_task(episode.task_id)
        logger.debug(f"Executing step for session '{session_id}', action: {action.tool_name}")

        return self.episode_manager.step(
            session_id=session_id,
            action=action,
            tool_result=tool_result,
            task_subtask_count=len(task.subtasks),
            task_description=task.description,
            current_objective=self._get_current_objective(task, episode),
            all_subtask_ids=task.get_all_subtask_ids(),
            task=task,
            progression_engine=self.progression_engine,
        )

    def reset(self, session_id: str, task_id: str) -> Episode:
        """
        RL gym-style reset function for starting new episodes.

        Args:
            session_id: ID of the session
            task_id: ID of the task to execute

        Returns:
            New Episode instance

        Raises:
            TaskNotFoundException: If task is not found
        """
        logger.info(f"RL reset for session '{session_id}' with task '{task_id}'")

        # End current episode if exists
        current_episode = self.episode_manager.get_current_episode(session_id)
        if current_episode:
            self.episode_manager.end_episode(session_id, "reset")

        # Start new episode
        episode = self.start_episode(session_id, task_id)

        # Initialize subtask state for the episode
        task = self.get_task(task_id)
        self.progression_engine.initialize_episode_subtasks(task, episode)

        return episode

    def end_episode(self, session_id: str, reason: str) -> EpisodeResult:
        """
        End the current episode for a session.

        Args:
            session_id: ID of the session
            reason: Reason for ending the episode

        Returns:
            EpisodeResult with completion information
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

    def _get_current_objective(self, task: Task, episode: Episode) -> Optional[str]:
        """
        Get the current objective for the episode based on active subtasks.

        Args:
            task: Task definition
            episode: Current episode

        Returns:
            Current objective string or None
        """
        # Get the first in-progress subtask's objective
        for subtask_id in episode.in_progress_subtasks:
            subtask = task.get_subtask_by_id(subtask_id)
            if subtask:
                return subtask.objective

        # If no in-progress subtasks, get the first available subtask
        for subtask_id in episode.not_visited_subtasks:
            subtask = task.get_subtask_by_id(subtask_id)
            if subtask and subtask.check_entry_conditions(episode):
                return subtask.objective

        # Default to task description if no specific objective
        return task.description
