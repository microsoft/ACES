"""TaskManager implementation for task management system."""

from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..execution.base import CommandResult
from .base import Action
from .core.subtask import SubTask
from .core.task import Task
from .core.task_config_loader import TaskConfigLoader
from .episodes.episode import Episode, Step
from .episodes.episode_manager import EpisodeManager
from .exceptions import EpisodeNotFoundException, SubTaskNotFoundException, TaskNotFoundException

logger = getLogger(__name__)


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

        # Get the task instance
        task = self.get_task(task_id)

        # Start episode with task reference and initial context
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

    def step(self, session_id: str, action: Action, command_result: CommandResult) -> Step:
        """
        RL gym-style step function that records episode steps and returns observations.

        Args:
            session_id: ID of the session
            action: Action that was taken
            command_result: CommandResult from tool execution

        Returns:
            Step object with all step information

        Raises:
            EpisodeNotFoundException: If session has no active episode
        """
        episode = self.episode_manager.get_current_episode(session_id)
        if not episode:
            raise EpisodeNotFoundException(f"No active episode found for session '{session_id}'")

        logger.debug(f"Executing step for session '{session_id}', action: {action.tool_name}")

        step = self.episode_manager.step(
            session_id=session_id,
            action=action,
            command_result=command_result,
        )

        return step

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
        logger.info(f"RL reset for session '{session_id}' with task_id '{task_id}'")

        # End current episode if exists
        current_episode = self.episode_manager.get_current_episode(session_id)
        if current_episode:
            self.episode_manager.end_episode(session_id, "reset")

        # Start new episode (which will automatically initialize with task)
        episode = self.start_episode(session_id, task_id)

        return episode

    def end_episode(self, session_id: str, reason: str) -> Episode:
        """
        End the current episode for a session.

        Args:
            session_id: ID of the session
            reason: Reason for ending the episode

        Returns:
            Episode with completion information
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

        # Get current episode to extract task_id
        current_episode = self.episode_manager.get_current_episode(session_id)
        if not current_episode:
            raise EpisodeNotFoundException(session_id)

        # Get the task for the reset
        task = self.get_task(current_episode.task_id)

        return self.episode_manager.reset_episode(session_id, task.task_id)

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
