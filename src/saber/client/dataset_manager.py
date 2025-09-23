"""
SABER Dataset Manager

Manages task discovery and dataset creation for SABER evaluations.
Extracted from SABEREvalManager to separate dataset concerns from agent management.

BREAKING CHANGE: Supports mult            raise DatasetCreationError(
                f"Failed to create dataset from {len(tasks_data)} tasks: {e}",
                details={"task_count": len(tasks_data), "error_type": type(e).__name__},
                suggestion="Check that task data is valid, agent assignments are configured, "
                          "and inspect_ai conversion is working",
            ) from ek datasets keyed by task_id for agent assignment.
"""

import hashlib
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .agent.task_agent_resolver import TaskAgentResolver

from ..models import TaskInfo  # Use server models directly
from .client_session import ClientSessionManager
from .exceptions import DatasetCreationError
from .models import SABERConfig

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages SABER task discovery and dataset creation with multi-task support.

    Responsibilities:
    - Task discovery via shared session manager
    - Multi-task dataset creation keyed by task_id
    - Task filtering and agent assignment coordination
    - Dataset caching/optimization (future)

    Breaking Change: Now creates datasets keyed by task_id to support
    task-specific agent assignment.

    This manager uses ClientSessionManager for all HTTP communication
    and focuses purely on data transformation logic.
    """

    def __init__(self, session_manager: ClientSessionManager, config: SABERConfig):
        """
        Initialize dataset manager with shared session manager.

        Args:
            session_manager: Shared ClientSessionManager for HTTP communication
            config: SABER configuration for dataset preferences
        """
        self.session_manager = session_manager
        self.config = config
        self._initialized = False

        logger.debug("Initialized DatasetManager with shared ClientSessionManager")

    async def __aenter__(self) -> "DatasetManager":
        """
        Enter async context manager and initialize dataset manager.

        Returns:
            Initialized dataset manager
        """
        if self._initialized:
            logger.debug("DatasetManager already initialized")
            return self

        logger.info("Initializing dataset manager")

        # Future: Add any initialization logic here (caching, etc.)
        self._initialized = True
        logger.info("Dataset manager initialized successfully")
        return self

    async def __aexit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Exit async context manager and cleanup dataset resources.
        """
        if not self._initialized:
            logger.debug("No dataset manager resources to cleanup")
            return

        logger.info("Cleaning up dataset manager resources")

        # Future: Add cleanup logic here (cache cleanup, etc.)

        self._initialized = False
        logger.info("Dataset manager cleanup completed")

    async def create_dataset(self, tasks_data: List[TaskInfo]) -> List[Any]:
        """
        Create SABER dataset from task data using inspect_ai dataset conversion.

        Args:
            tasks_data: List of TaskInfo objects from session manager

        Returns:
            inspect_ai dataset

        Raises:
            DatasetCreationError: If dataset creation fails
        """
        if not self._initialized:
            raise RuntimeError("DatasetManager not initialized - use as async context manager")

        if not tasks_data:
            raise DatasetCreationError(
                "Cannot create dataset from empty task data",
                details={"task_count": 0},
                suggestion="Ensure tasks are available on the server",
            )

        logger.info(f"Creating dataset from {len(tasks_data)} tasks")

        try:
            # Import conversion function
            from .inspect_ai.saber_dataset import create_saber_dataset

            # Create inspect_ai dataset using the conversion function with TaskInfo objects
            dataset = await create_saber_dataset(tasks_data)

            logger.info(f"Successfully created dataset with {len(dataset)} samples")
            return dataset

        except Exception as e:
            logger.error(f"Dataset creation failed: {e}")
            raise DatasetCreationError(
                f"Failed to create dataset from {len(tasks_data)} tasks: {e}",
                details={"task_count": len(tasks_data), "error_type": type(e).__name__},
                suggestion="Check that task data is valid and inspect_ai conversion is working",
            ) from e

    async def create_agent_datasets(
        self, tasks_data: List[TaskInfo], task_agent_resolver: "TaskAgentResolver"
    ) -> Dict[str, List[Any]]:
        """
        Create SABER datasets grouped by agent assignment instead of task_id.

        This groups tasks by which agent will handle them, creating combined datasets
        for agents that handle multiple tasks.

        Args:
            tasks_data: List of TaskInfo objects from session manager
            task_agent_resolver: TaskAgentResolver to determine agent assignments

        Returns:
            Dictionary mapping agent_id to combined inspect_ai dataset

        Raises:
            DatasetCreationError: If dataset creation fails
        """
        if not self._initialized:
            raise RuntimeError("DatasetManager not initialized - use as async context manager")

        if not tasks_data:
            raise DatasetCreationError(
                "Cannot create datasets from empty task data",
                details={"task_count": 0},
                suggestion="Ensure tasks are available on the server",
            )

        logger.info(f"Creating agent-grouped datasets from {len(tasks_data)} tasks")

        try:
            # Import conversion function
            from .inspect_ai.saber_dataset import create_saber_dataset

            # Group tasks by agent assignment using composite key (agent_id + task_hash)
            agent_task_groups: Dict[str, List[TaskInfo]] = {}

            for task_info in tasks_data:
                task_id = task_info.task_id

                # Get agent assignment for this task
                assignment = task_agent_resolver.get_assignment_for_task(task_id)

                # Create composite key: agent_id + hash of sorted tasks
                # This ensures agents with same ID but different task sets get separate datasets
                tasks_sorted = sorted(assignment.tasks)
                tasks_str = "+".join(tasks_sorted)
                tasks_hash = hashlib.sha256(tasks_str.encode()).hexdigest()[:8]  # Use first 8 chars
                agent_composite_key = f"{assignment.id}_{tasks_hash}"

                # Group tasks by composite agent key
                if agent_composite_key not in agent_task_groups:
                    agent_task_groups[agent_composite_key] = []
                agent_task_groups[agent_composite_key].append(task_info)

                logger.debug(
                    f"Assigned task '{task_id}' to agent group '{agent_composite_key}' "
                    f"(from assignment '{assignment.id}' with tasks {assignment.tasks})"
                )

            # Create combined datasets for each agent group
            agent_datasets: Dict[str, List[Any]] = {}

            for agent_composite_key, agent_tasks in agent_task_groups.items():
                # Create combined dataset from all tasks assigned to this agent group
                combined_dataset = await create_saber_dataset(agent_tasks)
                agent_datasets[agent_composite_key] = combined_dataset

                task_ids = [task.task_id for task in agent_tasks]
                logger.info(
                    f"Created combined dataset for agent group '{agent_composite_key}' "
                    f"from tasks {task_ids} with {len(combined_dataset)} total samples"
                )

            logger.info(f"Successfully created {len(agent_datasets)} agent-grouped datasets")
            return agent_datasets

        except Exception as e:
            logger.error(f"Agent dataset creation failed: {e}")
            raise DatasetCreationError(
                f"Failed to create agent datasets from {len(tasks_data)} tasks: {e}",
                details={"task_count": len(tasks_data), "error_type": type(e).__name__},
                suggestion="Check that task data is valid, agent assignments are configured, "
                "and inspect_ai conversion is working",
            ) from e
