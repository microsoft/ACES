"""
SABER Agent Manager - Direct Agent Creation

Manager that creates agents directly during task creation.
"""

from types import TracebackType
from typing import Dict, List, Optional, Set

from inspect_ai import Task
from inspect_ai.dataset import Sample

from ...logging_config import get_agent_logger
from ..client_session import ClientSessionManager
from ..exceptions import AgentInitializationError
from ..models import SABERConfig
from .task_agent_resolver import TaskAgentResolver

logger = get_agent_logger(__name__)


class AgentManager:
    """
    Simplified manager that creates agents directly during task creation.

    This replaces complex meta-agent management with direct agent creation
    during task setup. Agents are resolved from dataset metadata and created
    upfront using SABERAgentFactory.

    Responsibilities:
    - Create agents directly during task creation
    - Resolve agent assignments from dataset metadata
    - Handle agent lifecycle and cleanup
    - No agent instance caching or management

    BREAKING CHANGE: Simplified architecture with direct agent creation only.
    """

    def __init__(self, config: SABERConfig, session_manager: ClientSessionManager):
        """
        Initialize agent manager with configuration and shared session manager.

        Args:
            config: SABER configuration with agents assignments
            session_manager: Shared ClientSessionManager instance

        Raises:
            ValueError: If required parameters are missing
        """
        if not config:
            raise ValueError("SABERConfig is required")
        if not session_manager:
            raise ValueError("ClientSessionManager is required")

        self.config = config
        self.session_manager = session_manager
        self.resolver = TaskAgentResolver(config)
        self._available_task_ids: Optional[Set[str]] = None
        self._initialized = False

        logger.debug(
            "Agent manager initialized",
            extra={
                "event": "agent_manager_initialized",
                "has_task_ids": bool(config.task_ids),
            },
        )

    async def __aenter__(self, available_task_ids: Optional[List[str]] = None) -> "AgentManager":
        """
        Enter async context manager and initialize agent manager.

        Args:
            available_task_ids: List of task IDs available from the server.
                               If None, will attempt to derive from configuration.

        Returns:
            Initialized agent manager ready for direct agent creation

        Raises:
            AgentInitializationError: If initialization fails
        """
        if self._initialized:
            logger.debug(
                "Agent manager already initialized",
                extra={"event": "agent_manager_init_skipped"},
            )
            return self

        logger.info(
            "Initializing agent manager",
            extra={"event": "agent_manager_initializing"},
        )

        try:
            # Store available task IDs for agent resolution
            if available_task_ids is not None:
                self._available_task_ids = set(available_task_ids)
                logger.info(
                    "Using provided task identifiers",
                    extra={
                        "event": "agent_manager_tasks_provided",
                        "task_count": len(self._available_task_ids),
                        "tasks": sorted(self._available_task_ids),
                    },
                )
            else:
                # Fallback: Get available task IDs from configuration
                task_ids_from_config = self.config.task_ids or []
                if not task_ids_from_config:
                    logger.warning(
                        "Missing explicit task identifiers; extracting from assignments",
                        extra={
                            "event": "agent_manager_tasks_missing",
                            "assignment_count": len(self.config.agents),
                        },
                    )
                    # Extract task IDs from agents if not specified in config
                    task_set = set()
                    for assignment in self.config.agents:
                        for task in assignment.tasks:
                            if task != "*":  # Skip wildcard
                                task_set.add(task)
                    task_ids_from_config = list(task_set)

                self._available_task_ids = set(task_ids_from_config)

            if not self._available_task_ids:
                raise AgentInitializationError(
                    "No task IDs available for agent manager initialization. "
                    "Provide either available_task_ids parameter or ensure configuration "
                    "has task_ids or valid agent assignments."
                )

            # Initialize the resolver with available task IDs
            self.resolver.initialize(list(self._available_task_ids))

            logger.info(
                "Agent manager ready",
                extra={
                    "event": "agent_manager_ready",
                    "task_count": len(self._available_task_ids),
                },
            )
            self._initialized = True
            return self

        except Exception as exc:
            logger.error(
                "Agent manager initialization failed",
                extra={
                    "event": "agent_manager_init_failed",
                    "error": str(exc),
                },
            )
            raise AgentInitializationError(f"Agent manager initialization failed: {exc}") from exc

    async def __aexit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Exit async context manager and cleanup agent manager resources.
        """
        if not self._initialized:
            logger.debug(
                "Agent manager cleanup skipped",
                extra={"event": "agent_manager_cleanup_skipped"},
            )
            return

        logger.info(
            "Cleaning up agent manager",
            extra={"event": "agent_manager_cleanup"},
        )

        try:
            # Reset stored state
            self._available_task_ids = None
        except Exception as exc:
            logger.error(
                "Agent manager cleanup error",
                extra={
                    "event": "agent_manager_cleanup_failed",
                    "error": str(exc),
                },
            )
            # Don't raise - cleanup failures shouldn't break the main flow
        finally:
            self._initialized = False
            # Note: Don't cleanup session_manager here - it's shared

    async def create_agent_tasks(self, agent_datasets: Dict[str, List[Sample]]) -> List[Task]:
        """
        Create inspect_ai Tasks with directly created agents.

        Each agent group gets one Task containing all samples from all tasks assigned to it.
        Agents are created directly during task creation, not at execution time.

        Args:
            agent_datasets: Dictionary mapping agent_composite_key to combined dataset
                          where composite_key = "agent_id_taskhash"

        Returns:
            List of configured inspect_ai Tasks (one per agent group)

        Raises:
            RuntimeError: If agent manager not initialized
            AgentInitializationError: If task creation fails
        """
        if not self._initialized or self._available_task_ids is None:
            raise RuntimeError("Agent manager not initialized - use as async context manager")

        logger.info(
            "Creating agent-specific tasks",
            extra={
                "event": "agent_manager_create_tasks",
                "task_count": len(agent_datasets),
            },
        )

        tasks = []
        for agent_composite_key, dataset in agent_datasets.items():
            # Create task with agent_composite_key as identifier
            task = await self.create_agent_task(agent_composite_key, dataset)
            tasks.append(task)

        logger.info(
            "Agent-specific tasks created",
            extra={
                "event": "agent_manager_tasks_created",
                "task_count": len(tasks),
            },
        )
        return tasks

    async def create_agent_task(self, agent_composite_key: str, dataset: List[Sample]) -> Task:
        """
        Create inspect_ai Task with directly created agent.

        Args:
            agent_composite_key: Agent composite key (agent_id_taskhash)
            dataset: Combined dataset containing samples from all tasks assigned to this agent

        Returns:
            Configured inspect_ai Task with directly created agent

        Raises:
            RuntimeError: If agent manager not initialized
            AgentInitializationError: If task creation fails
        """
        if not self._initialized or self._available_task_ids is None:
            raise RuntimeError("Agent manager not initialized - use as async context manager")

        logger.debug(
            "Creating inspect_ai task",
            extra={
                "event": "agent_manager_task_create",
                "agent_composite_key": agent_composite_key,
                "sample_count": len(dataset),
            },
        )

        try:
            # Validate dataset
            if not dataset:
                raise AgentInitializationError(f"Empty dataset for agent group: {agent_composite_key}")

            # Extract task_id from first sample to resolve agent assignment
            first_sample = dataset[0]
            task_id = first_sample.metadata.get("task_id")
            if not task_id:
                available_keys = list(first_sample.metadata.keys())
                raise AgentInitializationError(
                    "No task_id in sample metadata for agent group: "
                    f"{agent_composite_key}. Available metadata keys: {available_keys}"
                )

            # 3. Resolve agent assignment using class resolver
            assignment = self.resolver.get_assignment_for_task(task_id)

            logger.info(
                "Resolved agent assignment",
                extra={
                    "event": "agent_manager_assignment_resolved",
                    "agent_id": assignment.id,
                    "task_id": task_id,
                    "agent_composite_key": agent_composite_key,
                },
            )

            # Create agent instance using SABERAgentFactory
            from ..inspect_ai.saber_scorer import saber_scorer
            from .factory import SABERAgentFactory

            factory = SABERAgentFactory()
            agent_instance = await factory.create_agent(
                agent_class=assignment.id,
                config=self.config,
                session_manager=self.session_manager,
                **assignment.kwargs,  # Pass agent-specific kwargs
            )

            logger.info(
                "Agent instance created",
                extra={
                    "event": "agent_manager_agent_created",
                    "agent_id": assignment.id,
                    "agent_composite_key": agent_composite_key,
                },
            )

            # Extract task_ids from all samples to track which tasks are included
            task_ids = set()
            for sample in dataset:
                if "task_id" in sample.metadata:
                    task_ids.add(sample.metadata["task_id"])

            # Create Task with resolved agent and proper metadata
            task = Task(
                dataset=dataset,
                model=assignment.model,  # Pass agent-specific model if configured
                solver=agent_instance,  # Direct agent, not meta-agent
                scorer=saber_scorer(),
                display_name=assignment.id,
                name=f"SABER Agent: {assignment.id} ({len(task_ids)} tasks)",
                metadata={
                    "saber_agent_id": assignment.id,
                    "saber_agent_name": assignment.id,
                    "saber_architecture": "direct_agent",
                    "saber_task_id": f"saber_task_{assignment.id}",
                    "saber_included_task_ids": list(task_ids),
                    "saber_sample_count": len(dataset),
                },
            )

            logger.info(
                "Agent task created",
                extra={
                    "event": "agent_manager_task_created",
                    "agent_id": assignment.id,
                    "agent_composite_key": agent_composite_key,
                    "sample_count": len(dataset),
                    "included_task_ids": sorted(task_ids),
                },
            )
            return task

        except Exception as exc:
            logger.error(
                "Agent task creation failed",
                extra={
                    "event": "agent_manager_task_failed",
                    "agent_composite_key": agent_composite_key,
                    "error": str(exc),
                },
            )
            raise AgentInitializationError(f"Agent task creation failed for {agent_composite_key}: {exc}") from exc

    @property
    def is_initialized(self) -> bool:
        """Check if agent manager is initialized."""
        return self._initialized

    @property
    def available_task_ids(self) -> Optional[Set[str]]:
        """Get available task IDs for debugging/testing."""
        return self._available_task_ids
