"""
SABER Agent Manager - Direct Agent Creation

Manager that creates agents directly during task creation.
"""

import logging
from types import TracebackType
from typing import Dict, List, Optional, Set

from inspect_ai import Task
from inspect_ai.dataset import Sample

from ..client_session import ClientSessionManager
from ..exceptions import AgentInitializationError
from ..models import SABERConfig
from .task_agent_resolver import TaskAgentResolver

logger = logging.getLogger(__name__)


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

        logger.debug("Initialized AgentManager with direct agent creation support")

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
            logger.debug("AgentManager already initialized")
            return self

        logger.info("Initializing SABER agent manager with direct agent creation")

        try:
            # Store available task IDs for agent resolution
            if available_task_ids is not None:
                self._available_task_ids = set(available_task_ids)
                logger.info(f"Using provided available task IDs: {available_task_ids}")
            else:
                # Fallback: Get available task IDs from configuration
                task_ids_from_config = self.config.task_ids or []
                if not task_ids_from_config:
                    logger.warning("No task_ids specified in configuration - extracting from agent assignments")
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

            logger.info(f"Agent manager ready to handle {len(self._available_task_ids)} tasks")
            self._initialized = True
            return self

        except Exception as e:
            logger.error(f"Failed to initialize agent manager: {e}")
            raise AgentInitializationError(f"Agent manager initialization failed: {e}") from e

    async def __aexit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Exit async context manager and cleanup agent manager resources.
        """
        if not self._initialized:
            logger.debug("No agent manager resources to cleanup")
            return

        logger.info("Cleaning up agent manager")

        try:
            # Reset stored state
            self._available_task_ids = None
        except Exception as e:
            logger.error(f"Error during agent manager cleanup: {e}")
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

        logger.info(f"Creating {len(agent_datasets)} agent-specific tasks with direct agent creation")

        tasks = []
        for agent_composite_key, dataset in agent_datasets.items():
            # Create task with agent_composite_key as identifier
            task = await self.create_agent_task(agent_composite_key, dataset)
            tasks.append(task)

        logger.info(f"Created {len(tasks)} agent-specific tasks with direct agents")
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

        logger.debug(f"Creating inspect_ai Task for agent group: {agent_composite_key}")

        try:
            # Validate dataset
            if not dataset:
                raise AgentInitializationError(f"Empty dataset for agent group: {agent_composite_key}")

            # Extract task_id from first sample to resolve agent assignment
            first_sample = dataset[0]
            task_id = first_sample.metadata.get("task_id")
            if not task_id:
                raise AgentInitializationError(
                    f"No task_id in sample metadata for agent group: {agent_composite_key}. "
                    f"Available metadata keys: {list(first_sample.metadata.keys())}"
                )

            # 3. Resolve agent assignment using class resolver
            assignment = self.resolver.get_assignment_for_task(task_id)

            logger.info(f"Resolved agent '{assignment.id}' for task '{task_id}' in group '{agent_composite_key}'")

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

            logger.info(f"Created agent instance '{assignment.id}' for group '{agent_composite_key}'")

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
                f"Created agent task for '{assignment.id}' with {len(dataset)} samples from tasks: {list(task_ids)}"
            )
            return task

        except Exception as e:
            logger.error(f"Failed to create agent task for {agent_composite_key}: {e}")
            raise AgentInitializationError(f"Agent task creation failed for {agent_composite_key}: {e}") from e

    @property
    def is_initialized(self) -> bool:
        """Check if agent manager is initialized."""
        return self._initialized

    @property
    def available_task_ids(self) -> Optional[Set[str]]:
        """Get available task IDs for debugging/testing."""
        return self._available_task_ids
