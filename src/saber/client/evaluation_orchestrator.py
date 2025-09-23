"""
SABER Evaluation Orchestrator

Top-level orchestration component for SABER evaluations with proper resource management,
upfront validation, and fail-fast error handling.

This component replaces the function-based eval_async approach with a clean
async context manager that handles component lifecycle and error recovery.

BREAKING CHANGE: No backwards compatibility with old run_saber_eval_async function.
"""

import logging
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, List, Optional

from ..models import TaskInfo
from .agent import AgentManager
from .client_session import ClientSessionManager
from .dataset_manager import DatasetManager
from .exceptions import AgentInitializationError, ConfigurationValidationError, DatasetCreationError
from .models import SABERConfig

logger = logging.getLogger(__name__)


class SABEREvaluationOrchestrator:
    """
    Orchestrates SABER evaluation workflow with proper resource management.

    This is the main entry point for SABER evaluations, replacing the old
    function-based approach with a clean async context manager.

    Responsibilities:
    - Upfront configuration validation (fail fast)
    - Component lifecycle management
    - Resource management via async context manager
    - Error handling and recovery
    - Evaluation orchestration

    Usage:
        async with SABEREvaluationOrchestrator(config) as orchestrator:
            eval_log = await orchestrator.execute()
    """

    def __init__(self, config: SABERConfig):
        """
        Initialize orchestrator with SABER configuration.

        Args:
            config: Complete SABER configuration

        Note: No validation is performed here - validation happens in __aenter__
        """
        self.config = config
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session_manager: Optional[ClientSessionManager] = None
        self._agent_manager: Optional["AgentManager"] = None
        self._dataset_manager: Optional["DatasetManager"] = None
        self._validated = False

        logger.debug("Initialized SABEREvaluationOrchestrator")

    async def __aenter__(self) -> "SABEREvaluationOrchestrator":
        """
        Enter async context manager and initialize all components.

        Performs upfront validation and component initialization:
        1. Validate configuration completely
        2. Test server connectivity
        3. Initialize shared session manager
        4. Initialize agent and dataset managers
        5. Perform final validation

        Returns:
            Initialized orchestrator ready for execution

        Raises:
            ConfigurationValidationError: Invalid configuration
            ServerConnectivityError: Cannot reach SABER server
            AgentInitializationError: Agent setup failed
        """
        logger.info("Starting SABER evaluation orchestrator initialization")

        self._exit_stack = AsyncExitStack()

        try:
            # Validate configuration completely
            await self._validate_configuration()

            # Initialize shared session manager
            logger.info("Initializing shared session manager")
            if self.config.session_config is None:
                raise ValueError("Session config is required but not provided")
            self._session_manager = ClientSessionManager(config=self.config.session_config)
            await self._exit_stack.enter_async_context(self._session_manager_context())

            # Discover available tasks early to provide to agent manager
            logger.info("Discovering available tasks from server")
            available_tasks = await self.discover_tasks()
            available_task_ids = [task.task_id for task in available_tasks]
            logger.info(f"Discovered {len(available_task_ids)} tasks: {available_task_ids}")

            # Initialize agent manager with available task IDs
            logger.info("Initializing agent manager")
            self._agent_manager = AgentManager(self.config, self._session_manager)
            await self._exit_stack.enter_async_context(self._agent_manager_context(available_task_ids))

            # Initialize dataset manager
            logger.info("Initializing dataset manager")
            self._dataset_manager = DatasetManager(self._session_manager, self.config)
            await self._exit_stack.enter_async_context(self._dataset_manager)

            # Final validation
            await self._validate_components()

            self._validated = True
            logger.info("SABER evaluation orchestrator initialized successfully")
            return self

        except Exception as e:
            # Cleanup on initialization failure
            if self._exit_stack:
                await self._exit_stack.__aexit__(type(e), e, e.__traceback__)
                self._exit_stack = None
            raise

    async def __aexit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Exit async context manager and cleanup all resources.

        Handles cleanup gracefully and logs any cleanup errors without
        raising them (cleanup errors shouldn't mask the original error).
        """
        logger.info("Cleaning up SABER evaluation orchestrator")

        cleanup_errors = []

        # Cleanup all managed resources
        if self._exit_stack:
            try:
                await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                cleanup_errors.append(f"Exit stack cleanup: {e}")
                logger.error(f"Error during exit stack cleanup: {e}")

        # Reset state
        self._exit_stack = None
        self._session_manager = None
        self._agent_manager = None
        self._dataset_manager = None
        self._validated = False

        # Log cleanup errors but don't raise (don't mask original exception)
        if cleanup_errors:
            logger.warning(f"Cleanup completed with {len(cleanup_errors)} errors: {cleanup_errors}")
        else:
            logger.info("SABER evaluation orchestrator cleanup completed successfully")

    # Validation Methods

    async def _validate_configuration(self) -> None:
        """
        Validate SABER configuration completely.

        Fails fast on any configuration issues with clear error messages.

        Raises:
            ConfigurationValidationError: Configuration is invalid
        """
        logger.debug("Validating SABER configuration")

        errors = []

        # Validate model configuration - ALL agents must specify their own models
        agents_without_models = []

        if self.config.agents:
            for i, agent in enumerate(self.config.agents):
                if not hasattr(agent, "model") or not agent.model:
                    agents_without_models.append(f"agents[{i}].id='{agent.id}'")

        if agents_without_models:
            errors.append(f"All agents must specify a 'model' field. Agents missing models: {agents_without_models}")

        # Global model is no longer required (or used)

        # Validate session configuration
        if not self.config.session_config:
            errors.append("session_config is required")
        elif not self.config.session_config.base_url:
            errors.append("session_config.base_url is required")

        # Validate agent assignments (new format only)
        if not self.config.agents:
            errors.append("at least one agent assignment is required")

        # Validate numeric configurations
        if hasattr(self.config, "max_parallel_tasks") and self.config.max_parallel_tasks <= 0:
            errors.append(f"max_parallel_tasks must be positive, got: {self.config.max_parallel_tasks}")

        if errors:
            raise ConfigurationValidationError(
                f"Configuration validation failed: {len(errors)} errors found",
                details={"validation_errors": errors},
                suggestion="Fix configuration errors and retry",
            )

        logger.debug("Configuration validation passed")

    async def _validate_components(self) -> None:
        """
        Validate all components are properly initialized.

        Raises:
            AgentInitializationError: Components not properly initialized
        """
        logger.debug("Validating component initialization")

        if not self._session_manager:
            raise AgentInitializationError("Session manager not initialized")

        if not self._agent_manager:
            raise AgentInitializationError("Agent manager not initialized")

        if not self._dataset_manager:
            raise AgentInitializationError("Dataset manager not initialized")

        logger.debug("Component validation passed")

    # Execution Methods

    async def discover_tasks(self) -> List[TaskInfo]:
        """
        Discover tasks by coordinating between session manager and configuration.

        Returns:
            List of TaskInfo objects available for evaluation

        Raises:
            DatasetCreationError: Task discovery failed
        """
        try:
            if self.config.task_ids:
                logger.info(f"Using configured task IDs: {self.config.task_ids}")
                # For configured task IDs, we need to fetch the full TaskInfo objects
                if self._session_manager is None:
                    raise RuntimeError("Session manager not initialized")
                all_tasks_data = await self._session_manager.get_available_tasks()
                # Filter to only the configured task IDs
                configured_tasks = [task for task in all_tasks_data if task.task_id in self.config.task_ids]
                if len(configured_tasks) != len(self.config.task_ids):
                    found_ids = [task.task_id for task in configured_tasks]
                    missing_ids = set(self.config.task_ids) - set(found_ids)
                    raise ValueError(f"Configured task IDs not found on server: {missing_ids}")
                return configured_tasks
            else:
                logger.info("Discovering available tasks")
                # Session manager handles HTTP communication
                if self._session_manager is None:
                    raise RuntimeError("Session manager not initialized")
                tasks_data = await self._session_manager.get_available_tasks()
                logger.info(f"Discovered {len(tasks_data)} available tasks")
                return tasks_data
        except Exception as e:
            raise DatasetCreationError(
                f"Task discovery failed: {e}",
                details={
                    "configured_task_ids": self.config.task_ids,
                    "server_url": self.config.session_config.base_url if self.config.session_config else "unknown",
                },
                suggestion="Verify server is running and has available tasks",
            ) from e

    async def create_dataset(
        self, task_ids: List[str]
    ) -> List[Any]:  # TODO: Should be List[Sample] when import available
        """
        Create inspect_ai dataset by coordinating between session manager and dataset manager.

        Args:
            task_ids: List of task IDs to include

        Returns:
            inspect_ai dataset

        Raises:
            DatasetCreationError: Dataset creation failed
        """
        try:
            # Session manager gets full task data via HTTP
            if self._session_manager is None:
                raise RuntimeError("Session manager not initialized")
            tasks_data = await self._session_manager.get_tasks(task_ids)

            # Dataset manager transforms data to inspect_ai format
            if self._dataset_manager is None:
                raise RuntimeError("Dataset manager not initialized")
            return await self._dataset_manager.create_dataset(tasks_data)
        except Exception as e:
            raise DatasetCreationError(
                f"Dataset creation failed: {e}",
                details={"task_ids": task_ids, "task_count": len(task_ids)},
                suggestion="Check that all task IDs exist on the server",
            ) from e

    async def create_multi_task_evaluation(
        self, task_ids: List[str]
    ) -> List[Any]:  # TODO: Should be List[Task] when available
        """
        Create inspect_ai Tasks using direct agent creation approach.

        This method works for any number of tasks (1 or many) by grouping them
        by agent assignment and creating unified datasets per agent.

        Args:
            task_ids: List of task IDs to create tasks for

        Returns:
            List of configured inspect_ai Tasks with direct agents

        Raises:
            DatasetCreationError: Dataset creation failed
            AgentInitializationError: Agent assignment failed
        """
        try:
            # Session manager gets full task data via HTTP
            if self._session_manager is None:
                raise RuntimeError("Session manager not initialized")
            tasks_data = await self._session_manager.get_tasks(task_ids)

            # Create TaskAgentResolver for agent-grouped datasets
            from .agent.task_agent_resolver import TaskAgentResolver

            task_agent_resolver = TaskAgentResolver(self.config)
            task_agent_resolver.initialize(task_ids)

            # Dataset manager creates agent-grouped datasets
            if self._dataset_manager is None:
                raise RuntimeError("Dataset manager not initialized")
            agent_datasets = await self._dataset_manager.create_agent_datasets(tasks_data, task_agent_resolver)

            # Agent manager creates tasks with direct agents (agent-grouped approach)
            if self._agent_manager is None:
                raise RuntimeError("Agent manager not initialized")

            # Use the new create_agent_tasks method for agent-grouped datasets
            tasks = await self._agent_manager.create_agent_tasks(agent_datasets)

            logger.info(f"Successfully created {len(tasks)} agent-grouped tasks with direct agents")
            return tasks

        except Exception as e:
            raise DatasetCreationError(
                f"Multi-task evaluation creation failed: {e}",
                details={"task_ids": task_ids, "task_count": len(task_ids)},
                suggestion="Check that all task IDs exist and agent assignments are valid",
            ) from e

    async def get_session_id(self) -> str:
        """
        Get the current session ID.

        Returns:
            Current session ID

        Raises:
            RuntimeError: If session manager not initialized or no active session
        """
        if self._session_manager is None:
            raise RuntimeError("Session manager not initialized")

        session_id = self._session_manager.get_current_session_id()
        if session_id is None:
            raise RuntimeError("No active session")
        return session_id

    # Helper Methods
    def _session_manager_context(self) -> Any:  # TODO: Better type annotation
        """
        Create async context manager for session manager.

        This wrapper ensures proper cleanup of session manager resources.
        """

        class SessionManagerContext:
            def __init__(self, session_manager: Any) -> None:
                self.session_manager = session_manager

            async def __aenter__(self) -> Any:
                # Initialize the session manager by creating a session
                await self.session_manager.create_session()
                return self.session_manager

            async def __aexit__(
                self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
            ) -> None:
                try:
                    await self.session_manager.cleanup()
                except Exception as e:
                    logger.error(f"Session manager cleanup error: {e}")
                    # Don't raise - cleanup errors shouldn't mask original exception

        return SessionManagerContext(self._session_manager)

    def _agent_manager_context(self, available_task_ids: List[str]) -> Any:
        """
        Create async context manager for agent manager with task IDs.

        This wrapper ensures agent manager gets available task IDs during initialization.
        """

        class AgentManagerContext:
            def __init__(self, agent_manager: Any, task_ids: List[str]) -> None:
                self.agent_manager = agent_manager
                self.task_ids = task_ids

            async def __aenter__(self) -> Any:
                # Agent manager needs available task IDs for TaskAgentResolver initialization
                await self.agent_manager.__aenter__(available_task_ids=self.task_ids)
                return self.agent_manager

            async def __aexit__(
                self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
            ) -> None:
                try:
                    await self.agent_manager.__aexit__(exc_type, exc_val, exc_tb)
                except Exception as e:
                    logger.error(f"Agent manager cleanup error: {e}")

        return AgentManagerContext(self._agent_manager, available_task_ids)
