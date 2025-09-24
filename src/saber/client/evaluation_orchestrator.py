"""
SABER Evaluation Orchestrator

Top-level orchestration component for SABER evaluations with proper resource management,
upfront validation, and fail-fast error handling.

Logging category: EVALUATION.

This component replaces the function-based eval_async approach with a clean
async context manager that handles component lifecycle and error recovery.

BREAKING CHANGE: No backwards compatibility with old run_saber_eval_async function.
"""

from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, List, Optional

from ..logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ..models import TaskInfo
from .agent import AgentManager
from .client_session import ClientSessionManager
from .dataset_manager import DatasetManager
from .exceptions import AgentInitializationError, ConfigurationValidationError, DatasetCreationError
from .models import SABERConfig

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


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

        logger.debug(
            "Evaluation orchestrator instantiated",
            extra={"event": "orchestrator_instantiated"},
        )

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
        log_operation_start(logger, "orchestrator_initialization")

        self._exit_stack = AsyncExitStack()

        try:
            # Validate configuration completely
            await self._validate_configuration()

            # Initialize shared session manager
            logger.info(
                "Initializing shared session manager",
                extra={"event": "session_manager_initializing"},
            )
            if self.config.session_config is None:
                raise ValueError("Session config is required but not provided")
            self._session_manager = ClientSessionManager(config=self.config.session_config)
            await self._exit_stack.enter_async_context(self._session_manager_context())

            # Discover available tasks early to provide to agent manager
            logger.info(
                "Discovering available tasks from server",
                extra={"event": "task_discovery_request"},
            )
            available_tasks = await self.discover_tasks()
            available_task_ids = [task.task_id for task in available_tasks]
            logger.info(
                "Available tasks discovered",
                extra={
                    "event": "task_discovery_result",
                    "task_count": len(available_task_ids),
                    "task_ids": available_task_ids,
                },
            )

            # Initialize agent manager with available task IDs
            logger.info(
                "Initializing agent manager",
                extra={"event": "agent_manager_initializing"},
            )
            self._agent_manager = AgentManager(self.config, self._session_manager)
            await self._exit_stack.enter_async_context(self._agent_manager_context(available_task_ids))

            # Initialize dataset manager
            logger.info(
                "Initializing dataset manager",
                extra={"event": "dataset_manager_initializing"},
            )
            self._dataset_manager = DatasetManager(self._session_manager, self.config)
            await self._exit_stack.enter_async_context(self._dataset_manager)

            # Final validation
            await self._validate_components()

            self._validated = True
            log_operation_success(
                logger,
                "orchestrator_initialization",
                session_id=(self._session_manager.get_current_session_id() if self._session_manager else None),
                components_initialized=3,
            )
            return self

        except Exception as e:
            # Cleanup on initialization failure
            if self._exit_stack:
                await self._exit_stack.__aexit__(type(e), e, e.__traceback__)
                self._exit_stack = None
            log_operation_failure(
                logger,
                "orchestrator_initialization",
                e,
                stage="initialization",
            )
            raise

    async def __aexit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Exit async context manager and cleanup all resources.

        Handles cleanup gracefully and logs any cleanup errors without
        raising them (cleanup errors shouldn't mask the original error).
        """
        logger.info(
            "Cleaning up evaluation orchestrator",
            extra={"event": "orchestrator_cleanup_start"},
        )

        cleanup_errors = []

        # Cleanup all managed resources
        if self._exit_stack:
            try:
                await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                cleanup_errors.append({"component": "exit_stack", "error": str(e)})
                logger.error(
                    "Error during exit stack cleanup",
                    extra={
                        "event": "exit_stack_cleanup_error",
                        "error": str(e),
                    },
                )

        # Reset state
        self._exit_stack = None
        self._session_manager = None
        self._agent_manager = None
        self._dataset_manager = None
        self._validated = False

        # Log cleanup errors but don't raise (don't mask original exception)
        if cleanup_errors:
            logger.warning(
                "Cleanup completed with errors",
                extra={
                    "event": "orchestrator_cleanup_errors",
                    "error_count": len(cleanup_errors),
                    "errors": cleanup_errors,
                },
            )
        else:
            logger.info(
                "Evaluation orchestrator cleanup completed successfully",
                extra={"event": "orchestrator_cleanup_complete"},
            )

    # Validation Methods

    async def _validate_configuration(self) -> None:
        """
        Validate SABER configuration completely.

        Fails fast on any configuration issues with clear error messages.

        Raises:
            ConfigurationValidationError: Configuration is invalid
        """
        logger.debug(
            "Validating evaluation configuration",
            extra={"event": "config_validation_start"},
        )

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
            logger.error(
                "Configuration validation failed",
                extra={
                    "event": "config_validation_failed",
                    "error_count": len(errors),
                    "errors": errors,
                },
            )
            raise ConfigurationValidationError(
                f"Configuration validation failed: {len(errors)} errors found",
                details={"validation_errors": errors},
                suggestion="Fix configuration errors and retry",
            )

        logger.debug(
            "Configuration validation passed",
            extra={"event": "config_validation_complete"},
        )

    async def _validate_components(self) -> None:
        """
        Validate all components are properly initialized.

        Raises:
            AgentInitializationError: Components not properly initialized
        """
        logger.debug(
            "Validating component initialization",
            extra={"event": "component_validation_start"},
        )

        if not self._session_manager:
            logger.error(
                "Session manager missing during component validation",
                extra={
                    "event": "component_validation_failed",
                    "missing_component": "session_manager",
                },
            )
            raise AgentInitializationError("Session manager not initialized")

        if not self._agent_manager:
            logger.error(
                "Agent manager missing during component validation",
                extra={
                    "event": "component_validation_failed",
                    "missing_component": "agent_manager",
                },
            )
            raise AgentInitializationError("Agent manager not initialized")

        if not self._dataset_manager:
            logger.error(
                "Dataset manager missing during component validation",
                extra={
                    "event": "component_validation_failed",
                    "missing_component": "dataset_manager",
                },
            )
            raise AgentInitializationError("Dataset manager not initialized")

        logger.debug(
            "Component validation passed",
            extra={"event": "component_validation_complete"},
        )

    # Execution Methods

    async def discover_tasks(self) -> List[TaskInfo]:
        """
        Discover tasks by coordinating between session manager and configuration.

        Returns:
            List of TaskInfo objects available for evaluation

        Raises:
            DatasetCreationError: Task discovery failed
        """
        configured_task_ids = list(self.config.task_ids or [])
        session_id: Optional[str] = None
        operation_details: dict[str, Any] = {}
        if configured_task_ids:
            operation_details["configured_task_ids"] = configured_task_ids

        try:
            if self._session_manager is None:
                raise RuntimeError("Session manager not initialized")

            session_id = self._session_manager.get_current_session_id()
            log_operation_start(
                logger,
                "task_discovery",
                session_id=session_id,
                **operation_details,
            )

            if configured_task_ids:
                logger.info(
                    "Using configured task IDs",
                    extra={
                        "event": "task_discovery_configured",
                        "configured_task_ids": configured_task_ids,
                        "task_count": len(configured_task_ids),
                    },
                )
                all_tasks_data = await self._session_manager.get_available_tasks()
                configured_tasks = [task for task in all_tasks_data if task.task_id in configured_task_ids]
                if len(configured_tasks) != len(configured_task_ids):
                    found_ids = [task.task_id for task in configured_tasks]
                    missing_ids = sorted(set(configured_task_ids) - set(found_ids))
                    logger.error(
                        "Configured task IDs missing on server",
                        extra={
                            "event": "task_discovery_missing_tasks",
                            "requested_task_ids": configured_task_ids,
                            "found_task_ids": found_ids,
                            "missing_task_ids": missing_ids,
                        },
                    )
                    raise ValueError(f"Configured task IDs not found on server: {missing_ids}")

                log_operation_success(
                    logger,
                    "task_discovery",
                    session_id=session_id,
                    requested_task_count=len(configured_task_ids),
                    resolved_task_count=len(configured_tasks),
                    **operation_details,
                )
                return configured_tasks

            logger.info(
                "Discovering available tasks",
                extra={"event": "task_discovery_request_all"},
            )
            tasks_data = await self._session_manager.get_available_tasks()
            log_operation_success(
                logger,
                "task_discovery",
                session_id=session_id,
                task_count=len(tasks_data),
            )
            return tasks_data

        except Exception as e:
            failure_context = dict(operation_details)
            log_operation_failure(
                logger,
                "task_discovery",
                e,
                session_id=session_id,
                **failure_context,
            )
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
        task_count = len(task_ids)
        session_id: Optional[str] = None
        if self._session_manager is not None:
            session_id = self._session_manager.get_current_session_id()

        log_operation_start(
            logger,
            "dataset_creation",
            session_id=session_id,
            task_count=task_count,
            requested_task_ids=task_ids,
        )
        try:
            # Session manager gets full task data via HTTP
            if self._session_manager is None:
                raise RuntimeError("Session manager not initialized")
            tasks_data = await self._session_manager.get_tasks(task_ids)

            # Dataset manager transforms data to inspect_ai format
            if self._dataset_manager is None:
                raise RuntimeError("Dataset manager not initialized")
            dataset = await self._dataset_manager.create_dataset(tasks_data)
            log_operation_success(
                logger,
                "dataset_creation",
                session_id=session_id,
                task_count=task_count,
            )
            return dataset
        except Exception as e:
            log_operation_failure(
                logger,
                "dataset_creation",
                e,
                session_id=session_id,
                task_count=task_count,
                requested_task_ids=task_ids,
            )
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
        task_count = len(task_ids)
        session_id: Optional[str] = None
        if self._session_manager is not None:
            session_id = self._session_manager.get_current_session_id()

        log_operation_start(
            logger,
            "multi_task_evaluation_build",
            session_id=session_id,
            task_count=task_count,
            requested_task_ids=task_ids,
        )
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

            logger.info(
                "Created agent-grouped tasks",
                extra={
                    "event": "multi_task_evaluation_built",
                    "requested_task_count": len(task_ids),
                    "generated_task_count": len(tasks),
                },
            )
            log_operation_success(
                logger,
                "multi_task_evaluation_build",
                session_id=session_id,
                task_count=task_count,
                generated_task_count=len(tasks),
            )
            return tasks

        except Exception as e:
            log_operation_failure(
                logger,
                "multi_task_evaluation_build",
                e,
                session_id=session_id,
                task_count=task_count,
                requested_task_ids=task_ids,
            )
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
                    logger.error(
                        "Session manager cleanup error",
                        extra={
                            "event": "session_manager_cleanup_error",
                            "error": str(e),
                        },
                    )
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
                    logger.error(
                        "Agent manager cleanup error",
                        extra={
                            "event": "agent_manager_cleanup_error",
                            "error": str(e),
                        },
                    )

        return AgentManagerContext(self._agent_manager, available_task_ids)
