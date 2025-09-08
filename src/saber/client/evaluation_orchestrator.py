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

from .agent_manager import AgentManager
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

            # Initialize agent manager
            logger.info("Initializing agent manager")
            self._agent_manager = AgentManager(self.config, self._session_manager)
            await self._exit_stack.enter_async_context(self._agent_manager)

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

        # Validate model configuration
        if not self.config.model:
            errors.append("model specification is required")

        # Validate session configuration
        if not self.config.session_config:
            errors.append("session_config is required")
        elif not self.config.session_config.base_url:
            errors.append("session_config.base_url is required")

        # Validate agent specification
        if not self.config.agent_id and not self.config.agent_path:
            errors.append("either agent_id or agent_path must be provided")

        # Validate agent file exists if specified
        if self.config.agent_path:
            from pathlib import Path

            if not Path(self.config.agent_path).exists():
                errors.append(f"agent file not found: {self.config.agent_path}")

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

    async def discover_tasks(self) -> List[str]:
        """
        Discover tasks by coordinating between session manager and configuration.

        Returns:
            List of task IDs to evaluate

        Raises:
            DatasetCreationError: Task discovery failed
        """
        try:
            if self.config.task_ids:
                logger.info(f"Using configured task IDs: {self.config.task_ids}")
                return self.config.task_ids
            else:
                logger.info("Discovering all available tasks")
                # Session manager handles HTTP communication
                if self._session_manager is None:
                    raise RuntimeError("Session manager not initialized")
                tasks_data = await self._session_manager.get_available_tasks()
                # Extract just the task IDs - tasks_data is a list of TaskInfo objects
                task_ids = [task.task_id for task in tasks_data]
                logger.info(f"Discovered {len(task_ids)} available tasks")
                return task_ids
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

    async def create_task(self, dataset: List[Any]) -> Any:  # TODO: Better types when available
        """
        Create inspect_ai Task using agent manager.

        Args:
            dataset: inspect_ai dataset

        Returns:
            Configured inspect_ai Task
        """
        if self._agent_manager is None:
            raise RuntimeError("Agent manager not initialized")
        return await self._agent_manager.create_task(dataset)

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
