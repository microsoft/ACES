"""
SABER Agent Manager

Manages SABER agent lifecycle for eval_async execution.
Refined from SABEREvalManager to focus solely on agent responsibilities.

BREAKING CHANGE: Removed all dataset-related functionality.
"""

import logging
from types import TracebackType
from typing import Any, List, Optional

from inspect_ai import Task

from .client_session import ClientSessionManager
from .exceptions import AgentInitializationError
from .models import AgentInfo, SABERConfig

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Manages SABER agent lifecycle for eval_async execution.

    Responsibilities (ONLY agent-related):
    - Agent discovery and initialization from registry
    - Agent lifecycle management
    - Task creation with agent configuration
    - Agent resource cleanup

    Removed responsibilities:
    - Dataset creation (moved to DatasetManager)
    - Task discovery (moved to DatasetManager)
    - HTTP client creation (use shared ClientSessionManager)
    """

    def __init__(self, config: SABERConfig, session_manager: ClientSessionManager):
        """
        Initialize agent manager with configuration and shared session manager.

        Args:
            config: SABER configuration
            session_manager: Shared ClientSessionManager instance
        """
        self.config = config
        self.session_manager = session_manager
        self.saber_agent: Optional["AgentInfo"] = None
        self._initialized = False

        logger.debug("Initialized AgentManager with shared ClientSessionManager")

    async def __aenter__(self) -> "AgentManager":
        """
        Enter async context manager and initialize agent.

        Returns:
            Initialized agent manager

        Raises:
            AgentInitializationError: Agent initialization failed
        """
        if self._initialized:
            logger.debug("AgentManager already initialized")
            return self

        logger.info("Initializing SABER agent")

        try:
            # For now, we'll skip the complex registry and just validate the agent_id
            if not hasattr(self.config, "agent_id") or not self.config.agent_id:
                raise AgentInitializationError("Agent not specified in config")

            # Create a simple agent info object instead of using registry
            self.saber_agent = AgentInfo(
                agent_id=self.config.agent_id,
                name=f"{self.config.agent_id.title()} Agent",
                description=f"SABER {self.config.agent_id} agent for security domain benchmarking",
                capabilities=["reasoning", "tool_use"],
                tags=[self.config.agent_id, "saber", "security"],
            )

            # Create session using shared session manager
            session_id = await self.session_manager.create_session()
            logger.info(f"Created evaluation session: {session_id}")

            # Note: MCP clients are now created per-episode in ensure_episode_mcp_client()

            self._initialized = True
            logger.info(f"SABER agent '{self.saber_agent.name}' initialized successfully")
            return self

        except Exception as e:
            logger.error(f"Failed to initialize SABER agent: {e}")
            raise AgentInitializationError(f"Agent initialization failed: {e}") from e

    async def __aexit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Exit async context manager and cleanup agent resources.
        """
        if not self._initialized:
            logger.debug("No SABER agent resources to cleanup")
            return

        logger.info("Cleaning up SABER agent resources")

        try:
            # AgentInfo is just metadata, no cleanup needed
            pass
        except Exception as e:
            logger.error(f"Error during SABER agent cleanup: {e}")
            # Don't raise - cleanup failures shouldn't break the main flow
        finally:
            self._initialized = False
            self.saber_agent = None
            # Note: Don't cleanup session_manager here - it's shared

    async def create_task(self, dataset: List[Any]) -> Task:
        """
        Create inspect_ai Task configured with SABER agent.

        Args:
            dataset: inspect_ai dataset

        Returns:
            Configured inspect_ai Task

        Raises:
            RuntimeError: If agent not initialized
        """
        if not self._initialized or not self.saber_agent:
            raise RuntimeError("SABER agent not initialized - use as async context manager")

        logger.debug("Creating inspect_ai Task with SABER react agent")

        # Import here to avoid circular imports
        from .inspect_ai.saber_react import saber_react
        from .inspect_ai.saber_scorer import SABERTaskScorer

        # Create SABER react agent with session manager
        saber_agent = saber_react(
            attempts=1,
            agent_id=self.saber_agent.agent_id,
            session_manager=self.session_manager,
        )

        # Create task with SABER context in metadata
        task = Task(
            dataset=dataset,
            solver=saber_agent,
            scorer=SABERTaskScorer.create_default_scorer(),
            metadata={
                "saber_agent_id": self.saber_agent.agent_id,
                "saber_agent_name": self.saber_agent.name,
                "saber_architecture": "direct_agent",
                "saber_task_id": f"saber_task_{self.saber_agent.agent_id}",  # Provide task_id for episodes
            },
        )

        return task
