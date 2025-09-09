"""
SABER Agent Base Infrastructure

Provides common SABER functionality that all agents need:
- Episode management with tool call limits from server
- Dynamic prompt fetching from policy endpoint
- Session and context management
- Fail-fast error handling

MCP tools are now handled natively by inspect_ai via mcp_server_http().

Following SABER best practices:
- No backwards compatibility
- Fail-fast design
- Type-safe implementation
- Async context management
"""

import logging
from typing import Any, Dict, List, Optional

from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageSystem
from inspect_ai.tool import Tool
from inspect_ai.util import store

from ...models import EpisodeCreateResponse
from ..client_session import ClientSessionManager

logger = logging.getLogger(__name__)


class SABERAgentError(Exception):
    """Base exception for SABER agent errors."""

    pass


class SABERAgentContext:
    """
    SABER agent context manager for common infrastructure.

    Handles:
    - Episode creation and management
    - Dynamic prompt fetching
    - Session lifecycle management
    - Fail-fast error handling

    Note: MCP tools are now handled natively by inspect_ai via mcp_server_http().
    """

    def __init__(
        self,
        session_manager: ClientSessionManager,
        agent_id: str,
        default_prompt: str = "You are a security domain agent.",
    ):
        """Initialize SABER agent context.

        Args:
            session_manager: SABER session manager
            agent_id: Agent identifier
            default_prompt: Default prompt if policy fetch fails
        """
        self.session_manager = session_manager
        self.agent_id = agent_id
        self.default_prompt = default_prompt
        self.episode: Optional[EpisodeCreateResponse] = None
        self.enhanced_prompt: str = default_prompt

    async def __aenter__(self) -> "SABERAgentContext":
        """Initialize SABER context and episode."""
        await self._initialize_saber_context()
        await self._fetch_dynamic_prompt()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup SABER context and end episode."""
        if self.episode and self.session_manager:
            try:
                # Determine result based on exception
                if exc_type is None:
                    result = "success"
                    reason = "completed"
                else:
                    result = str(exc_val) if exc_val else "error"
                    reason = "error"

                await self.session_manager.end_episode(
                    self.episode.session_id, self.episode.episode_id, reason=reason, result=result
                )
            except Exception as e:
                logger.error(f"Failed to end episode: {e}")
                # Don't raise - cleanup failures shouldn't break the main flow

    def get_enhanced_state(self, state: AgentState) -> AgentState:
        """Enhance state with SABER prompt if available.

        Args:
            state: Original agent state

        Returns:
            Enhanced agent state with SABER prompt
        """
        if self.enhanced_prompt == self.default_prompt:
            return state  # No enhancement needed

        enhanced_messages = list(state.messages)
        enhanced_messages.insert(0, ChatMessageSystem(content=self.enhanced_prompt))
        enhanced_state = AgentState(messages=enhanced_messages)
        enhanced_state.output = state.output
        return enhanced_state

    def get_all_tools(self, base_tools: List[Tool]) -> List[Tool]:
        """Return base tools (MCP tools now handled natively by inspect_ai).

        Args:
            base_tools: Base inspect_ai tools

        Returns:
            Base tool list (MCP integration handled by inspect_ai)
        """
        return list(base_tools)

    def get_mcp_headers(self) -> Dict[str, str]:
        """Get MCP headers for session and episode context.

        Returns:
            Headers dict with session_id and episode_id for MCP server
        """
        if not self.episode:
            raise SABERAgentError("Episode not initialized - cannot create MCP headers")

        return {
            "X-SABER-Session-ID": self.episode.session_id,
            "X-SABER-Episode-ID": self.episode.episode_id,
        }

    async def _initialize_saber_context(self) -> None:
        """Initialize SABER context in the task store."""
        # Configure inspect_ai debug logging
        self._configure_inspect_ai_debug_logging()

        # Get SABER context from store
        task_store = store()

        # Store session manager
        task_store.set("saber_session_manager", self.session_manager)

        if self.session_manager is None:
            raise SABERAgentError("Session manager is required but not provided")

        session_id = self.session_manager.get_current_session_id()
        if session_id is None:
            raise SABERAgentError("Session ID not available from session manager")
        task_store.set("saber_session_id", session_id)

        # Get task_id from sample metadata - FAIL FAST, NO FALLBACKS
        current_sample = task_store.get("sample")
        if current_sample is None:
            raise SABERAgentError("Current sample not found in task store")

        if not hasattr(current_sample, "metadata") or current_sample.metadata is None:
            raise SABERAgentError("Sample metadata is missing")

        task_id = current_sample.metadata.get("task_id")
        if task_id is None:
            raise SABERAgentError("task_id not found in sample metadata")

        task_store.set("saber_task_id", task_id)
        logger.info(f"Using task_id: {task_id}")

        try:
            self.episode = await self.session_manager.create_episode(session_id, task_id)
            task_store.set("saber_current_episode", self.episode)
            logger.info(f"Created SABER episode: {self.episode.episode_id}")
        except Exception as e:
            logger.error(f"Failed to create SABER episode: {e}")
            raise SABERAgentError(f"Episode creation failed: {e}") from e

        # Mark as initialized
        task_store.set("saber_initialized", True)
        task_store.set("saber_agent_id", self.agent_id)

    async def _fetch_dynamic_prompt(self) -> None:
        """Fetch dynamic prompt from policy endpoint."""
        if not self.episode:
            return

        try:
            policy_response = await self.session_manager.get_policy_response(
                self.episode.session_id, self.episode.episode_id
            )
            if policy_response and policy_response.prompt:
                self.enhanced_prompt = policy_response.prompt
                logger.info("Using dynamic prompt from policy endpoint")
        except Exception as e:
            logger.warning(f"Failed to fetch policy prompt: {e}")
            # Continue with default prompt

    def _configure_inspect_ai_debug_logging(self) -> None:
        """Configure inspect_ai loggers to DEBUG level for SABER debugging."""
        import logging

        # Set the specific inspect_ai tool loggers to DEBUG
        inspect_ai_tool_logger = logging.getLogger("inspect_ai.tool._tool_def")
        inspect_ai_tool_logger.setLevel(logging.DEBUG)

        inspect_ai_info_logger = logging.getLogger("inspect_ai.tool._tool_info")
        inspect_ai_info_logger.setLevel(logging.DEBUG)

        logger.debug("SABER DEBUG: Configured inspect_ai tool loggers to DEBUG level")
