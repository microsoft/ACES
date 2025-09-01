#!/usr/bin/env python3
"""
Base Agent Adapter

Abstract base class for agent adapters that provide unified execution interface
for different agent implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AgentAdapter(ABC):
    """
    Abstract base class for agent adapters.

    Provides unified interface for executing different types of agents
    (class-based, function-based, async/sync) with MCP client access.
    """

    def __init__(self, agent: Any, mcp_client: Any):
        """
        Initialize agent adapter.

        Args:
            agent: Agent object or function to adapt
            mcp_client: MCP client for tool access
        """
        self.agent = agent
        self.mcp_client = mcp_client
        self.shutdown_requested = False

    @abstractmethod
    async def run(self, initial_prompt: str) -> Any:
        """
        Execute agent with initial prompt.

        Args:
            initial_prompt: Initial prompt to send to agent

        Returns:
            Agent execution result
        """
        pass

    def request_shutdown(self) -> None:
        """Request graceful shutdown of agent."""
        self.shutdown_requested = True
        logger.info("🛑 Agent shutdown requested")

    def _prepare_agent_context(self) -> Dict[str, Any]:
        """
        Prepare context for agent execution.

        Returns:
            Context dictionary with MCP client and other utilities
        """
        return {
            "mcp_client": self.mcp_client,
            "tools": self.mcp_client,  # Alias for compatibility
            "shutdown_requested": lambda: self.shutdown_requested,
        }
