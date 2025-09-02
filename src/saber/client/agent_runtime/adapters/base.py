#!/usr/bin/env python3
"""
Base Agent Adapter

Abstract base class for agent adapters that provide unified execution interface
for different agent implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, cast

logger = logging.getLogger(__name__)


class AgentAdapter(ABC):
    """
    Abstract base class for agent adapters.

    Provides unified interface for executing different types of agents
    (class-based, function-based, async/sync) with tool injection.
    """

    def __init__(self, agent: Any, tool_injector: Any):
        """
        Initialize agent adapter.

        Args:
            agent: Agent object or function to adapt
            tool_injector: Tool injector for function injection
        """
        self.agent = agent
        self.tool_injector = tool_injector
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
        Prepare context for agent execution with injected tool functions.

        Returns:
            Context dictionary with tool functions and utilities
        """
        context = {
            "shutdown_requested": lambda: self.shutdown_requested,
        }

        # Inject tool functions directly into context
        context = self.tool_injector.inject_into_context(context)

        return cast(Dict[str, Any], context)
