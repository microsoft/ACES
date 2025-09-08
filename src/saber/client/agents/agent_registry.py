"""
SABER Agent Registry

Simple agent registration system for SABER agents.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..models import AgentInfo

logger = logging.getLogger(__name__)


@dataclass
class AgentRegistryEntry:
    """Entry in the agent registry."""

    agent_info: AgentInfo
    factory_func: Callable[..., Any]


class AgentRegistry:
    """Registry for SABER agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentRegistryEntry] = {}
        self._initialized = False

    def register_agent(self, agent_info: AgentInfo, factory_func: Callable[..., Any]) -> None:
        """Register an agent with the registry."""
        logger.debug(f"Registering agent: {agent_info.agent_id}")

        self._agents[agent_info.agent_id] = AgentRegistryEntry(agent_info=agent_info, factory_func=factory_func)

    def get_agent_factory(self, agent_id: str) -> Optional[Callable[..., Any]]:
        """Get agent factory function by ID."""
        entry = self._agents.get(agent_id)
        return entry.factory_func if entry else None

    def get_available_agents(self) -> List[AgentInfo]:
        """Get list of all available agents."""
        return [entry.agent_info for entry in self._agents.values()]

    def create_agent(self, agent_id: str, **kwargs: Any) -> Optional[Any]:
        """Create an agent instance by ID."""
        factory = self.get_agent_factory(agent_id)
        if factory is None:
            logger.error(f"No factory found for agent: {agent_id}")
            return None

        try:
            agent = factory(**kwargs)
            logger.debug(f"Created agent: {agent_id}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent {agent_id}: {e}")
            return None

    def has_agent(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def list_agent_ids(self) -> List[str]:
        return list(self._agents.keys())


# Global registry instance
agent_registry = AgentRegistry()
