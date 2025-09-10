"""
SABER Agent Factory - Composition Engine

Following SABER best practices:
- Single responsibility: creates configured agents
- Fail-fast validation
- Type-safe implementation
- Clear dependency injection

This factory composes low-level agent implementations with SABER infrastructure.
It delegates to implementation-specific factories but provides the unified interface.
"""

import logging
from typing import Any, Type

from ..client_session import ClientSessionManager
from ..models import SABERConfig
from .registry import SABERAgentNotFoundError, SABERAgentRegistrationError, SABERAgentRegistry

logger = logging.getLogger(__name__)


class SABERAgentFactory:
    """
    SABER agent factory following best practices.

    Features:
    - Dynamic agent creation based on config
    - Fail-fast validation
    - Type-safe implementation
    - Integration with SABER infrastructure
    - Clear separation from implementation details
    """

    def __init__(self, registry: Type[SABERAgentRegistry] = SABERAgentRegistry):
        """Initialize factory with registry.

        Args:
            registry: Agent registry class (for dependency injection)
        """
        self.registry = registry

    async def create_agent(
        self, agent_id: str, config: SABERConfig, session_manager: ClientSessionManager, **kwargs: Any
    ) -> Any:
        """
        Create SABER agent instance from configuration.

        Args:
            agent_id: Agent identifier from config
            config: SABER configuration
            session_manager: Session manager for SABER infrastructure
            **kwargs: Additional agent-specific parameters

        Returns:
            Agent instance (type depends on implementation)

        Raises:
            SABERAgentNotFoundError: If agent not found in registry
            SABERAgentRegistrationError: If agent creation fails
        """
        logger.info(f"Creating SABER agent: {agent_id}")

        # Get agent specification (fail-fast if not found)
        try:
            spec = self.registry.get_agent_spec(agent_id)
            factory_func = self.registry.get_factory_func(agent_id)
        except SABERAgentNotFoundError as e:
            logger.error(f"Failed to find SABER agent '{agent_id}': {e}")
            raise

        # Create agent using factory function
        try:
            agent = await factory_func(config=config, session_manager=session_manager, agent_id=agent_id, **kwargs)

            logger.info(
                f"Successfully created SABER agent: {agent_id} "
                f"(type: {spec.implementation_type}, factory: {spec.factory_func})"
            )
            return agent

        except Exception as e:
            logger.error(f"Failed to create SABER agent '{agent_id}': {e}")
            raise SABERAgentRegistrationError(f"SABER agent creation failed for '{agent_id}': {e}") from e

    def list_available_agents(self) -> list[str]:
        """List all available agent IDs.

        Returns:
            List of agent identifiers
        """
        return [spec.name for spec in self.registry.list_agents()]

    def list_agents_by_implementation(self, implementation_type: str) -> list[str]:
        """List agents by implementation type.

        Args:
            implementation_type: Implementation type to filter by

        Returns:
            List of agent identifiers
        """
        return [spec.name for spec in self.registry.list_agents_by_type(implementation_type)]
