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
        self, agent_class: str, config: SABERConfig, session_manager: ClientSessionManager, **kwargs: Any
    ) -> Any:
        """
        Create SABER agent instance from agent class with kwargs validation.

        Args:
            agent_class: Agent class identifier (e.g., "mcp_agent_claude")
            config: SABER configuration
            session_manager: Session manager for SABER infrastructure
            **kwargs: Additional agent-specific parameters (validated and forwarded)

        Returns:
            Agent instance (type depends on implementation)

        Raises:
            SABERAgentNotFoundError: If agent class not found in registry
            SABERAgentRegistrationError: If agent creation fails
            ValueError: If kwargs validation fails
        """
        logger.info(f"Creating SABER agent: {agent_class}")
        logger.debug(f"Agent creation kwargs: {list(kwargs.keys())}")

        # Get agent specification (fail-fast if not found)
        try:
            spec = self.registry.get_agent_spec(agent_class)
            factory_func = self.registry.get_factory_func(agent_class)
        except SABERAgentNotFoundError as e:
            logger.error(f"Failed to find SABER agent '{agent_class}': {e}")
            raise

        # Validate kwargs against agent specification if validation is available
        validated_kwargs = self._validate_kwargs(agent_class, spec, kwargs)

        # Create agent using factory function with validated kwargs
        try:
            agent = await factory_func(
                config=config, session_manager=session_manager, agent_id=agent_class, **validated_kwargs
            )

            logger.info(
                f"Successfully created SABER agent: {agent_class} "
                f"(type: {spec.implementation_type}, factory: {spec.factory_func})"
            )
            return agent

        except Exception as e:
            logger.error(f"Failed to create SABER agent '{agent_class}': {e}")
            raise SABERAgentRegistrationError(f"SABER agent creation failed for '{agent_class}': {e}") from e

    def _validate_kwargs(self, agent_class: str, spec: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and prepare kwargs for agent creation.

        Args:
            agent_class: Agent class identifier
            spec: Agent specification from registry
            kwargs: Raw kwargs to validate

        Returns:
            Validated kwargs dictionary

        Raises:
            ValueError: If validation fails
        """
        if not kwargs:
            logger.debug(f"No kwargs provided for agent '{agent_class}'")
            return {}

        # For now, do basic validation - could be enhanced with schema validation
        validated = kwargs.copy()

        # Log potentially problematic kwargs
        common_conflicts = ["config", "session_manager", "agent_id"]
        for conflict in common_conflicts:
            if conflict in validated:
                logger.warning(
                    f"Agent '{agent_class}' kwargs contains '{conflict}' which may conflict with factory parameters"
                )

        logger.debug(f"Validated {len(validated)} kwargs for agent '{agent_class}': {list(validated.keys())}")
        return validated

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
