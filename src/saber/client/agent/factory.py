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

from typing import Any, Type

from ...logging_config import get_agent_logger
from ..client_session import ClientSessionManager
from ..models import SABERConfig
from .registry import SABERAgentNotFoundError, SABERAgentRegistrationError, SABERAgentRegistry

logger = get_agent_logger(__name__)


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
        logger.info(
            "Creating SABER agent",
            extra={
                "event": "agent_creation_started",
                "agent_class": agent_class,
                "config_id": getattr(config, "name", None),
                "kwarg_keys": sorted(kwargs.keys()),
            },
        )

        if kwargs:
            logger.debug(
                "Agent creation kwargs detailed",
                extra={
                    "agent_class": agent_class,
                    "kwargs": {key: type(value).__name__ for key, value in kwargs.items()},
                },
            )

        # Get agent specification (fail-fast if not found)
        try:
            spec = self.registry.get_agent_spec(agent_class)
            factory_func = self.registry.get_factory_func(agent_class)
        except SABERAgentNotFoundError as exc:
            logger.error(
                "Agent registration missing",
                extra={
                    "event": "agent_creation_missing",
                    "agent_class": agent_class,
                    "error": str(exc),
                },
            )
            raise

        # Validate kwargs against agent specification if validation is available
        validated_kwargs = self._validate_kwargs(agent_class, spec, kwargs)

        # Create agent using factory function with validated kwargs
        try:
            agent = await factory_func(
                config=config, session_manager=session_manager, agent_id=agent_class, **validated_kwargs
            )

            logger.info(
                "Agent created",
                extra={
                    "event": "agent_creation_completed",
                    "agent_class": agent_class,
                    "implementation_type": getattr(spec, "implementation_type", None),
                    "factory": getattr(spec, "factory_func", None),
                },
            )
            return agent

        except Exception as exc:  # pragma: no cover - propagate with enriched context
            logger.error(
                "Agent creation failed",
                extra={
                    "event": "agent_creation_failed",
                    "agent_class": agent_class,
                    "error": str(exc),
                },
            )
            raise SABERAgentRegistrationError(f"SABER agent creation failed for '{agent_class}': {exc}") from exc

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
            logger.debug(
                "No agent kwargs provided",
                extra={
                    "agent_class": agent_class,
                },
            )
            return {}

        # For now, do basic validation - could be enhanced with schema validation
        validated = kwargs.copy()

        # Log potentially problematic kwargs
        common_conflicts = ["config", "session_manager", "agent_id"]
        for conflict in common_conflicts:
            if conflict in validated:
                logger.warning(
                    "Agent kwargs overlaps with factory parameters",
                    extra={
                        "event": "agent_kwargs_conflict",
                        "agent_class": agent_class,
                        "conflicting_key": conflict,
                    },
                )

        logger.debug(
            "Validated agent kwargs",
            extra={
                "agent_class": agent_class,
                "validated_keys": sorted(validated.keys()),
                "validated_count": len(validated),
            },
        )
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
