"""
SABER Agent Registry - Public API

Following SABER best practices:
- Fail-fast design with strict validation
- Type-safe implementation with Pydantic models
- No backwards compatibility - clean modern API
- Clear separation of concerns
- External API for SABER agent management

This is the HIGH-LEVEL registry that client code should interact with.
It composes low-level agent implementations with SABER infrastructure.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from ..client_session import ClientSessionManager
from ..models import SABERConfig

if TYPE_CHECKING:
    from inspect_ai.agent import Agent

logger = logging.getLogger(__name__)


class SABERAgentSpec(BaseModel):
    """Type-safe SABER agent specification for registry."""

    name: str = Field(..., description="Agent name/identifier")
    description: str = Field(..., description="Agent description")
    factory_func: str = Field(..., description="Fully qualified factory function name")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    tags: List[str] = Field(default_factory=list, description="Agent tags")
    implementation_type: str = Field(..., description="Underlying implementation type (e.g., 'inspect_ai')")

    class Config:
        extra = "forbid"  # Fail on unknown fields - SABER best practice


class SABERAgentFactoryFunc(ABC):
    """Abstract base for SABER agent factory functions."""

    @abstractmethod
    async def create_agent(self, config: SABERConfig, session_manager: ClientSessionManager, **kwargs: Any) -> Any:
        """Create a SABER agent instance.

        Args:
            config: SABER configuration
            session_manager: Session manager for SABER infrastructure
            **kwargs: Additional agent-specific parameters

        Returns:
            Agent instance (type depends on implementation)
        """
        pass


class SABERAgentRegistrationError(Exception):
    """Raised when SABER agent registration fails."""

    pass


class SABERAgentNotFoundError(Exception):
    """Raised when requested SABER agent is not found in registry."""

    pass


class SABERAgentRegistry:
    """
    Central SABER agent registry - PUBLIC API.

    This is the main registry that client code should interact with.
    It manages SABER-integrated agents that compose low-level implementations
    with SABER infrastructure (MCP tools, session management, etc.).

    Features:
    - Fail-fast registration validation
    - Type-safe agent specifications
    - Clear separation from implementation details
    - No backwards compatibility concerns
    """

    _agents: Dict[str, SABERAgentSpec] = {}
    _factory_funcs: Dict[str, Callable[..., Awaitable["Agent"]]] = {}

    @classmethod
    def register_agent(
        cls,
        name: str,
        factory_func: Callable[..., Awaitable["Agent"]],
        description: str,
        implementation_type: str,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a SABER agent in the registry.

        Args:
            name: Agent identifier (must be unique)
            factory_func: Factory function that creates the agent
            description: Human-readable description
            implementation_type: Type of underlying implementation (e.g., 'inspect_ai')
            capabilities: List of agent capabilities
            tags: List of agent tags

        Raises:
            SABERAgentRegistrationError: If registration fails
        """
        if not name:
            raise SABERAgentRegistrationError("Agent name cannot be empty")

        if not callable(factory_func):
            raise SABERAgentRegistrationError(f"Factory function for '{name}' must be callable")

        if name in cls._agents:
            raise SABERAgentRegistrationError(f"Agent '{name}' is already registered")

        if not implementation_type:
            raise SABERAgentRegistrationError("Implementation type cannot be empty")

        # Create agent spec with validation
        try:
            spec = SABERAgentSpec(
                name=name,
                description=description,
                factory_func=f"{factory_func.__module__}.{factory_func.__name__}",
                implementation_type=implementation_type,
                capabilities=capabilities or [],
                tags=tags or [],
            )
        except Exception as e:
            raise SABERAgentRegistrationError(f"Invalid agent specification for '{name}': {e}") from e

        # Store both spec and factory function
        cls._agents[name] = spec
        cls._factory_funcs[name] = factory_func

        logger.info(f"Registered SABER agent: {name} (type: {implementation_type}, factory: {spec.factory_func})")
        logger.debug(f"Total registered agents: {list(cls._agents.keys())}")

    @classmethod
    def get_agent_spec(cls, name: str) -> SABERAgentSpec:
        """Get SABER agent specification by name.

        Args:
            name: Agent name

        Returns:
            Agent specification

        Raises:
            SABERAgentNotFoundError: If agent not found
        """
        if name not in cls._agents:
            available = list(cls._agents.keys())
            logger.error(f"Agent '{name}' not found. Available agents: {available}")
            logger.debug(f"All registered agents: {cls._agents}")
            raise SABERAgentNotFoundError(f"SABER agent '{name}' not found. Available: {available}")

        return cls._agents[name]

    @classmethod
    def get_factory_func(cls, name: str) -> Callable[..., Awaitable["Agent"]]:
        """Get factory function by agent name.

        Args:
            name: Agent name

        Returns:
            Factory function

        Raises:
            SABERAgentNotFoundError: If agent not found
        """
        if name not in cls._factory_funcs:
            available = list(cls._factory_funcs.keys())
            raise SABERAgentNotFoundError(f"SABER agent factory for '{name}' not found. Available: {available}")

        return cls._factory_funcs[name]

    @classmethod
    def list_agents(cls) -> List[SABERAgentSpec]:
        """List all registered SABER agents.

        Returns:
            List of agent specifications
        """
        return list(cls._agents.values())

    @classmethod
    def list_agents_by_type(cls, implementation_type: str) -> List[SABERAgentSpec]:
        """List registered agents by implementation type.

        Args:
            implementation_type: Implementation type to filter by

        Returns:
            List of agent specifications matching the type
        """
        return [spec for spec in cls._agents.values() if spec.implementation_type == implementation_type]

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered SABER agents (primarily for testing)."""
        cls._agents.clear()
        cls._factory_funcs.clear()
        logger.debug("Cleared SABER agent registry")


def register_saber_agent(
    name: str,
    description: str,
    implementation_type: str,
    capabilities: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> Callable[[Callable[..., Awaitable["Agent"]]], Callable[..., Awaitable["Agent"]]]:
    """
    Decorator for registering SABER agent factory functions.

    Args:
        name: Agent identifier
        description: Human-readable description
        implementation_type: Type of underlying implementation
        capabilities: List of agent capabilities
        tags: List of agent tags

    Returns:
        Decorator function

    Example:
        @register_saber_agent("custom_agent", "My custom agent", "inspect_ai")
        async def create_custom_agent(config, session_manager, **kwargs):
            return custom_agent_instance
    """

    def decorator(factory_func: Callable[..., Awaitable["Agent"]]) -> Callable[..., Awaitable["Agent"]]:
        SABERAgentRegistry.register_agent(
            name=name,
            factory_func=factory_func,
            description=description,
            implementation_type=implementation_type,
            capabilities=capabilities,
            tags=tags,
        )
        return factory_func

    return decorator
