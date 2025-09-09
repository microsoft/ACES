"""
SABER Agent Registry and Factory System - Inspect AI Integration

Following SABER best practices:
- Fail-fast design with strict validation
- Type-safe implementation with Pydantic models
- No backwards compatibility - clean modern API
- Explicit registration pattern
- Dynamic agent construction based on config

This module contains inspect_ai-specific implementations that were moved from the core
client to maintain healthy separation between external dependencies and SABER core code.

Architecture:
- AgentRegistry: Central registry for inspect_ai agent types
- AgentFactory: Creates inspect_ai agents based on configuration
- AgentSpec: Type-safe agent specification
- Registration decorators for easy agent registration
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

from inspect_ai.agent import Agent
from pydantic import BaseModel, Field

from ..client_session import ClientSessionManager
from ..models import SABERConfig

logger = logging.getLogger(__name__)


class InspectAIAgentSpec(BaseModel):
    """Type-safe inspect_ai agent specification for registry."""

    name: str = Field(..., description="Agent name/identifier")
    description: str = Field(..., description="Agent description")
    factory_func: str = Field(..., description="Fully qualified factory function name")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    tags: List[str] = Field(default_factory=list, description="Agent tags")

    class Config:
        extra = "forbid"  # Fail on unknown fields - SABER best practice


class InspectAIAgentFactoryFunc(ABC):
    """Abstract base for inspect_ai agent factory functions."""

    @abstractmethod
    async def create_agent(self, config: SABERConfig, session_manager: ClientSessionManager, **kwargs: Any) -> Agent:
        """Create an inspect_ai agent instance.

        Args:
            config: SABER configuration
            session_manager: Session manager for SABER infrastructure
            **kwargs: Additional agent-specific parameters

        Returns:
            inspect_ai Agent instance
        """
        pass


class InspectAIAgentRegistrationError(Exception):
    """Raised when inspect_ai agent registration fails."""

    pass


class InspectAIAgentNotFoundError(Exception):
    """Raised when requested inspect_ai agent is not found in registry."""

    pass


class InspectAIAgentRegistry:
    """
    Central inspect_ai agent registry following SABER best practices.

    Features:
    - Fail-fast registration validation
    - Type-safe agent specifications
    - Dynamic agent discovery
    - No backwards compatibility concerns
    """

    _agents: Dict[str, InspectAIAgentSpec] = {}
    _factory_funcs: Dict[str, Callable] = {}

    @classmethod
    def register_agent(
        cls,
        name: str,
        factory_func: Callable,
        description: str,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register an inspect_ai agent in the registry.

        Args:
            name: Agent identifier (must be unique)
            factory_func: Factory function that creates the agent
            description: Human-readable description
            capabilities: List of agent capabilities
            tags: List of agent tags

        Raises:
            InspectAIAgentRegistrationError: If registration fails
        """
        if not name:
            raise InspectAIAgentRegistrationError("Agent name cannot be empty")

        if not callable(factory_func):
            raise InspectAIAgentRegistrationError(f"Factory function for '{name}' must be callable")

        if name in cls._agents:
            raise InspectAIAgentRegistrationError(f"Agent '{name}' is already registered")

        # Create agent spec with validation
        try:
            spec = InspectAIAgentSpec(
                name=name,
                description=description,
                factory_func=f"{factory_func.__module__}.{factory_func.__name__}",
                capabilities=capabilities or [],
                tags=tags or [],
            )
        except Exception as e:
            raise InspectAIAgentRegistrationError(f"Invalid agent specification for '{name}': {e}") from e

        # Store both spec and factory function
        cls._agents[name] = spec
        cls._factory_funcs[name] = factory_func

        logger.info(f"Registered inspect_ai agent: {name} ({spec.factory_func})")
        logger.debug(f"Total registered agents: {list(cls._agents.keys())}")

    @classmethod
    def get_agent_spec(cls, name: str) -> InspectAIAgentSpec:
        """Get inspect_ai agent specification by name.

        Args:
            name: Agent name

        Returns:
            Agent specification

        Raises:
            InspectAIAgentNotFoundError: If agent not found
        """
        if name not in cls._agents:
            available = list(cls._agents.keys())
            logger.error(f"Agent '{name}' not found. Available agents: {available}")
            logger.debug(f"All registered agents: {cls._agents}")
            raise InspectAIAgentNotFoundError(f"Inspect AI agent '{name}' not found. Available: {available}")

        return cls._agents[name]

    @classmethod
    def get_factory_func(cls, name: str) -> Callable:
        """Get factory function by agent name.

        Args:
            name: Agent name

        Returns:
            Factory function

        Raises:
            InspectAIAgentNotFoundError: If agent not found
        """
        if name not in cls._factory_funcs:
            available = list(cls._factory_funcs.keys())
            raise InspectAIAgentNotFoundError(
                f"Inspect AI agent factory for '{name}' not found. Available: {available}"
            )

        return cls._factory_funcs[name]

    @classmethod
    def list_agents(cls) -> List[InspectAIAgentSpec]:
        """List all registered inspect_ai agents.

        Returns:
            List of agent specifications
        """
        return list(cls._agents.values())

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered inspect_ai agents (primarily for testing)."""
        cls._agents.clear()
        cls._factory_funcs.clear()
        logger.debug("Cleared inspect_ai agent registry")


class InspectAIAgentFactory:
    """
    Inspect AI agent factory following SABER best practices.

    Features:
    - Dynamic agent creation based on config
    - Fail-fast validation
    - Type-safe implementation
    - Integration with SABER infrastructure
    """

    def __init__(self, registry: Type[InspectAIAgentRegistry] = InspectAIAgentRegistry):
        """Initialize factory with registry.

        Args:
            registry: Agent registry class (for dependency injection)
        """
        self.registry = registry

    async def create_agent(
        self, agent_id: str, config: SABERConfig, session_manager: ClientSessionManager, **kwargs: Any
    ) -> Agent:
        """
        Create inspect_ai agent instance from configuration.

        Args:
            agent_id: Agent identifier from config
            config: SABER configuration
            session_manager: Session manager for SABER infrastructure
            **kwargs: Additional agent-specific parameters

        Returns:
            inspect_ai Agent instance

        Raises:
            InspectAIAgentNotFoundError: If agent not found in registry
            InspectAIAgentRegistrationError: If agent creation fails
        """
        logger.info(f"Creating inspect_ai agent: {agent_id}")

        # Get agent specification (fail-fast if not found)
        try:
            spec = self.registry.get_agent_spec(agent_id)
            factory_func = self.registry.get_factory_func(agent_id)
        except InspectAIAgentNotFoundError as e:
            logger.error(f"Failed to find inspect_ai agent '{agent_id}': {e}")
            raise

        # Create agent using factory function
        try:
            agent = await factory_func(config=config, session_manager=session_manager, agent_id=agent_id, **kwargs)

            logger.info(f"Successfully created inspect_ai agent: {agent_id} ({spec.factory_func})")
            return agent

        except Exception as e:
            logger.error(f"Failed to create inspect_ai agent '{agent_id}': {e}")
            raise InspectAIAgentRegistrationError(f"Inspect AI agent creation failed for '{agent_id}': {e}") from e


def register_inspect_ai_agent(
    name: str, description: str, capabilities: Optional[List[str]] = None, tags: Optional[List[str]] = None
) -> Callable:
    """
    Decorator for registering inspect_ai agent factory functions.

    Args:
        name: Agent identifier
        description: Human-readable description
        capabilities: List of agent capabilities
        tags: List of agent tags

    Returns:
        Decorator function

    Example:
        @register_inspect_ai_agent("custom_agent", "My custom agent")
        async def create_custom_agent(config, session_manager, **kwargs):
            return custom_agent_instance
    """

    def decorator(factory_func: Callable) -> Callable:
        InspectAIAgentRegistry.register_agent(
            name=name, factory_func=factory_func, description=description, capabilities=capabilities, tags=tags
        )
        return factory_func

    return decorator
