"""
Inspect AI Agent Implementations Registry - Low Level

Following SABER best practices:
- Single responsibility: manages inspect_ai agent implementations
- Clear separation from SABER infrastructure
- Fail-fast design
- Type-safe implementation

This is the LOW-LEVEL registry that maps names to raw inspect_ai agent functions.
It should NOT contain any SABER-specific logic - just pure inspect_ai mappings.
"""

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class InspectAIImplementationNotFoundError(Exception):
    """Raised when inspect_ai implementation not found."""

    pass


class InspectAIImplementationRegistry:
    """
    Registry for raw inspect_ai agent implementations.

    This is the low-level registry that maps simple names to actual inspect_ai
    agent functions without any SABER integration. Pure inspect_ai logic only.

    Examples:
    - "react" -> inspect_ai.agent.react
    - "chain" -> inspect_ai.agent.chain (when available)
    """

    _implementations: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register_implementation(cls, name: str, agent_func: Callable[..., Any]) -> None:
        """Register a raw inspect_ai agent implementation.

        Args:
            name: Implementation name (e.g., "react", "chain")
            agent_func: Raw inspect_ai agent function

        Raises:
            ValueError: If implementation already registered
        """
        if not name:
            raise ValueError("Implementation name cannot be empty")

        if not callable(agent_func):
            raise ValueError(f"Agent function for '{name}' must be callable")

        if name in cls._implementations:
            raise ValueError(f"inspect_ai implementation '{name}' is already registered")

        cls._implementations[name] = agent_func
        logger.info(f"Registered inspect_ai implementation: {name}")

    @classmethod
    def get_implementation(cls, name: str) -> Callable[..., Any]:
        """Get raw inspect_ai agent implementation by name.

        Args:
            name: Implementation name

        Returns:
            Raw inspect_ai agent function

        Raises:
            InspectAIImplementationNotFoundError: If implementation not found
        """
        if name not in cls._implementations:
            available = list(cls._implementations.keys())
            raise InspectAIImplementationNotFoundError(
                f"inspect_ai implementation '{name}' not found. Available: {available}"
            )

        return cls._implementations[name]

    @classmethod
    def list_implementations(cls) -> List[str]:
        """List available implementation names.

        Returns:
            List of registered implementation names
        """
        return list(cls._implementations.keys())

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all implementations (primarily for testing)."""
        cls._implementations.clear()
        logger.debug("Cleared inspect_ai implementation registry")


def register_inspect_ai_implementation(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for registering inspect_ai implementations.

    Args:
        name: Implementation name

    Returns:
        Decorator function

    Example:
        @register_inspect_ai_implementation("custom_react")
        def my_custom_react():
            return some_inspect_ai_agent
    """

    def decorator(agent_func: Callable[..., Any]) -> Callable[..., Any]:
        InspectAIImplementationRegistry.register_implementation(name, agent_func)
        return agent_func

    return decorator


# Register built-in inspect_ai implementations
def _register_builtin_implementations() -> None:
    """Register standard inspect_ai agent implementations."""
    try:
        from inspect_ai.agent import react

        InspectAIImplementationRegistry.register_implementation("react", react)
        logger.info("Registered built-in inspect_ai.agent.react")
    except ImportError:
        logger.warning("Could not import inspect_ai.agent.react - may not be available")

    # Register other inspect_ai agents as they become available
    # try:
    #     from inspect_ai.agent import chain
    #     InspectAIImplementationRegistry.register_implementation("chain", chain)
    # except ImportError:
    #     logger.debug("inspect_ai.agent.chain not available")


# Auto-register built-in implementations when this module is imported
_register_builtin_implementations()
