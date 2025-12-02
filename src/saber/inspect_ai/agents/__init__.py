"""SABER core agent registry and agent-related components.

This module provides the core registry of agent implementations available
across all SABER domains. Domain-specific agents can be defined in each
domain's client/ folder.

Agent Discovery Order:
1. Domain-local agents (domains/{domain}/client/{agent}.py)
2. Core SABER agents (saber.inspect_ai.agents.{agent})
3. Error if not found

Core agents are maintained here for reusability across domains.
"""

from pathlib import Path
from typing import Callable, Dict, Optional

from ...logging_config import LogCategory, get_saber_logger

# Import and re-export agent-related components
from .agent_resolver import resolve_agent_implementation
from .role_config_processor import all_roles_have_models, process_role_configuration
from .solver_factory import create_saber_solver

logger = get_saber_logger(LogCategory.AGENT, __name__)


class AgentNotFoundError(Exception):
    """Raised when requested agent implementation is not found."""

    pass


class SABERAgentRegistry:
    """Registry for core SABER agent implementations.

    This registry maintains reusable agent implementations that can be
    used across all domains. Domain-specific agents should be placed in
    the domain's client/ folder.
    """

    _agents: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, agent_factory: Callable) -> None:
        """Register a core agent implementation.

        Args:
            name: Agent name (e.g., "react", "chain")
            agent_factory: Callable that returns an Agent

        Raises:
            ValueError: If agent name already registered
        """
        if name in cls._agents:
            raise ValueError(f"Agent '{name}' is already registered in core registry")

        cls._agents[name] = agent_factory
        logger.info(f"Core agent '{name}' registered", extra={"event": "core_agent_registered", "agent": name})

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """Get a core agent implementation by name.

        Args:
            name: Agent name

        Returns:
            Agent factory callable, or None if not found
        """
        return cls._agents.get(name)

    @classmethod
    def list_agents(cls) -> list[str]:
        """List all registered core agent names.

        Returns:
            List of agent names
        """
        return list(cls._agents.keys())


# Auto-discover and register core agents from this directory
def _register_core_agents() -> None:
    """Auto-discover and register core agent implementations."""
    agents_dir = Path(__file__).parent / "registry"

    if not agents_dir.exists():
        logger.debug("No registry directory found, skipping agent auto-registration")
        return

    # Look for Python files in the registry directory (excluding __init__.py)
    for agent_file in agents_dir.glob("*.py"):
        if agent_file.name.startswith("_"):
            continue

        agent_name = agent_file.stem  # filename without .py

        try:
            # Dynamically import the module
            module_name = f"saber.inspect_ai.agents.registry.{agent_name}"
            import importlib

            module = importlib.import_module(module_name)

            # Look for create_agent function
            if hasattr(module, "create_agent"):
                SABERAgentRegistry.register(agent_name, module.create_agent)
                logger.debug(f"Auto-registered core agent from {agent_file.name}", extra={"agent": agent_name})
            else:
                logger.warning(
                    f"Agent file {agent_file.name} missing create_agent() function", extra={"file": str(agent_file)}
                )
        except Exception as e:
            logger.warning(
                f"Failed to auto-register agent from {agent_file.name}: {e}",
                extra={"file": str(agent_file), "error": str(e)},
            )


# Auto-register on import
_register_core_agents()


__all__ = [
    "SABERAgentRegistry",
    "AgentNotFoundError",
    "resolve_agent_implementation",
    "create_saber_solver",
    "all_roles_have_models",
    "process_role_configuration",
]
