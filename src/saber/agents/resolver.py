"""Agent name resolution with helpful error messages."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from saber.agents import AgentNotFoundError, AgentRegistry
from saber.logging import get_logger

logger = get_logger(__name__)


def register_agent_package(agent_name: str, module_path: str) -> None:
    """Dynamically import and register an external agent module.

    Use this during development when the external package is not installed
    with entry points.  For example::

        register_agent_package("my_agent", "my_package.saber_adapter")

    The module must export a ``create_agent()`` callable.

    Args:
        agent_name: Name to register the agent under.
        module_path: Dotted Python module path to import.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the module has no ``create_agent``.
    """
    module = importlib.import_module(module_path)
    factory = getattr(module, "create_agent", None)
    if factory is None or not callable(factory):
        raise AttributeError(
            f"Module '{module_path}' does not export a callable create_agent()"
        )
    if AgentRegistry.get(agent_name) is not None:
        logger.warning(
            "Agent '%s' already registered — overriding with %s",
            agent_name,
            module_path,
        )
        AgentRegistry._agents[agent_name] = factory
    else:
        AgentRegistry.register(agent_name, factory)
    logger.info("Agent '%s' registered from module '%s'", agent_name, module_path)


def resolve_agent(agent_name: str) -> Callable[..., object]:
    """Resolve agent factory by name from the registry.

    Args:
        agent_name: Registered agent name (e.g., "react", "copilot").

    Returns:
        Agent factory callable.

    Raises:
        AgentNotFoundError: With helpful message listing available agents.
    """
    factory = AgentRegistry.get(agent_name)
    if factory is not None:
        return factory  # type: ignore[no-any-return]

    available = AgentRegistry.list_agents()
    raise AgentNotFoundError(
        f"Agent '{agent_name}' not found.\n\n"
        f"Available agents: {', '.join(available) if available else '(none)'}\n\n"
        f"To add a built-in agent: create agents/registry/{agent_name}.py with create_agent()\n"
        f"To register an external agent: add a 'saber.agents' entry point in your package's pyproject.toml:\n"
        f"  [project.entry-points.\"saber.agents\"]\n"
        f"  {agent_name} = \"your_package.module:create_agent\""
    )
