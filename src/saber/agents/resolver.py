"""Agent name resolution with helpful error messages."""

from __future__ import annotations

from collections.abc import Callable

from saber.agents import AgentNotFoundError, AgentRegistry


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
        f"To add a new agent: create agents/registry/{agent_name}.py with create_agent()"
    )
