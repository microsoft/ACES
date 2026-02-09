"""Agent implementation resolution for SABER domains.

This module provides agent discovery from core SABER agents registry.

Key features:
- Core agent registry lookup
- Detailed error messages with fix suggestions
"""

from collections.abc import Callable
from pathlib import Path

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


def resolve_agent_implementation(
    domain_slug: str,
    domains_root: Path,
    agent_name: str,
) -> Callable:
    """Resolve agent implementation from core SABER agents.

    Args:
        domain_slug: Domain name
        domains_root: Path to domains directory (unused, kept for API compatibility)
        agent_name: Name of agent to resolve

    Returns:
        Agent factory callable

    Raises:
        AgentNotFoundError: If agent not found in registry
    """
    logger.debug(
        f"Resolving agent '{agent_name}' for domain '{domain_slug}'", extra={"agent": agent_name, "domain": domain_slug}
    )

    from . import SABERAgentRegistry

    core_agent = SABERAgentRegistry.get(agent_name)
    if core_agent is not None:
        logger.info(
            f"Using core SABER agent '{agent_name}'",
            extra={"agent": agent_name, "domain": domain_slug, "source": "core-registry"},
        )
        return core_agent

    # Not found - provide helpful error
    core_agents = SABERAgentRegistry.list_agents()

    error_msg = (
        f"Agent '{agent_name}' not found for domain '{domain_slug}'.\n\n"
        f"Available core SABER agents: {', '.join(core_agents) if core_agents else '(none)'}\n\n"
        f"To fix:\n"
        f"  - Use a core agent: -T agent={core_agents[0] if core_agents else 'react'}\n"
        f"  - Check spelling: '{agent_name}'"
    )

    logger.error(
        f"Agent '{agent_name}' not found",
        extra={
            "agent": agent_name,
            "domain": domain_slug,
            "core_agents": core_agents,
        },
    )

    from . import AgentNotFoundError

    raise AgentNotFoundError(error_msg)
