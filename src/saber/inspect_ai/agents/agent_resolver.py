"""Agent implementation resolution for SABER domains.

This module provides three-tier agent discovery:
1. Domain-local agents (domains/{domain}/client/{agent}.py)
2. Core SABER agents (saber.inspect_ai.agents.{agent})
3. Error with helpful guidance if not found

Key features:
- Polymorphic agent loading
- Domain-specific agent overrides
- Detailed error messages with fix suggestions
"""

from pathlib import Path
from typing import Callable, Optional

from ...logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


def resolve_agent_implementation(
    domain_slug: str,
    domains_root: Path,
    agent_name: str,
) -> Callable:
    """Resolve agent implementation with three-tier discovery.

    Discovery order:
    1. Domain-local agents (domains/{domain}/client/{agent}.py)
    2. Core SABER agents (saber.inspect_ai.agents.{agent})
    3. Error if not found

    Args:
        domain_slug: Domain name
        domains_root: Path to domains directory
        agent_name: Name of agent to resolve

    Returns:
        Agent factory callable

    Raises:
        AgentNotFoundError: If agent not found in any registry
    """
    logger.debug(
        f"Resolving agent '{agent_name}' for domain '{domain_slug}'", extra={"agent": agent_name, "domain": domain_slug}
    )

    # 1. Try domain-local agent first
    domain_agent = load_domain_agent(domain_slug, domains_root, agent_name)
    if domain_agent is not None:
        logger.info(
            f"Using domain-local agent '{agent_name}' from {domain_slug}/client/",
            extra={"agent": agent_name, "domain": domain_slug, "source": "domain-local"},
        )
        return domain_agent

    # 2. Try core SABER agent registry
    from . import SABERAgentRegistry

    core_agent = SABERAgentRegistry.get(agent_name)
    if core_agent is not None:
        logger.info(
            f"Using core SABER agent '{agent_name}'",
            extra={"agent": agent_name, "domain": domain_slug, "source": "core-registry"},
        )
        return core_agent

    # 3. Not found - provide helpful error
    core_agents = SABERAgentRegistry.list_agents()
    domain_client_dir = domains_root / domain_slug / "client"

    error_msg = (
        f"Agent '{agent_name}' not found for domain '{domain_slug}'.\n\n"
        f"Agent discovery order:\n"
        f"  1. Domain-local: {domain_client_dir}/{agent_name}.py\n"
        f"  2. Core SABER agents: {', '.join(core_agents) if core_agents else '(none)'}\n\n"
        f"To fix:\n"
        f"  - Use a core agent: -T agent={core_agents[0] if core_agents else 'react'}\n"
        f"  - Create domain agent: {domain_client_dir}/{agent_name}.py with create_agent() function\n"
        f"  - Check spelling: '{agent_name}'"
    )

    logger.error(
        f"Agent '{agent_name}' not found",
        extra={
            "agent": agent_name,
            "domain": domain_slug,
            "core_agents": core_agents,
            "domain_client_dir": str(domain_client_dir),
        },
    )

    from . import AgentNotFoundError

    raise AgentNotFoundError(error_msg)


def load_domain_agent(
    domain_slug: str,
    domains_root: Path,
    agent_name: str,
) -> Optional[Callable]:
    """Load agent from domain's client folder.

    Searches: domains/{domain_slug}/client/{agent_name}.py
    Expects: A module with create_agent() function

    Args:
        domain_slug: Domain name
        domains_root: Path to domains directory
        agent_name: Agent name (filename without .py)

    Returns:
        Agent factory callable, or None if not found
    """
    agent_file = domains_root / domain_slug / "client" / f"{agent_name}.py"

    if not agent_file.exists():
        logger.debug(
            f"Domain-local agent file not found: {agent_file}", extra={"agent": agent_name, "domain": domain_slug}
        )
        return None

    try:
        # Dynamically import the module
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(f"domain_{domain_slug}_agent_{agent_name}", agent_file)
        if spec is None or spec.loader is None:
            logger.warning(
                f"Failed to load agent module spec: {agent_file}", extra={"agent": agent_name, "domain": domain_slug}
            )
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Look for create_agent function
        if hasattr(module, "create_agent"):
            logger.debug(
                f"Loaded domain-local agent from {agent_file}", extra={"agent": agent_name, "domain": domain_slug}
            )
            return module.create_agent  # type: ignore[no-any-return]
        else:
            logger.warning(
                f"Domain agent file {agent_file} missing create_agent() function",
                extra={"agent": agent_name, "domain": domain_slug, "file": str(agent_file)},
            )
            return None

    except Exception as e:
        logger.warning(
            f"Failed to load domain agent from {agent_file}: {e}",
            extra={"agent": agent_name, "domain": domain_slug, "error": str(e)},
        )
        return None
