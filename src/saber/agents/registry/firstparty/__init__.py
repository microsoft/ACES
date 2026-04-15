"""First-party agent runtime — sandbox_agent_bridge integration for SABER."""

from .solver import create_agent

# Import runtimes to trigger auto-registration with RuntimeRegistry.
import saber.agents.registry.firstparty.runtimes  # noqa: F401

__all__ = ["create_agent"]
