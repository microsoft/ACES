"""Agent registry with auto-discovery.

Agents are auto-discovered from the registry/ directory at import time.
Each module or package exporting a create_agent() function is registered.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

from saber.logging import get_logger

logger = get_logger(__name__)


class AgentNotFoundError(Exception):
    """Raised when requested agent implementation is not found."""


class AgentRegistry:
    """Registry of pluggable agent implementations.

    Class-level singleton mapping agent names to factory callables.
    """

    _agents: dict[str, Callable[..., object]] = {}

    @classmethod
    def register(cls, name: str, agent_factory: Callable[..., object]) -> None:
        """Register an agent implementation.

        Args:
            name: Agent name for CLI lookup (e.g., "react", "copilot").
            agent_factory: Factory callable — called as factory() -> create_with_prompts.

        Raises:
            ValueError: If name is already registered.
        """
        if name in cls._agents:
            raise ValueError(f"Agent '{name}' is already registered")
        cls._agents[name] = agent_factory
        logger.info("Agent '%s' registered", name)

    @classmethod
    def get(cls, name: str) -> Callable[..., object] | None:
        """Get agent factory by name. Returns None if not found."""
        return cls._agents.get(name)

    @classmethod
    def list_agents(cls) -> list[str]:
        """List all registered agent names."""
        return sorted(cls._agents.keys())

    @classmethod
    def _reset(cls) -> None:
        """Reset registry. For testing only."""
        cls._agents = {}


def _register_core_agents() -> None:
    """Auto-discover agents from agents/registry/ directory."""
    agents_dir = Path(__file__).parent / "registry"
    if not agents_dir.exists():
        return

    # Single-file agents: registry/react.py -> "react"
    for agent_file in sorted(agents_dir.glob("*.py")):
        if agent_file.name.startswith("_"):
            continue
        _try_register_module(
            f"saber.agents.registry.{agent_file.stem}",
            agent_file.stem,
        )

    # Package agents: registry/copilot/ -> "copilot"
    for agent_dir in sorted(agents_dir.iterdir()):
        if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
            continue
        if not (agent_dir / "__init__.py").exists():
            continue
        if AgentRegistry.get(agent_dir.name) is not None:
            continue  # Already registered as single-file
        _try_register_module(
            f"saber.agents.registry.{agent_dir.name}",
            agent_dir.name,
        )


def _try_register_module(module_path: str, name: str) -> None:
    """Import a module and register its create_agent function.

    Raises:
        Exception: If the module cannot be imported.
    """
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, "create_agent"):
            AgentRegistry.register(name, module.create_agent)
    except Exception:
        logger.error("Failed to load agent module '%s'", module_path, exc_info=True)
        raise


# Auto-register on import
_register_core_agents()
