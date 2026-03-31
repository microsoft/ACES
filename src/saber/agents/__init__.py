"""Agent registry with auto-discovery.

Agents are discovered from two sources at import time:

1. **Built-in agents**: Modules/packages in ``agents/registry/`` exporting
   ``create_agent()``.
2. **Plugin agents**: External packages registered via the
   ``saber.agents`` entry-point group.  Any installed package with::

       [project.entry-points."saber.agents"]
       my_agent = "my_package.saber_adapter:create_agent"

   will be auto-discovered alongside built-in agents.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from saber.agents.models import AgentFactory
from saber.logging import get_logger

logger = get_logger(__name__)


class AgentNotFoundError(Exception):
    """Raised when requested agent implementation is not found."""


class AgentRegistry:
    """Registry of pluggable agent implementations.

    Class-level singleton mapping agent names to factory callables.
    """

    _agents: dict[str, AgentFactory] = {}

    @classmethod
    def register(cls, name: str, agent_factory: AgentFactory, *, override: bool = False) -> None:
        """Register an agent implementation.

        Args:
            name: Agent name for CLI lookup (e.g., "react", "copilot").
            agent_factory: Factory callable — called as factory() -> create_with_prompts.
            override: If True, replace an existing registration silently.

        Raises:
            ValueError: If name is already registered and override is False.
        """
        if name in cls._agents and not override:
            raise ValueError(f"Agent '{name}' is already registered")
        cls._agents[name] = agent_factory
        logger.info("Agent '%s' registered", name)

        # Invalidate capability cache so late registrations are picked up.
        try:
            from saber.agents.solver_factory import _resolve_capabilities

            _resolve_capabilities.cache_clear()
        except ImportError:
            pass

    @classmethod
    def get(cls, name: str) -> AgentFactory | None:
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
            if not callable(module.create_agent):
                logger.error(
                    "Agent module '%s' exports non-callable create_agent: %s",
                    module_path,
                    type(module.create_agent).__name__,
                )
                return
            AgentRegistry.register(name, module.create_agent)
    except Exception:
        logger.error("Failed to load agent module '%s'", module_path, exc_info=True)
        raise


def _discover_plugin_agents() -> None:
    """Discover external agents registered via entry points.

    External packages register agents by adding an entry point in the
    ``saber.agents`` group::

        [project.entry-points."saber.agents"]
        my_agent = "my_package.adapter:create_agent"

    The entry point name becomes the agent name.  The entry point value
    must resolve to a ``create_agent()`` callable following the standard
    SABER two-level factory contract.

    Agents that conflict with already-registered names (built-in or
    earlier plugins) are skipped with a warning.
    """
    from importlib.metadata import entry_points

    eps = entry_points(group="saber.agents")
    for ep in eps:
        if AgentRegistry.get(ep.name) is not None:
            logger.warning(
                "Plugin agent '%s' (from %s) skipped — name already registered",
                ep.name,
                ep.value,
            )
            continue
        try:
            factory = ep.load()
            if not callable(factory):
                logger.error(
                    "Plugin agent '%s' entry point did not resolve to a callable: %s",
                    ep.name,
                    type(factory).__name__,
                )
                continue
            AgentRegistry.register(ep.name, factory)
            logger.info("Plugin agent '%s' loaded from %s", ep.name, ep.value)
        except Exception:
            logger.error(
                "Failed to load plugin agent '%s' from %s",
                ep.name,
                ep.value,
                exc_info=True,
            )


# Auto-register on import
_register_core_agents()
_discover_plugin_agents()
