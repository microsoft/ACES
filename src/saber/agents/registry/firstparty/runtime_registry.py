"""Registry of available first-party runtime specifications."""

from __future__ import annotations

from saber.agents.registry.firstparty.runtime_spec import RuntimeSpec
from saber.logging import get_logger

logger = get_logger(__name__)


class RuntimeRegistry:
    """Registry of available 1P runtime specifications.

    Class-level singleton mapping runtime names to their RuntimeSpec.
    """

    _runtimes: dict[str, RuntimeSpec] = {}

    @classmethod
    def register(cls, spec: RuntimeSpec) -> None:
        """Register a RuntimeSpec.

        Args:
            spec: The runtime specification to register.

        Raises:
            ValueError: If a runtime with the same name is already registered.
        """
        if spec.name in cls._runtimes:
            raise ValueError(f"Runtime '{spec.name}' is already registered")
        cls._runtimes[spec.name] = spec
        logger.info("Runtime '%s' registered", spec.name)

    @classmethod
    def get(cls, name: str) -> RuntimeSpec | None:
        """Get a RuntimeSpec by name.

        Args:
            name: The runtime name to look up.

        Returns:
            The RuntimeSpec if found, else None.
        """
        return cls._runtimes.get(name)

    @classmethod
    def list_runtimes(cls) -> list[str]:
        """List all registered runtime names, sorted."""
        return sorted(cls._runtimes.keys())

    @classmethod
    def _reset(cls) -> None:
        """Reset registry. For testing only."""
        cls._runtimes = {}
