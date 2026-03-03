"""Tool registry — resolves tool names from YAML config to inspect_ai Tool instances."""

from __future__ import annotations

import inspect as _inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType

from inspect_ai.tool import Tool, bash, python
from pydantic import BaseModel, ConfigDict, Field, model_validator

from saber.config.models import ToolConfig
from saber.tools.security import ToolSecurityConfig


def _supports_timeout(factory: Callable[..., Tool]) -> bool:
    """Check whether *factory* accepts a ``timeout`` keyword argument."""
    try:
        sig = _inspect.signature(factory)
        return "timeout" in sig.parameters
    except (ValueError, TypeError):
        return False


class ToolRegistry:
    """Resolves tool names from YAML task config to inspect_ai Tool instances.

    Built-in tools: bash, python.
    Domain-local tools are registered via .register().

    Usage:
        registry = ToolRegistry()
        registry.register("my_tool", my_tool_factory)
        tools = registry.resolve({"bash": ToolConfig(timeout=180), "my_tool": ToolConfig()})
    """

    _BUILTINS: dict[str, Callable[..., Tool]] = {
        "bash": bash,
        "python": python,
    }

    def __init__(self) -> None:
        self._custom: dict[str, Callable[..., Tool]] = {}

    def register(self, name: str, factory: Callable[..., Tool]) -> None:
        """Register a domain-local tool factory.

        Args:
            name: Tool name matching the YAML config key.
            factory: Callable that accepts at least timeout kwarg and returns a Tool.

        Raises:
            ValueError: If name is already registered (built-in or custom).
        """
        if name in self._BUILTINS:
            raise ValueError(f"Cannot override built-in tool '{name}'")
        if name in self._custom:
            raise ValueError(f"Tool '{name}' is already registered")
        self._custom[name] = factory

    def _create_tool(self, factory: Callable[..., Tool], config: ToolConfig) -> Tool:
        """Invoke *factory* passing ``timeout`` only when the factory accepts it."""
        if _supports_timeout(factory):
            return factory(timeout=config.timeout)
        return factory()

    def resolve(self, tool_configs: dict[str, ToolConfig]) -> ResolvedTools:
        """Resolve tool name→config mapping to a ResolvedTools bundle.

        The ``submit`` key is reserved — it is skipped (not instantiated)
        because submit is handled by the react agent's AgentSubmit.

        Args:
            tool_configs: Mapping of tool names to their ToolConfig.

        Returns:
            ResolvedTools containing tool instances, submit flag, and security configs.

        Raises:
            ValueError: If a tool name is not found in builtins or custom.
        """
        tools: list[Tool] = []
        security_configs: dict[str, ToolSecurityConfig] = {}

        for name, config in tool_configs.items():
            if name == "submit":
                # submit is handled by react agent, not tool registry
                continue

            if name in self._BUILTINS:
                tools.append(self._create_tool(self._BUILTINS[name], config))
            elif name in self._custom:
                tools.append(self._create_tool(self._custom[name], config))
            else:
                available = sorted({*self._BUILTINS, *self._custom})
                raise ValueError(f"Unknown tool: {name!r}. Available: {available}")

            # Collect security config if present
            if config.security is not None:
                security_configs[name] = config.security

        return ResolvedTools(
            tools=tuple(tools),
            submit_enabled="submit" in tool_configs,
            security_configs=MappingProxyType(security_configs),
        )

    def available(self) -> list[str]:
        """Return sorted list of all registered tool names."""
        return sorted({*self._BUILTINS, *self._custom})


class ResolvedTools(BaseModel):
    """Result of ToolRegistry.resolve() — tools + submit flag + security configs."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tools: tuple[Tool, ...]
    submit_enabled: bool
    security_configs: Mapping[str, ToolSecurityConfig] = Field(default_factory=lambda: MappingProxyType({}))

    @model_validator(mode="after")
    def _freeze_security_configs(self) -> ResolvedTools:
        """Wrap mutable dicts in MappingProxyType to enforce immutability."""
        if isinstance(self.security_configs, dict):
            object.__setattr__(self, "security_configs", MappingProxyType(self.security_configs))
        return self
