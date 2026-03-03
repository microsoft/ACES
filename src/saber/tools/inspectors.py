"""Tool inspectors for extracting validatable content from tool call arguments.

Provides a :class:`ToolInspector` protocol, concrete implementations
(:class:`KeyBasedInspector`, :class:`FallbackInspector`), factory functions
for common tools, and a :class:`ToolInspectorRegistry` for lookup.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field  # noqa: I001

# ── Enums ───────────────────────────────────────────────────────────


class ToolArgumentKey(StrEnum):
    """Keys used to identify inspectable arguments in tool calls."""

    CMD = "cmd"
    CODE = "code"
    COMMAND = "command"
    INPUT = "input"


# ── Models ──────────────────────────────────────────────────────────


class ToolInspectorConfig(BaseModel):
    """Configuration for inspecting a specific tool's arguments.

    Attributes:
        tool_name: Name of the tool to inspect.
        argument_keys: Non-empty tuple of argument key names to check.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    argument_keys: tuple[str, ...] = Field(min_length=1)


# ── Protocol ────────────────────────────────────────────────────────


@runtime_checkable
class ToolInspector(Protocol):
    """Protocol for extracting inspectable content from tool call arguments."""

    def extract_content(
        self,
        arguments: dict[str, Any],  # Any required: matches inspect_ai ToolCall.arguments type
    ) -> list[str]:
        """Extract content strings that should be validated from tool arguments.

        Args:
            arguments: The tool call's argument dict.

        Returns:
            List of content strings to validate. May be empty if no
            inspectable content is found.
        """
        ...


# ── Implementations ─────────────────────────────────────────────────


class KeyBasedInspector:
    """Inspector that extracts content from specific argument keys.

    This single class replaces the need for separate BashInspector,
    PythonInspector, SqlInspector classes — they all do the same thing
    (look up known keys in the argument dict and return their string values).
    """

    __slots__ = ("_keys",)

    def __init__(self, keys: tuple[str, ...]) -> None:
        if not keys:
            msg = "At least one argument key is required"
            raise ValueError(msg)
        self._keys = keys

    @property
    def keys(self) -> tuple[str, ...]:
        """The argument keys this inspector looks for."""
        return self._keys

    def extract_content(
        self,
        arguments: dict[str, Any],  # Any required: matches inspect_ai ToolCall.arguments type
    ) -> list[str]:
        """Extract string values for the configured keys.

        Args:
            arguments: The tool call's argument dict.

        Returns:
            List of non-empty string values found at the configured keys,
            in key definition order.
        """
        result: list[str] = []
        for key in self._keys:
            value = arguments.get(key)
            if isinstance(value, str) and value:
                result.append(value)
        return result


class FallbackInspector:
    """Inspector that extracts ALL string-valued arguments.

    Used for unknown tools where we don't know which arguments
    contain executable content. Conservative approach: validate everything.
    """

    __slots__ = ()

    def extract_content(
        self,
        arguments: dict[str, Any],  # Any required: matches inspect_ai ToolCall.arguments type
    ) -> list[str]:
        """Extract all non-empty string values from the arguments.

        Args:
            arguments: The tool call's argument dict.

        Returns:
            List of all non-empty string values in dict-iteration order.
        """
        return [v for v in arguments.values() if isinstance(v, str) and v]


# ── Factory Functions ───────────────────────────────────────────────


def bash_inspector() -> KeyBasedInspector:
    """Create an inspector for bash tool calls (keys: cmd, command)."""
    return KeyBasedInspector(keys=(ToolArgumentKey.CMD.value, ToolArgumentKey.COMMAND.value))


def python_inspector() -> KeyBasedInspector:
    """Create an inspector for python tool calls (key: code)."""
    return KeyBasedInspector(keys=(ToolArgumentKey.CODE.value,))


# ── Registry ────────────────────────────────────────────────────────


class ToolInspectorRegistry:
    """Registry mapping tool names to their inspectors.

    Pre-registers known tools (bash, python) at construction.
    Falls back to :class:`FallbackInspector` for unknown tools.
    Supports custom registration via :class:`ToolInspectorConfig`.
    """

    def __init__(self) -> None:
        self._fallback: FallbackInspector = FallbackInspector()
        self._inspectors: dict[str, ToolInspector] = {
            "bash": bash_inspector(),
            "python": python_inspector(),
        }

    def get(self, tool_name: str) -> ToolInspector:
        """Get the inspector for a tool, falling back to FallbackInspector.

        Args:
            tool_name: The name of the tool to look up.

        Returns:
            The registered inspector, or the shared FallbackInspector.
        """
        return self._inspectors.get(tool_name, self._fallback)

    def register(self, tool_name: str, inspector: ToolInspector) -> None:
        """Register a custom inspector for a tool name.

        Args:
            tool_name: The tool name to register.
            inspector: Any object satisfying the ToolInspector protocol.
        """
        self._inspectors[tool_name] = inspector

    def register_from_config(self, config: ToolInspectorConfig) -> None:
        """Register an inspector from a ToolInspectorConfig.

        Args:
            config: The tool inspector configuration.
        """
        self._inspectors[config.tool_name] = KeyBasedInspector(keys=config.argument_keys)

    @property
    def registered_tools(self) -> frozenset[str]:
        """Return the set of explicitly registered tool names."""
        return frozenset(self._inspectors.keys())
