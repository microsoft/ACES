"""Tool registry and domain tools."""

from saber.tools.security import ToolSecurityConfig

__all__ = ["ToolRegistry", "ToolSecurityConfig"]


def __getattr__(name: str) -> object:
    """Lazy import to break circular dependency with config.models."""
    if name == "ToolRegistry":
        from saber.tools.registry import ToolRegistry

        return ToolRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
