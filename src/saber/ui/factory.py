#!/usr/bin/env python3
"""
SABER UI Factory

Factory for creating UI adapters with isolated dependencies.
"""

from enum import Enum
from typing import Optional

from .interfaces import SABERUIAdapter


class UIBackendType(Enum):
    """Available UI backend types."""

    AUTO = "auto"
    INSPECT_AI_RICH = "inspect_ai_rich"
    INSPECT_AI_TEXTUAL = "inspect_ai_textual"
    INSPECT_AI_PLAIN = "inspect_ai_plain"
    CONSOLE = "console"
    WEB = "web"
    NONE = "none"


def create_ui_adapter(backend: UIBackendType = UIBackendType.AUTO, config: Optional[dict] = None) -> SABERUIAdapter:
    """
    Factory to create UI adapter - isolated from inspect-ai imports.

    Args:
        backend: Desired UI backend type
        config: Optional configuration for the adapter

    Returns:
        Configured UI adapter instance
    """
    config = config or {}

    if backend == UIBackendType.NONE:
        from .adapters.null_adapter import NullUIAdapter

        return NullUIAdapter()

    if backend == UIBackendType.CONSOLE:
        from .adapters.console_adapter import ConsoleUIAdapter

        return ConsoleUIAdapter(config)

    if backend == UIBackendType.WEB:
        from .adapters.web_adapter import WebUIAdapter

        return WebUIAdapter(config)  # type: ignore[no-any-return]

    # For inspect-ai backends
    if backend in [UIBackendType.INSPECT_AI_RICH, UIBackendType.INSPECT_AI_TEXTUAL, UIBackendType.INSPECT_AI_PLAIN]:
        try:
            from .adapters.inspect_ai_adapter import InspectAIAdapter

            return InspectAIAdapter(backend, config)  # type: ignore[no-any-return]
        except ImportError:
            # Fallback to console adapter if inspect-ai not available
            from .adapters.console_adapter import ConsoleUIAdapter

            return ConsoleUIAdapter(config)

    # AUTO mode - detect best available backend
    if backend == UIBackendType.AUTO:
        # Try inspect-ai first
        try:
            from .adapters.inspect_ai_adapter import InspectAIAdapter

            return InspectAIAdapter(UIBackendType.INSPECT_AI_RICH, config)  # type: ignore[no-any-return]
        except ImportError:
            pass

        # Fall back to console
        from .adapters.console_adapter import ConsoleUIAdapter

        return ConsoleUIAdapter(config)

    # Default fallback
    from .adapters.null_adapter import NullUIAdapter

    return NullUIAdapter()
