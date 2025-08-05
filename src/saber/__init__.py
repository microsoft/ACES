"""SABER - Security Agent Benchmarking and Evaluation Research."""

from typing import Any

__version__ = "0.1.0"
__author__ = "SABER Team"
__description__ = "A distributed system for benchmarking agentic workflows in cybersecurity domains"


# Lazy imports to avoid loading server dependencies when only client is needed
def __getattr__(name: str) -> Any:
    """Lazy import mechanism to load modules only when accessed."""
    if name == "server":
        from . import server

        return server
    elif name == "client":
        from . import client

        return client
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["server", "client"]
