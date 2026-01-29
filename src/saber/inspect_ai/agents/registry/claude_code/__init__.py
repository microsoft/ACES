"""Claude Code SDK agent implementation for SABER.

This package provides Anthropic's Claude Code SDK integration for SABER benchmark tasks.

Exports:
    create_agent: Factory function for creating Claude Code agent solvers
"""

from .solver import create_agent

__all__ = ["create_agent"]
