"""Copilot agent implementation for SABER.

This package provides the GitHub Copilot SDK integration for SABER benchmark tasks.

Exports:
    create_agent: Factory function for creating Copilot agent solvers
    copilot_solver: The main solver implementation
"""

from .solver import copilot_solver, create_agent

__all__ = ["create_agent", "copilot_solver"]
