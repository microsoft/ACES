"""
SABER Container Management System

Provides containerized agent execution with shared MCP sidecar architecture.
Enables standard MCP library compatibility while maintaining robust process management.
"""

from .agent_manager import AgentManager
from .sidecar_manager import SidecarManager

__all__ = ["SidecarManager", "AgentManager"]
