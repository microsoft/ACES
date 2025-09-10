"""
SABER Agent System

Core agent registry and factory system for SABER.
This is the main public API for agent management.

Usage:
    from saber.client.agent import SABERAgentRegistry, SABERAgentFactory

    factory = SABERAgentFactory()
    agent = await factory.create_agent("react", config, session_manager)
"""

from .factory import SABERAgentFactory
from .manager import AgentManager
from .registry import SABERAgentRegistry, register_saber_agent

__all__ = [
    "SABERAgentRegistry",
    "SABERAgentFactory",
    "register_saber_agent",
    "AgentManager",
]
