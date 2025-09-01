"""
Agent Adapters

Adapters for different agent interface patterns to provide unified execution interface.
Supports class-based, function-based, async/sync agents with automatic detection.
"""

from typing import Any

from .async_adapter import AsyncAdapter
from .base import AgentAdapter
from .class_adapter import ClassBasedAdapter
from .function_adapter import FunctionBasedAdapter


def create_agent_adapter(agent: Any, mcp_client: Any) -> AgentAdapter:
    """
    Create appropriate agent adapter based on agent type and interface.

    Args:
        agent: Agent object or function
        mcp_client: MCP client for tool access

    Returns:
        Appropriate agent adapter
    """
    # Check if agent is a function
    if (
        callable(agent)
        and not hasattr(agent, "__class__")
        or hasattr(agent, "__call__")
        and not hasattr(agent, "__dict__")
    ):
        return FunctionBasedAdapter(agent, mcp_client)

    # Check if agent is a class instance with common methods
    if hasattr(agent, "run") or hasattr(agent, "process") or hasattr(agent, "chat") or hasattr(agent, "respond"):
        return ClassBasedAdapter(agent, mcp_client)

    # Check if agent has async methods
    import inspect

    if hasattr(agent, "run") and callable(getattr(agent, "run", None)) and inspect.iscoroutinefunction(agent.run):
        return AsyncAdapter(agent, mcp_client)

    # Default to class-based adapter
    return ClassBasedAdapter(agent, mcp_client)


__all__ = ["AgentAdapter", "ClassBasedAdapter", "FunctionBasedAdapter", "AsyncAdapter", "create_agent_adapter"]
