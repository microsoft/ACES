"""Agent-related type definitions."""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, ConfigDict


class AgentPromptKwargs(TypedDict, total=False):
    """Typed kwargs for agent prompt injection."""

    instruction_prompt: str
    assistant_prompt: str
    max_steps: int  # Tool-call limit for graceful stop


class AgentCapabilities(BaseModel):
    """Capabilities supported by an agent implementation.

    Used by solver_factory to gate which kwargs are forwarded to each agent.

    All three built-in agents support tools:
    - ``react``: tools passed directly to ``react(tools=...)``
    - ``claude_code``: tools converted to ``BridgedToolsSpec`` for MCP inside sandbox
    - ``copilot``: tools converted to ``BridgedToolsSpec`` for MCP inside sandbox
    """

    model_config = ConfigDict(frozen=True)

    supports_tools: bool = True
