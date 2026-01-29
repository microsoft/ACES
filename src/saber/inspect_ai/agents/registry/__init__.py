"""Registry of core SABER agent implementations.

This package contains agent implementations organized by SDK/provider:
- copilot/: GitHub Copilot SDK integration
- claude_code/: Anthropic Claude Code SDK integration
- react.py: Inspect AI's built-in React agent (default)

Shared utilities:
- models.py: Shared enums and TypedDicts (AgentType, AgentPromptKwargs, etc.)
- custom_agent.py: Custom agent persona parsing and tool mapping

Agent implementations are automatically discovered and registered
with SABERAgentRegistry on import.
"""

# Re-export shared models
from .models import (
    AgentPromptKwargs,
    AgentType,
    ModelPrefix,
    TranscriptConfig,
    TranscriptWebSocketConfig,
)

__all__ = [
    # Shared models
    "TranscriptWebSocketConfig",
    "TranscriptConfig",
    "AgentPromptKwargs",
    "AgentType",
    "ModelPrefix",
]
