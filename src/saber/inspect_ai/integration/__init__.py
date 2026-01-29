"""External integrations (MCP tools, model wrappers).

Module structure:
- model_wrapper: Main wrapper class using TranscriptSyncClient
- tools: MCP tool integration
- agent_transcript_sync: Manual transcript sync for custom agents

Note: Copilot SDK tool bridge is now in agents/registry/copilot/tools.py
"""

from .agent_transcript_sync import AgentTranscriptSync
from .model_wrapper import WebSocketTranscriptSyncingModelWrapper
from .tools import SABERToolSource, saber_tools

__all__ = [
    # Main wrapper
    "WebSocketTranscriptSyncingModelWrapper",
    # Manual sync for custom agents
    "AgentTranscriptSync",
    # Tools
    "SABERToolSource",
    "saber_tools",
]
