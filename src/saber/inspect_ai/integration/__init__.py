"""External integrations (MCP tools, model wrappers).

Module structure:
- model_wrapper: Main wrapper class using TranscriptSyncClient
- tools: MCP tool integration

The transcript sync components (connection, events, sync operations) have been
moved to saber.client.transcript for harness-agnostic reuse.
"""

from .model_wrapper import InspectAIMessageSerializer, WebSocketTranscriptSyncingModelWrapper
from .tools import SABERToolSource, saber_tools

__all__ = [
    # Main wrapper
    "WebSocketTranscriptSyncingModelWrapper",
    # Serializer for Inspect AI messages
    "InspectAIMessageSerializer",
    # Tools
    "SABERToolSource",
    "saber_tools",
]
