"""External integrations (MCP tools, model wrappers, Copilot SDK).

Module structure:
- model_wrapper: Main wrapper class using TranscriptSyncClient
- tools: MCP tool integration
- copilot_tools: Copilot SDK tool bridge for MCP tools

The transcript sync components (connection, events, sync operations) have been
moved to saber.client.transcript for harness-agnostic reuse.
"""

from .model_wrapper import InspectAIMessageSerializer, WebSocketTranscriptSyncingModelWrapper
from .tools import SABERToolSource, saber_tools
from .copilot_tools import (
    Tool as CopilotTool,
    mcp_tool_to_copilot_tool,
    convert_mcp_tools_to_copilot,
    create_submit_tool,
    get_saber_mcp_tools,
)

__all__ = [
    # Main wrapper
    "WebSocketTranscriptSyncingModelWrapper",
    # Serializer for Inspect AI messages
    "InspectAIMessageSerializer",
    # Tools
    "SABERToolSource",
    "saber_tools",
    # Copilot SDK tools
    "CopilotTool",
    "mcp_tool_to_copilot_tool",
    "convert_mcp_tools_to_copilot",
    "create_submit_tool",
    "get_saber_mcp_tools",
]
