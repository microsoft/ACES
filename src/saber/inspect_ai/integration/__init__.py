"""External integrations (MCP tools, model wrappers, Copilot SDK).

Module structure:
- model_wrapper: Main wrapper class using TranscriptSyncClient
- tools: MCP tool integration
- copilot_tools: Copilot SDK tool bridge for MCP tools
- agent_transcript_sync: Manual transcript sync for custom agents

The transcript sync components (connection, events, sync operations) have been
moved to saber.client.transcript for harness-agnostic reuse.
"""

from .agent_transcript_sync import AgentTranscriptSync
from .copilot_tools import (
    Tool as CopilotTool,
    ToolCallRecord,
    ToolCallTracker,
    convert_mcp_tools_to_copilot,
    create_submit_tool,
    get_saber_mcp_tools,
    mcp_tool_to_copilot_tool,
)
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
    # Copilot SDK tools
    "CopilotTool",
    "mcp_tool_to_copilot_tool",
    "convert_mcp_tools_to_copilot",
    "create_submit_tool",
    "get_saber_mcp_tools",
    # Tool call tracking for transcripts
    "ToolCallTracker",
    "ToolCallRecord",
]
