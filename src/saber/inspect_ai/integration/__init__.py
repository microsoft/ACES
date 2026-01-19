"""External integrations (MCP tools, model wrappers, Copilot SDK).

Module structure:
- model_wrapper: Main wrapper class using TranscriptSyncClient
- tools: MCP tool integration
- copilot_tools: Copilot SDK tool bridge for MCP tools

The transcript sync components (connection, events, sync operations) have been
moved to saber.client.transcript for harness-agnostic reuse.
"""

<<<<<<< HEAD
from .model_wrapper import InspectAIMessageSerializer, WebSocketTranscriptSyncingModelWrapper
from .tools import SABERToolSource, saber_tools
from .copilot_tools import (
    Tool as CopilotTool,
    mcp_tool_to_copilot_tool,
    convert_mcp_tools_to_copilot,
    create_submit_tool,
    get_saber_mcp_tools,
)
=======
# Re-export component modules for direct access if needed
from .copilot_tools import (
    Tool as CopilotTool,
)
from .copilot_tools import (
    convert_mcp_tools_to_copilot,
    create_submit_tool,
    get_saber_mcp_tools,
    mcp_tool_to_copilot_tool,
)
from .event_processor import WebSocketEventProcessor
from .message_serialization import (
    deserialize_message,
    is_websocket_closed,
    serialize_message,
)
from .model_wrapper import WebSocketTranscriptSyncingModelWrapper
from .sync_operations import TranscriptSyncOperations
from .tools import SABERToolSource, saber_tools
from .websocket_connection import WebSocketConnectionManager
>>>>>>> 83c16c6 (Copilot agent connection fixes and QoL code improvements)

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
