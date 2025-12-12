"""External integrations (MCP tools, model wrappers).

Module structure:
- message_serialization: Message serialization/deserialization utilities
- websocket_connection: WebSocket connection lifecycle management
- event_processor: WebSocket event listening and processing
- sync_operations: Transcript push/pull sync operations
- model_wrapper: Main wrapper class using composition of above modules
- tools: MCP tool integration
"""

# Re-export component modules for direct access if needed
from .event_processor import WebSocketEventProcessor
from .message_serialization import deserialize_message, is_websocket_closed, serialize_message
from .model_wrapper import WebSocketTranscriptSyncingModelWrapper
from .sync_operations import TranscriptSyncOperations
from .tools import SABERToolSource, saber_tools
from .websocket_connection import WebSocketConnectionManager

__all__ = [
    # Main wrapper
    "WebSocketTranscriptSyncingModelWrapper",
    # Component modules
    "WebSocketConnectionManager",
    "WebSocketEventProcessor",
    "TranscriptSyncOperations",
    # Serialization utilities
    "serialize_message",
    "deserialize_message",
    "is_websocket_closed",
    # Tools
    "SABERToolSource",
    "saber_tools",
]
