"""
SABER Transcript Sync Client - Generic, Harness-Agnostic Transcript Synchronization

This module provides a generic transcript synchronization client that works with
any evaluation harness via the MessageSerializer protocol.

Components:
- TranscriptSyncClient: Main orchestrator for WebSocket-based transcript sync
- MessageSerializer: Protocol for harness-specific message serialization
- WebSocketConnectionManager: Connection lifecycle management
- WebSocketEventProcessor: Event listening and processing
- GenericTranscriptSyncOperations: Push sync operations

Example usage:
    from saber.client.transcript import TranscriptSyncClient, MessageSerializer

    class MyMessageSerializer(MessageSerializer[MyMessage]):
        def serialize(self, msg: MyMessage) -> dict:
            return {"role": msg.role, "content": msg.text}
        def deserialize(self, data: dict) -> MyMessage:
            return MyMessage(role=data["role"], text=data["content"])
        def get_role(self, msg: MyMessage) -> str:
            return msg.role
        def has_tool_calls(self, msg: MyMessage) -> bool:
            return False
        def get_tool_call_ids(self, msg: MyMessage) -> list[str]:
            return []
        def get_tool_call_id(self, msg: MyMessage) -> str | None:
            return None

    async with TranscriptSyncClient(
        episode_id="ep-123",
        rest_url="http://localhost:8000",
        serializer=MyMessageSerializer(),
    ) as client:
        await client.push_message(response_msg)
"""

from .client import TranscriptSyncClient
from .connection import WebSocketConnectionManager
from .event_processor import WebSocketEventProcessor
from .protocols import MessageSerializer
from .sync_operations import GenericTranscriptSyncOperations
from .utils import is_websocket_closed

__all__ = [
    # Main client
    "TranscriptSyncClient",
    # Protocol
    "MessageSerializer",
    # Components (for advanced usage)
    "WebSocketConnectionManager",
    "WebSocketEventProcessor",
    "GenericTranscriptSyncOperations",
    # Utilities
    "is_websocket_closed",
]
