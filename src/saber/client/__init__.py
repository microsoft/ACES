"""
SABER Client - Harness-Agnostic Client Code

This module contains generic, harness-agnostic client code for interacting with
a SABER server. It provides the base API client and configuration models that
can be used by any evaluation harness (e.g., inspect_ai, custom harnesses).

For harness-specific integration code, see the respective modules:
- saber.inspect_ai: Inspect AI-specific integration (sandbox, tools, dataset)

Client-side harness and utilities for running agent benchmarks against a SABER server
using containerized agent execution with a shared MCP sidecar.

Programmatic usage:
    from saber.client import SABERRestClient

    client = SABERRestClient(server_url="http://localhost:8000")
    # Use client for API calls

Transcript synchronization:
    from saber.client.transcript import TranscriptSyncClient, MessageSerializer

    # Implement MessageSerializer for your harness
    async with TranscriptSyncClient(...) as client:
        messages = await client.sync_transcript()
"""

from ..models import BenchmarkInfo, PolicyInfo, SessionInfo, StepResponse, TaskInfo
from .api import SABERRestClient
from .models import SABERConfig

# Import transcript module exports
from .transcript import (
    GenericTranscriptSyncOperations,
    MessageSerializer,
    SyncResult,
    TranscriptSyncClient,
    WebSocketConnectionManager,
    WebSocketEventProcessor,
    is_websocket_closed,
)

__all__ = [
    # Configuration models
    "SABERConfig",
    # API client
    "SABERRestClient",
    # API models
    "StepResponse",
    "TaskInfo",
    "PolicyInfo",
    "BenchmarkInfo",
    "SessionInfo",
    # Transcript sync
    "TranscriptSyncClient",
    "MessageSerializer",
    "SyncResult",
    "WebSocketConnectionManager",
    "WebSocketEventProcessor",
    "GenericTranscriptSyncOperations",
    "is_websocket_closed",
]
