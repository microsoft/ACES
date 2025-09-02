"""
SABER Client-Side Container Logging System

Provides persistent logging capabilities for client-side containers including:
- Client containers
- MCP sidecar containers
- Agent runner containers

Enables debugging of container execution issues with structured, searchable logs.
"""

from .config import ClientLoggingConfig, SessionLoggingContext
from .container_log_manager import ContainerLogManager
from .events import (
    CleanupEvent,
    EpisodeQueueEvent,
    ErrorEvent,
    ExecutionCompleteEvent,
    ExecutionStartEvent,
    HarnessInitializationEvent,
    InfrastructureEvent,
    SessionCreationEvent,
    TaskResolutionEvent,
    UIEvent,
)
from .harness_execution_logger import HarnessExecutionLogger
from .log_stream_collector import LogStreamCollector

__all__ = [
    "ClientLoggingConfig",
    "SessionLoggingContext",
    "ContainerLogManager",
    "HarnessExecutionLogger",
    "LogStreamCollector",
    # Event classes
    "HarnessInitializationEvent",
    "SessionCreationEvent",
    "TaskResolutionEvent",
    "EpisodeQueueEvent",
    "ExecutionStartEvent",
    "ExecutionCompleteEvent",
    "InfrastructureEvent",
    "UIEvent",
    "CleanupEvent",
    "ErrorEvent",
]
