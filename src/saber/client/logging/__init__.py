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
from .docker_log_collector import DockerLogCollector
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

__all__ = [
    "ClientLoggingConfig",
    "SessionLoggingContext",
    "ContainerLogManager",
    "HarnessExecutionLogger",
    "DockerLogCollector",
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
