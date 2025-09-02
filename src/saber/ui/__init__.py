#!/usr/bin/env python3
"""
SABER UI Module

Provides beautiful CLI interfaces for SABER harness execution.
All inspect-ai dependencies are isolated in adapter implementations.
"""

from .factory import UIBackendType, create_ui_adapter
from .interfaces import (
    MCPToolCall,
    MCPToolCallStatus,
    MCPToolMonitor,
    SABERTaskState,
    SABERUIAdapter,
    SABERUIProgressUpdate,
    SABERUISessionInfo,
    SABERUISessionSummary,
    SABERUITaskInfo,
    SABERUITaskResult,
)

__all__ = [
    "SABERUIAdapter",
    "SABERUISessionInfo",
    "SABERUITaskInfo",
    "SABERUIProgressUpdate",
    "SABERUITaskResult",
    "SABERUISessionSummary",
    "SABERTaskState",
    "MCPToolCall",
    "MCPToolCallStatus",
    "MCPToolMonitor",
    "create_ui_adapter",
    "UIBackendType",
]
