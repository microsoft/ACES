"""
SABER API Models - Shared data models for client-server communication.

These models define the API contract between SABER clients and servers.
They should be used by both client and server implementations to ensure
consistent data structures.
"""

from enum import Enum


# MCP Protocol Constants
class MCPHeaders:
    """HTTP header names for MCP session mapping."""

    SESSION_ID = "X-SABER-Session-ID"
    TASK_ID = "X-SABER-Task-ID"
    CLIENT_ID = "X-SABER-Client-ID"


# SSE Event Types
class SSEEventType(str, Enum):
    """Server-Sent Event types used for client-server communication."""

    # Episode termination events
    MAX_STEPS_REACHED = "max_steps_reached"
    EPISODE_TIMEOUT = "episode_timeout"
    MANUAL_TERMINATION = "manual_termination"
    EPISODE_COMPLETED = "episode_completed"
    EPISODE_FAILED = "episode_failed"

    # Environment transition events
    ENVIRONMENT_RESET = "environment_reset"

    # General episode events
    EPISODE_STARTED = "episode_started"
    EPISODE_PROGRESS = "episode_progress"
