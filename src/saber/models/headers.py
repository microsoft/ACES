"""
SABER HTTP Headers - Shared constants for HTTP communication.

These headers are used across REST and MCP protocols for session mapping
and context passing between client and server.
"""


class HTTPHeaders:
    """HTTP header names for SABER session and context mapping."""

    SESSION_ID = "X-SABER-Session-ID"
    TASK_ID = "X-SABER-Task-ID"
    EPISODE_ID = "X-SABER-Episode-ID"
    CLIENT_ID = "X-SABER-Client-ID"
