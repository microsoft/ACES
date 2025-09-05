"""
SABER API Models - Shared data models for client-server communication.

These models define the API contract between SABER clients and servers.
They should be used by both client and server implementations to ensure
consistent data structures.
"""


# MCP Protocol Constants
class MCPHeaders:
    """HTTP header names for MCP session mapping."""

    SESSION_ID = "X-SABER-Session-ID"
    TASK_ID = "X-SABER-Task-ID"
    EPISODE_ID = "X-SABER-Episode-ID"
    CLIENT_ID = "X-SABER-Client-ID"
