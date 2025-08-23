"""
SABER API Models - Shared data models for client-server communication.

These models define the API contract between SABER clients and servers.
They should be used by both client and server implementations to ensure
consistent data structures.
"""


# MCP Protocol Constants
class MCPHeaders:
    """HTTP header names for MCP session mapping."""

    SESSION_ID = "x-saber-session-id"
    TASK_ID = "x-saber-task-id"
    CLIENT_ID = "x-saber-client-id"
