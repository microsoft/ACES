"""
SABER API Client

Provides REST client functionality for SABER dataset operations and benchmark info.
MCP integration is now handled natively by inspect_ai.
"""

from .rest_client import SABERRestClient

__all__ = [
    "SABERRestClient",
]
