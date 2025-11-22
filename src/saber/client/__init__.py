"""
SABER Client

Client-side harness and utilities for running agent benchmarks against a SABER server
using containerized agent execution with a shared MCP sidecar.

Programmatic usage:
    from saber.client import SABERRestClient

    client = SABERRestClient(server_url="http://localhost:8000")
    # Use client for API calls
"""

from ..models import BenchmarkInfo, PolicyInfo, SessionInfo, StepResponse, TaskInfo
from .api import SABERRestClient
from .models import SABERConfig

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
]
