"""
SABER Client

Client-side harness and utilities for running agent benchmarks against a SABER server
using containerized agent execution with a shared MCP sidecar.

Programmatic usage:
    from saber.client import SABERHarness, SABERHarnessConfig

    config = SABERHarnessConfig(server_url="http://localhost:8000")
    harness = SABERHarness(config)
    await harness.initialize(my_agent)
    results = await harness.run()
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
