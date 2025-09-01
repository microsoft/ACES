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

from ..api_models import EpisodeInfo, PolicyInfo, SessionInfo, StepResponse, TaskInfo
from .api import SABERRestClient
from .harness_models import EpisodeResult, HarnessRunResult, SABERHarnessConfig
from .llm import LLMClientFactory, create_llm_client
from .saber_harness import SABERHarness

__all__ = [
    "SABERHarness",
    "SABERHarnessConfig",
    "EpisodeResult",
    "HarnessRunResult",
    "SABERRestClient",
    "LLMClientFactory",
    "create_llm_client",
    "StepResponse",
    "TaskInfo",
    "PolicyInfo",
    "EpisodeInfo",
    "SessionInfo",
]
