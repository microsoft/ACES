"""
SABER Client - Enhanced Agent Testing Framework

Enhanced client framework for testing security agents against SABER server.
Supports both autonomous agents with MCP integration and traditional step-by-step agents.

Key Features:
- Autonomous agent execution with MCP tool integration
- Session and episode management via REST API
- LLM client factory with extensible provider support
- Episode status monitoring via SSE
- Comprehensive logging and error handling
- Flexible agent adaptation

CLI Usage:
    python -m saber.client --agent <path_to_agent> --task task_id

Programmatic Usage:
    from saber.client import SABERHarness, SABERHarnessConfig

    config = SABERHarnessConfig(server_url="http://localhost:8000")
    harness = SABERHarness(config)
    await harness.initialize(agent)
    results = await harness.run_test()
"""

from ..api_models import EpisodeInfo, PolicyInfo, SessionInfo, StepResponse, TaskInfo
from .api import SABERMCPClient, SABERRestClient
from .llm import AzureOpenAIClient, LLMClientFactory, create_llm_client
from .saber_harness import SABERHarness, SABERHarnessConfig

__all__ = [
    "SABERHarness",
    "SABERHarnessConfig",
    "SABERMCPClient",
    "SABERRestClient",
    "AzureOpenAIClient",
    "LLMClientFactory",
    "create_llm_client",
    "StepResponse",
    "TaskInfo",
    "PolicyInfo",
    "EpisodeInfo",
    "SessionInfo",
]
