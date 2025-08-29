"""
SABER Client - Enhanced Agent Testing Framework

Enhanced client framework for testing security agents against SABER server.
Supports both autonomous agents with MCP integration and traditional step-by-step agents.

Key Features:
- Universal agent adapter without code changes (R1)
- Task selection (supplied vs auto-fetch all) (R2/R2b)
- Episode-level policy prompts (R3)
- Native MCP client exposure (R4)
- Termination detection and step counting (R5/R6)
- Thread-based parallelism (R7)

CLI Usage:
    python -m saber.client --agent <path_to_agent> --task task_id

Programmatic Usage:
    from saber.client import SABERHarness, SABERHarnessConfig

    config = SABERHarnessConfig(server_url="http://localhost:8000")
    harness = SABERHarness(config)
    await harness.initialize(agent)
    results = await harness.run()
"""

from ..api_models import EpisodeInfo, PolicyInfo, SessionInfo, StepResponse, TaskInfo
from .agent_wrapper import AgentWrapper
from .agents import ReActAgent
from .api import SABERMCPClient, SABERRestClient
from .harness_models import EpisodeResult, HarnessRunResult, SABERHarnessConfig
from .llm import AzureOpenAIClient, LLMClientFactory, create_llm_client
from .saber_harness import SABERHarness

__all__ = [
    "SABERHarness",
    "SABERHarnessConfig",
    "EpisodeResult",
    "HarnessRunResult",
    "AgentWrapper",
    "ReActAgent",
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
