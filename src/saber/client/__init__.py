"""
SABER Client - Minimal test harness for connecting agents to SABER server.

This client provides a simple interface for agents to interact with the SABER
security benchmarking server. The server handles all complex logic while the
client facilitates communication and prompt building.

CLI Usage:
    python -m saber.client --agent <path_to_agent> --task task_id --episodes 2

Customers can use any agent - no interface requirements!
"""

from ..api_models import EpisodeInfo, PolicyInfo, SessionInfo, StepResponse, TaskInfo
from .agent_wrapper import AgentLoader, AgentWrapper
from .prompt_builder import PromptBuilder
from .server_client import ServerClient
from .test_harness import TestHarness, TestHarnessConfig

__all__ = [
    "TestHarness",
    "TestHarnessConfig",
    "ServerClient",
    "StepResponse",
    "TaskInfo",
    "PolicyInfo",
    "EpisodeInfo",
    "SessionInfo",
    "AgentWrapper",
    "AgentLoader",
    "PromptBuilder",
]
