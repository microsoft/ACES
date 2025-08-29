#!/usr/bin/env python3
"""
SABER Harness Data Models

Data models and configuration for the SABER test harness.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SABERHarnessConfig:
    """Configuration for SABER harness."""

    # Server connection
    server_url: str = "http://localhost:8000"
    mcp_url: str = "http://localhost:8001"
    client_id: str = "saber-client"
    request_timeout: float = 300.0

    # Logging
    log_level: str = "INFO"

    # Task configuration
    task_ids: Optional[List[str]] = None  # None → auto-fetch and run all
    parallelism: int = 1  # Number of parallel episodes

    # Safety limits
    max_steps_client_safety: int = 100  # Local step cap fallback

    # LLM configuration (optional)
    llm_provider: Optional[str] = None
    llm_config: Dict[str, Any] = field(default_factory=dict)

    # Agent configuration (optional)
    agent_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Result of a single episode execution."""

    task_id: str
    episode_id: str
    attempt: int
    success: bool
    termination_reason: Optional[str] = None
    flag: Optional[str] = None
    iterations: int = 0
    error: Optional[str] = None


@dataclass
class HarnessRunResult:
    """Result of complete harness run."""

    session_id: str
    total_episodes: int
    successful_episodes: int
    episode_results: List[EpisodeResult]
    success: bool
