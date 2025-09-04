#!/usr/bin/env python3
"""
SABER Harness Data Models

Data models and configuration for the SABER test harness.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DockerCommand:
    """Configuration for additional Docker commands to execute during agent container setup."""

    type: str  # "copy", "exec"
    description: Optional[str] = None

    # For type="copy"
    source: Optional[str] = None
    destination: Optional[str] = None

    # For type="exec"
    command: Optional[List[str]] = None
    user: Optional[str] = None  # User to run exec command as (default: container default user)

    def __post_init__(self) -> None:
        """Validate docker command configuration."""
        if self.type == "copy":
            if not self.source or not self.destination:
                raise ValueError(
                    f"Docker copy command requires both 'source' and 'destination'. "
                    f"Got source='{self.source}', destination='{self.destination}'"
                )
        elif self.type == "exec":
            if not self.command or not isinstance(self.command, list):
                raise ValueError(f"Docker exec command requires 'command' as a list of strings. Got: {self.command}")
        else:
            raise ValueError(f"Unsupported docker command type: '{self.type}'. Supported types: 'copy', 'exec'")


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

    # Container logging configuration
    client_log_dir: Optional[Path] = None
    enable_container_logging: bool = True
    log_retention_days: int = 30
    max_log_size_mb: int = 50
    compress_old_logs: bool = True

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
    agent_path: Optional[str] = None
    docker_commands: List[DockerCommand] = field(default_factory=list)  # Additional Docker commands for agent setup

    # UI Configuration
    ui_backend: str = "auto"  # auto, console, inspect_ai_rich, inspect_ai_textual, inspect_ai_plain, none
    ui_enabled: bool = True  # Master switch for UI
    ui_internal_only: bool = True  # Internal debugging vs external monitoring
    ui_tool_detail_level: str = "full"  # none, basic, full (input/output capture)


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
