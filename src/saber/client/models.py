"""
SABER Client Models - Type definitions for client-side operations.

These models provide strict typing for SABER client operations, including
solver execution, agent configuration, and client configuration, replacing
raw dictionaries with proper Pydantic models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SessionManagerConfig(BaseModel):
    """Simplified configuration for ClientSessionManager REST API layer."""

    # REST API configuration
    base_url: str = Field(description="SABER server REST API base URL")
    client_id: str = Field(default="saber_client", description="Client identifier for session creation")
    rest_timeout: float = Field(default=30.0, description="REST API request timeout in seconds")

    # MCP server URL for agent tasks (used by inspect_ai native integration)
    mcp_server_url: str = Field(description="SABER MCP server URL for agent tools")

    @classmethod
    def from_urls(
        cls, rest_url: str, mcp_url: str, client_id: str = "saber_client", **kwargs: Any
    ) -> "SessionManagerConfig":
        """
        Create SessionManagerConfig from separate REST and MCP URLs.

        Args:
            rest_url: SABER server REST API URL
            mcp_url: SABER server MCP endpoint URL
            client_id: Client identifier
            **kwargs: Additional configuration options

        Returns:
            Configured SessionManagerConfig instance
        """
        return cls(base_url=rest_url, mcp_server_url=mcp_url, client_id=client_id, **kwargs)

    class Config:
        extra = "forbid"  # Don't allow extra fields for strict typing


class AgentConfig(BaseModel):
    """Strictly typed agent configuration for container execution."""

    # Core agent behavior settings
    debug_mode: bool = Field(default=False, description="Enable debug mode")

    # Allow additional fields for extensibility
    class Config:
        extra = "allow"


class AgentInfo(BaseModel):
    """Information about a SABER agent."""

    agent_id: str = Field(description="Unique agent identifier")
    name: str = Field(description="Human-readable agent name")
    description: str = Field(description="Agent description")
    version: str = Field(default="1.0.0", description="Agent version")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    tags: List[str] = Field(default_factory=list, description="Agent tags")


class SABERTask(BaseModel):
    """Typed SABER task definition."""

    id: str = Field(description="Unique task identifier")
    title: Optional[str] = Field(None, description="Task title")
    description: str = Field(description="Task description")
    environment: str = Field(description="Environment description")
    subtasks: List[Dict[str, Any]] = Field(default_factory=list, description="Task subtasks")
    success_criteria: Optional[str] = Field(None, description="Success criteria")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")


class AgentExecutionParams(BaseModel):
    """Parameters for agent container execution."""

    agent_file: str = Field(description="Path to agent file")
    initial_prompt: str = Field(description="Initial prompt for agent")
    task_id: str = Field(description="Task identifier")
    agent_config: AgentConfig = Field(description="Agent configuration")
    timeout: int = Field(default=300, description="Execution timeout in seconds")


class ContainerExecutionResult(BaseModel):
    """Result from container execution via AgentManager."""

    status: str = Field(description="Execution status (completed/failed)")
    result: str = Field(description="Execution result or output")
    task_id: str = Field(description="Task identifier")
    success: bool = Field(description="Whether execution succeeded")
    exit_code: int = Field(description="Container exit code")
    execution_time: float = Field(description="Execution time in seconds")
    container_id: str = Field(description="Container identifier")
    termination_reason: str = Field(description="How container terminated")
    error: Optional[str] = Field(None, description="Error message if failed")
    agent_id: str = Field(description="Agent identifier")
    episode_id: str = Field(description="Episode identifier")


# =============================================================================
# Configuration Models
# =============================================================================


@dataclass
class SABERConfig:
    """
    Main SABER configuration using inspect_ai model specifications.
    Clean configuration with no legacy support - fail fast design.
    """

    model: str  # Required model spec, no default
    model_args: Dict[str, Any] = field(default_factory=dict)

    # Session manager configuration (unified REST + MCP)
    session_config: Optional[SessionManagerConfig] = field(default=None)

    # Task configuration
    task_ids: Optional[List[str]] = None

    # Agent configuration - modernized, no legacy support
    agent_id: Optional[str] = None  # Agent ID from registry (preferred)
    agent_path: Optional[str] = None  # Path to agent Python file
    agent_config: Optional[AgentConfig] = field(default=None)

    # Container configuration
    container_timeout: int = 300

    # Execution configuration
    ui_enabled: bool = True
    log_level: str = "INFO"
    log_dir: Optional[str] = None

    # Log upload configuration
    log_upload_enabled: bool = True
    log_upload_max_retries: int = 3
    log_upload_timeout: float = 30.0
    log_upload_fail_on_error: bool = False

    # eval_async specific configuration
    max_samples: Optional[int] = None
    max_subprocesses: int = 1
    parallel_execution: bool = True
    max_parallel_tasks: int = 4

    @classmethod
    def create(
        cls,
        model: str,
        rest_url: str,
        mcp_url: str,
        client_id: str = "saber-client",
        model_args: Optional[Dict[str, Any]] = None,
        task_ids: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        agent_path: Optional[str] = None,
        debug_mode: bool = False,
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        ui_enabled: bool = True,
        container_timeout: int = 300,
        max_samples: Optional[int] = None,
        max_subprocesses: int = 1,
        parallel_execution: bool = True,
        max_parallel_tasks: int = 4,
        log_upload_enabled: bool = True,
        log_upload_max_retries: int = 3,
        log_upload_timeout: float = 30.0,
        log_upload_fail_on_error: bool = False,
    ) -> "SABERConfig":
        """
        Factory method to create SABERConfig with proper validation.

        Args:
            model: Model specification (required)
            rest_url: SABER server REST API URL
            mcp_url: SABER server MCP URL
            client_id: Client identifier
            model_args: Model arguments dictionary
            task_ids: List of task IDs to execute
            agent_id: Agent ID from registry
            agent_path: Path to agent Python file
            debug_mode: Enable debug mode
            log_level: Logging level
            log_dir: Log directory path
            ui_enabled: Enable UI
            container_timeout: Container timeout in seconds
            max_samples: Maximum samples to process
            max_subprocesses: Maximum subprocess count
            parallel_execution: Enable parallel execution
            max_parallel_tasks: Maximum parallel tasks
            log_upload_enabled: Enable automatic log file upload to server
            log_upload_max_retries: Maximum retry attempts for log upload
            log_upload_timeout: Timeout for log upload requests in seconds
            log_upload_fail_on_error: Whether to fail evaluation if log upload fails

        Returns:
            Configured SABERConfig instance

        Raises:
            ValueError: If configuration is invalid
        """
        # Create session config
        session_config = SessionManagerConfig.from_urls(rest_url=rest_url, mcp_url=mcp_url, client_id=client_id)

        # Create agent config
        agent_config = AgentConfig(debug_mode=debug_mode)

        return cls(
            model=model,
            model_args=model_args or {},
            session_config=session_config,
            task_ids=task_ids,
            agent_id=agent_id,
            agent_path=agent_path,
            agent_config=agent_config,
            container_timeout=container_timeout,
            ui_enabled=ui_enabled,
            log_level=log_level,
            log_dir=log_dir,
            max_samples=max_samples,
            max_subprocesses=max_subprocesses,
            parallel_execution=parallel_execution,
            max_parallel_tasks=max_parallel_tasks,
            log_upload_enabled=log_upload_enabled,
            log_upload_max_retries=log_upload_max_retries,
            log_upload_timeout=log_upload_timeout,
            log_upload_fail_on_error=log_upload_fail_on_error,
        )

    def __post_init__(self) -> None:
        """Validate configuration on creation - fail fast design."""
        if not self.model:
            raise ValueError("model specification is required")

        if not self.session_config:
            raise ValueError("session_config is required")

        # Validate agent specification - exactly one required
        if not self.agent_id and not self.agent_path:
            raise ValueError("Either agent_id or agent_path must be provided")

        if self.agent_id and self.agent_path:
            raise ValueError("Cannot specify both agent_id and agent_path - choose one")

        if self.agent_path:
            agent_path_obj = Path(self.agent_path)
            if not agent_path_obj.exists():
                raise FileNotFoundError(f"Agent file not found: {self.agent_path}")

        if not self.agent_config:
            raise ValueError("agent_config is required")

        # Validate container_timeout
        if self.container_timeout <= 0:
            raise ValueError(f"container_timeout must be positive, got: {self.container_timeout}")

        # Validate max_parallel_tasks
        if self.max_parallel_tasks <= 0:
            raise ValueError(f"max_parallel_tasks must be positive, got: {self.max_parallel_tasks}")

        # Validate max_subprocesses
        if self.max_subprocesses <= 0:
            raise ValueError(f"max_subprocesses must be positive, got: {self.max_subprocesses}")

        # Validate model_args
        if not isinstance(self.model_args, dict):
            raise TypeError("model_args must be a dictionary")

        # Validate agent_config type
        if not isinstance(self.agent_config, AgentConfig):
            raise TypeError("agent_config must be an AgentConfig instance")

        # Validate log upload configuration
        if self.log_upload_max_retries < 0:
            raise ValueError(f"log_upload_max_retries must be non-negative, got: {self.log_upload_max_retries}")

        if self.log_upload_timeout <= 0:
            raise ValueError(f"log_upload_timeout must be positive, got: {self.log_upload_timeout}")

    # Legacy property accessors for backward compatibility during transition
    @property
    def saber_rest_url(self) -> str:
        """Legacy property accessor for REST URL."""
        return self.session_config.base_url if self.session_config else ""

    @property
    def saber_mcp_url(self) -> str:
        """Legacy property accessor for MCP URL."""
        return self.session_config.mcp_server_url if self.session_config else ""

    @property
    def client_id(self) -> str:
        """Legacy property accessor for client ID."""
        return self.session_config.client_id if self.session_config else ""
