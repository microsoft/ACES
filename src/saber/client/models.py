"""
SABER Client Models - Type definitions for client-side operations.

These models provide strict typing for SABER client operations, including
solver execution, agent configuration, and client configuration, replacing
raw dictionaries with proper Pydantic models.
"""

from dataclasses import dataclass, field
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


class AgentAssignment(BaseModel):
    """Agent assignment with kwargs support for task-specific agent assignment."""

    id: str = Field(..., description="Agent identifier from registry")
    model: Optional[str] = Field(default=None, description="Model to use for this agent (overrides global model)")
    tasks: List[str] = Field(..., description="Task IDs or '*' for wildcard assignment")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific parameters")

    class Config:
        extra = "forbid"  # Fail fast on unknown fields

    def model_post_init(self, __context: Any) -> None:
        """Validate assignment after creation."""
        if not self.id:
            raise ValueError("Agent ID cannot be empty")
        if not self.tasks:
            raise ValueError("Tasks list cannot be empty")
        if "*" in self.tasks and len(self.tasks) > 1:
            raise ValueError("Wildcard '*' cannot be combined with specific task IDs")


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
    Main SABER configuration with per-agent model specification.
    No backwards compatibility - enforces new multi-agent configuration with agent-specific models.
    """

    # Global model is optional - agents specify their own models
    model: Optional[str] = None  # Deprecated - use per-agent models instead
    model_args: Dict[str, Any] = field(default_factory=dict)

    # Session manager configuration (unified REST + MCP)
    session_config: Optional[SessionManagerConfig] = field(default=None)

    # Task configuration
    task_ids: Optional[List[str]] = None

    # Multi-agent configuration (required - new format only)
    agents: List[AgentAssignment] = field(default_factory=list)

    # Domain configuration (optional - for logging organization)
    domain: Optional[str] = field(default=None)

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

    # Endpoint configuration for model inference
    endpoint_timeout: Optional[int] = None  # Model API request timeout in seconds
    endpoint_max_retries: Optional[int] = None  # Maximum retry attempts for model API
    endpoint_max_connections: Optional[int] = None  # Maximum concurrent connections to model API

    @classmethod
    def create(
        cls,
        model: str,
        rest_url: str,
        mcp_url: str,
        agents: List[AgentAssignment],
        client_id: str = "saber-client",
        model_args: Optional[Dict[str, Any]] = None,
        task_ids: Optional[List[str]] = None,
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        domain: Optional[str] = None,
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
        endpoint_timeout: Optional[int] = None,
        endpoint_max_retries: Optional[int] = None,
        endpoint_max_connections: Optional[int] = None,
    ) -> "SABERConfig":
        """
        Factory method to create SABERConfig with multi-agent assignments.

        Args:
            model: Model specification (required)
            rest_url: SABER server REST API URL
            mcp_url: SABER server MCP URL
            agents: List of agent task assignments (required)
            client_id: Client identifier
            model_args: Model arguments dictionary
            task_ids: List of task IDs to execute
            log_level: Logging level
            log_dir: Log directory path
            domain: Domain name for logging organization
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
        # Create session config only if URLs are provided (not in auto mode)
        session_config = None
        if rest_url and mcp_url:
            session_config = SessionManagerConfig.from_urls(rest_url=rest_url, mcp_url=mcp_url, client_id=client_id)

        return cls(
            model=model,
            model_args=model_args or {},
            session_config=session_config,
            task_ids=task_ids,
            agents=agents,
            domain=domain,
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
            endpoint_timeout=endpoint_timeout,
            endpoint_max_retries=endpoint_max_retries,
            endpoint_max_connections=endpoint_max_connections,
        )

    def __post_init__(self) -> None:
        """Validate configuration on creation - fail fast design."""
        if not self.model:
            raise ValueError("model specification is required")

        # Allow None session_config for auto mode - will be hydrated later
        # Validation will happen at runtime when trying to use the config

        # Validate agent configuration - new format required
        if not self.agents:
            raise ValueError("agents configuration is required")

        if not isinstance(self.agents, list):
            raise TypeError("agents must be a list")

        # Validate agent assignments
        self._validate_agent_assignments()

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

        # Validate log upload configuration
        if self.log_upload_max_retries < 0:
            raise ValueError(f"log_upload_max_retries must be non-negative, got: {self.log_upload_max_retries}")

        if self.log_upload_timeout <= 0:
            raise ValueError(f"log_upload_timeout must be positive, got: {self.log_upload_timeout}")

    def _validate_agent_assignments(self) -> None:
        """Validate agent assignments following the design requirements."""
        if not self.agents:
            raise ValueError("At least one agent assignment is required")

        # NOTE: Allow duplicate agent IDs with different kwargs/tasks
        # This enables the same agent type to be configured differently for different tasks

        # Check for multiple wildcard assignments
        wildcard_count = sum(1 for assignment in self.agents if "*" in assignment.tasks)
        if wildcard_count > 1:
            raise ValueError("Only one agent can have wildcard '*' assignment")

        # Check for task conflicts (same task assigned to multiple agents)
        explicit_tasks = set()
        for assignment in self.agents:
            for task in assignment.tasks:
                if task != "*":
                    if task in explicit_tasks:
                        raise ValueError(f"Task '{task}' assigned to multiple agents")
                    explicit_tasks.add(task)

    # Legacy property accessor with deprecation warning
    @property
    def agent_assignments(self) -> List[AgentAssignment]:
        """
        Legacy property for backwards compatibility (deprecated).

        Returns:
            List of agent assignments

        Note: This property is deprecated. Use .agents directly.
        """
        import warnings

        warnings.warn("agent_assignments is deprecated, use .agents instead", DeprecationWarning, stacklevel=2)
        return self.agents
