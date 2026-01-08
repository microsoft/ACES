"""
SABER Client Models - Type definitions for client-side operations.

These models provide strict typing for SABER client operations, including
solver execution, agent configuration, and client configuration, replacing
raw dictionaries with proper Pydantic models.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, root_validator, validator


@dataclass(frozen=True)
class AgentCompositeKey:
    """Type-safe composite key for agent datasets.

    Composite keys uniquely identify agent-task assignments, optionally including
    role information for orchestrated tasks. Uses '|' separator to avoid conflicts
    with agent IDs that may contain underscores.

    Attributes:
        agent_id: Agent identifier (e.g., 'react', 'custom_react')
        tasks_hash: SHA256 hash of sorted task patterns (8 chars)
        role: Optional role name for orchestrated tasks (e.g., 'red', 'blue')

    Examples:
        >>> key = AgentCompositeKey(agent_id="react", tasks_hash="a1b2c3d4")
        >>> key.to_string()
        'react|a1b2c3d4'

        >>> key_with_role = AgentCompositeKey(
        ...     agent_id="react", tasks_hash="a1b2c3d4", role="red"
        ... )
        >>> key_with_role.to_string()
        'react|red|a1b2c3d4'
    """

    agent_id: str
    tasks_hash: str
    role: str | None = None

    def to_string(self) -> str:
        """Convert composite key to string representation.

        Returns:
            String format: 'agent_id|tasks_hash' or 'agent_id|role|tasks_hash'
        """
        if self.role:
            return f"{self.agent_id}|{self.role}|{self.tasks_hash}"
        return f"{self.agent_id}|{self.tasks_hash}"

    @classmethod
    def from_string(cls, key: str) -> "AgentCompositeKey":
        """Parse composite key from string representation.

        Args:
            key: String in format 'agent_id|tasks_hash' or 'agent_id|role|tasks_hash'

        Returns:
            AgentCompositeKey instance

        Raises:
            ValueError: If key format is invalid

        Examples:
            >>> key = AgentCompositeKey.from_string("react|a1b2c3d4")
            >>> key.agent_id
            'react'
            >>> key.role is None
            True

            >>> key = AgentCompositeKey.from_string("react|red|a1b2c3d4")
            >>> key.role
            'red'
        """
        parts = key.split("|")

        if len(parts) == 2:
            return cls(agent_id=parts[0], tasks_hash=parts[1])
        elif len(parts) == 3:
            return cls(agent_id=parts[0], role=parts[1], tasks_hash=parts[2])
        else:
            raise ValueError(
                f"Invalid composite key format: '{key}'. Expected 'agent_id|tasks_hash' or 'agent_id|role|tasks_hash'"
            )

    @staticmethod
    def create_hash(tasks: list[str]) -> str:
        """Create consistent hash from task list.

        Args:
            tasks: List of task patterns

        Returns:
            8-character hex hash
        """
        tasks_sorted = sorted(tasks)
        tasks_str = "+".join(tasks_sorted)
        return hashlib.sha256(tasks_str.encode()).hexdigest()[:8]

    @classmethod
    def create(cls, agent_id: str, tasks: list[str], role: str | None = None) -> "AgentCompositeKey":
        """Factory method to create composite key from agent assignment.

        Args:
            agent_id: Agent identifier
            tasks: List of task patterns (will be sorted and hashed)
            role: Optional role name

        Returns:
            AgentCompositeKey instance

        Examples:
            >>> AgentCompositeKey.create("react", ["task1", "task2"])
            AgentCompositeKey(agent_id='react', tasks_hash='...', role=None)
        """
        tasks_sorted = sorted(tasks)
        tasks_str = "+".join(tasks_sorted)
        tasks_hash = hashlib.sha256(tasks_str.encode()).hexdigest()[:8]
        return cls(agent_id=agent_id, tasks_hash=tasks_hash, role=role)


class SessionManagerConfig(BaseModel):
    """Simplified configuration for ClientSessionManager REST API layer."""

    # REST API configuration
    base_url: str = Field(..., description="Base URL for SABER server REST API")
    client_id: str = Field(default="saber_client", description="Client identifier for session creation")
    rest_timeout: float = Field(default=300.0, description="REST API request timeout in seconds")
    rest_max_retries: int = Field(default=3, description="Maximum number of REST API retry attempts")

    # MCP configuration - INCREASED TIMEOUTS FOR LONG-RUNNING COMMANDS
    mcp_timeout: float = Field(
        default=3600.0, description="MCP HTTP operations timeout in seconds (1 hour for long commands)"
    )
    mcp_sse_read_timeout: float = Field(
        default=3600.0, description="MCP SSE read timeout in seconds (1 hour for long commands)"
    )
    mcp_max_retries: int = Field(default=3, description="Maximum number of MCP connection retry attempts")

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

    model_config = ConfigDict(extra="forbid")  # Don't allow extra fields for strict typing


class AgentAssignment(BaseModel):
    """Agent assignment with role and advanced configuration support."""

    id: str = Field(..., description="Agent identifier from registry")
    model: str | None = Field(default=None, description="Model to use for this agent (overrides global model)")
    tasks: list[str] = Field(..., description="Task IDs or '*' for wildcard assignment")
    role: str | None = Field(default=None, description="Role this agent handles in orchestrated tasks")
    attempts: int | None = Field(default=None, description="Episode attempts override for this agent")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Agent-specific parameters")

    model_config = ConfigDict(extra="forbid")  # Fail fast on unknown fields

    def model_post_init(self, __context: Any) -> None:
        """Validate assignment after creation."""
        if not self.id:
            raise ValueError("Agent ID cannot be empty")
        if not self.tasks:
            raise ValueError("Tasks list cannot be empty")
        if "*" in self.tasks and len(self.tasks) > 1:
            raise ValueError("Wildcard '*' cannot be combined with specific task IDs")


class RoleAgentConfig(BaseModel):
    """Configuration for a specific role in orchestrated tasks.

    Defines agent, model, and behavior for one role (e.g., 'blue', 'red').
    """

    agent: str = Field(default="react", description="Agent implementation to use for this role")

    model: str | None = Field(default=None, description="Model to use for this role (overrides global --model)")

    attempts: int | None = Field(default=None, description="Episode attempts for this role (overrides task default)")

    submit: bool | None = Field(
        default=None,
        description="Whether to enable the submit tool for this role. "
        "Set to False for continuous monitoring agents (like blue team) that should never submit. "
        "Default (None) means submit is enabled.",
    )

    kwargs: dict[str, Any] = Field(default_factory=dict, description="Additional agent-specific parameters")

    # Documentation
    description: str | None = Field(default=None, description="Human-readable description of this role's purpose")

    # Future extensions
    tools: list[str] | None = Field(default=None, description="Role-specific tool restrictions")

    timeout: int | None = Field(default=None, description="Role-specific timeout in seconds")

    model_config = ConfigDict(extra="forbid")  # Fail on unknown fields


class RoleBasedConfig(BaseModel):
    """Complete role-based configuration for orchestrated tasks.

    Supports both per-role settings and defaults.
    """

    roles: dict[str, RoleAgentConfig] = Field(
        default_factory=dict, description="Configuration for each role (key = role name)"
    )

    defaults: RoleAgentConfig | None = Field(
        default=None, description="Default configuration for roles not explicitly defined"
    )

    @validator("roles")
    def validate_role_names(cls, v: dict[str, RoleAgentConfig]) -> dict[str, RoleAgentConfig]:
        """Validate role names are non-empty strings."""
        for role_name in v.keys():
            if not role_name or not isinstance(role_name, str):
                raise ValueError(f"Invalid role name: {role_name}")
        return v

    @root_validator(skip_on_failure=True)
    def validate_has_configuration(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure at least one role or defaults is configured."""
        roles = values.get("roles", {})
        defaults = values.get("defaults")

        if not roles and not defaults:
            raise ValueError(
                "RoleBasedConfig must have at least one role defined or defaults. Got empty roles dict and no defaults."
            )

        return values

    def get_config_for_role(self, role: str) -> RoleAgentConfig:
        """Get configuration for a specific role with defaults fallback.

        Args:
            role: Role name (e.g., 'blue', 'red')

        Returns:
            RoleAgentConfig for this role, merged with defaults

        Raises:
            ValueError: If role not found and no defaults configured
        """
        # Validate role exists or defaults are available
        if role not in self.roles and not self.defaults:
            available_roles = list(self.roles.keys())
            raise ValueError(
                f"Role '{role}' not found in configuration and no defaults provided. Available roles: {available_roles}"
            )

        # Get role-specific config or empty config
        role_config = self.roles.get(role, RoleAgentConfig())

        # If we have defaults, merge them
        if self.defaults:
            # Create merged config: defaults + role-specific overrides
            merged_data = self.defaults.model_dump(exclude_unset=True)
            merged_data.update(role_config.model_dump(exclude_unset=True))

            # Merge kwargs separately (dict merge, not replace)
            if self.defaults.kwargs or role_config.kwargs:
                merged_kwargs = {**self.defaults.kwargs}
                merged_kwargs.update(role_config.kwargs)
                merged_data["kwargs"] = merged_kwargs

            return RoleAgentConfig(**merged_data)

        return role_config

    def to_agent_assignments(self) -> list[AgentAssignment]:
        """Convert role-based config to AgentAssignment list.

        Creates one AgentAssignment per role with appropriate settings.

        Returns:
            List of AgentAssignment objects
        """
        assignments = []

        for role_name, role_config in self.roles.items():
            assignment = AgentAssignment(
                id=role_config.agent,
                model=role_config.model,
                role=role_name,
                tasks=["*"],  # Role assignments apply to all tasks
                attempts=role_config.attempts,
                kwargs=role_config.kwargs,
            )
            assignments.append(assignment)

        return assignments


class AgentInfo(BaseModel):
    """Information about a SABER agent."""

    agent_id: str = Field(description="Unique agent identifier")
    name: str = Field(description="Human-readable agent name")
    description: str = Field(description="Agent description")
    version: str = Field(default="1.0.0", description="Agent version")
    capabilities: list[str] = Field(default_factory=list, description="Agent capabilities")
    tags: list[str] = Field(default_factory=list, description="Agent tags")


class SABERTask(BaseModel):
    """Typed SABER task definition."""

    id: str = Field(description="Unique task identifier")
    title: str | None = Field(None, description="Task title")
    description: str = Field(description="Task description")
    environment: str = Field(description="Environment description")
    subtasks: list[dict[str, Any]] = Field(default_factory=list, description="Task subtasks")
    success_criteria: str | None = Field(None, description="Success criteria")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")


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
    error: str | None = Field(None, description="Error message if failed")
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
    model: str | None = None  # Deprecated - use per-agent models instead
    model_args: dict[str, Any] = field(default_factory=dict)

    # Session manager configuration (unified REST + MCP)
    session_config: SessionManagerConfig | None = field(default=None)

    # Task configuration
    task_ids: list[str] | None = None

    # Multi-agent configuration (required - new format only)
    agents: list[AgentAssignment] = field(default_factory=list)

    # Role-based configuration for orchestrated tasks
    role_config: Optional["RoleBasedConfig"] = field(default=None)

    # Domain configuration (optional - for logging organization)
    domain: str | None = field(default=None)

    # Container configuration
    container_timeout: int = 300

    # Execution configuration
    ui_enabled: bool = True
    log_level: str = "INFO"
    log_dir: str | None = None

    # Log upload configuration
    log_upload_enabled: bool = True
    log_upload_max_retries: int = 3
    log_upload_timeout: float = 30.0
    log_upload_fail_on_error: bool = False

    # eval_async specific configuration
    max_subprocesses: int = 1
    parallel_execution: bool = True
    max_parallel_samples: int = 4

    # Endpoint configuration for model inference
    endpoint_timeout: int | None = None  # Model API request timeout in seconds
    endpoint_max_retries: int | None = None  # Maximum retry attempts for model API
    endpoint_max_connections: int | None = None  # Maximum concurrent connections to model API

    @classmethod
    def create(
        cls,
        model: str,
        rest_url: str,
        mcp_url: str,
        agents: list[AgentAssignment],
        client_id: str = "saber-client",
        model_args: dict[str, Any] | None = None,
        task_ids: list[str] | None = None,
        log_level: str = "INFO",
        log_dir: str | None = None,
        domain: str | None = None,
        ui_enabled: bool = True,
        container_timeout: int = 300,
        max_subprocesses: int = 1,
        parallel_execution: bool = True,
        max_parallel_samples: int = 4,
        log_upload_enabled: bool = True,
        log_upload_max_retries: int = 3,
        log_upload_timeout: float = 30.0,
        log_upload_fail_on_error: bool = False,
        endpoint_timeout: int | None = None,
        endpoint_max_retries: int | None = None,
        endpoint_max_connections: int | None = None,
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
            max_subprocesses: Maximum subprocess count
            parallel_execution: Enable parallel execution
            max_parallel_samples: Maximum parallel sample executions
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
            max_subprocesses=max_subprocesses,
            parallel_execution=parallel_execution,
            max_parallel_samples=max_parallel_samples,
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

        # Validate max_parallel_samples
        if self.max_parallel_samples <= 0:
            raise ValueError(f"max_parallel_samples must be positive, got: {self.max_parallel_samples}")

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

    def get_agent_assignments(self) -> list[AgentAssignment]:
        """Get all agent assignments including role-based ones.

        Combines:
        1. Explicit agent assignments (self.agents)
        2. Role-based assignments (from role_config)

        Returns:
            Complete list of agent assignments
        """
        assignments = list(self.agents)

        # Add role-based assignments if configured
        if self.role_config:
            role_assignments = self.role_config.to_agent_assignments()
            assignments.extend(role_assignments)

        return assignments

    # Legacy property accessor with deprecation warning
    @property
    def agent_assignments(self) -> list[AgentAssignment]:
        """
        Legacy property for backwards compatibility (deprecated).

        Returns:
            List of agent assignments

        Note: This property is deprecated. Use .agents directly.
        """
        import warnings

        warnings.warn("agent_assignments is deprecated, use .agents instead", DeprecationWarning, stacklevel=2)
        return self.agents
