"""
Strongly-typed models for the SABER execution system.

This module provides typed dataclasses that replace untyped dict parameters
throughout the executor system, enabling IDE autocomplete, static type checking,
and centralized field definitions.
"""

from dataclasses import dataclass, field, fields, replace
from typing import Any, Protocol, TypeVar, runtime_checkable

from typing_extensions import Self


@runtime_checkable
class ExecutorParameters(Protocol):
    """
    Protocol that all executor parameter dataclasses must implement.

    This enables generic typing in CommandExecutor and consistent
    conversion from dict at the ExecutionManager boundary.
    """

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary with validation."""
        ...


# TypeVar bound to ExecutorParameters for generic executor typing
P = TypeVar("P", bound=ExecutorParameters)


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """
    Strongly-typed execution context passed to all executors.

    This replaces the untyped `context: dict[str, Any]` parameter.
    The episode_id is required for Docker execution; other fields are optional.
    """

    episode_id: str
    session_id: str | None = None
    task_id: str | None = None

    # Orchestration fields (for red/blue team scenarios)
    target_episode_ids: list[str] | None = None
    role: str | None = None  # "red" | "blue"

    # Additional metadata that doesn't fit typed fields
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionContext":
        """
        Create ExecutionContext from a dictionary.

        Args:
            data: Dictionary with context data

        Returns:
            ExecutionContext instance

        Raises:
            ValueError: If required fields are missing
        """
        episode_id = data.get("episode_id")
        if not episode_id:
            raise ValueError("episode_id is required in execution context")

        # Extract known fields
        known_fields = {"episode_id", "session_id", "task_id", "target_episode_ids", "role"}
        extra = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            episode_id=episode_id,
            session_id=data.get("session_id"),
            task_id=data.get("task_id"),
            target_episode_ids=data.get("target_episode_ids"),
            role=data.get("role"),
            extra=extra,
        )


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    """
    Base configuration for all executors.

    Subclasses should add their own typed fields. Do not use extras or get().
    """

    timeout: float = 300.0

    def with_overrides(self, overrides: dict[str, Any]) -> Self:
        """
        Create a new config instance with overrides applied.

        Only fields that exist on this config class are applied.
        Unknown keys in overrides are silently ignored.

        Args:
            overrides: Dictionary of field values to override

        Returns:
            New config instance with overrides applied
        """
        # Get valid field names for this specific config class
        valid_fields = {f.name for f in fields(self)}
        # Filter to only valid overrides
        valid_overrides = {k: v for k, v in overrides.items() if k in valid_fields}
        # Use dataclasses.replace to create new immutable instance
        return replace(self, **valid_overrides)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Self:
        """Create config from dictionary, using defaults for missing values."""
        if data is None:
            return cls()
        # Filter to only valid fields for this class
        valid_fields = {f.name for f in fields(cls)}
        valid_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**valid_data)


@dataclass(frozen=True, slots=True)
class ViewExecutorConfig(ExecutorConfig):
    """Configuration for view executor."""

    large_file_threshold: int = 2000


@dataclass(frozen=True, slots=True)
class GrepExecutorConfig(ExecutorConfig):
    """Configuration for grep executor."""

    max_results: int = 500


@dataclass(frozen=True, slots=True)
class GlobExecutorConfig(ExecutorConfig):
    """Configuration for glob executor."""

    max_results: int = 1000


@dataclass(frozen=True, slots=True)
class SqlExecutorConfig(ExecutorConfig):
    """Configuration for SQL executor."""

    allow_schema_queries: bool = True
    max_rows: int = 1000


@dataclass(frozen=True, slots=True)
class PythonExecutorConfig(ExecutorConfig):
    """Configuration for Python executor."""

    allowed_modules: list[str] = field(default_factory=list)
    script_templates: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BashExecutorConfig(ExecutorConfig):
    """Configuration for Bash executor."""

    timeout: float = 60.0  # Override default: reduced timeout for Bash
    allowed_commands: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EditExecutorConfig(ExecutorConfig):
    """Configuration for Edit executor."""

    timeout: float = 30.0  # Override default: file edits should be quick


@dataclass(frozen=True, slots=True)
class CreateExecutorConfig(ExecutorConfig):
    """Configuration for Create executor."""

    timeout: float = 30.0  # Override default: file creation should be quick


# =============================================================================
# Docker-specific models
# =============================================================================


@dataclass(frozen=True, slots=True)
class ResourceLimitsDocker:
    """
    Docker container resource limits configuration.

    Used by DockerConfig to replace resource_limits: dict[str, Any].
    """

    memory_limit: str | None = None  # e.g., "512m", "1g"
    cpu_limit: float | None = None  # e.g., 0.5, 1.0, 2.0
    cpu_shares: int | None = None
    pids_limit: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResourceLimitsDocker":
        """Create from dictionary."""
        if data is None:
            return cls()
        return cls(
            memory_limit=data.get("memory_limit"),
            cpu_limit=data.get("cpu_limit"),
            cpu_shares=data.get("cpu_shares"),
            pids_limit=data.get("pids_limit"),
        )


@dataclass(frozen=True, slots=True)
class DockerConfig:
    """Docker container configuration details."""

    image: str | None = None
    network_mode: str | None = None
    read_only_root: bool | None = None
    user: str | None = None
    resource_limits: ResourceLimitsDocker | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DockerConfig":
        """Create from dictionary."""
        if data is None:
            return cls()
        resource_limits_data = data.get("resource_limits")
        return cls(
            image=data.get("image"),
            network_mode=data.get("network_mode"),
            read_only_root=data.get("read_only_root"),
            user=data.get("user"),
            resource_limits=ResourceLimitsDocker.from_dict(resource_limits_data) if resource_limits_data else None,
        )


@dataclass(frozen=True, slots=True)
class DockerInfo:
    """Docker-specific configuration information returned by executors."""

    execution_environment: str = "docker_container"
    timeout: float = 300.0
    docker_config: DockerConfig | None = None


# =============================================================================
# Executor-specific parameter dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class EditParameters:
    """Parameters for the edit executor (str_replace)."""

    path: str
    old_str: str
    new_str: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditParameters":
        """Create from dictionary with validation."""
        path = data.get("path")
        old_str = data.get("old_str")
        if not path:
            raise ValueError("path is required")
        if old_str is None:
            raise ValueError("old_str is required")
        return cls(
            path=path,
            old_str=old_str,
            new_str=data.get("new_str", ""),
        )


@dataclass(frozen=True, slots=True)
class ViewParameters:
    """Parameters for the view executor."""

    path: str
    view_range: tuple[int, int] | None = None
    force_read_large_files: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ViewParameters":
        """Create from dictionary with validation."""
        path = data.get("path")
        if not path:
            raise ValueError("path is required")

        view_range = data.get("view_range")
        if view_range is not None:
            if isinstance(view_range, (list, tuple)) and len(view_range) == 2:
                view_range = (int(view_range[0]), int(view_range[1]))
            else:
                raise ValueError("view_range must be a [start, end] array")

        return cls(
            path=path,
            view_range=view_range,
            force_read_large_files=data.get("forceReadLargeFiles", False),
        )


@dataclass(frozen=True, slots=True)
class CreateParameters:
    """Parameters for the create executor."""

    path: str
    file_text: str = ""
    overwrite: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CreateParameters":
        """Create from dictionary with validation."""
        path = data.get("path")
        if not path:
            raise ValueError("path is required")
        return cls(
            path=path,
            file_text=data.get("file_text", ""),
            overwrite=data.get("overwrite", False),
        )


@dataclass(frozen=True, slots=True)
class GrepParameters:
    """Parameters for the grep executor."""

    pattern: str
    path: str | None = None
    glob: str | None = None
    file_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GrepParameters":
        """Create from dictionary with validation."""
        pattern = data.get("pattern")
        if not pattern:
            raise ValueError("pattern is required")
        return cls(
            pattern=pattern,
            path=data.get("path"),
            glob=data.get("glob"),
            file_type=data.get("type"),
        )


@dataclass(frozen=True, slots=True)
class GlobParameters:
    """Parameters for the glob executor."""

    pattern: str
    path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlobParameters":
        """Create from dictionary with validation."""
        pattern = data.get("pattern")
        if not pattern:
            raise ValueError("pattern is required")
        return cls(
            pattern=pattern,
            path=data.get("path"),
        )


@dataclass(frozen=True, slots=True)
class BashParameters:
    """Parameters for the bash executor."""

    command: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BashParameters":
        """Create from dictionary with validation."""
        command = data.get("command")
        if not command:
            raise ValueError("command is required")
        return cls(command=command)


@dataclass(frozen=True, slots=True)
class PythonParameters:
    """Parameters for the python executor."""

    code: str
    template: str | None = None
    working_dir: str = "/workspace"
    timeout: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PythonParameters":
        """Create from dictionary with validation."""
        code = data.get("code")
        if not code:
            raise ValueError("code is required")
        return cls(
            code=code,
            template=data.get("template"),
            working_dir=data.get("working_dir", "/workspace"),
            timeout=data.get("timeout"),
        )


@dataclass(frozen=True, slots=True)
class SqlParameters:
    """Parameters for the SQL executor."""

    query: str
    connection_string: str | None = None
    max_rows: int = 1000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SqlParameters":
        """Create from dictionary with validation."""
        query = data.get("query")
        if not query:
            raise ValueError("query is required")
        return cls(
            query=query,
            connection_string=data.get("connection_string"),
            max_rows=data.get("max_rows", 1000),
        )


@dataclass(frozen=True, slots=True)
class InjectPromptParameters:
    """Parameters for the inject_prompt executor."""

    message: str
    strategy: str = "append"
    wait_for_user: bool = True
    max_wait_seconds: float = 120.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InjectPromptParameters":
        """Create from dictionary with validation."""
        message = data.get("message")
        if not message:
            raise ValueError("message is required")
        return cls(
            message=message,
            strategy=data.get("strategy", "append"),
            wait_for_user=data.get("wait_for_user", True),
            max_wait_seconds=data.get("max_wait_seconds", 120.0),
        )


@dataclass(frozen=True, slots=True)
class GetTargetTranscriptParameters:
    """Parameters for the get_target_transcript executor."""

    wait_for_user: bool = True
    max_wait_seconds: float = 120.0
    retrieval_mode: str = "full"
    tail_count: int = 10
    since_version: int = 0
    include_metadata: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GetTargetTranscriptParameters":
        """Create from dictionary."""
        return cls(
            wait_for_user=data.get("wait_for_user", True),
            max_wait_seconds=data.get("max_wait_seconds", 120.0),
            retrieval_mode=data.get("retrieval_mode", "full"),
            tail_count=data.get("tail_count", 10),
            since_version=data.get("since_version", 0),
            include_metadata=data.get("include_metadata", True),
        )


# =============================================================================
# Configuration dataclasses - Replace untyped dict[str, Any] throughout codebase
# =============================================================================


@dataclass(frozen=True, slots=True)
class ExecutorMetadata:
    """
    Strongly-typed executor metadata.

    Replaces untyped _executor_metadata dicts on executor classes.
    """

    name: str
    description: str
    domain: str = "general"
    security_level: str = "medium"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutorMetadata":
        """Create from dictionary with defaults."""
        return cls(
            name=data.get("name", "unknown"),
            description=data.get("description", "No description"),
            domain=data.get("domain", "general"),
            security_level=data.get("security_level", "medium"),
        )


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """
    Configuration for sandbox environment manager.

    Replaces sandbox_config: dict[str, Any] in SandboxEnvironmentManager.__init__().
    """

    domain: str
    config_dir: str | None = None
    logs_dir: str | None = None
    enable_container_logging: bool = True
    enable_logging: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SandboxConfig":
        """Create from dictionary with validation."""
        domain = data.get("domain")
        if not domain:
            raise ValueError("domain is required in sandbox config")
        return cls(
            domain=domain,
            config_dir=data.get("config_dir"),
            logs_dir=data.get("logs_dir"),
            enable_container_logging=data.get("enable_container_logging", True),
            enable_logging=data.get("enable_logging", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for legacy compatibility."""
        result: dict[str, Any] = {
            "domain": self.domain,
            "enable_container_logging": self.enable_container_logging,
            "enable_logging": self.enable_logging,
        }
        if self.config_dir:
            result["config_dir"] = self.config_dir
        if self.logs_dir:
            result["logs_dir"] = self.logs_dir
        return result


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """
    Configuration for container logging.

    Replaces logging_config dicts passed to orchestrators.
    """

    logs_directory: str
    domain: str = "unknown"
    enable_logging: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        """Create from dictionary."""
        logs_directory = data.get("logs_directory")
        if not logs_directory:
            raise ValueError("logs_directory is required in logging config")
        return cls(
            logs_directory=logs_directory,
            domain=data.get("domain", "unknown"),
            enable_logging=data.get("enable_logging", True),
        )

    def to_dict(self) -> dict[str, str | bool]:
        """Convert to dictionary for legacy compatibility."""
        return {
            "logs_directory": self.logs_directory,
            "domain": self.domain,
            "enable_logging": self.enable_logging,
        }


@dataclass(frozen=True, slots=True)
class PermanentEnvironmentConfig:
    """
    Configuration for permanent environment manager.

    Replaces config: dict[str, Any] in PermanentEnvironmentManager.__init__().
    """

    domain: str
    config_dir: str | None = None
    logs_dir: str | None = None
    enable_logging: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PermanentEnvironmentConfig":
        """Create from dictionary with validation."""
        domain = data.get("domain")
        if not domain:
            raise ValueError("domain is required in permanent environment config")
        return cls(
            domain=domain,
            config_dir=data.get("config_dir"),
            logs_dir=data.get("logs_dir"),
            enable_logging=data.get("enable_logging", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for legacy compatibility."""
        result: dict[str, Any] = {
            "domain": self.domain,
            "enable_logging": self.enable_logging,
        }
        if self.config_dir:
            result["config_dir"] = self.config_dir
        if self.logs_dir:
            result["logs_dir"] = self.logs_dir
        return result


@dataclass(frozen=True, slots=True)
class EpisodeConfiguration:
    """
    Episode-specific executor configuration.

    Replaces _episode_configurations: dict[str, dict[str, Any]] values
    in ExecutorFactory.
    """

    allowed_executors: list[str]
    config: dict[str, Any] = field(default_factory=dict)  # Per-executor configs, nested by executor type

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeConfiguration":
        """Create from dictionary."""
        allowed_executors = data.get("allowed_executors", [])
        config = data.get("config", {})
        return cls(
            allowed_executors=list(allowed_executors),
            config=dict(config),
        )


@dataclass(frozen=True, slots=True)
class ExecutionStats:
    """
    Statistics about active executions.

    Replaces the dict returned by ExecutionManager.get_execution_stats().
    """

    total_active_executions: int
    active_episodes: int
    max_concurrent_per_episode: int
    episode_execution_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class CommandInfo:
    """
    Information about an available command/executor.

    Replaces the dicts returned by ExecutionManager.list_commands().
    """

    executor_type: str
    name: str
    description: str
    domain: str
    security_level: str
    parameters: list[str]

    @classmethod
    def from_executor(
        cls,
        executor_type: str,
        metadata: "ExecutorMetadata",
        parameters: list[str],
    ) -> "CommandInfo":
        """Create from executor metadata."""
        return cls(
            executor_type=executor_type,
            name=metadata.name,
            description=metadata.description,
            domain=metadata.domain,
            security_level=metadata.security_level,
            parameters=parameters,
        )


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """
    Result of cleanup operations.

    Replaces the dict returned by ExecutionManager.cleanup_all_containers().
    """

    ephemeral_episodes_cleaned: int
    permanent_environment_stopped: bool
    total_cleanup_success: bool
