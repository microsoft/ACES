"""SABER Sandbox Environment for Inspect AI.

Implements the SandboxEnvironment interface by delegating to SABER's DomainOrchestrator
and managing per-sample REST sessions and MCP clients for tool execution.

This module realizes Phase 3 of the SABER sandbox integration, providing lifecycle
management (task_init, sample_init, sample_cleanup, task_cleanup) with class-level
domain ownership semantics.

Concurrency is controlled at the Inspect AI level via --max-samples (default: 8).
"""

import inspect
import threading
import time
from pathlib import Path
from typing import Any

import anyio
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.log._samples import sample_active
from inspect_ai.tool import Tool
from inspect_ai.tool._mcp._local import MCPServerLocal
from inspect_ai.util import sandboxenv, store
from pydantic import BaseModel

from saber.client.client_session import ClientSessionManager
from saber.client.models import SessionManagerConfig
from saber.debug_logging import (
    clear_episode_context,
    enable_debug_logging,
    log_lifecycle_summary,
    log_sample_cleanup_called,
    log_sample_init_complete,
    log_sample_init_start,
)
from saber.logging_config import LogCategory, get_saber_logger
from saber.models import (
    BenchmarkTask,
    ClientIdentifiers,
    ExecutionMode,
    MetadataKeys,
    OrchestratedTask,
    SingleEpisodeTask,
    TaskExecutionMode,
    TaskInitMode,
)

from .config.env_loader import deserialize_config, get_config_files, get_default_concurrency, load_environment
from .constants import InspectStoreKeys, SandboxTimeouts
from .core.orchestration_coordinator import OrchestrationCoordinator
from .core.orchestration_init import OrchestrationInitializer
from .core.task_handlers import get_benchmark_task_handler
from .core.types import HandlerState, OrchestrationSubTaskState
from .server.domain_manager import get_active_domain, remove_active_domain
from .server.episode_manager import EpisodeLifecycleManager
from .server.sandbox_registry import SandboxRegistry
from .server.server import DomainContext, DomainController
from .server.session_manager import SessionLifecycleManager

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Load environment variables from .env file for eval-retry support
load_environment()


class SandboxError(Exception):
    """Exception raised for SABER sandbox lifecycle errors."""

    pass


@sandboxenv("saber")
class SABERSandboxEnvironment:
    """SABER sandbox environment with lifecycle management and single ownership semantics.

    Implements Inspect AI sandbox lifecycle hooks to integrate with SABER's domain orchestrator.
    Enforces single ownership per domain, with per-task sessions and per-sample episodes.

    Concurrency is controlled at the Inspect AI level via --max-samples (default: 8).
    """

    # Backward compatibility: expose SandboxRegistry._registry as class attribute
    # This allows tests to access SABERSandboxEnvironment._registry directly
    _registry = SandboxRegistry._registry
    _lock = SandboxRegistry._lock  # Backward compatibility for tests

    # Stateless lifecycle managers (reused across all samples for efficiency)
    _orchestration_initializer: Any = None  # OrchestrationInitializer instance (TYPE: ignore None check via assertion)
    _episode_mapping_lock = threading.Lock()  # Thread-safe lock for episode mapping operations

    def __init__(
        self,
        domain_slug: str,
        domains_root: Path,
        rest_port: int = 8000,
        mcp_port: int = 8001,
        compose_template_path: Path | None = None,
        mcp_timeout: float = SandboxTimeouts.MCP_CONNECTION_SECONDS,
        rest_base_url: str | None = None,
        mcp_url: str | None = None,
        enable_debug_logging: bool = False,
    ):
        """Initialize SABER sandbox instance.

        Args:
            domain_slug: SABER domain slug to use
            domains_root: Path to SABER domains directory
            rest_port: REST API port (default: 8000)
            mcp_port: MCP API port (default: 8001)
            compose_template_path: Optional custom compose template
            mcp_timeout: Timeout for MCP operations (default: SandboxTimeouts.MCP_CONNECTION_SECONDS)
            rest_base_url: Override REST base URL (default: http://localhost:{rest_port})
            mcp_url: Override MCP URL (default: http://localhost:{mcp_port})
            enable_debug_logging: Enable detailed episode lifecycle debug logging (default: False)

        Note:
            Concurrency is controlled via Inspect AI's --max-samples flag (default: 8).
            Use --max-samples to adjust parallel sample execution.
        """
        self._domain_slug = domain_slug
        self._domains_root = Path(domains_root)
        self._rest_port = rest_port
        self._mcp_port = mcp_port
        self._compose_template_path = compose_template_path
        self._mcp_timeout = mcp_timeout
        self._enable_debug_logging = enable_debug_logging

        self._rest_base_url = rest_base_url or f"http://localhost:{rest_port}"
        self._mcp_url = mcp_url or f"http://localhost:{mcp_port}"

        self._episode_id: str | None = None
        self._task_id: str | None = None
        self._sample_id: str | None = None
        self._mcp_client: Tool | None = None
        self._handler: Any | None = None
        self._handler_state: HandlerState | None = None
        self._episode_ids: list[str] | None = None
        self._primary_episode_id: str | None = None
        self._session_id: str | None = None
        self._context: DomainContext | None = None
        self._session_manager: ClientSessionManager | None = None
        self._episode_manager: EpisodeLifecycleManager | None = None

    @classmethod
    def default_concurrency(cls) -> int | None:
        """Default max_sandboxes for SABER provider."""
        return get_default_concurrency()

    @classmethod
    def config_files(cls) -> list[str]:
        """Return list of config files for SABER sandbox."""
        return get_config_files()

    @classmethod
    def config_deserialize(cls, config: dict[str, Any]) -> BaseModel:
        """Deserialize SABER sandbox config from dict."""
        return deserialize_config(config)

    @classmethod
    def clear_stale_ownership(cls, domain_slug: str, force: bool = False) -> bool:
        """Clear stale ownership for a domain (useful for debugging/recovery)."""
        return SandboxRegistry.clear_stale_ownership(domain_slug, force)

    @classmethod
    async def task_init_environment(
        cls,
        config: BaseModel | None,
        metadata: dict[str, str],
    ) -> dict[str, str]:
        """Return environment variables for task initialization (empty for SABER)."""
        return {}

    @classmethod
    async def task_init(
        cls,
        task_name: str,
        config: BaseModel | None,
    ) -> None:
        """Initialize SABER domain for task execution.

        Supports three modes:
        1. Ownership transfer: If domain already started by factory, reuse it
        2. Fresh start: Start domain if not already running (backward compat)
        3. Eval-retry: If domain owned by same task (eval-retry scenario), reuse existing session

        Only one task can own a domain at a time. If domain is already owned
        by a different task, raises SandboxError.

        Args:
            task_name: Name of the task owning this domain
            config: SABER sandbox configuration (BaseModel with domain_slug, domains_root, etc.)

        Raises:
            SandboxError: If domain already owned by different task or startup fails
        """
        if config is None:
            raise SandboxError("SABER sandbox config is required for task_init")

        domain_slug = config.domain_slug  # type: ignore[attr-defined]
        domains_root = config.domains_root  # type: ignore[attr-defined]
        rest_port = getattr(config, "rest_port", 8000)
        mcp_port = getattr(config, "mcp_port", 8001)

        enable_debug = getattr(config, "enable_debug_logging", False)
        if enable_debug:
            enable_debug_logging()
            logger.info("Episode lifecycle debug logging enabled")

        existing_entry = SandboxRegistry.get_domain_entry(domain_slug)
        if existing_entry:
            owner = existing_entry.owner

            if owner == task_name:
                logger.info(
                    f"Domain '{domain_slug}' already initialized by task '{task_name}' "
                    "(likely eval-retry scenario). Reusing existing session.",
                    extra={
                        "domain": domain_slug,
                        "task": task_name,
                        "mode": TaskInitMode.EVAL_RETRY_REUSE,
                    },
                )
                return
            else:
                raise SandboxError(
                    f"SABER sandbox for domain '{domain_slug}' already owned by task '{owner}'. "
                    "Only one task can own a domain at a time. Ensure previous task cleanup completed."
                )

        active_domain = get_active_domain(domain_slug)

        if active_domain:
            logger.info(
                f"Domain '{domain_slug}' already started by factory, transferring ownership to sandbox",
                extra={
                    "domain": domain_slug,
                    "task": task_name,
                    "mode": TaskInitMode.OWNERSHIP_TRANSFER,
                },
            )

            session_id = await cls._create_session_for_task(active_domain["rest_url"], task_name)

            SandboxRegistry.register_domain(
                domain_slug=domain_slug,
                task_name=task_name,
                controller=active_domain["controller"],
                context=active_domain["context"],
                ownership=True,
                rest_port=active_domain["rest_port"],
                mcp_port=active_domain["mcp_port"],
                rest_url=active_domain["rest_url"],
                mcp_url=active_domain["mcp_url"],
                session_id=session_id,
            )

            # Initialize orchestration initializer for multi-sample orchestration support
            cls._orchestration_initializer = OrchestrationInitializer()
            return

        controller = DomainController(Path(domains_root))

        try:
            logger.info(
                f"Starting SABER domain '{domain_slug}' for task '{task_name}'",
                extra={"domain": domain_slug, "task": task_name, "rest_port": rest_port, "mcp_port": mcp_port},
            )

            context = await controller.start(
                domain=domain_slug,
                rest_port=rest_port,
                mcp_port=mcp_port,
                log_level="INFO",
            )

            session_id = await cls._create_session_for_task(context.rest_url, task_name)

            SandboxRegistry.register_domain(
                domain_slug=domain_slug,
                task_name=task_name,
                controller=controller,
                context=context,
                ownership=True,
                rest_port=rest_port,
                mcp_port=mcp_port,
                rest_url=context.rest_url,
                mcp_url=context.mcp_url,
                session_id=session_id,
            )

            logger.info(
                f"Domain '{domain_slug}' started for task '{task_name}'",
                extra={"domain": domain_slug, "task": task_name, "session_id": session_id},
            )

            # Initialize orchestration initializer for multi-sample orchestration support
            cls._orchestration_initializer = OrchestrationInitializer()

        except Exception as e:
            SandboxRegistry.unregister_domain(domain_slug)

            try:
                await controller.stop(domain_slug)
            except Exception as stop_error:
                logger.warning(
                    f"Failed to stop domain after startup error: {stop_error}", extra={"domain": domain_slug}
                )

            raise SandboxError(f"Failed to start SABER sandbox for domain '{domain_slug}': {e}") from e

    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: BaseModel | None,
        metadata: dict[str, str],
    ) -> dict[str, "SABERSandboxEnvironment"]:
        """Initialize sandbox environment for sample execution.

        Creates a SABER sandbox instance, establishes REST session, starts episode,
        and constructs MCP client for tool execution.

        Handles eval-retry scenario: If sample already has SABER IDs in metadata
        (from a previous completed run), creates a minimal instance without
        initializing new episode.

        Args:
            task_name: Name of task using the sandbox
            config: SABER sandbox configuration
            metadata: Sample metadata dict, must contain 'task_id' key

        Returns:
            Dict with single 'default' key mapping to initialized SABERSandboxEnvironment

        Raises:
            ValueError: If metadata missing 'task_id'/'benchmark_task' or config is None
            SandboxError: If session/episode creation fails
        """
        if config is None:
            raise ValueError("SABER sandbox config is required for sample_init")

        # Check for either benchmark_task (new format) or task_id (legacy format)
        if "benchmark_task" not in metadata and "task_id" not in metadata:
            raise ValueError(
                "Missing 'task_id' or 'benchmark_task' in metadata. "
                "Dataset must provide task identifier to start SABER episode."
            )

        # Create instance
        instance = cls(
            domain_slug=config.domain_slug,  # type: ignore[attr-defined]
            domains_root=config.domains_root,  # type: ignore[attr-defined]
            rest_port=getattr(config, "rest_port", 8000),
            mcp_port=getattr(config, "mcp_port", 8001),
            compose_template_path=getattr(config, "compose_template_path", None),
        )

        # Check if this is a completed sample from eval-retry
        # (inspect_ai's eval_retry will skip these samples, but we still need to provide an instance)
        if MetadataKeys.SABER_SESSION_ID in metadata and MetadataKeys.SABER_EPISODE_ID in metadata:
            logger.info(
                "Sample already has SABER IDs (eval-retry completed sample), creating minimal instance",
                extra={
                    "task_id": metadata[MetadataKeys.TASK_ID],
                    "session_id": metadata[MetadataKeys.SABER_SESSION_ID],
                    "episode_id": metadata[MetadataKeys.SABER_EPISODE_ID],
                    "mode": TaskInitMode.EVAL_RETRY_COMPLETED_SAMPLE,
                },
            )
            # Set IDs from metadata but don't create new episode
            instance._session_id = metadata[MetadataKeys.SABER_SESSION_ID]
            instance._episode_id = metadata[MetadataKeys.SABER_EPISODE_ID]
            instance._task_id = metadata[MetadataKeys.TASK_ID]
            # Don't create MCP client - sample won't be executed
            return {"default": instance}

        # Normal case: Initialize session and episode for new sample
        await instance._init_sample(metadata)

        return {"default": instance}

    async def _init_sample(self, metadata: dict[str, str]) -> None:
        """Internal method to initialize episode(s) for sample using task handlers.

        This method now supports both single and orchestrated tasks through polymorphic handlers.

        Args:
            metadata: Sample metadata dict with 'benchmark_task' or 'task_id'
        """
        init_start_time = time.time()

        # Extract task information from metadata
        benchmark_task = None
        task_id: str | None = None

        if MetadataKeys.BENCHMARK_TASK in metadata:
            # New polymorphic task format
            task_data = metadata[MetadataKeys.BENCHMARK_TASK]
            if not isinstance(task_data, dict):  # type: ignore[unreachable]
                raise SandboxError(f"Invalid benchmark task data type: {type(task_data)}")
            benchmark_task = self._deserialize_benchmark_task(task_data)  # type: ignore[unreachable]
            task_id = benchmark_task.benchmark_task_id

        # Set task_id with fallback
        self._task_id = task_id or metadata.get(MetadataKeys.SAMPLE_ID) or "unknown"

        self._sample_id = metadata.get(MetadataKeys.SAMPLE_ID, self._task_id)

        # Get registry entry to access URLs and session_id
        with self._lock:
            entry = self._registry.get(self._domain_slug)
            if not entry:
                raise SandboxError(f"Domain '{self._domain_slug}' not initialized. Call task_init before sample_init.")
            rest_base_url = entry.rest_url
            mcp_url_base = entry.mcp_url
            self._session_id = entry.session_id  # Get shared session from registry

        # Create session manager if not already created
        if self._session_manager is None:
            config = SessionManagerConfig(
                base_url=rest_base_url,
                mcp_server_url=mcp_url_base,
                client_id=ClientIdentifiers.INSPECT_AI_SANDBOX,
                rest_timeout=SandboxTimeouts.REST_API_SECONDS,
            )
            self._session_manager = ClientSessionManager(config=config)
            self._session_manager._current_session_id = self._session_id

            # Initialize episode manager with session manager
            self._episode_manager = EpisodeLifecycleManager(self._session_manager)

        try:
            # Check execution mode to determine how to handle episode creation
            execution_mode = metadata.get(MetadataKeys.EXECUTION_MODE)

            if execution_mode == ExecutionMode.ORCHESTRATED_SUB_TASK:
                # Each sub-task is a separate sample coordinated via OrchestrationCoordinator
                assert (
                    type(self)._orchestration_initializer is not None
                ), "orchestration_initializer must be initialized"
                self._handler_state = await type(self)._orchestration_initializer.init_orchestrated_sub_task(
                    metadata=metadata,
                    session_id=self._session_id,
                    session_manager=self._session_manager,
                    sample_id=self._sample_id,
                )
                # Extract episode information from handler state
                self._episode_ids = self._handler_state.episode_ids
                self._primary_episode_id = self._handler_state.primary_episode_id
                self._episode_id = self._primary_episode_id

                # Register ActiveSample with coordinator for sibling interruption
                active = sample_active()
                if active is not None and isinstance(self._handler_state, OrchestrationSubTaskState):
                    OrchestrationCoordinator().set_active_sample(
                        orchestration_id=self._handler_state.orchestration_id,
                        role=self._handler_state.sub_task_role,
                        active_sample=active,
                    )
                    logger.debug(
                        "Registered ActiveSample with OrchestrationCoordinator",
                        extra={
                            "orchestration_id": self._handler_state.orchestration_id,
                            "role": self._handler_state.sub_task_role,
                            "active_sample_id": active.id,
                        },
                    )
            elif benchmark_task is not None:
                # Use task handler pattern for single episode tasks
                # (OrchestratedTask should not reach here with new dataset creation)
                self._handler = get_benchmark_task_handler(benchmark_task)  # type: ignore[unreachable]

                # Initialize episode(s) using handler
                # Concurrency is controlled by Inspect AI --max-samples (default: 8)
                self._handler_state = await self._handler.initialize(
                    benchmark_task=benchmark_task,
                    session_id=self._session_id,
                    session_manager=self._session_manager,
                )

                # Extract episode information from handler state
                self._episode_ids = self._handler_state.episode_ids
                self._primary_episode_id = self._handler_state.primary_episode_id
                self._episode_id = self._primary_episode_id

            log_sample_init_start(
                task_id=self._task_id,
                sample_id=self._sample_id,
                episode_id=self._primary_episode_id,
                session_id=self._session_id,
                domain_slug=self._domain_slug,
            )

            # Store session manager and context in inspect_ai store for scorer and solver
            # Note: We store to task_store instead of metadata because inspect_ai creates
            # state.metadata as a deepcopy of sample.metadata BEFORE calling sample_init,
            # so any updates to the metadata parameter here are lost. The solver retrieves
            # SABER context from state.store instead.
            task_store = store()
            task_store.set(InspectStoreKeys.SESSION_MANAGER, self._session_manager)
            task_store.set(InspectStoreKeys.SESSION_ID, self._session_id)
            task_store.set(InspectStoreKeys.TASK_ID, self._task_id)
            task_store.set(InspectStoreKeys.DOMAIN_SLUG, self._domain_slug)

            # Store episode mapping
            sample_id = metadata.get(MetadataKeys.SAMPLE_ID, self._task_id)
            primary_episode_id = self._primary_episode_id
            session_id = self._session_id
            task_id = self._task_id

            # Ensure all values are strings for store_episode_mapping
            assert isinstance(sample_id, str), f"sample_id must be str, got {type(sample_id)}"
            assert isinstance(primary_episode_id, str), f"episode_id must be str, got {type(primary_episode_id)}"
            assert isinstance(session_id, str), f"session_id must be str, got {type(session_id)}"
            assert isinstance(task_id, str), f"task_id must be str, got {type(task_id)}"

            SandboxRegistry.store_episode_mapping(
                sample_id=sample_id,
                episode_id=primary_episode_id,
                session_id=session_id,
                task_id=task_id,
            )

            # Construct MCP client with headers and retry logic
            from .core.mcp_factory import MCPClientFactory as MCPFactory

            # Create factory once per instance (reused for cleanup)
            if not hasattr(self, "_mcp_factory") or self._mcp_factory is None:
                self._mcp_factory: MCPFactory = MCPFactory(
                    mcp_url_base=mcp_url_base,
                    domain_slug=self._domain_slug,
                    timeout=self._mcp_timeout,
                )

            # Type assertion: _primary_episode_id is guaranteed to be str at this point
            assert self._primary_episode_id is not None, "primary_episode_id must be set"
            self._mcp_client = await self._mcp_factory.create_mcp_client(
                session_id=self._session_id,
                episode_id=self._primary_episode_id,
                task_id=self._task_id,
                sample_id=sample_id,
            )

            # Log completion of sample initialization
            init_duration = time.time() - init_start_time
            mcp_url = f"{mcp_url_base}/mcp"
            log_sample_init_complete(
                duration=init_duration,
                sample_id=self._sample_id,
                mcp_url=mcp_url,
            )

        except Exception as e:
            # Use handler for cleanup if available
            if self._handler is not None and self._handler_state is not None:
                try:
                    cleanup_result = await self._handler.cleanup(
                        state=self._handler_state,
                        session_id=self._session_id,
                        session_manager=self._session_manager,
                    )
                    if cleanup_result.has_errors:
                        logger.warning(
                            "Handler cleanup completed with errors during init error",
                            extra={
                                "task_id": self._task_id,
                                "error_count": cleanup_result.error_count,
                                "errors": cleanup_result.errors,
                            },
                        )
                except Exception as cleanup_err:
                    logger.error(
                        "Handler cleanup failed during init error",
                        extra={
                            "task_id": self._task_id,
                            "error": str(cleanup_err),
                        },
                    )

            # Cleanup partial state
            await self._cleanup_partial_state(interrupted=True)
            raise PrerequisiteError(
                f"Failed to initialize SABER sandbox for sample (task_id={self._task_id}): {e}"
            ) from e

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: BaseModel | None,
        environments: dict[str, "SABERSandboxEnvironment"],
        interrupted: bool,
    ) -> None:
        """Cleanup sandbox environments after sample execution (best-effort)."""
        # Capture caller info for debugging
        caller_frame = inspect.currentframe()
        caller_info = "unknown"
        if caller_frame and caller_frame.f_back:
            caller_info = f"{caller_frame.f_back.f_code.co_filename}:{caller_frame.f_back.f_lineno}"

        for name, env in environments.items():
            # Log sample cleanup call
            log_sample_cleanup_called(
                interrupted=interrupted,
                caller=caller_info,
                task_name=task_name,
                environment_name=name,
            )
            try:
                await env._cleanup_sample(interrupted=interrupted)
            except Exception as e:
                logger.warning(
                    f"Error during sample cleanup for environment '{name}': {e}",
                    extra={
                        "environment": name,
                        "session_id": getattr(env, "_session_id", None),
                        "episode_id": getattr(env, "_episode_id", None),
                    },
                    exc_info=True,
                )

    async def _cleanup_sample(self, interrupted: bool = False) -> None:
        """Internal method to cleanup sample state (handlers, episodes, mappings)."""
        logger.info(
            "[SAMPLE_CLEANUP] _cleanup_sample() CALLED",
            extra={"episode_id": self._episode_id, "interrupted": interrupted},
        )
        with anyio.CancelScope(shield=True):
            try:
                # Cleanup WebSocket model wrapper if present (before episode cleanup)
                # Use the class-level registry to find the wrapper by episode_id
                from .integration.model_wrapper import WebSocketTranscriptSyncingModelWrapper

                wrapper = (
                    WebSocketTranscriptSyncingModelWrapper.get_wrapper_for_episode(self._episode_id)
                    if self._episode_id
                    else None
                )
                logger.info(
                    "[SAMPLE_CLEANUP] Looking up wrapper in class registry",
                    extra={
                        "episode_id": self._episode_id,
                        "wrapper_found": wrapper is not None,
                        "wrapper_type": type(wrapper).__name__ if wrapper else None,
                    },
                )

                if wrapper and hasattr(wrapper, "cleanup"):
                    try:
                        logger.info(
                            "[SAMPLE_CLEANUP] Calling wrapper.cleanup()",
                            extra={"episode_id": self._episode_id},
                        )
                        await wrapper.cleanup()
                        logger.info(
                            "[SAMPLE_CLEANUP] wrapper.cleanup() COMPLETED",
                            extra={"episode_id": self._episode_id},
                        )
                    except Exception as e:
                        logger.warning(
                            "[SAMPLE_CLEANUP] Failed to cleanup WebSocket model wrapper",
                            extra={"episode_id": self._episode_id, "error": str(e)},
                        )
                else:
                    logger.info(
                        "[SAMPLE_CLEANUP] No wrapper to cleanup or no cleanup method",
                        extra={
                            "episode_id": self._episode_id,
                            "has_wrapper": wrapper is not None,
                            "has_cleanup": hasattr(wrapper, "cleanup") if wrapper else False,
                        },
                    )

                # Check if this is an orchestrated sub-task by looking for orchestration_id field
                if (
                    self._handler_state
                    and isinstance(self._handler_state, dict)
                    and "orchestration_id" in self._handler_state
                ):
                    # Old dict-based orchestration (backward compat)
                    assert (
                        type(self)._orchestration_initializer is not None
                    ), "orchestration_initializer must be initialized"
                    await type(self)._orchestration_initializer.cleanup_orchestrated_sub_task(
                        handler_state=self._handler_state,
                        session_id=self._session_id,
                        session_manager=self._session_manager,
                    )
                elif (
                    self._handler_state
                    and hasattr(self._handler_state, "to_dict")
                    and "orchestration_id" in self._handler_state.to_dict()
                ):
                    # New object-based orchestration
                    assert (
                        type(self)._orchestration_initializer is not None
                    ), "orchestration_initializer must be initialized"
                    await type(self)._orchestration_initializer.cleanup_orchestrated_sub_task(
                        handler_state=self._handler_state,
                        session_id=self._session_id,
                        session_manager=self._session_manager,
                    )
                elif self._handler is not None and self._handler_state is not None:
                    cleanup_result = await self._handler.cleanup(
                        state=self._handler_state,
                        session_id=self._session_id,
                        session_manager=self._session_manager,
                    )
                    if cleanup_result.has_errors:
                        logger.warning(
                            "Handler cleanup completed with errors",
                            extra={"task_id": self._task_id, "error_count": cleanup_result.error_count},
                        )
                else:
                    await self._cleanup_partial_state(interrupted=interrupted)
            except Exception as e:
                logger.warning(
                    f"Error during sample cleanup: {e}",
                    extra={
                        "session_id": self._session_id,
                        "episode_id": self._episode_id,
                    },
                    exc_info=True,
                )
            finally:
                log_lifecycle_summary()

                if hasattr(self, "_sample_id") and self._sample_id:
                    try:
                        SandboxRegistry.remove_episode_mapping(
                            sample_id=self._sample_id,
                            task_id=self._task_id or "unknown",
                        )
                    except Exception as cleanup_err:
                        logger.warning(
                            f"Failed to clean up episode mapping for sample {self._sample_id}: {cleanup_err}",
                            extra={"sample_id": self._sample_id, "task_id": self._task_id},
                        )

                # Clean up MCP session cache to prevent memory leaks
                if hasattr(self, "_mcp_client") and self._mcp_client is not None:
                    try:
                        task_id = anyio.get_current_task().id
                        server_name = f"SABER {self._domain_slug} Tools - {self._sample_id}"
                        cache_key = f"{task_id}_{server_name}"

                        if cache_key in MCPServerLocal._task_sessions:
                            cached_session = MCPServerLocal._task_sessions.pop(cache_key)
                            if hasattr(cached_session, "_session") and cached_session._session is not None:
                                await cached_session.__aexit__(None, None, None)
                    except Exception as cache_cleanup_err:
                        logger.warning(
                            f"Failed to clean up MCP session cache for sample {self._sample_id}: {cache_cleanup_err}",
                            extra={"sample_id": self._sample_id, "task_id": self._task_id},
                        )

                self._reset_state()
                clear_episode_context()

    @classmethod
    async def task_cleanup(
        cls,
        task_name: str,
        config: BaseModel | None,
        cleanup: bool = True,
    ) -> None:
        """Cleanup SABER domain after task completion (terminate session, optionally stop domain)."""
        if config is None:
            logger.warning("task_cleanup called with no config, skipping")
            return

        domain_slug = config.domain_slug  # type: ignore[attr-defined]
        cleanup_requested = getattr(config, "cleanup", cleanup)

        entry = SandboxRegistry.get_domain_entry(domain_slug)
        if not entry:
            logger.info(
                f"Domain '{domain_slug}' not in registry during task_cleanup. "
                "May indicate partial failure during task_init or already cleaned up.",
                extra={"domain": domain_slug, "task": task_name},
            )
            remove_active_domain(domain_slug)
            return

        controller = entry.controller
        ownership = entry.ownership
        session_id = entry.session_id
        rest_url = entry.rest_url
        owner = entry.owner

        # Log cleanup decision for debugging eval-retry scenarios
        logger.info(
            f"Cleaning up domain '{domain_slug}' for task '{task_name}'",
            extra={
                "domain": domain_slug,
                "task": task_name,
                "owner": owner,
                "cleanup_requested": cleanup_requested,
                "has_session": session_id is not None,
            },
        )

        if not cleanup_requested and not cleanup:
            logger.info(
                f"Preserving domain '{domain_slug}' ownership for potential eval-retry (--no-sandbox-cleanup)",
                extra={
                    "domain": domain_slug,
                    "task": task_name,
                    "session_id": session_id,
                    "note": "Domain and session preserved - episodes remain alive on server",
                },
            )
            # NOTE: We do NOT terminate the session here when --no-sandbox-cleanup is used.
            # Terminating the session would cause the server to end all episodes,
            # defeating the purpose of --no-sandbox-cleanup.
            # The domain, session, and episodes all remain alive for debugging/inspection.
            return

        SandboxRegistry.unregister_domain(domain_slug)
        if session_id and rest_url:
            session_mgr = SessionLifecycleManager(rest_url)
            session_mgr.terminate_session_sync(session_id)

        remove_active_domain(domain_slug)

        if ownership and cleanup_requested:
            try:
                await controller.stop(domain_slug)
                logger.info(f"Stopped domain '{domain_slug}'", extra={"domain": domain_slug})
            except Exception as e:
                logger.warning(f"Failed to stop domain '{domain_slug}': {e}", extra={"domain": domain_slug})

    def _reset_state(self) -> None:
        """Reset per-sample state."""
        self._episode_id = None
        self._mcp_client = None
        self._task_id = None
        self._sample_id = None

    @staticmethod
    async def _create_session_for_task(rest_base_url: str, task_name: str) -> str:
        """Create SABER session for a task via REST API.

        This is a backward-compatibility helper that delegates to SessionLifecycleManager.

        Args:
            rest_base_url: Base URL for SABER REST API
            task_name: Name of the task

        Returns:
            Session ID

        Raises:
            PrerequisiteError: If session creation fails
        """
        session_mgr = SessionLifecycleManager(rest_base_url)
        return await session_mgr.create_session(task_name)

    async def _cleanup_partial_state(self, interrupted: bool = False) -> None:
        """Best-effort cleanup of episode with robust retry logic.

        Unified episode ending for all paths - delegates to EpisodeLifecycleManager.
        Note: Session is NOT terminated here - it's shared across all samples and cleaned up in task_cleanup.

        Args:
            interrupted: Whether this is an interrupted (non-happy path) cleanup
        """
        if self._episode_id and self._episode_manager:
            # Get submission from store if available (happy path after scorer)
            submission = None
            if not interrupted:
                try:
                    task_store = store()
                    submission = task_store.get(InspectStoreKeys.EPISODE_SUBMISSION)
                except Exception:
                    # Store might not be available or submission not set - that's ok
                    pass

            # Delegate to episode manager for unified cleanup (best-effort)
            try:
                assert self._session_id is not None, "session_id must be set for cleanup"
                await self._episode_manager.cleanup_episode(
                    session_id=self._session_id,
                    episode_id=self._episode_id,
                    interrupted=interrupted,
                    submission=submission,
                )
            except Exception as e:
                # Log but don't re-raise - this is best-effort cleanup
                logger.warning(
                    f"Episode cleanup failed (best-effort): {e}",
                    extra={
                        "session_id": self._session_id,
                        "episode_id": self._episode_id,
                        "interrupted": interrupted,
                    },
                )

    def _deserialize_benchmark_task(self, data: dict[str, Any]) -> BenchmarkTask:
        """Deserialize BenchmarkTask from metadata dict.

        Args:
            data: Serialized benchmark task data from sample metadata

        Returns:
            SingleEpisodeTask or OrchestratedTask instance

        Raises:
            ValueError: If task_type/execution_mode is unknown or data is invalid
        """
        # Try task_type first (new discriminator field), fall back to execution_mode
        task_type = data.get(MetadataKeys.TASK_TYPE.value) or data.get(MetadataKeys.EXECUTION_MODE.value)

        if task_type is None:
            raise ValueError(
                f"Missing '{MetadataKeys.TASK_TYPE.value}' or "
                f"'{MetadataKeys.EXECUTION_MODE.value}' in benchmark task data. "
                f"Available keys: {list(data.keys())}"
            )

        if task_type == TaskExecutionMode.SINGLE.value:
            return SingleEpisodeTask(**data)
        elif task_type == TaskExecutionMode.ORCHESTRATED.value:
            return OrchestratedTask(**data)
        else:
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Expected '{TaskExecutionMode.SINGLE.value}' or '{TaskExecutionMode.ORCHESTRATED.value}'. "
                f"Data keys: {list(data.keys())}"
            )

    async def connection(self, *, user: str | None = None) -> Any:
        """Not supported - use saber_tools() for MCP protocol access."""
        raise NotImplementedError("connection() not supported - use saber_tools() for SABER access via MCP protocol.")

    async def exec(self, *args: Any, **kwargs: Any) -> Any:
        """Not supported - use saber_tools() for command execution."""
        raise NotImplementedError("exec() not supported - use saber_tools() for command execution via MCP.")

    async def read_file(self, *args: Any, **kwargs: Any) -> Any:
        """Not supported - use saber_tools() for file operations."""
        raise NotImplementedError("read_file() not supported - use saber_tools() for file operations via MCP.")

    async def write_file(self, *args: Any, **kwargs: Any) -> Any:
        """Not supported - use saber_tools() for file operations."""
        raise NotImplementedError("write_file() not supported - use saber_tools() for file operations via MCP.")
