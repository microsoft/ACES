"""SABER Sandbox Environment for Inspect AI.

Implements the SandboxEnvironment interface by delegating to SABER's DomainOrchestrator
and managing per-sample REST sessions and MCP clients for tool execution.

This module realizes Phase 3 of the SABER sandbox integration, providing lifecycle
management (task_init, sample_init, sample_cleanup, task_cleanup) with class-level
domain ownership semantics.
"""

import asyncio
import inspect
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import anyio
import requests  # type: ignore[import-untyped]
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.tool import Tool, mcp_server_http
from inspect_ai.tool._mcp._local import MCPServerLocal
from inspect_ai.util import sandboxenv, store
from pydantic import BaseModel, ConfigDict, create_model

from saber.client.client_session import ClientSessionManager
from saber.client.models import SessionManagerConfig
from saber.debug_logging import (
    EpisodeDebugLogger,
    clear_episode_context,
    enable_debug_logging,
    log_episode_end_complete,
    log_episode_end_request,
    log_lifecycle_summary,
    log_sample_cleanup_called,
    log_sample_init_complete,
    log_sample_init_start,
)
from saber.logging_config import LogCategory, get_saber_logger
from saber.models.mcp import OrchestrationEnvironment

from .server import DomainContext, DomainController
from .tasks import get_active_domain, remove_active_domain

logger = get_saber_logger(LogCategory.AGENT, __name__)

# Load environment variables from .env file for eval-retry support
# This ensures API keys and other config are available when retrying evals
try:
    from dotenv import load_dotenv

    # Try to find .env file in saber package directory
    saber_dir = Path(__file__).parent.parent.parent
    env_file = saber_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.debug(f"Loaded environment from {env_file}", extra={"env_file": str(env_file)})
    else:
        # Try current working directory
        load_dotenv()
        logger.debug("Loaded environment from default locations")
except ImportError:
    logger.debug("python-dotenv not available, skipping .env loading")
except Exception as e:
    logger.debug(f"Failed to load .env file: {e}")


# SABER MCP Header Constants
HEADER_SESSION_ID = "X-SABER-Session-ID"
HEADER_EPISODE_ID = "X-SABER-Episode-ID"
HEADER_TASK_ID = "X-SABER-Task-ID"
HEADER_ORCHESTRATION_ENV = "X-SABER-Orchestration-Env"


class SandboxError(Exception):
    """Exception raised for SABER sandbox lifecycle errors."""

    pass


@sandboxenv("saber")
class SABERSandboxEnvironment:
    """SABER sandbox environment with lifecycle management and single ownership semantics.

    This class implements the Inspect AI sandbox lifecycle hooks to integrate with
    SABER's domain orchestrator. Key design points:

    - Class-level registry enforces single ownership per domain (one task owns a domain at a time)
    - Per-task session shared across all samples (created in task_init, terminated in task_cleanup)
    - Per-sample episodes for individual execution (created in sample_init, ended in sample_cleanup)
    - Delegates domain lifecycle to DomainController (Phase 2)
    - Constructs MCP client for tool execution via SABER's REST/MCP APIs
    - Raises NotImplementedError for exec/read_file/write_file (users should use saber_tools())

    Lifecycle flow:
    1. task_init: Start domain once per task, create shared session, acquire ownership
    2. sample_init: Create episode (reusing session), construct MCP client with headers
    3. sample_cleanup: End episode, reset per-sample state
    4. task_cleanup: Terminate session, stop domain if owned and cleanup flag set
    """

    # Class-level registry: domain_slug -> metadata
    _registry: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()
    _episode_mapping_lock = threading.Lock()  # Protects episode_mapping read-modify-write

    # Class-level semaphore for episode concurrency control
    # Limits how many episodes can be created/run simultaneously across all domains
    _episode_semaphore: Optional[asyncio.Semaphore] = None
    _max_concurrent_episodes: int = 8  # Default: 8 concurrent episodes

    def __init__(
        self,
        domain_slug: str,
        domains_root: Path,
        rest_port: int = 8000,
        mcp_port: int = 8001,
        compose_template_path: Optional[Path] = None,
        mcp_timeout: float = 300.0,
        rest_base_url: Optional[str] = None,
        mcp_url: Optional[str] = None,
        max_concurrent_episodes: Optional[int] = None,
        enable_debug_logging: bool = False,
    ):
        """Initialize SABER sandbox instance.

        Args:
            domain_slug: SABER domain slug to use
            domains_root: Path to SABER domains directory
            rest_port: REST API port (default: 8000)
            mcp_port: MCP API port (default: 8001)
            compose_template_path: Optional custom compose template
            mcp_timeout: Timeout for MCP operations (default: 300s)
            rest_base_url: Override REST base URL (default: http://localhost:{rest_port})
            mcp_url: Override MCP URL (default: http://localhost:{mcp_port})
            max_concurrent_episodes: Max concurrent episodes (default: 8, None = unlimited)
            enable_debug_logging: Enable detailed episode lifecycle debug logging (default: False)
        """
        self._domain_slug = domain_slug
        self._domains_root = Path(domains_root)
        self._rest_port = rest_port
        self._mcp_port = mcp_port
        self._compose_template_path = compose_template_path
        self._mcp_timeout = mcp_timeout
        self._enable_debug_logging = enable_debug_logging

        # Set max concurrent episodes (instance can override class default)
        if max_concurrent_episodes is not None:
            type(self)._max_concurrent_episodes = max_concurrent_episodes

        # Construct URLs
        self._rest_base_url = rest_base_url or f"http://localhost:{rest_port}"
        self._mcp_url = mcp_url or f"http://localhost:{mcp_port}"

        # Per-instance state (managed during sample lifecycle)
        self._episode_id: Optional[str] = None
        self._task_id: Optional[str] = None
        self._sample_id: Optional[str] = None
        self._mcp_client: Optional[Tool] = None

        # Session state (set during task_init, shared across all samples)
        self._session_id: Optional[str] = None

        # Domain context (set during task_init)
        self._context: Optional[DomainContext] = None

        # Session manager for API interactions (created lazily when needed)
        self._session_manager: Optional[ClientSessionManager] = None

    @classmethod
    def default_concurrency(cls) -> Optional[int]:
        """Default max_sandboxes for SABER provider.

        Returns the configured max_concurrent_episodes to control how many
        samples (episodes) can run concurrently. This tells Inspect AI to
        limit concurrent sample execution to match SABER's episode semaphore.

        Returns None for unlimited concurrency (if max_concurrent_episodes <= 0).
        """
        if cls._max_concurrent_episodes is None or cls._max_concurrent_episodes <= 0:
            return None
        return cls._max_concurrent_episodes

    @classmethod
    def config_files(cls) -> list[str]:
        """Return list of config files for SABER sandbox.

        SABER doesn't use file-based config (all config is passed programmatically),
        so return empty list.
        """
        return []

    @classmethod
    def config_deserialize(cls, config: dict[str, Any]) -> BaseModel:
        """Deserialize SABER sandbox config from dict.

        Since SABER passes config as constructor kwargs (not a BaseModel),
        we create a simple BaseModel wrapper to satisfy the interface.
        """
        # Create a dynamic model with the config fields (frozen=True for hashability)
        SABERConfig = create_model(
            "SABERConfig",
            domain_slug=(str, ...),
            domains_root=(Path, ...),
            rest_port=(int, 8000),
            mcp_port=(int, 8001),
            compose_template_path=(Optional[Path], None),
            cleanup=(bool, False),  # Default to False - keep server running
            max_concurrent_episodes=(Optional[int], None),  # Limit concurrent episodes
            __config__=ConfigDict(frozen=True),
        )

        return SABERConfig(**config)

    @classmethod
    def clear_stale_ownership(cls, domain_slug: str, force: bool = False) -> bool:
        """Clear stale ownership for a domain (useful for debugging/recovery).

        This method allows manual cleanup of domain ownership in cases where:
        - A previous eval was interrupted and didn't clean up properly
        - eval-retry is failing due to ownership conflicts
        - Manual intervention is needed for development/debugging

        Args:
            domain_slug: The domain to clear ownership for
            force: If True, clear ownership even if entry looks valid

        Returns:
            True if ownership was cleared, False if domain wasn't in registry
        """
        with cls._lock:
            if domain_slug not in cls._registry:
                return False

            entry = cls._registry[domain_slug]
            logger.warning(
                f"Clearing ownership for domain '{domain_slug}' (owner: {entry.get('owner')})",
                extra={
                    "domain": domain_slug,
                    "owner": entry.get("owner"),
                    "force": force,
                    "reason": "manual_cleanup",
                },
            )
            del cls._registry[domain_slug]
            return True

    @classmethod
    def _get_episode_semaphore(cls) -> Optional[asyncio.Semaphore]:
        """Get or create the class-level episode concurrency semaphore.

        Returns:
            Semaphore limiting concurrent episodes, or None if unlimited
        """
        if cls._max_concurrent_episodes is None or cls._max_concurrent_episodes <= 0:
            return None

        if cls._episode_semaphore is None:
            cls._episode_semaphore = asyncio.Semaphore(cls._max_concurrent_episodes)
            logger.info(
                f"Created episode concurrency semaphore (limit: {cls._max_concurrent_episodes})",
                extra={"max_concurrent_episodes": cls._max_concurrent_episodes},
            )

        return cls._episode_semaphore

    @classmethod
    async def task_init_environment(
        cls,
        config: Optional[BaseModel],
        metadata: dict[str, str],
    ) -> dict[str, str]:
        """Return environment variables for task initialization.

        SABER doesn't require per-sample environment variables for task_init,
        so we return an empty dict (single task_init call for all samples).

        Args:
            config: SABER sandbox configuration (unused)
            metadata: Sample metadata (unused)

        Returns:
            Empty dict (no per-sample environment variables needed)
        """
        return {}

    @classmethod
    async def task_init(
        cls,
        task_name: str,
        config: Optional[BaseModel],
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
        # Extract config fields
        if config is None:
            raise SandboxError("SABER sandbox config is required for task_init")

        domain_slug = config.domain_slug  # type: ignore[attr-defined]
        domains_root = config.domains_root  # type: ignore[attr-defined]
        rest_port = getattr(config, "rest_port", 8000)
        mcp_port = getattr(config, "mcp_port", 8001)

        # Enable debug logging if requested
        enable_debug = getattr(config, "enable_debug_logging", False)
        if enable_debug:
            enable_debug_logging()
            logger.info("Episode lifecycle debug logging enabled")

        # Set max_concurrent_episodes from config EARLY (before Inspect AI queries default_concurrency)
        max_concurrent_episodes = getattr(config, "max_concurrent_episodes", None)
        if max_concurrent_episodes is not None:
            cls._max_concurrent_episodes = max_concurrent_episodes
            logger.info(
                f"Set max_concurrent_episodes to {max_concurrent_episodes} from task config",
                extra={"max_concurrent_episodes": max_concurrent_episodes},
            )

        with cls._lock:
            # Check if domain already in sandbox registry
            if domain_slug in cls._registry:
                owner = cls._registry[domain_slug]["owner"]

                # Handle eval-retry scenario: same task re-initializing
                if owner == task_name:
                    logger.info(
                        f"Domain '{domain_slug}' already initialized by task '{task_name}' "
                        "(likely eval-retry scenario). Reusing existing session.",
                        extra={
                            "domain": domain_slug,
                            "task": task_name,
                            "mode": "eval_retry_reuse",
                        },
                    )
                    # Session already exists, just return
                    return
                else:
                    raise SandboxError(
                        f"SABER sandbox for domain '{domain_slug}' already owned by task '{owner}'. "
                        "Only one task can own a domain at a time. Ensure previous task cleanup completed."
                    )

            # Check if domain started by factory (ownership transfer mode)
            active_domain = get_active_domain(domain_slug)

            if active_domain:
                # Ownership transfer: reuse running server
                logger.info(
                    f"Domain '{domain_slug}' already started by factory, transferring ownership to sandbox",
                    extra={
                        "domain": domain_slug,
                        "task": task_name,
                        "mode": "ownership_transfer",
                    },
                )

                # Register in sandbox with transferred ownership
                cls._registry[domain_slug] = {
                    "owner": task_name,
                    "domain_slug": domain_slug,
                    "controller": active_domain["controller"],
                    "context": active_domain["context"],
                    "ownership": True,  # Sandbox now owns cleanup
                    "rest_port": active_domain["rest_port"],
                    "mcp_port": active_domain["mcp_port"],
                    "rest_url": active_domain["rest_url"],
                    "mcp_url": active_domain["mcp_url"],
                }

        # Create session outside lock (for both ownership transfer and fresh start)
        with cls._lock:
            entry = cls._registry.get(domain_slug)
            if entry:
                rest_url = entry["rest_url"]
            else:
                rest_url = None

        if rest_url:
            # Create session for this task (ownership transfer case)
            session_id = await cls._create_session_for_task(rest_url, task_name)

            # Store session_id in registry
            with cls._lock:
                cls._registry[domain_slug]["session_id"] = session_id

            logger.info(
                f"Created SABER session for task '{task_name}' (ownership transfer)",
                extra={
                    "domain": domain_slug,
                    "task": task_name,
                    "session_id": session_id,
                },
            )
            return

        # Fresh start case continues below
        # Create controller (CLI-based, no compose_template_path needed)
        controller = DomainController(Path(domains_root))

        # Start domain outside the lock to avoid blocking
        try:
            logger.info(
                f"Starting SABER domain '{domain_slug}' for task '{task_name}'",
                extra={"domain": domain_slug, "task": task_name, "rest_port": rest_port, "mcp_port": mcp_port},
            )

            # Start domain asynchronously
            context = await controller.start(
                domain=domain_slug,
                rest_port=rest_port,
                mcp_port=mcp_port,
                log_level="INFO",  # Use default log level
            )

            # Register ownership
            with cls._lock:
                entry = {
                    "owner": task_name,
                    "domain_slug": domain_slug,
                    "controller": controller,
                    "context": context,
                    "ownership": True,
                    "rest_port": rest_port,
                    "mcp_port": mcp_port,
                    "rest_url": context.rest_url,
                    "mcp_url": context.mcp_url,
                }
                cls._registry[domain_slug] = entry

            logger.info(
                f"SABER domain '{domain_slug}' started and registered for task '{task_name}'",
                extra={"domain": domain_slug, "task": task_name},
            )

            # Create session for this task (shared across all samples)
            session_id = await cls._create_session_for_task(context.rest_url, task_name)

            # Store session_id in registry
            with cls._lock:
                cls._registry[domain_slug]["session_id"] = session_id

            logger.info(
                f"Created SABER session for task '{task_name}'",
                extra={
                    "domain": domain_slug,
                    "task": task_name,
                    "session_id": session_id,
                },
            )

        except Exception as e:
            # Cleanup on failure
            with cls._lock:
                if domain_slug in cls._registry:
                    del cls._registry[domain_slug]

            # Best-effort domain stop
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
        config: Optional[BaseModel],
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
            ValueError: If metadata missing 'task_id' or config is None
            SandboxError: If session/episode creation fails
        """
        if config is None:
            raise ValueError("SABER sandbox config is required for sample_init")

        if "task_id" not in metadata:
            raise ValueError(
                "Missing 'task_id' in metadata. Please ensure your dataset provides a 'task_id'. "
                "The task_id is required to start a SABER episode for this sample."
            )

        # Create instance
        instance = cls(
            domain_slug=config.domain_slug,  # type: ignore[attr-defined]
            domains_root=config.domains_root,  # type: ignore[attr-defined]
            rest_port=getattr(config, "rest_port", 8000),
            mcp_port=getattr(config, "mcp_port", 8001),
            compose_template_path=getattr(config, "compose_template_path", None),
            max_concurrent_episodes=getattr(config, "max_concurrent_episodes", None),
        )

        # Check if this is a completed sample from eval-retry
        # (inspect_ai's eval_retry will skip these samples, but we still need to provide an instance)
        if "saber_session_id" in metadata and "saber_episode_id" in metadata:
            logger.info(
                "Sample already has SABER IDs (eval-retry completed sample), creating minimal instance",
                extra={
                    "task_id": metadata["task_id"],
                    "session_id": metadata["saber_session_id"],
                    "episode_id": metadata["saber_episode_id"],
                    "mode": "eval_retry_completed_sample",
                },
            )
            # Set IDs from metadata but don't create new episode
            instance._session_id = metadata["saber_session_id"]
            instance._episode_id = metadata["saber_episode_id"]
            instance._task_id = metadata["task_id"]
            # Don't create MCP client - sample won't be executed
            return {"default": instance}

        # Normal case: Initialize session and episode for new sample
        await instance._init_sample(metadata)

        return {"default": instance}

    async def _init_sample(self, metadata: dict[str, str]) -> None:
        """Internal method to initialize episode for sample.

        Args:
            metadata: Sample metadata dict with 'task_id'
        """
        init_start_time = time.time()
        self._task_id = metadata["task_id"]
        self._sample_id = metadata.get("sample_id", self._task_id)  # Store sample_id for cleanup

        # Get registry entry to access URLs and session_id
        with self._lock:
            entry = self._registry.get(self._domain_slug)
            if not entry:
                raise SandboxError(
                    f"Domain '{self._domain_slug}' not initialized. " "Call task_init before sample_init."
                )
            rest_base_url = entry["rest_url"]
            mcp_url_base = entry["mcp_url"]
            self._session_id = entry["session_id"]  # Get shared session from registry

        # Create session manager if not already created
        if self._session_manager is None:
            config = SessionManagerConfig(
                base_url=rest_base_url,
                mcp_server_url=mcp_url_base,
                client_id="inspect_ai_sandbox",
                rest_timeout=180.0,  # Increased to 3 minutes for episode creation with queueing
            )
            self._session_manager = ClientSessionManager(config=config)
            self._session_manager._current_session_id = self._session_id

        # Track semaphore outside try block so exception handler can release it
        semaphore_acquired = False
        semaphore = self._get_episode_semaphore()

        try:
            # Acquire semaphore to limit concurrent episode creation
            # This controls how many episodes can run simultaneously
            if semaphore is not None:
                logger.debug(
                    f"Acquiring episode semaphore for task {self._task_id} (available: {semaphore._value})",
                    extra={
                        "task_id": self._task_id,
                        "max_concurrent": self._max_concurrent_episodes,
                        "semaphore_available": semaphore._value,
                    },
                )

                # Use asyncio.wait_for to add timeout to semaphore acquisition
                # This prevents indefinite waiting if all slots are leaked
                try:
                    await asyncio.wait_for(semaphore.acquire(), timeout=300.0)  # 5 minute timeout
                    semaphore_acquired = True
                    logger.debug(
                        f"Episode semaphore acquired for task {self._task_id} (remaining: {semaphore._value})",
                        extra={
                            "task_id": self._task_id,
                            "semaphore_remaining": semaphore._value,
                        },
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"Timeout waiting for episode semaphore for task {self._task_id} (waited 300s)",
                        extra={
                            "task_id": self._task_id,
                            "max_concurrent": self._max_concurrent_episodes,
                            "event": "semaphore_acquisition_timeout",
                        },
                    )
                    raise PrerequisiteError(
                        f"Timeout waiting for episode semaphore (waited 300s). "
                        f"This may indicate semaphore slot leakage or insufficient capacity. "
                        f"Max concurrent episodes: {self._max_concurrent_episodes}"
                    )

            # Create episode via REST API (using shared session manager)
            self._episode_id = await self._create_episode(self._session_id, self._task_id)
            logger.debug(
                "Created SABER episode for sample",
                extra={
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                    "task_id": self._task_id,
                },
            )

            log_sample_init_start(
                task_id=self._task_id,
                sample_id=self._sample_id,
                episode_id=self._episode_id,
                session_id=self._session_id,
                domain_slug=self._domain_slug,
            )

            # Store session and episode IDs in sample metadata for eval-retry support
            # This allows completed samples to be identified and skipped during retries
            # Note: metadata is passed by reference, so these updates persist to the Sample
            metadata["saber_session_id"] = self._session_id
            metadata["saber_episode_id"] = self._episode_id
            metadata["saber_domain_slug"] = self._domain_slug

            # Log metadata configuration for debugging episode lifecycle issues
            debug_logger = EpisodeDebugLogger("metadata")
            debug_logger.info(
                "Sample metadata configured with SABER IDs",
                sample_id=metadata.get("sample_id"),
                saber_episode_id_set=metadata.get("saber_episode_id"),
                saber_session_id_set=metadata.get("saber_session_id"),
                episode_id_instance_var=self._episode_id,
                session_id_instance_var=self._session_id,
            )

            # Store session manager and context in inspect_ai store for scorer
            # The scorer needs this to fetch evaluation data from the server
            task_store = store()
            task_store.set("saber_session_manager", self._session_manager)
            task_store.set("saber_session_id", self._session_id)
            task_store.set("saber_task_id", self._task_id)

            # Store episode mapping using centralized helper (prevents race conditions)
            sample_id = metadata.get("sample_id", self._task_id)
            self._store_episode_mapping(
                sample_id=sample_id,
                episode_id=self._episode_id,
                session_id=self._session_id,
                task_id=self._task_id,
            )

            logger.debug(
                "Stored SABER context in inspect_ai store for scorer",
                extra={
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                    "task_id": self._task_id,
                    "sample_id": sample_id,
                    "event": "saber_context_stored",
                },
            )

            # Construct MCP client with headers and retry logic
            mcp_headers = {
                HEADER_SESSION_ID: self._session_id,
                HEADER_EPISODE_ID: self._episode_id,
                HEADER_TASK_ID: self._task_id,
                HEADER_ORCHESTRATION_ENV: OrchestrationEnvironment.INSPECT.value,
            }

            # Log MCP headers for debugging episode lifecycle issues
            debug_logger = EpisodeDebugLogger("mcp_client")
            debug_logger.info(
                "MCP client headers configured",
                sample_id=sample_id,
                episode_id_in_headers=mcp_headers[HEADER_EPISODE_ID],
                episode_id_instance_var=self._episode_id,
                task_id=self._task_id,
            )

            mcp_url = f"{mcp_url_base}/mcp"
            logger.debug(
                f"Creating MCP client for URL: {mcp_url}",
                extra={
                    "mcp_url": mcp_url,
                    "headers": mcp_headers,
                },
            )

            # Create MCP client with retry logic for transient failures
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    logger.debug(
                        f"Creating MCP client (attempt {attempt + 1}/{max_retries})",
                        extra={
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "session_id": self._session_id,
                            "episode_id": self._episode_id,
                        },
                    )

                    # CRITICAL: Name must be unique per sample to prevent MCP client caching!
                    # mcp_server_http() caches sessions by (task_id, name), so if multiple
                    # samples use the same name, they share the same MCP session with the
                    # FIRST sample's headers (causing episode ID cross-contamination).
                    # Using sample_id ensures each sample gets its own MCP client.
                    self._mcp_client = mcp_server_http(
                        name=f"SABER {self._domain_slug} Tools - {sample_id}",
                        url=mcp_url,
                        headers=mcp_headers,
                        timeout=self._mcp_timeout,
                    )

                    logger.debug(
                        "MCP client created successfully",
                        extra={
                            "attempt": attempt + 1,
                            "session_id": self._session_id,
                            "episode_id": self._episode_id,
                        },
                    )

                    # Don't enter the context here - let Inspect AI manage it
                    # The MCP server will be entered when Inspect AI calls .tools() on it

                    break  # Success, exit retry loop

                except Exception as e:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s
                        wait_time = 2**attempt
                        logger.warning(
                            f"MCP client creation failed, retrying in {wait_time}s",
                            extra={
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "wait_time": wait_time,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            "MCP client creation failed after all retries",
                            extra={
                                "attempts": max_retries,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )
                        raise Exception(
                            f"Failed to create MCP client after {max_retries} attempts. "
                            f"Last error: {type(e).__name__}: {str(e)}"
                        ) from e

            logger.info(
                "SABER sandbox ready for sample execution",
                extra={
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                    "task_id": self._task_id,
                },
            )

            # Log completion of sample initialization
            init_duration = time.time() - init_start_time
            log_sample_init_complete(
                duration=init_duration,
                sample_id=self._sample_id,
                mcp_url=mcp_url,
            )

        except Exception as e:
            # CRITICAL: Release semaphore BEFORE cleanup to prevent permanent slot leakage
            # This fixes the bug where failed episode creation would leak semaphore slots,
            # causing capacity to degrade from 8 -> 7 -> 6 -> 5 over time during long test runs
            if semaphore_acquired and semaphore is not None:
                semaphore.release()
                logger.debug(
                    f"Released episode semaphore after init failure for task {self._task_id} "
                    f"(available: {semaphore._value})",
                    extra={
                        "task_id": self._task_id,
                        "semaphore_available": semaphore._value,
                        "event": "semaphore_released_on_init_failure",
                    },
                )

            # Cleanup partial state (initialization failure counts as interrupted)
            await self._cleanup_partial_state(interrupted=True)
            raise PrerequisiteError(
                f"Failed to initialize SABER sandbox for sample (task_id={self._task_id}): {e}"
            ) from e

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: Optional[BaseModel],
        environments: dict[str, "SABERSandboxEnvironment"],
        interrupted: bool,
    ) -> None:
        """Cleanup sandbox environments after sample execution.

        Ends episode and terminates session for each environment.
        Best-effort cleanup - logs errors but doesn't raise.

        Args:
            task_name: Name of task (unused, for compatibility)
            config: SABER sandbox configuration (unused)
            environments: Dict of SABERSandboxEnvironment instances to cleanup
            interrupted: Whether the sample was interrupted
        """
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
        """Internal method to cleanup sample state.

        Ends episode and terminates session.

        Args:
            interrupted: Whether the sample was interrupted (Ctrl+C, error, etc.)
                If True, end the episode with reason="interrupted"
                If False, assume scorer already ended the episode (happy path)
        """
        # Shield cleanup from cancellation - we need this to complete even during shutdown
        with anyio.CancelScope(shield=True):
            try:
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
                # Log lifecycle summary before cleanup
                log_lifecycle_summary()

                # Clean up episode mapping using centralized helper (prevents race conditions)
                if hasattr(self, "_sample_id") and self._sample_id:
                    try:
                        self._remove_episode_mapping(
                            sample_id=self._sample_id,
                            task_id=self._task_id or "unknown",
                        )
                    except Exception as cleanup_err:
                        logger.warning(
                            f"Failed to clean up episode mapping for sample {self._sample_id}: {cleanup_err}",
                            extra={"sample_id": self._sample_id, "task_id": self._task_id},
                        )

                # MEMORY LEAK FIX: Clean up MCP client session cache to prevent accumulation
                # With unique server names, each sample creates a cache entry that never gets cleaned.
                # For 10,000+ samples, this would cause significant memory leaks.
                if hasattr(self, "_mcp_client") and self._mcp_client is not None:
                    try:
                        # Build the cache key that would have been used for this sample
                        task_id = anyio.get_current_task().id
                        server_name = f"SABER {self._domain_slug} Tools - {self._sample_id}"
                        cache_key = f"{task_id}_{server_name}"

                        # Remove from cache if it exists
                        if cache_key in MCPServerLocal._task_sessions:
                            cached_session = MCPServerLocal._task_sessions.pop(cache_key)

                            # Properly close the cached session if it's active
                            if hasattr(cached_session, "_session") and cached_session._session is not None:
                                await cached_session.__aexit__(None, None, None)

                            logger.debug(
                                f"Cleaned up MCP session cache for sample {self._sample_id}",
                                extra={
                                    "cache_key": cache_key,
                                    "remaining_cache_entries": len(MCPServerLocal._task_sessions),
                                    "event": "mcp_cache_cleanup",
                                },
                            )
                    except Exception as cache_cleanup_err:
                        logger.warning(
                            f"Failed to clean up MCP session cache for sample {self._sample_id}: {cache_cleanup_err}",
                            extra={"sample_id": self._sample_id, "task_id": self._task_id},
                        )

                # Release episode semaphore to allow next sample to run
                semaphore = self._get_episode_semaphore()
                if semaphore is not None:
                    semaphore.release()
                    logger.debug(
                        f"Released episode semaphore for task {self._task_id} (available: {semaphore._value})",
                        extra={
                            "task_id": self._task_id,
                            "semaphore_available": semaphore._value,
                            "event": "semaphore_released_on_cleanup",
                        },
                    )

                self._reset_state()

                # Clear episode context after cleanup
                clear_episode_context()

    @classmethod
    async def task_cleanup(
        cls,
        task_name: str,
        config: Optional[BaseModel],
        cleanup: bool = True,
    ) -> None:
        """Cleanup SABER domain after task completion.

        Stops the domain if current task owns it and cleanup flag is set.
        Handles missing registry entry gracefully (partial failure case).
        Also removes domain from the factory's active domains registry.

        For eval-retry scenarios: If cleanup=False (no interrupt), preserves the domain
        for faster retry execution.

        Args:
            task_name: Name of the task (unused, for compatibility)
            config: SABER sandbox configuration
            cleanup: Whether to stop the domain (from Inspect AI, typically True for interrupts)
        """
        if config is None:
            logger.warning("task_cleanup called with no config, skipping")
            return

        domain_slug = config.domain_slug  # type: ignore[attr-defined]
        # Get cleanup preference from config (stop_saber_after flag)
        # If config has cleanup field, use it; otherwise use the cleanup parameter
        cleanup_requested = getattr(config, "cleanup", cleanup)

        with cls._lock:
            entry = cls._registry.get(domain_slug)
            if not entry:
                logger.info(
                    f"Domain '{domain_slug}' not in registry during task_cleanup. "
                    "May indicate partial failure during task_init or already cleaned up.",
                    extra={"domain": domain_slug, "task": task_name},
                )
                # Still try to remove from active domains
                remove_active_domain(domain_slug)
                return

            controller = entry["controller"]
            ownership = entry["ownership"]
            session_id = entry.get("session_id")
            rest_url = entry.get("rest_url")
            owner = entry.get("owner")

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

            # Don't remove from registry yet if cleanup=False (eval-retry scenario)
            # This allows the domain to be reused on retry
            if not cleanup_requested and not cleanup:
                logger.info(
                    f"Preserving domain '{domain_slug}' ownership for potential eval-retry",
                    extra={
                        "domain": domain_slug,
                        "task": task_name,
                        "session_id": session_id,
                        "note": "Domain will remain in registry for faster retry",
                    },
                )
                # Still terminate the session but keep ownership
                if session_id and rest_url:
                    logger.info(
                        "Terminating SABER session for task cleanup (preserving domain)",
                        extra={
                            "domain": domain_slug,
                            "session_id": session_id,
                        },
                    )
                    cls._terminate_session_sync(rest_url, session_id)
                return

            # Remove from sandbox registry
            del cls._registry[domain_slug]

        # Terminate session (shared across all samples)
        # Use synchronous request with fire-and-forget to avoid event loop cancellation issues during shutdown
        if session_id and rest_url:
            logger.info(
                "Terminating SABER session for task cleanup",
                extra={
                    "domain": domain_slug,
                    "session_id": session_id,
                },
            )
            cls._terminate_session_sync(rest_url, session_id)

        # Remove from factory's active domains registry
        remove_active_domain(domain_slug)

        # Stop domain if owned and cleanup requested
        if ownership and cleanup_requested:
            try:
                logger.info(
                    f"Stopping SABER domain '{domain_slug}' for task cleanup (stop_saber_after=True)",
                    extra={"domain": domain_slug},
                )
                await controller.stop(domain_slug)
                logger.info(f"SABER domain '{domain_slug}' stopped successfully", extra={"domain": domain_slug})
            except Exception as e:
                logger.warning(
                    f"Failed to stop SABER domain '{domain_slug}': {e}", extra={"domain": domain_slug}, exc_info=True
                )
        elif ownership:
            logger.info(
                f"SABER domain '{domain_slug}' kept running (stop_saber_after=False). "
                f"Server will remain active at REST port {entry.get('rest_port', 8000)} "
                f"and MCP port {entry.get('mcp_port', 8001)} for faster re-runs.",
                extra={"domain": domain_slug},
            )

    def _reset_state(self) -> None:
        """Reset per-sample state."""
        self._episode_id = None
        self._mcp_client = None
        self._task_id = None
        self._sample_id = None  # Reset sample_id for next sample
        # Note: _session_id is NOT reset - it's shared across all samples

    @classmethod
    def _store_episode_mapping(
        cls,
        sample_id: str,
        episode_id: str,
        session_id: str,
        task_id: str,
    ) -> None:
        """Store episode mapping in Inspect AI store with thread-safe locking.

        This centralizes all episode mapping storage to prevent race conditions
        from concurrent read-modify-write operations on the shared store.

        Args:
            sample_id: Unique sample identifier (includes attempt suffix)
            episode_id: SABER episode ID
            session_id: SABER session ID
            task_id: Task ID (for debugging)
        """
        task_store = store()
        with cls._episode_mapping_lock:
            episode_mapping = task_store.get("saber_episode_mapping", {})
            episode_mapping[sample_id] = type(
                "Episode",
                (),
                {
                    "episode_id": episode_id,
                    "session_id": session_id,
                    "attached_to_episode_id": None,
                    "task_id": task_id,
                    "sample_id": sample_id,
                },
            )()
            task_store.set("saber_episode_mapping", episode_mapping)

            logger.debug(
                f"Stored episode mapping for sample {sample_id}",
                extra={
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "total_mappings": len(episode_mapping),
                    "event": "episode_mapping_stored",
                },
            )

    @classmethod
    def _remove_episode_mapping(cls, sample_id: str, task_id: str) -> None:
        """Remove episode mapping from Inspect AI store with thread-safe locking.

        This centralizes all episode mapping cleanup to prevent race conditions.

        Args:
            sample_id: Unique sample identifier to remove
            task_id: Task ID (for debugging/logging)
        """
        task_store = store()
        with cls._episode_mapping_lock:
            episode_mapping = task_store.get("saber_episode_mapping", {})
            if sample_id in episode_mapping:
                del episode_mapping[sample_id]
                task_store.set("saber_episode_mapping", episode_mapping)
                logger.debug(
                    f"Removed episode mapping for sample {sample_id}",
                    extra={
                        "sample_id": sample_id,
                        "task_id": task_id,
                        "remaining_mappings": len(episode_mapping),
                        "event": "episode_mapping_removed",
                    },
                )

    @classmethod
    async def _create_session_for_task(cls, rest_base_url: str, task_name: str) -> str:
        """Create SABER session for a task (shared across all samples).

        Args:
            rest_base_url: Base URL for REST API
            task_name: Name of the task

        Returns:
            Session ID
        """
        url = f"{rest_base_url}/api/v1/session"
        params = {"client_id": f"inspect_ai_{task_name}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    text = await response.text()
                    raise PrerequisiteError(f"Failed to create SABER session: {response.status} - {text}")
                data = await response.json()
                return str(data["session_id"])

    @classmethod
    def _terminate_session_sync(cls, rest_base_url: str, session_id: str) -> None:
        """Terminate SABER session synchronously with fire-and-forget.

        Uses synchronous requests library with background thread to avoid event loop
        cancellation issues during shutdown. Doesn't wait for response since this is
        called during cleanup.

        Args:
            rest_base_url: Base URL for REST API
            session_id: Session ID to terminate
        """
        url = f"{rest_base_url}/api/v1/session/{session_id}"

        def send_delete_request() -> None:
            """Send DELETE request in background thread."""
            try:
                response = requests.delete(url, timeout=2.0)
                if response.status_code == 200:
                    logger.info(
                        "Session terminated successfully",
                        extra={
                            "session_id": session_id,
                            "event": "session_terminated_sync",
                        },
                    )
                else:
                    logger.debug(
                        f"Session termination returned {response.status_code}",
                        extra={
                            "session_id": session_id,
                            "status_code": response.status_code,
                        },
                    )
            except Exception as e:
                logger.debug(
                    f"Session termination exception (server may still process): {e}",
                    extra={"session_id": session_id},
                )

        # Start in non-daemon thread and wait briefly for completion
        thread = threading.Thread(target=send_delete_request, daemon=False)
        thread.start()
        thread.join(timeout=1.5)  # Wait max 1.5 seconds

        logger.info(
            "Session termination request sent",
            extra={
                "session_id": session_id,
                "event": "session_termination_initiated",
            },
        )

    async def _create_episode(self, session_id: str, task_id: str) -> str:
        """Create SABER episode via REST API and wait for it to be ready.

        Episode creation is asynchronous - the POST returns immediately with an episode ID,
        but Docker container setup can take several minutes. We use the ClientSessionManager's
        create_episode_and_wait method which has proper timeout and polling logic.

        Timeline:
        - Episode POST returns immediately with episode_id
        - Docker compose starts (serialized by global lock)
        - Background finalization: health checks (200s timeout), prompt generation, etc.
        - Client polls for is_ready=True

        Timeout must exceed server-side health check timeout (200s) plus buffer for other
        finalization steps. Using 400s (6.67 minutes) to ensure server can complete or fail.
        """
        if self._session_manager is None:
            raise SandboxError("Session manager not initialized")

        try:
            response = await self._session_manager.create_episode_and_wait(
                session_id=session_id,
                task_id=task_id,
                timeout_seconds=400,  # Increased from 300s to exceed server health check timeout
            )
            episode_id: str = response.episode_id
            return episode_id
        except TimeoutError as e:
            # Provide clear timeout diagnostic
            error_msg = (
                f"Episode creation timed out after 400s for task '{task_id}'. "
                f"This usually indicates Docker health checks are taking too long or containers failed to start. "
                f"Check server logs for details. Error: {str(e)}"
            )
            logger.error(
                error_msg,
                extra={
                    "event": "episode_creation_timeout",
                    "task_id": task_id,
                    "session_id": session_id,
                    "timeout_seconds": 400,
                },
            )
            raise PrerequisiteError(error_msg) from e
        except Exception as e:
            # Provide context for other errors
            error_msg = f"Failed to create episode for task '{task_id}': {type(e).__name__}: {str(e)}"
            logger.error(
                error_msg,
                extra={
                    "event": "episode_creation_failed",
                    "task_id": task_id,
                    "session_id": session_id,
                    "error_type": type(e).__name__,
                },
            )
            raise PrerequisiteError(error_msg) from e

    async def _cleanup_partial_state(self, interrupted: bool = False) -> None:
        """Best-effort cleanup of episode with robust retry logic.

        Unified episode ending for all paths - this is the single place where episodes end.
        Note: Session is NOT terminated here - it's shared across all samples and cleaned up in task_cleanup.

        Args:
            interrupted: Whether this is an interrupted (non-happy path) cleanup
        """
        # End episode if it exists (both happy and interrupted paths)
        if self._episode_id and self._session_manager:
            try:
                # Determine reason based on interrupted flag
                reason = "interrupted" if interrupted else "completed"

                logger.info(
                    f"Ending episode (reason={reason})",
                    extra={
                        "session_id": self._session_id,
                        "episode_id": self._episode_id,
                        "reason": reason,
                        "event": "sandbox_cleanup_end_episode",
                    },
                )

                # Log episode end request
                log_episode_end_request(
                    reason=reason,
                    interrupted=interrupted,
                )

                # For interrupted cleanup, use fire-and-forget HTTP to avoid event loop shutdown issues
                if interrupted:
                    # Use synchronous requests library for reliability during shutdown
                    # NOW WITH RETRY for better reliability
                    base_url = self._session_manager.base_url
                    session_id = self._session_id
                    episode_id = self._episode_id
                    url = f"{base_url}/api/v1/session/{session_id}/episodes/{episode_id}"
                    verify_url = f"{base_url}/api/v1/session/{session_id}/episodes/{episode_id}/status"
                    params = {
                        "reason": reason,
                        "cascade_end_attached_episodes": "false",
                    }

                    def send_delete_request_with_retry() -> bool:
                        """Send delete request with retry logic in background thread."""
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                # Longer timeout than before - 5s instead of 2s
                                response = requests.delete(url, params=params, timeout=5.0)

                                if response.status_code == 200:
                                    # Verify episode actually ended
                                    time.sleep(0.5)  # Brief wait for server processing
                                    verify_response = requests.get(verify_url, timeout=2.0)
                                    if verify_response.status_code == 404:
                                        logger.info(
                                            f"Episode end verified successfully (attempt {attempt + 1}/{max_retries})",
                                            extra={
                                                "session_id": session_id,
                                                "episode_id": episode_id,
                                                "reason": reason,
                                            },
                                        )
                                        return True
                                    else:
                                        logger.debug(
                                            f"Episode ended but verification inconclusive "
                                            f"(status {verify_response.status_code})",
                                            extra={
                                                "session_id": session_id,
                                                "episode_id": episode_id,
                                                "verify_status": verify_response.status_code,
                                            },
                                        )
                                        return True  # Accept success even if verification unclear

                                elif response.status_code == 400:
                                    # Might be already ended
                                    logger.info(
                                        f"Episode already ended (attempt {attempt + 1}/{max_retries})",
                                        extra={
                                            "session_id": session_id,
                                            "episode_id": episode_id,
                                        },
                                    )
                                    return True

                                else:
                                    logger.debug(
                                        f"Episode end request returned {response.status_code} "
                                        f"(attempt {attempt + 1}/{max_retries})",
                                        extra={
                                            "session_id": session_id,
                                            "episode_id": episode_id,
                                            "status_code": response.status_code,
                                            "attempt": attempt + 1,
                                        },
                                    )

                            except Exception as e:
                                logger.debug(
                                    f"Episode end request attempt {attempt + 1}/{max_retries} failed: {e}",
                                    extra={
                                        "session_id": session_id,
                                        "episode_id": episode_id,
                                        "attempt": attempt + 1,
                                        "error": str(e),
                                    },
                                )

                            # Retry with backoff
                            if attempt < max_retries - 1:
                                backoff = 1.0 * (attempt + 1)
                                time.sleep(backoff)

                        # All retries exhausted
                        logger.warning(
                            f"Episode end failed after {max_retries} attempts (interrupted cleanup)",
                            extra={
                                "session_id": session_id,
                                "episode_id": episode_id,
                            },
                        )
                        return False

                    # Start in non-daemon thread and wait for completion
                    # Non-daemon prevents process exit until thread completes
                    # Increased timeout to allow for retries: 3 attempts * ~6s each = 18s max
                    thread = threading.Thread(target=send_delete_request_with_retry, daemon=False)
                    thread.start()
                    thread.join(timeout=20.0)  # Wait up to 20s for retries to complete

                    logger.info(
                        f"Episode end request initiated with retry in background (reason={reason})",
                        extra={
                            "session_id": self._session_id,
                            "episode_id": self._episode_id,
                            "reason": reason,
                            "note": "Used retry logic with up to 3 attempts",
                        },
                    )
                else:
                    # Happy path - use async session manager WITH RETRY
                    # Get submission from store if available (happy path after scorer)
                    submission = None
                    try:
                        task_store = store()
                        # Scorer may have stored the submission for us
                        submission = task_store.get("saber_episode_submission")
                    except Exception:
                        # Store might not be available or submission not set - that's ok
                        pass

                    # End episode with submission - USING ROBUST RETRY
                    success = await self._session_manager.end_episode_with_retry(
                        session_id=self._session_id,
                        episode_id=self._episode_id,
                        reason=reason,
                        result=submission,
                        cascade_end_attached_episodes=False,
                        max_retries=5,  # More retries for happy path
                        initial_backoff=1.0,
                        max_backoff=30.0,
                    )

                    if success:
                        logger.info(
                            f"Episode ended successfully with verification (reason={reason})",
                            extra={
                                "session_id": self._session_id,
                                "episode_id": self._episode_id,
                                "reason": reason,
                            },
                        )

                        # Log successful episode end
                        log_episode_end_complete(success=True)
                    else:
                        logger.error(
                            f"Episode ending failed after retries - may be orphaned (reason={reason})",
                            extra={
                                "session_id": self._session_id,
                                "episode_id": self._episode_id,
                                "reason": reason,
                                "event": "episode_end_failed_with_retry",
                            },
                        )

                        # Log failed episode end
                        log_episode_end_complete(success=False)

            except (asyncio.CancelledError, asyncio.TimeoutError):
                # Cancellation/timeout during shutdown - log but don't raise
                # The server will clean up the episode eventually
                reason_str = reason if "reason" in locals() else "unknown"
                logger.info(
                    f"Episode ending cancelled/timed out during shutdown (reason={reason_str})",
                    extra={
                        "session_id": self._session_id,
                        "episode_id": self._episode_id,
                        "event": "episode_end_cancelled",
                        "note": "Server will clean up episode eventually",
                    },
                )
            except Exception as e:
                logger.warning(
                    f"Failed to end episode with retry (server will eventually clean up): {e}",
                    extra={
                        "session_id": self._session_id,
                        "episode_id": self._episode_id,
                        "reason": reason if "reason" in locals() else "unknown",
                    },
                    exc_info=True,
                )

    async def connection(self, *, user: Optional[str] = None) -> Any:
        """Get connection information for SABER sandbox.

        SABER doesn't provide SSH/shell connections - all interaction happens
        through the MCP tool interface.

        Args:
            user: User to connect as (ignored for SABER)

        Raises:
            NotImplementedError: SABER uses MCP tools, not direct connections
        """
        raise NotImplementedError(
            "SABERSandboxEnvironment.connection() is not supported. "
            "SABER provides tools via MCP protocol, not direct shell connections. "
            "Use the tools() method to access SABER capabilities."
        )

    # Unsupported operations - users should use saber_tools()

    async def exec(self, *args: Any, **kwargs: Any) -> Any:
        """Not supported for SABER sandbox."""
        raise NotImplementedError(
            "SABERSandboxEnvironment.exec() is not supported. "
            "Use saber_tools() to access SABER tool execution via MCP. "
            "Tools are available through the MCP client created during sample_init."
        )

    async def read_file(self, *args: Any, **kwargs: Any) -> Any:
        """Not supported for SABER sandbox."""
        raise NotImplementedError(
            "SABERSandboxEnvironment.read_file() is not supported. "
            "Use saber_tools() for file access via MCP tools. "
            "SABER provides file operations through its MCP tool interface."
        )

    async def write_file(self, *args: Any, **kwargs: Any) -> Any:
        """Not supported for SABER sandbox."""
        raise NotImplementedError(
            "SABERSandboxEnvironment.write_file() is not supported. "
            "Use saber_tools() for file writing via MCP tools. "
            "SABER provides file operations through its MCP tool interface."
        )
