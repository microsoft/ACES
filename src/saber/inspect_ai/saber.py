"""SABER Sandbox Environment for Inspect AI.

Implements the SandboxEnvironment interface by delegating to SABER's DomainOrchestrator
and managing per-sample REST sessions and MCP clients for tool execution.

This module realizes Phase 3 of the SABER sandbox integration, providing lifecycle
management (task_init, sample_init, sample_cleanup, task_cleanup) with class-level
domain ownership semantics.
"""

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from inspect_ai._util.error import PrerequisiteError
from inspect_ai.tool import Tool, mcp_server_http
from inspect_ai.util import sandboxenv, store
from pydantic import BaseModel

from saber.client.client_session import ClientSessionManager
from saber.logging_config import LogCategory, get_saber_logger
from saber.models.mcp import OrchestrationEnvironment

from .server import DomainContext, DomainController

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
        """
        self._domain_slug = domain_slug
        self._domains_root = Path(domains_root)
        self._rest_port = rest_port
        self._mcp_port = mcp_port
        self._compose_template_path = compose_template_path
        self._mcp_timeout = mcp_timeout

        # Construct URLs
        self._rest_base_url = rest_base_url or f"http://localhost:{rest_port}"
        self._mcp_url = mcp_url or f"http://localhost:{mcp_port}"

        # Per-instance state (managed during sample lifecycle)
        self._episode_id: Optional[str] = None
        self._task_id: Optional[str] = None
        self._mcp_client: Optional[Tool] = None

        # Session state (set during task_init, shared across all samples)
        self._session_id: Optional[str] = None

        # Domain context (set during task_init)
        self._context: Optional[DomainContext] = None

    @classmethod
    def default_concurrency(cls) -> Optional[int]:
        """Default max_sandboxes for SABER provider.

        Returns None to indicate no maximum concurrency limit.
        SABER handles resource management internally via Docker.
        """
        return None

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
        from pydantic import ConfigDict, create_model

        # Create a dynamic model with the config fields (frozen=True for hashability)
        SABERConfig = create_model(
            "SABERConfig",
            domain_slug=(str, ...),
            domains_root=(Path, ...),
            rest_port=(int, 8000),
            mcp_port=(int, 8001),
            compose_template_path=(Optional[Path], None),
            cleanup=(bool, False),  # Default to False - keep server running
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
        # Import here to avoid circular dependency
        from .tasks import get_active_domain

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
        self._task_id = metadata["task_id"]

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

        try:
            # Create episode via REST API (using shared session)
            self._episode_id = await self._create_episode(rest_base_url, self._session_id, self._task_id)
            logger.debug(
                "Created SABER episode for sample",
                extra={
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                    "task_id": self._task_id,
                },
            )

            # Store session and episode IDs in sample metadata for eval-retry support
            # This allows completed samples to be identified and skipped during retries
            # Note: metadata is passed by reference, so these updates persist to the Sample
            metadata["saber_session_id"] = self._session_id
            metadata["saber_episode_id"] = self._episode_id
            metadata["saber_domain_slug"] = self._domain_slug

            # Create session manager for scorer to use
            # The scorer needs this to fetch evaluation data from the server
            from saber.client.models import SessionManagerConfig

            session_manager_config = SessionManagerConfig(
                base_url=rest_base_url,
                mcp_server_url=mcp_url_base,
                client_id="inspect_ai_sandbox",
                rest_timeout=30.0,
            )
            session_manager = ClientSessionManager(config=session_manager_config)
            session_manager._current_session_id = self._session_id  # Set the session ID

            # Store session manager and context in inspect_ai store for scorer
            task_store = store()
            task_store.set("saber_session_manager", session_manager)
            task_store.set("saber_session_id", self._session_id)
            task_store.set("saber_task_id", self._task_id)
            # Store episode info that will be updated by agent/solver
            task_store.set(
                "saber_current_episode",
                type(
                    "Episode",
                    (),
                    {
                        "episode_id": self._episode_id,
                        "session_id": self._session_id,
                        "attached_to_episode_id": None,
                    },
                )(),
            )

            logger.debug(
                "Stored SABER context in inspect_ai store for scorer",
                extra={
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
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

                    self._mcp_client = mcp_server_http(
                        name=f"SABER {self._domain_slug} Tools",
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

        except Exception as e:
            # Cleanup partial state (initialization failure counts as interrupted)
            await self._cleanup_partial_state(rest_base_url, interrupted=True)
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
        for name, env in environments.items():
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
        import anyio

        # Shield cleanup from cancellation - we need this to complete even during shutdown
        with anyio.CancelScope(shield=True):
            rest_base_url = None
            try:
                # Get REST URL from registry
                with self._lock:
                    entry = self._registry.get(self._domain_slug)
                    if entry:
                        rest_base_url = entry["rest_url"]

                if rest_base_url:
                    await self._cleanup_partial_state(rest_base_url, interrupted=interrupted)
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
                self._reset_state()

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

        # Import here to avoid circular dependency
        from .tasks import remove_active_domain

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
        # Note: _session_id is NOT reset - it's shared across all samples

    @classmethod
    async def _create_session_for_task(cls, rest_base_url: str, task_name: str) -> str:
        """Create SABER session for a task (shared across all samples).

        Args:
            rest_base_url: Base URL for REST API
            task_name: Name of the task

        Returns:
            Session ID
        """
        import aiohttp

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
        import threading

        import requests  # type: ignore[import-untyped]

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

    @classmethod
    async def _terminate_session(cls, rest_base_url: str, session_id: str) -> None:
        """Terminate SABER session (async version - unused, kept for compatibility).

        Args:
            rest_base_url: Base URL for REST API
            session_id: Session ID to terminate
        """
        import aiohttp

        url = f"{rest_base_url}/api/v1/session/{session_id}"

        async with aiohttp.ClientSession() as session:
            async with session.delete(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Failed to terminate SABER session: {response.status} - {text}")

    async def _create_episode(self, rest_base_url: str, session_id: str, task_id: str) -> str:
        """Create SABER episode via REST API and wait for it to be ready."""
        import aiohttp

        # Create episode
        url = f"{rest_base_url}/api/v1/session/{session_id}/episodes"
        params = {"task_id": task_id}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    text = await response.text()
                    raise PrerequisiteError(f"Failed to create SABER episode: {response.status} - {text}")
                data = await response.json()
                episode_id = data["episode_id"]

        # Wait for episode to be ready (poll status endpoint)
        max_wait = 30  # seconds
        poll_interval = 0.5  # seconds
        elapsed = 0.0

        logger.debug(f"Waiting for episode {episode_id} to become ready...")

        async with aiohttp.ClientSession() as session:
            while elapsed < max_wait:
                status_url = f"{rest_base_url}/api/v1/session/{session_id}/episodes/{episode_id}/status"
                try:
                    async with session.get(status_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            status_data = await response.json()
                            state = status_data.get("state", "unknown")

                            if state in ("ready", "active"):
                                logger.debug(f"Episode {episode_id} is {state}")
                                return str(episode_id)
                            elif state == "error":
                                raise PrerequisiteError(f"Episode {episode_id} entered error state")
                            # Otherwise keep polling (pending, initializing, etc.)
                except Exception as e:
                    logger.debug(f"Status check failed (will retry): {e}")

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        raise PrerequisiteError(f"Episode {episode_id} did not become ready within {max_wait}s")

    async def _cleanup_partial_state(self, rest_base_url: str, interrupted: bool = False) -> None:
        """Best-effort cleanup of episode.

        Unified episode ending for all paths - this is the single place where episodes end.
        Note: Session is NOT terminated here - it's shared across all samples and cleaned up in task_cleanup.

        Args:
            rest_base_url: Base URL for REST API
            interrupted: Whether this is an interrupted (non-happy path) cleanup
        """
        # End episode if it exists (both happy and interrupted paths)
        if self._episode_id:
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

                # For interrupted cleanup, use fire-and-forget HTTP to avoid event loop shutdown issues
                if interrupted:
                    # Use synchronous requests library for reliability during shutdown
                    # Fire-and-forget: send request without waiting for response
                    import threading

                    import requests

                    url = f"{rest_base_url}/api/v1/session/{self._session_id}/episodes/{self._episode_id}"
                    params = {
                        "reason": reason,
                        "cascade_end_attached_episodes": "false",
                    }

                    def send_delete_request() -> None:
                        """Fire-and-forget delete request in background thread."""
                        try:
                            # Short timeout - we just want to send the request, not wait for full processing
                            response = requests.delete(url, params=params, timeout=2.0)
                            if response.status_code == 200:
                                logger.info(
                                    f"Episode end request sent successfully (reason={reason})",
                                    extra={
                                        "session_id": self._session_id,
                                        "episode_id": self._episode_id,
                                        "reason": reason,
                                    },
                                )
                            else:
                                logger.debug(
                                    f"Episode end request returned {response.status_code}",
                                    extra={
                                        "session_id": self._session_id,
                                        "episode_id": self._episode_id,
                                        "status_code": response.status_code,
                                    },
                                )
                        except Exception as e:
                            # Log at debug level - server might still receive and process the request
                            logger.debug(
                                f"Episode end request exception (server may still process): {e}",
                                extra={
                                    "session_id": self._session_id,
                                    "episode_id": self._episode_id,
                                },
                            )

                    # Start in non-daemon thread and wait briefly for completion
                    # Non-daemon prevents process exit until thread completes or timeout
                    thread = threading.Thread(target=send_delete_request, daemon=False)
                    thread.start()
                    thread.join(timeout=1.5)  # Wait max 1.5 seconds for request to send

                    logger.info(
                        f"Episode end request initiated in background (reason={reason})",
                        extra={
                            "session_id": self._session_id,
                            "episode_id": self._episode_id,
                            "reason": reason,
                            "note": "Waited up to 1.5s for request to send",
                        },
                    )
                else:
                    # Happy path - use async client
                    from saber.client.client_session import ClientSessionManager
                    from saber.client.models import SessionManagerConfig

                    config = SessionManagerConfig(
                        base_url=rest_base_url,
                        mcp_server_url="",  # Not needed for episode ending
                        client_id="inspect_ai_sandbox_cleanup",
                        rest_timeout=10.0,
                    )
                    session_manager = ClientSessionManager(config=config)

                    # Get submission from store if available (happy path after scorer)
                    submission = None
                    try:
                        from inspect_ai.util import store

                        task_store = store()
                        # Scorer may have stored the submission for us
                        submission = task_store.get("saber_episode_submission")
                    except Exception:
                        # Store might not be available or submission not set - that's ok
                        pass

                    # End episode with submission
                    await session_manager.end_episode(
                        session_id=self._session_id,
                        episode_id=self._episode_id,
                        reason=reason,
                        result=submission,
                        cascade_end_attached_episodes=False,
                    )

                    logger.info(
                        f"Episode ended successfully (reason={reason})",
                        extra={
                            "session_id": self._session_id,
                            "episode_id": self._episode_id,
                            "reason": reason,
                        },
                    )

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
                    f"Failed to end episode (server will eventually clean up): {e}",
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
