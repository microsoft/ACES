"""
SessionManager implementation for SABER domain server.

The SessionManager serves as the central orchestrator for managing client sessions
and coordinating all server components. REST API functionality is handled by SessionAPI.
"""

import asyncio
import os
import subprocess
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from ..logging_config import (
    get_cleanup_logger,
    get_session_manager_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
    log_session_end,
)
from ..models import BenchmarkInfo, EpisodeEndResponse, EvalSubmission, MetadataKeys
from .api.session_mcp_api import SessionMCPAPI
from .api.session_rest_api import SessionRestAPI
from .base import Action, CommandResult, Episode, EpisodeState
from .benchmarks.benchmark_manager import BenchmarkManager
from .benchmarks.task import Task

# CleanupReason removed - using direct component cleanup
from .db.event_repository import EpisodeEventRepository
from .db.redis_manager import RedisManager
from .episodes.constants import EpisodeTerminationReason
from .episodes.episode_manager import EpisodeManager
from .evaluation.evaluation_manager import EvaluationManager
from .evaluation.session_evaluation_service import SessionEvaluationService
from .execution.execution_manager import ExecutionManager
from .policy.policy_manager import PolicyDocument, PolicyManager
from .time_source import TimeSource, UTCTimeSource
from .time_source import utc_now as _utc_now

logger = get_session_manager_logger(__name__)
cleanup_logger = get_cleanup_logger(__name__)


class ClientSession(BaseModel):
    """Represents an active client session with support for multiple episodes."""

    model_config = ConfigDict()

    session_id: str = Field(..., description="Unique session identifier")
    client_id: str = Field(..., description="Client identifier")

    # Multi-episode support
    creating_episode_ids: list[str] = Field(default_factory=list, description="Episodes in creation (holding lock)")
    active_episode_ids: list[str] = Field(default_factory=list, description="Currently active episode IDs")
    episode_history: list[str] = Field(
        default_factory=list, description="All completed episode IDs in chronological order"
    )

    # Task orchestration support
    task_queue: list[str] = Field(default_factory=list, description="Queued tasks for orchestration")

    # Session metadata
    created_at: datetime = Field(default_factory=_utc_now, description="Session creation time")
    last_activity: datetime = Field(default_factory=_utc_now, description="Last activity timestamp")
    is_active: bool = Field(default=True, description="Whether session is active")
    context: dict[str, Any] = Field(default_factory=dict, description="Session context data")

    @field_serializer("created_at", "last_activity")
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime fields to ISO format."""
        return value.isoformat()

    def update_activity(self, current_time: datetime | None = None) -> None:
        """Update the last activity timestamp.

        Args:
            current_time: Current time (defaults to UTC now if None)
        """
        if current_time is None:
            current_time = datetime.now(UTC)
        self.last_activity = current_time

    def add_creating_episode(self, episode_id: str) -> None:
        """Add an episode to the creating episodes list (while holding creation lock)."""
        if episode_id not in self.creating_episode_ids:
            self.creating_episode_ids.append(episode_id)
            self.update_activity()

    def move_to_active_episode(self, episode_id: str) -> None:
        """Move an episode from creating to active (after Docker startup completes)."""
        if episode_id in self.creating_episode_ids:
            self.creating_episode_ids.remove(episode_id)
        if episode_id not in self.active_episode_ids:
            self.active_episode_ids.append(episode_id)
            self.update_activity()

    def add_active_episode(self, episode_id: str) -> None:
        """Add an episode to the active episodes list."""
        if episode_id not in self.active_episode_ids:
            self.active_episode_ids.append(episode_id)
            self.update_activity()

    def remove_active_episode(self, episode_id: str) -> None:
        """Remove an episode from active episodes."""
        if episode_id in self.active_episode_ids:
            self.active_episode_ids.remove(episode_id)
            self.update_activity()

    def remove_creating_episode(self, episode_id: str) -> None:
        """Remove an episode from creating episodes (on error during creation)."""
        if episode_id in self.creating_episode_ids:
            self.creating_episode_ids.remove(episode_id)
            self.update_activity()

    def complete_episode(self, episode_id: str) -> None:
        """Move an episode from active to history."""
        if episode_id in self.active_episode_ids:
            self.active_episode_ids.remove(episode_id)
            if episode_id not in self.episode_history:
                self.episode_history.append(episode_id)
            self.update_activity()

    def add_task_to_queue(self, task_id: str) -> None:
        """Add a task to the orchestration queue."""
        if task_id not in self.task_queue:
            self.task_queue.append(task_id)
            self.update_activity()

    def get_next_task(self) -> str | None:
        """Get and remove the next task from the queue."""
        if self.task_queue:
            task_id = self.task_queue.pop(0)
            self.update_activity()
            return task_id
        return None

    def has_active_episodes(self) -> bool:
        """Check if session has any active or creating episodes."""
        return len(self.creating_episode_ids) > 0 or len(self.active_episode_ids) > 0

    def get_episode_count(self) -> dict[str, int]:
        """Get episode counts for analytics."""
        return {
            "creating": len(self.creating_episode_ids),
            "active": len(self.active_episode_ids),
            "completed": len(self.episode_history),
            "total": len(self.creating_episode_ids) + len(self.active_episode_ids) + len(self.episode_history),
        }


class SessionManager:
    """
    Central orchestrator for SABER domain server.

    Manages multiple concurrent client sessions and coordinates all server components.
    REST API functionality is delegated to SessionAPI.
    """

    def __init__(
        self,
        domain_name: str,
        config_dir: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        mcp_host: str = "0.0.0.0",
        mcp_port: int = 3001,
        session_timeout_minutes: int = 30,
        cleanup_interval_minutes: int = 5,
        manifest: dict[str, Any] | None = None,
        manifest_path: str | None = None,
        time_source: TimeSource | None = None,
        data_dir: str | None = None,
    ):
        """
        Initialize the SessionManager.

        Args:
            domain_name: Name of the security domain (e.g., 'malware_classification')
            config_dir: Path to configuration directory containing tasks.yaml and other config files
            host: REST server host address
            port: REST server port
            mcp_host: MCP server host address
            mcp_port: MCP server port
            session_timeout_minutes: Minutes of inactivity before a session times out
            cleanup_interval_minutes: Minutes between cleanup checks
            manifest: Domain manifest dictionary (optional)
            manifest_path: Path to domain manifest file (optional)
            time_source: Time source for getting current time (defaults to UTCTimeSource)
            data_dir: Path to data directory for evaluation results (defaults to server/data)
        """
        self.domain_name = domain_name
        self.config_dir = config_dir
        self.data_dir = data_dir or str(Path(config_dir).parent / "data")
        self.host = host
        self.port = port
        self.mcp_host = mcp_host
        self.mcp_port = mcp_port
        self.session_timeout_minutes = session_timeout_minutes
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.active_sessions: dict[str, ClientSession] = {}
        self.cleanup_task: asyncio.Task[None] | None = None
        self.shutdown_event = asyncio.Event()
        self._shutdown_in_progress = False

        # Semaphore for episode creation to limit concurrent Docker resource usage
        # This limits episode creation to prevent overwhelming the Docker daemon
        # with too many concurrent Docker Compose environment creations
        # Increased from Lock (1) to Semaphore(3) to allow moderate concurrency
        self._episode_creation_lock = asyncio.Semaphore(3)  # Max 3 concurrent episode creations

        # Track background finalization tasks for async episode creation
        self._episode_finalization_tasks: dict[str, asyncio.Task[None]] = {}
        # Semaphore to limit concurrent episode finalizations (health checks, not Docker compose)
        # Note: Docker compose operations are limited by _episode_creation_lock semaphore
        # This semaphore prevents overwhelming the system with concurrent health checks
        self._finalization_semaphore = asyncio.Semaphore(16)  # Max 16 concurrent finalizations

        # Track permanent environment startup status
        self._permanent_env_startup_task: asyncio.Task[None] | None = None
        self._permanent_env_startup_error: str | None = None
        self._permanent_env_startup_complete = False

        # Manifest information for health endpoints
        self.manifest = manifest or {}
        self.manifest_path = manifest_path
        self._time_source = time_source or UTCTimeSource()

        # Initialize server components
        logger.info(
            "Initializing SessionManager",
            extra={
                "event": "session_manager_initializing",
                "domain": domain_name,
                "config_dir": config_dir,
            },
        )

        self.benchmark_manager = BenchmarkManager(domain_name, config_dir)
        self.redis_manager = RedisManager()
        self._redis_auto_start = os.getenv("SABER_REDIS_AUTO_START", "true").lower() != "false"
        self._redis_compose_path: Path | None = None  # Set during start, used during stop
        self.episode_manager = EpisodeManager(time_source=self._time_source)

        # Initialize execution manager with self for cross-episode executor operations
        self.execution_manager = ExecutionManager(config_dir, session_manager=self)

        # Initialize permanent environment manager through ExecutionManager for unified container lifecycle
        permanent_config = {
            "domain": domain_name,
            "config_dir": config_dir,
            "logs_dir": str(Path(config_dir).parent / "logs"),  # Logs go to server/logs, not server/config/logs
            "enable_logging": True,
        }
        self.execution_manager.initialize_permanent_environment_manager(permanent_config)

        self.policy_manager = PolicyManager(domain_name)
        # Pass data_dir to evaluation manager for proper store path
        evaluation_store_path = Path(self.data_dir) / "evaluations"
        self.evaluation_manager = EvaluationManager(store=evaluation_store_path)

        # Initialize evaluation service for retrieval operations
        self.evaluation_service = SessionEvaluationService(self.evaluation_manager.store)

        # Initialize protocol handlers
        self.rest_api = SessionRestAPI(self, host, port)
        self.mcp_api = SessionMCPAPI(self, mcp_host, mcp_port)

        logger.info(
            "SessionManager initialized",
            extra={
                "event": "session_manager_initialized",
                "domain": domain_name,
                "rest_host": host,
                "rest_port": port,
                "mcp_host": mcp_host,
                "mcp_port": mcp_port,
                "manifest_available": bool(self.manifest),
            },
        )

    def get_health_metadata(self) -> dict[str, Any]:
        """Get health metadata including manifest information."""
        import hashlib
        import os
        from pathlib import Path

        metadata: dict[str, Any] = {
            "status": "healthy",
            "domain": self.domain_name,
        }

        # Add manifest information if available
        if self.manifest:
            domain_info = self.manifest.get("domain", {})
            manifest_data: dict[str, Any] = {
                "capabilities": self.manifest.get("capabilities", []),
            }

            # Add optional string fields only if they exist
            domain_slug = domain_info.get("slug")
            if domain_slug is not None:
                manifest_data["domain_slug"] = str(domain_slug)

            schema_version = self.manifest.get("schemaVersion")
            if schema_version is not None:
                manifest_data["schema_version"] = str(schema_version)

            if self.manifest_path is not None:
                manifest_data["manifest_path"] = str(self.manifest_path)

            metadata.update(manifest_data)

        # Calculate config directory checksum for drift detection
        try:
            config_path = Path(self.config_dir)
            if config_path.exists():
                # Simple checksum based on modification times of key files
                checksum_data = []
                for file_path in config_path.rglob("*.yaml"):
                    if file_path.is_file():
                        stat = file_path.stat()
                        checksum_data.append(f"{file_path.name}:{stat.st_mtime}:{stat.st_size}")

                if checksum_data:
                    config_checksum = hashlib.md5(":".join(sorted(checksum_data)).encode()).hexdigest()
                    metadata["config_checksum"] = config_checksum
        except Exception as e:
            logger.warning(
                "Failed to calculate config checksum", extra={"event": "config_checksum_failed", "error": str(e)}
            )

        # Add build metadata if available from environment
        build_metadata: dict[str, str] = {}
        for env_var in ["GIT_SHA", "BUILD_TIMESTAMP", "IMAGE_TAG"]:
            value = os.getenv(env_var)
            if value:
                build_metadata[env_var.lower()] = value

        if build_metadata:
            metadata["build_metadata"] = build_metadata

        # Check permanent environment health if configured
        perm_env_health = self._check_permanent_environment_health()
        perm_status = perm_env_health.get("status", "")

        if not perm_env_health["healthy"]:
            # Set status based on whether it's starting or actually unhealthy
            if perm_status == "starting":
                metadata["status"] = "starting"
            else:
                metadata["status"] = "unhealthy"
                # Only add error if present
                if "error" in perm_env_health:
                    metadata["permanent_environment_error"] = perm_env_health["error"]

        metadata["permanent_environment"] = perm_env_health

        return metadata

    def _check_permanent_environment_health(self) -> dict[str, Any]:
        """Check permanent environment health using ComposeHealthChecker.

        Returns:
            Dict with health status and details including:
            - healthy: True if all services healthy, False otherwise
            - status: 'starting', 'healthy', 'unhealthy', 'failed', etc.
            - services: Per-service health status (when available)
        """
        # Check if permanent environment is configured
        permanent_env_name = self.benchmark_manager.config_loader.get_permanent_environment()
        if not permanent_env_name:
            return {"healthy": True, "status": "not_configured", "message": "No permanent environment configured"}

        # Check if startup failed
        if self._permanent_env_startup_error:
            return {
                "healthy": False,
                "status": "startup_failed",
                "error": self._permanent_env_startup_error,
                "environment_name": permanent_env_name,
            }

        # Check if still starting up (background task running)
        if not self._permanent_env_startup_complete:
            # Get current health status to show progress
            try:
                from pathlib import Path

                from saber.server.execution.sandbox.compose_health_checker import ComposeHealthChecker

                permanent_compose_path = (
                    Path(self.config_dir) / "environments" / "permanent" / f"{permanent_env_name}.compose.yml"
                )
                if permanent_compose_path.exists() and self.execution_manager._permanent_environment_manager:
                    health_checker = ComposeHealthChecker()
                    project_name = self.execution_manager._permanent_environment_manager.compose_project_name
                    health_summary = health_checker.get_service_health_summary(
                        str(permanent_compose_path), project_name
                    )
                    return {
                        "healthy": False,
                        "status": "starting",
                        "message": (
                            f"Permanent environment starting: "
                            f"{health_summary['healthy_count']}/{health_summary['total_count']} services healthy"
                        ),
                        "environment_name": permanent_env_name,
                        "healthy_services": health_summary["healthy_count"],
                        "total_services": health_summary["total_count"],
                        "services": health_summary["services"],
                    }
            except Exception:
                pass  # Fall through to basic starting status

            return {
                "healthy": False,
                "status": "starting",
                "message": "Permanent environment is starting up...",
                "environment_name": permanent_env_name,
            }

        # Check if permanent environment manager exists and is running
        if not self.execution_manager._permanent_environment_manager:
            return {
                "healthy": False,
                "status": "manager_not_initialized",
                "error": "Permanent environment manager not initialized",
            }

        if not self.execution_manager._permanent_environment_manager.is_running():
            return {"healthy": False, "status": "not_running", "error": "Permanent environment not running"}

        # Get the compose file path for health checking
        from pathlib import Path

        permanent_compose_path = (
            Path(self.config_dir) / "environments" / "permanent" / f"{permanent_env_name}.compose.yml"
        )

        if not permanent_compose_path.exists():
            return {
                "healthy": False,
                "status": "compose_file_missing",
                "error": f"Permanent environment compose file not found: {permanent_compose_path}",
            }

        # Use ComposeHealthChecker to validate service health
        try:
            from saber.server.execution.sandbox.compose_health_checker import ComposeHealthChecker

            health_checker = ComposeHealthChecker()
            project_name = self.execution_manager._permanent_environment_manager.compose_project_name

            health_summary = health_checker.get_service_health_summary(str(permanent_compose_path), project_name)

            return {
                "healthy": health_summary["overall_healthy"],
                "status": "checked",
                "environment_name": permanent_env_name,
                "project_name": project_name,
                "healthy_services": health_summary["healthy_count"],
                "total_services": health_summary["total_count"],
                "services": health_summary["services"],
                "error": health_summary.get("error"),
            }

        except Exception as e:
            return {
                "healthy": False,
                "status": "health_check_failed",
                "error": f"Failed to check permanent environment health: {str(e)}",
            }

    async def _connect_redis(self) -> None:
        """Connect RedisManager and wire EpisodeEventRepository if successful.

        Redis is OPTIONAL — connection failure is logged as a warning
        and the system continues without DB-backed transcript storage.
        """
        try:
            await self.redis_manager.connect()
        except Exception:
            logger.warning(
                "Redis unavailable — continuing without DB transcript storage",
                extra={"event": "redis_optional_skip"},
                exc_info=True,
            )
            return

        if self.redis_manager.is_connected:
            event_repo = EpisodeEventRepository(self.redis_manager.client)
            self.episode_manager.wire_event_repository(event_repo)
            logger.info(
                "EpisodeEventRepository wired into TranscriptCoordinator",
                extra={"event": "event_repository_wired"},
            )

    def _start_redis_container(self) -> None:
        """Start Redis container via Docker Compose if auto-start is enabled.

        Uses the bundled redis-compose.yml from package_resources.
        Graceful degradation: failures are logged, never raised.
        """
        if not self._redis_auto_start:
            return

        try:
            from saber.domain.resources import resolve_redis_compose_path

            compose_path = resolve_redis_compose_path()
            self._redis_compose_path = compose_path
        except Exception as e:
            logger.warning(
                f"Redis compose file not found, skipping auto-start: {e}",
                extra={"event": "redis_compose_not_found", "error": str(e)},
            )
            return

        cmd = [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "-p",
            "saber-redis",
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            "30",
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(
                "Redis container started",
                extra={"event": "redis_container_started", "compose_path": str(compose_path)},
            )
        except FileNotFoundError:
            logger.warning(
                "Docker not found — cannot auto-start Redis container",
                extra={"event": "redis_docker_not_found"},
            )
            self._redis_compose_path = None
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Failed to start Redis container: {e.stderr}",
                extra={
                    "event": "redis_container_start_failed",
                    "returncode": e.returncode,
                    "stderr": e.stderr,
                },
            )
            self._redis_compose_path = None

    def _stop_redis_container(self) -> None:
        """Stop Redis container via Docker Compose.

        Only stops if we started it (compose_path is set from start).
        Graceful degradation: failures are logged, never raised.
        """
        if not self._redis_compose_path:
            return

        cmd = [
            "docker",
            "compose",
            "-f",
            str(self._redis_compose_path),
            "-p",
            "saber-redis",
            "down",
            "-t",
            "1",
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            logger.info(
                "Redis container stopped",
                extra={"event": "redis_container_stopped"},
            )
        except Exception as e:
            # Use string check because tests mock subprocess, making
            # subprocess.TimeoutExpired a MagicMock rather than a real class.
            if type(e).__name__ == "TimeoutExpired":
                logger.warning(
                    "Redis container stop timed out after 30s",
                    extra={"event": "redis_container_stop_timeout"},
                )
            else:
                logger.warning(
                    f"Failed to stop Redis container: {e}",
                    extra={"event": "redis_container_stop_failed", "error": str(e)},
                )
        finally:
            self._redis_compose_path = None

    async def start_server(self) -> None:
        """Start both REST and MCP servers concurrently with session cleanup.

        Permanent environment is started in the background so the REST API
        can respond to health checks immediately with startup progress.
        """
        import asyncio

        # Start managed Redis container if auto-start enabled
        # Run sync subprocess in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._start_redis_container)

        # Connect to Redis (optional — failure is non-fatal)
        await self._connect_redis()

        # Start permanent environment in background (non-blocking)
        # This allows REST API to respond to health checks during startup
        self._permanent_env_startup_task = asyncio.create_task(self._start_permanent_environment_async())

        # Start the session cleanup task
        self.cleanup_task = asyncio.create_task(self._session_cleanup_loop())

        # Start both servers concurrently using asyncio tasks
        rest_task = asyncio.create_task(self.rest_api.start_server())
        mcp_task = asyncio.create_task(self.mcp_api.start_mcp_server())

        # Wait for both to complete (they run indefinitely)
        await asyncio.gather(rest_task, mcp_task, self.cleanup_task)

    async def shutdown(self) -> None:
        """Shutdown the SessionManager and cleanup resources.

        This method is idempotent — concurrent or repeated calls are safely ignored.
        """
        if self._shutdown_in_progress:
            logger.info(
                "Shutdown already in progress; ignoring duplicate call",
                extra={"event": "session_manager_shutdown_duplicate", "domain": self.domain_name},
            )
            return
        self._shutdown_in_progress = True

        logger.info(
            "Shutting down SessionManager",
            extra={"event": "session_manager_shutdown_start", "domain": self.domain_name},
        )

        # Signal shutdown to stop cleanup loop
        self.shutdown_event.set()

        # Cancel cleanup task if running
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Stop permanent environment through ExecutionManager
        if self.execution_manager.is_permanent_environment_running():
            logger.info(
                "Stopping permanent environment",
                extra={"event": "permanent_environment_stop_requested", "domain": self.domain_name},
            )
            try:
                await self.execution_manager.stop_permanent_environment()
                logger.info(
                    "Permanent environment stopped",
                    extra={"event": "permanent_environment_stop_completed", "domain": self.domain_name},
                )
            except Exception as e:
                logger.error(
                    "Failed to stop permanent environment",
                    extra={
                        "event": "permanent_environment_stop_failed",
                        "domain": self.domain_name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

        # Shutdown MCP server
        await self.mcp_api.shutdown_mcp_server()

        # Cleanup all active sessions
        session_ids = list(self.active_sessions.keys())
        cleanup_logger.info(
            "Shutdown cleanup initiated",
            extra={
                "event": "shutdown_cleanup_start",
                "active_sessions_count": len(session_ids),
                "session_ids": session_ids,
            },
        )
        for session_id in session_ids:
            cleanup_logger.info(
                "Terminating session during shutdown",
                extra={"event": "shutdown_session_termination", "session_id": session_id},
            )
            await self.terminate_session(session_id)

        # Disconnect RedisManager and stop container BEFORE slow network cleanup
        # so the force-exit watchdog doesn't kill us before Redis is stopped
        await self.redis_manager.disconnect()
        self._stop_redis_container()

        # Clean up SABER episode networks (can be slow, best-effort)
        await self._cleanup_saber_episode_networks()

        logger.info(
            "SessionManager shutdown complete",
            extra={"event": "session_manager_shutdown_complete", "domain": self.domain_name},
        )

    async def _cleanup_episode_network(self, episode_id: str) -> None:
        """Clean up Docker network for a specific episode"""
        network_name = f"saber-episode-{episode_id}"
        try:
            # Check if network exists
            result = await asyncio.create_subprocess_exec(
                "docker",
                "network",
                "ls",
                "--format",
                "{{.Name}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                existing_networks = stdout.decode().strip().split("\n")
                if network_name in existing_networks:
                    logger.info(
                        "Removing episode network",
                        extra={"event": "episode_network_removal", "network": network_name},
                    )

                    # Remove any containers still connected to the network
                    inspect_result = await asyncio.create_subprocess_exec(
                        "docker",
                        "network",
                        "inspect",
                        network_name,
                        "--format",
                        "{{range .Containers}}{{.Name}} {{end}}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    inspect_stdout, inspect_stderr = await inspect_result.communicate()

                    if inspect_result.returncode == 0:
                        container_names = inspect_stdout.decode().strip().split()
                        for container_name in container_names:
                            if container_name:  # Skip empty strings
                                logger.info(
                                    "Force removing container attached to episode network",
                                    extra={
                                        "event": "episode_network_container_remove",
                                        "network": network_name,
                                        "container": container_name,
                                    },
                                )
                                rm_result = await asyncio.create_subprocess_exec(
                                    "docker",
                                    "rm",
                                    "-f",
                                    container_name,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE,
                                )
                                await rm_result.communicate()

                    # Remove the network
                    rm_network_result = await asyncio.create_subprocess_exec(
                        "docker",
                        "network",
                        "rm",
                        network_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, rm_stderr = await rm_network_result.communicate()

                    if rm_network_result.returncode == 0:
                        logger.info(
                            "Episode network removed",
                            extra={"event": "episode_network_removed", "network": network_name},
                        )
                    else:
                        stderr_text = rm_stderr.decode().strip() if rm_stderr else ""
                        logger.warning(
                            "Failed to remove episode network",
                            extra={
                                "event": "episode_network_remove_failed",
                                "network": network_name,
                                "stderr": stderr_text,
                            },
                        )

        except Exception as e:
            logger.error(
                "Episode network cleanup error",
                extra={
                    "event": "episode_network_cleanup_error",
                    "network": network_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def _cleanup_episode_network_with_retry(
        self, episode_id: str, max_retries: int = 3, base_delay: float = 1.0
    ) -> bool:
        """
        Clean up Docker network for a specific episode with retry logic.

        Uses exponential backoff to handle transient failures (e.g., containers still shutting down).

        Args:
            episode_id: Episode ID whose network should be cleaned up
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 1.0)

        Returns:
            bool: True if cleanup succeeded, False if all retries failed
        """
        for attempt in range(max_retries):
            try:
                await self._cleanup_episode_network(episode_id)
                if attempt > 0:
                    logger.info(
                        "Episode network cleanup succeeded after retry",
                        extra={
                            "event": "episode_network_cleanup_retry_success",
                            "episode_id": episode_id,
                            "attempt": attempt + 1,
                        },
                    )
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        "Episode network cleanup failed, retrying",
                        extra={
                            "event": "episode_network_cleanup_retry",
                            "episode_id": episode_id,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "retry_delay": delay,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Episode network cleanup failed after all retries",
                        extra={
                            "event": "episode_network_cleanup_retry_exhausted",
                            "episode_id": episode_id,
                            "attempts": max_retries,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    return False
        return False

    async def _cleanup_saber_episode_networks(self) -> None:
        """Clean up all SABER episode networks to prevent Docker subnet pool exhaustion."""

        logger.info(
            "Starting SABER episode network cleanup",
            extra={"event": "episode_network_bulk_cleanup_start"},
        )

        try:
            result = await asyncio.create_subprocess_exec(
                "docker",
                "network",
                "ls",
                "--filter",
                "name=saber-episode-",
                "--format",
                "{{.Name}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.warning(
                    "Failed to list SABER episode networks",
                    extra={
                        "event": "episode_network_list_failed",
                        "return_code": result.returncode,
                        "stderr": stderr.decode().strip(),
                    },
                )
                return

            network_names = stdout.decode().strip().split("\n")
            network_names = [name.strip() for name in network_names if name.strip()]

            if not network_names:
                logger.info(
                    "No SABER episode networks found for cleanup",
                    extra={"event": "episode_network_bulk_cleanup_empty"},
                )
                return

            logger.info(
                "Identified SABER episode networks for cleanup",
                extra={
                    "event": "episode_network_bulk_cleanup_identified",
                    "network_count": len(network_names),
                    "networks": network_names,
                },
            )

            for network_name in network_names:
                try:
                    inspect_result = await asyncio.create_subprocess_exec(
                        "docker",
                        "network",
                        "inspect",
                        network_name,
                        "--format",
                        "{{range .Containers}}{{.Name}} {{end}}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    containers_stdout, containers_stderr = await inspect_result.communicate()

                    if inspect_result.returncode == 0:
                        container_names = containers_stdout.decode().strip().split()
                        if container_names:
                            logger.info(
                                "Removing containers attached to episode network",
                                extra={
                                    "event": "episode_network_containers_remove",
                                    "network": network_name,
                                    "container_count": len(container_names),
                                    "containers": container_names,
                                },
                            )
                            for container_name in container_names:
                                await asyncio.create_subprocess_exec(
                                    "docker",
                                    "rm",
                                    "-f",
                                    container_name,
                                    stdout=asyncio.subprocess.DEVNULL,
                                    stderr=asyncio.subprocess.DEVNULL,
                                )
                    else:
                        logger.warning(
                            "Failed to inspect episode network",
                            extra={
                                "event": "episode_network_inspect_failed",
                                "network": network_name,
                                "return_code": inspect_result.returncode,
                                "stderr": containers_stderr.decode().strip(),
                            },
                        )
                except Exception as e:
                    logger.warning(
                        "Error cleaning containers for episode network",
                        extra={
                            "event": "episode_network_container_cleanup_error",
                            "network": network_name,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

            cleaned_count = 0
            for network_name in network_names:
                try:
                    remove_result = await asyncio.create_subprocess_exec(
                        "docker",
                        "network",
                        "rm",
                        network_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, rm_stderr = await remove_result.communicate()

                    if remove_result.returncode == 0:
                        cleaned_count += 1
                        logger.debug(
                            "Episode network removed during bulk cleanup",
                            extra={
                                "event": "episode_network_bulk_removed",
                                "network": network_name,
                            },
                        )
                    else:
                        logger.warning(
                            "Failed to remove episode network during bulk cleanup",
                            extra={
                                "event": "episode_network_bulk_remove_failed",
                                "network": network_name,
                                "return_code": remove_result.returncode,
                                "stderr": rm_stderr.decode().strip(),
                            },
                        )

                except Exception as e:
                    logger.warning(
                        "Error removing episode network during bulk cleanup",
                        extra={
                            "event": "episode_network_bulk_remove_error",
                            "network": network_name,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

            logger.info(
                "Completed SABER episode network cleanup",
                extra={
                    "event": "episode_network_bulk_cleanup_complete",
                    "network_count": len(network_names),
                    "networks_removed": cleaned_count,
                },
            )

        except Exception as e:
            logger.error(
                "SABER episode network cleanup failed",
                extra={
                    "event": "episode_network_bulk_cleanup_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def create_session(self, client_id: str) -> ClientSession:
        """
        Create a new client session.

        Args:
            client_id: Identifier for the connecting client

        Returns:
            ClientSession object for the new session

        Raises:
            HTTPException: If the server is not ready for session creation
        """
        # Check if the sandbox manager is ready before creating sessions
        if not self.execution_manager.is_sandbox_ready():
            raise HTTPException(status_code=503, detail="Server is still initializing. Please try again in a moment.")

        session_id = str(uuid.uuid4())
        session = ClientSession(session_id=session_id, client_id=client_id)

        self.active_sessions[session_id] = session

        # Log session creation with evaluation manager (ignore failures)
        try:
            await self.evaluation_manager.log_session_start(session_id, client_id)
        except Exception as e:
            logger.warning(
                "Failed to log session start",
                extra={
                    "event": "session_start_logging_failed",
                    "session_id": session_id,
                    "client_id": client_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

        logger.info(
            "Session created",
            extra={"event": "session_created", "session_id": session_id, "client_id": client_id},
        )
        return session

    async def terminate_session(self, session_id: str, reason: str = "unknown") -> None:
        """
        Terminate a client session and cleanup resources.

        Args:
            session_id: ID of the session to terminate
            reason: Reason for termination (e.g., 'client_request', 'timeout', 'error')
        """
        logger.info(
            "Session termination initiated",
            extra={
                "event": "session_termination_initiated",
                "session_id": session_id,
                "reason": reason,
            },
        )
        log_session_end(logger, session_id, f"SessionManager termination - {reason}")
        session = self._get_session(session_id)

        # Acquire episode creation lock to prevent race with episodes being added to lists
        async with self._episode_creation_lock:
            # Mark session as inactive to prevent new episodes from being created during termination
            session.is_active = False
            logger.info(
                "Session marked as inactive - no new episodes will be accepted",
                extra={
                    "event": "session_marked_inactive",
                    "session_id": session_id,
                },
            )

            # Capture episodes from BOTH creating and active lists while holding the lock
            # This ensures we clean up episodes that are mid-creation (holding lock, Docker starting)
            # as well as fully created episodes
            creating_episodes = list(session.creating_episode_ids)
            active_episodes = list(session.active_episode_ids)
            episodes_to_terminate = creating_episodes + active_episodes

            logger.info(
                "Captured all episodes for termination (while holding lock)",
                extra={
                    "event": "session_episodes_captured_for_termination",
                    "session_id": session_id,
                    "creating_episode_count": len(creating_episodes),
                    "active_episode_count": len(active_episodes),
                    "total_episode_count": len(episodes_to_terminate),
                    "creating_episode_ids": creating_episodes,
                    "active_episode_ids": active_episodes,
                },
            )

        # Release lock before terminating episodes (to avoid blocking new requests)
        # End all active episodes (this will trigger individual episode cleanup)
        if episodes_to_terminate:
            try:
                logger.info(
                    "Terminating active episodes during session shutdown",
                    extra={
                        "event": "session_active_episodes_terminating",
                        "session_id": session_id,
                        "active_episode_count": len(episodes_to_terminate),
                        "active_episode_ids": episodes_to_terminate,
                    },
                )
                # End all active episodes - mark them as terminated first
                episodes_needing_cleanup = []
                for episode_id in episodes_to_terminate:
                    try:
                        # Check if episode is already completed before forcing termination
                        episode = self.episode_manager.get_episode_by_id(episode_id)
                        if episode and episode.is_complete:
                            logger.info(
                                "Episode already complete; skipping termination override",
                                extra={
                                    "event": "session_episode_already_complete",
                                    "session_id": session_id,
                                    "episode_id": episode_id,
                                },
                            )
                            continue

                        # Call episode manager directly for session termination (no submission required)
                        await self.episode_manager.end_episode(
                            episode_id, EpisodeTerminationReason.SESSION_TERMINATED, None
                        )
                        logger.info(
                            "Episode terminated during session shutdown",
                            extra={
                                "event": "session_episode_terminated",
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "termination_reason": EpisodeTerminationReason.SESSION_TERMINATED,
                            },
                        )
                        episodes_needing_cleanup.append(episode_id)
                    except Exception as e:
                        logger.warning(
                            "Failed to terminate episode during session shutdown",
                            extra={
                                "event": "session_episode_termination_failed",
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )

                # Cleanup all episode Docker containers in parallel
                if episodes_needing_cleanup:

                    async def cleanup_episode_task(ep_id: str) -> None:
                        try:
                            await self.execution_manager.cleanup_episode(
                                ep_id,
                                {"episode_end_reason": EpisodeTerminationReason.SESSION_TERMINATED},
                            )
                            logger.info(
                                "Episode Docker cleanup completed during session shutdown",
                                extra={
                                    "event": "session_episode_cleanup_completed",
                                    "session_id": session_id,
                                    "episode_id": ep_id,
                                },
                            )
                        except Exception as cleanup_error:
                            logger.warning(
                                "Failed to cleanup episode Docker during session shutdown",
                                extra={
                                    "event": "session_episode_cleanup_failed",
                                    "session_id": session_id,
                                    "episode_id": ep_id,
                                    "error": str(cleanup_error),
                                    "error_type": type(cleanup_error).__name__,
                                },
                            )

                    # Run all cleanup tasks in parallel
                    await asyncio.gather(
                        *[cleanup_episode_task(ep_id) for ep_id in episodes_needing_cleanup], return_exceptions=True
                    )

                # Clear remaining state from both lists
                logger.info(
                    "Clearing all episode IDs from session state during termination",
                    extra={
                        "event": "session_episode_ids_cleared",
                        "session_id": session_id,
                        "creating_count_before_clear": len(session.creating_episode_ids),
                        "active_count_before_clear": len(session.active_episode_ids),
                        "creating_episode_ids": list(session.creating_episode_ids),
                        "active_episode_ids": list(session.active_episode_ids),
                    },
                )
                session.creating_episode_ids.clear()
                session.active_episode_ids.clear()
                session.task_queue.clear()
            except Exception as e:
                logger.warning(
                    "Failed to terminate active episodes during session shutdown",
                    extra={
                        "event": "session_active_episode_termination_error",
                        "session_id": session_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

        # Log session end with evaluation manager (ignore failures)
        try:
            await self.evaluation_manager.log_session_end(session_id)
        except Exception as e:
            logger.warning(
                "Failed to log session end",
                extra={
                    "event": "session_end_logging_failed",
                    "session_id": session_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

        # Check for any orphaned episodes that might need cleanup
        # Only clean up episodes that are still in creating or active state
        try:
            log_operation_start(logger, "Orphaned episode cleanup check", session_id)
            episodes_to_cleanup = list(set(session.creating_episode_ids) | set(session.active_episode_ids))
            logger.info(
                "Checking for orphaned episodes",
                extra={
                    "event": "orphaned_episode_cleanup_check",
                    "session_id": session_id,
                    "episode_ids": episodes_to_cleanup,
                    "creating_count": len(session.creating_episode_ids),
                    "active_count": len(session.active_episode_ids),
                },
            )

            cleanup_success = True
            for episode_id in episodes_to_cleanup:
                try:
                    logger.info(
                        "Evaluating orphaned episode cleanup",
                        extra={
                            "event": "orphaned_episode_cleanup_attempt",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )
                    episode_cleanup = self.execution_manager.cleanup_episode(
                        episode_id,
                        {"manual_termination": True, "orphaned_check": True},
                    )
                    if not episode_cleanup:
                        cleanup_success = False
                        logger.warning(
                            "Orphaned episode cleanup failed",
                            extra={
                                "event": "orphaned_episode_cleanup_failed",
                                "session_id": session_id,
                                "episode_id": episode_id,
                            },
                        )
                    else:
                        logger.info(
                            "Orphaned episode cleanup succeeded",
                            extra={
                                "event": "orphaned_episode_cleanup_succeeded",
                                "session_id": session_id,
                                "episode_id": episode_id,
                            },
                        )
                except Exception as e:
                    cleanup_success = False
                    logger.warning(
                        "Error during orphaned episode cleanup",
                        extra={
                            "event": "orphaned_episode_cleanup_error",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

            if cleanup_success:
                log_operation_success(logger, "Orphaned episode cleanup check", session_id)
            else:
                logger.warning(
                    "Orphaned episode cleanup reported failures",
                    extra={
                        "event": "orphaned_episode_cleanup_partial",
                        "session_id": session_id,
                    },
                )
        except Exception as e:
            log_operation_failure(logger, "Orphaned episode cleanup check", str(e), session_id)

        # CRITICAL FIX: Cancel pending finalization tasks for this session's episodes
        cancelled_tasks = []
        for episode_id in list(self._episode_finalization_tasks.keys()):
            episode = self.episode_manager.get_episode_by_id(episode_id)
            if episode and episode.session_id == session_id:
                task = self._episode_finalization_tasks.get(episode_id)
                if task and not task.done():
                    task.cancel()
                    cancelled_tasks.append(episode_id)
                    logger.info(
                        "Cancelled finalization task for episode",
                        extra={
                            "event": "finalization_task_cancelled",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )

        # Wait for cancellations to complete (with timeout)
        if cancelled_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *[
                            self._episode_finalization_tasks[eid]
                            for eid in cancelled_tasks
                            if eid in self._episode_finalization_tasks
                        ],
                        return_exceptions=True,
                    ),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning(
                    "Finalization task cancellation timed out",
                    extra={
                        "event": "finalization_cancellation_timeout",
                        "session_id": session_id,
                        "cancelled_count": len(cancelled_tasks),
                    },
                )
            except Exception as e:
                logger.warning(
                    "Error during finalization task cancellation",
                    extra={
                        "event": "finalization_cancellation_error",
                        "session_id": session_id,
                        "error": str(e),
                    },
                )

        # Mark session as inactive
        session.is_active = False

        # Remove session from active sessions
        logger.info(
            "Session removed from active sessions",
            extra={"event": "session_removed", "session_id": session_id},
        )
        del self.active_sessions[session_id]

        logger.info(
            "Session terminated",
            extra={"event": "session_terminated", "session_id": session_id},
        )

    async def start_episode(self, session_id: str, task_id: str) -> Episode:
        """
        Initialize episode for a task with automatic dependency resolution.

        This method creates the episode and starts the Docker environment under a GLOBAL lock,
        then releases the lock and continues finalization (health checks, file copying, prompts, etc.)
        in a background task.

        Returns immediately with episode in CREATING state. Clients must poll status
        until episode becomes READY.

        This method uses a GLOBAL lock to serialize Docker Compose environment creation
        across all sessions, preventing concurrent operations that can overwhelm the
        Docker daemon. The lock is released BEFORE health checks to allow parallelism.

        Args:
            session_id: ID of the client session
            task_id: ID of the task to start

        Returns:
            Episode object in CREATING state
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Get the raw task for environment/dependency configuration
        raw_task = self.benchmark_manager.get_task(task_id)
        # Get SingleEpisodeTask with rendered prompts for transcript initialization
        single_episode_task = self.benchmark_manager.get_single_episode_task(task_id)

        # Acquire GLOBAL lock to serialize episode creation (Docker compose only)
        async with self._episode_creation_lock:
            # Check again INSIDE the lock - session might have been terminated while waiting
            if not session.is_active:
                logger.warning(
                    "Cannot create episode - session terminated while waiting for lock",
                    extra={
                        "event": "episode_creation_rejected_session_terminated_during_wait",
                        "session_id": session_id,
                        "task_id": task_id,
                    },
                )
                raise HTTPException(
                    status_code=409, detail=f"Cannot create episode: session {session_id} is terminating"
                )
            logger.info(
                "🔒 Episode creation lock acquired (global serialization active)",
                extra={
                    "event": "episode_creation_lock_acquired",
                    "session_id": session_id,
                    "task_id": task_id,
                    "lock_scope": "global",
                },
            )

            # Create episode in CREATING state (changed from ACTIVE)
            # Initialize context with transcript via public API (using SingleEpisodeTask with rendered prompts)
            context = self.episode_manager.initialize_episode_context(single_episode_task)

            episode = Episode(
                task_id=task_id,
                session_id=session_id,
                state=EpisodeState.CREATING,  # Changed from ACTIVE
                context=context,
                creation_error=None,
                metadata={"created_at": self._time_source.now().isoformat()},
                end_time=None,
                eval_submission=None,
                completion_reason=None,
                submission=None,
                attached_to_episode_id=None,
            )

            # Add episode to episode manager tracking immediately
            self.episode_manager.add_episode_to_session(session_id, episode)

            # Add to creating_episode_ids IMMEDIATELY (while holding lock) to ensure it's tracked for cleanup
            session.add_creating_episode(episode.episode_id)

            logger.info(
                "Episode created in CREATING state and added to creating_episode_ids",
                extra={
                    "event": "episode_created",
                    "session_id": session_id,
                    "episode_id": episode.episode_id,
                    "task_id": task_id,
                    "state": "creating",
                    "creating_episode_ids_count": len(session.creating_episode_ids),
                },
            )

            # Handle automatic dependency resolution (using raw_task.depends_on_task_id)
            # Note: dependency_template is cleared after template expansion, depends_on_task_id is set instead
            # by template_expander.py via direct attribute assignment (bypassing __init__ validation)
            effective_attach_to_episode_id = None
            if raw_task.depends_on_task_id:
                logger.info(  # type: ignore[unreachable]
                    "Task dependency detected",
                    extra={
                        "event": "task_dependency_detected",
                        "session_id": session_id,
                        "episode_id": episode.episode_id,
                        "task_id": task_id,
                        "depends_on_task_id": raw_task.depends_on_task_id,
                    },
                )

                # Get dependency configuration
                dependency_config = self.benchmark_manager.get_dependency_config()

                # Find available episode (with retry logic)
                try:
                    available_episode_id = await self.episode_manager.find_available_episode_for_dependency_with_retry(
                        session_id=session_id,
                        target_task_id=raw_task.depends_on_task_id,
                        dependent_task_id=task_id,
                        max_wait_seconds=dependency_config["wait_seconds"],
                        retry_interval=dependency_config["retry_interval"],
                        max_retry_interval=dependency_config["max_retry_interval"],
                    )
                except ValueError as e:
                    dependency_error = ValueError(f"Dependency validation failed: {e}")
                    self.episode_manager.remove_episode_on_error(episode.episode_id, dependency_error)
                    # Remove from creating list on dependency error
                    session.remove_creating_episode(episode.episode_id)
                    raise ValueError(f"Cannot create episode for task {task_id}: {e}") from e

                if available_episode_id:
                    effective_attach_to_episode_id = available_episode_id
                    self.episode_manager.attach_episode_to_episode(episode.episode_id, available_episode_id)

                    # CRITICAL: Store target episode ID in context for orchestration-aware executors
                    # (inject_prompt, get_target_transcript) to resolve the target episode
                    episode.context[MetadataKeys.ORCHESTRATION_TARGET_EPISODES] = [available_episode_id]

                    logger.info(
                        "Episode attached to dependency",
                        extra={
                            "event": "episode_dependency_attached",
                            "session_id": session_id,
                            "episode_id": episode.episode_id,
                            "dependency_episode_id": available_episode_id,
                            "orchestration_target_episodes": [available_episode_id],
                        },
                    )
                else:
                    dependency_error = ValueError(
                        f"No available episodes with required dependency task_id {raw_task.depends_on_task_id} "
                        f"(waited {dependency_config['wait_seconds']}s)"
                    )
                    self.episode_manager.remove_episode_on_error(episode.episode_id, dependency_error)
                    # Remove from creating list on dependency error
                    session.remove_creating_episode(episode.episode_id)
                    raise ValueError(
                        f"Cannot create episode for task {task_id}: no available episodes with required dependency "
                        f"task_id {raw_task.depends_on_task_id} after waiting {dependency_config['wait_seconds']}s"
                    )

            # Start Docker environment (WITHOUT health checks) - runs in thread pool
            # Use raw_task for environment/executor configuration
            try:
                await asyncio.to_thread(
                    self.execution_manager.configure_for_task_async,
                    episode.episode_id,
                    raw_task,
                    session_id=session_id,
                    target_episode_id=effective_attach_to_episode_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to start Docker environment",
                    extra={
                        "event": "episode_docker_start_failed",
                        "session_id": session_id,
                        "episode_id": episode.episode_id,
                        "task_id": task_id,
                        "error": str(e),
                    },
                )

                # Clean up any partially-created Docker resources (networks, containers, etc.)
                # This is critical when 'docker compose up' fails partway through
                try:
                    logger.info(
                        "Attempting to clean up Docker resources after failed episode creation",
                        extra={
                            "event": "episode_failed_creation_cleanup_started",
                            "episode_id": episode.episode_id,
                            "task_id": task_id,
                        },
                    )
                    await self.execution_manager.cleanup_episode(episode.episode_id)
                    logger.info(
                        "Docker resource cleanup completed after failed episode creation",
                        extra={
                            "event": "episode_failed_creation_cleanup_completed",
                            "episode_id": episode.episode_id,
                            "task_id": task_id,
                        },
                    )
                except Exception as cleanup_error:
                    # Log cleanup failure but don't mask original error
                    logger.warning(
                        "Failed to clean up Docker resources after episode creation failure",
                        extra={
                            "event": "episode_failed_creation_cleanup_failed",
                            "episode_id": episode.episode_id,
                            "task_id": task_id,
                            "cleanup_error": str(cleanup_error),
                            "original_error": str(e),
                        },
                    )

                self.episode_manager.mark_episode_failed_creation(episode.episode_id, str(e))
                self.episode_manager.remove_episode_on_error(episode.episode_id, e)
                # Remove from creating list on error
                session.remove_creating_episode(episode.episode_id)
                raise HTTPException(status_code=500, detail=f"Failed to start Docker environment: {e}") from e

            # Move from creating to active IMMEDIATELY after Docker starts (health checks will run in background)
            session.move_to_active_episode(episode.episode_id)

            logger.info(
                "🔓 Episode creation lock released (Docker environment ready)",
                extra={
                    "event": "episode_creation_lock_released",
                    "episode_id": episode.episode_id,
                    "task_id": task_id,
                    "lock_scope": "global",
                },
            )

        # AFTER LOCK RELEASE: Spawn background task for remaining finalization
        # (prompts, policy, file copies, etc.)
        # Pass raw_task for environment/prompt rendering
        finalization_coro = self._finalize_episode_creation(
            episode_id=episode.episode_id,
            session_id=session_id,
            task_id=task_id,
            task=raw_task,
            attach_to_episode_id=effective_attach_to_episode_id,
        )
        self._schedule_episode_finalization(episode.episode_id, finalization_coro)

        logger.info(
            "Episode initiated (background finalization in progress)",
            extra={
                "event": "episode_initiated",
                "episode_id": episode.episode_id,
                "state": episode.state.value,
            },
        )

        return episode

    def _schedule_episode_finalization(self, episode_id: str, finalization_coro: Any) -> None:
        """
        Register background finalization work with concurrency limits.

        Args:
            episode_id: Episode identifier
            finalization_coro: Coroutine for finalization work
        """

        async def _run() -> None:
            # HIGH FIX: Semaphore controls concurrent health checks to prevent system overload
            # Docker compose operations are already serialized by _episode_creation_lock
            async with self._finalization_semaphore:
                logger.debug(
                    "Episode finalization acquired semaphore slot",
                    extra={
                        "event": "finalization_semaphore_acquired",
                        "episode_id": episode_id,
                    },
                )
                try:
                    await finalization_coro
                finally:
                    self._episode_finalization_tasks.pop(episode_id, None)
                    logger.debug(
                        "Episode finalization released semaphore slot",
                        extra={
                            "event": "finalization_semaphore_released",
                            "episode_id": episode_id,
                        },
                    )

        task = asyncio.create_task(_run(), name=f"episode-finalize-{episode_id}")
        self._episode_finalization_tasks[episode_id] = task
        task.add_done_callback(lambda t: self._handle_finalization_task_done(episode_id, t))

    def _handle_finalization_task_done(self, episode_id: str, task: asyncio.Task[None]) -> None:
        """
        Surface task errors and mark failures when needed.

        Args:
            episode_id: Episode identifier
            task: Asyncio task that finished
        """
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info(
                "Episode finalization task cancelled",
                extra={"event": "episode_finalization_cancelled", "episode_id": episode_id},
            )
        except Exception as e:
            logger.error(
                "Episode finalization crashed",
                extra={
                    "event": "episode_finalization_crashed",
                    "episode_id": episode_id,
                    "error": str(e),
                },
            )
            self.episode_manager.mark_episode_failed_creation(episode_id, f"Finalization crashed: {e}")

    async def _finalize_episode_creation(
        self,
        episode_id: str,
        session_id: str,
        task_id: str,
        task: Task,
        attach_to_episode_id: str | None = None,
    ) -> None:
        """
        Background task that finalizes episode creation after Docker compose.

        Runs OUTSIDE the global lock:
        1. Wait for health checks
        2. Generate prompts
        3. Configure policy
        4. Configure episode manager
        5. Copy initial files
        6. Configure evaluation
        7. Mark episode READY

        Args:
            episode_id: Episode identifier
            session_id: Session identifier
            task_id: Task identifier
            task: Task object
            attach_to_episode_id: Optional attached episode ID
        """
        try:
            logger.info(
                "Episode finalization started (waiting for health checks)",
                extra={
                    "event": "episode_finalization_started",
                    "episode_id": episode_id,
                },
            )

            # Cache the environment check — avoid repeated calls to has_episode_environment()
            has_env = self.execution_manager.has_episode_environment(episode_id)

            # Wait for health checks (runs in parallel with other episodes)
            # Only if this episode has a sandbox environment configured
            if has_env:
                try:
                    health_task = self.execution_manager.wait_for_episode_healthy(
                        episode_id=episode_id, timeout_seconds=180
                    )
                    await asyncio.wait_for(health_task, timeout=200)
                    logger.info(
                        "Episode health checks passed",
                        extra={
                            "event": "episode_health_check_passed",
                            "episode_id": episode_id,
                        },
                    )
                except TimeoutError:
                    error_msg = "Health check exceeded 200s timeout"
                    logger.error(
                        "Episode health check timed out",
                        extra={
                            "event": "episode_health_check_timeout",
                            "episode_id": episode_id,
                            "timeout_seconds": 200,
                        },
                    )
                    self.episode_manager.mark_episode_failed_creation(episode_id, error_msg)
                    await self._cleanup_failed_episode_environment(episode_id)
                    return
                except Exception as e:
                    logger.error(
                        "Episode health check failed",
                        extra={
                            "event": "episode_health_check_failed",
                            "episode_id": episode_id,
                            "error": str(e),
                        },
                    )
                    self.episode_manager.mark_episode_failed_creation(episode_id, f"Health check failed: {e}")
                    await self._cleanup_failed_episode_environment(episode_id)
                    return
            else:
                logger.info(
                    "No sandbox environment for episode - skipping health checks",
                    extra={
                        "event": "episode_health_check_skipped",
                        "episode_id": episode_id,
                    },
                )

            # Generate prompts
            try:
                rendered_prompts = self.benchmark_manager.prompt_generator.render_agent_prompts_for_task(task)
                instruction_prompt = rendered_prompts["instruction"]
                self.policy_manager.set_episode_policy(episode_id, session_id, instruction_prompt)
            except Exception as e:
                logger.error(
                    "Episode prompt generation failed",
                    extra={
                        "event": "episode_prompt_generation_failed",
                        "episode_id": episode_id,
                        "error": str(e),
                    },
                )
                self.episode_manager.mark_episode_failed_creation(episode_id, f"Prompt generation failed: {e}")
                await self._cleanup_failed_episode_environment(episode_id)
                return

            # Configure episode manager
            try:
                await self.episode_manager.configure_for_task(episode_id, task)
            except Exception as e:
                logger.error(
                    "Episode manager configuration failed",
                    extra={
                        "event": "episode_manager_config_failed",
                        "episode_id": episode_id,
                        "error": str(e),
                    },
                )
                self.episode_manager.mark_episode_failed_creation(episode_id, f"Episode configuration failed: {e}")
                await self._cleanup_failed_episode_environment(episode_id)
                return

            # Copy initial files to execution container (after health checks pass)
            # Only if task has initial files AND episode has a sandbox environment
            if task.initial_files and has_env:
                try:
                    await self.execution_manager.copy_initial_files_to_episode(episode_id, task)
                except Exception as e:
                    logger.error(
                        "Initial files copy failed",
                        extra={
                            "event": "episode_initial_files_copy_failed",
                            "episode_id": episode_id,
                            "error": str(e),
                        },
                    )
                    self.episode_manager.mark_episode_failed_creation(episode_id, f"File copy failed: {e}")
                    await self._cleanup_failed_episode_environment(episode_id)
                    return
            elif not has_env:
                logger.warning(
                    "No sandbox environment for episode - skipping initial file copy"
                    " (task has initial_files configured)",
                    extra={
                        "event": "episode_initial_files_copy_skipped_no_environment",
                        "episode_id": episode_id,
                        "initial_files_count": len(task.initial_files) if task.initial_files else 0,
                    },
                )

            # Configure evaluation
            try:
                self.evaluation_manager.configure_for_task(task)
            except Exception as e:
                logger.error(
                    "Evaluation configuration failed",
                    extra={
                        "event": "evaluation_config_failed",
                        "episode_id": episode_id,
                        "error": str(e),
                    },
                )
                self.episode_manager.mark_episode_failed_creation(episode_id, f"Evaluation configuration failed: {e}")
                await self._cleanup_failed_episode_environment(episode_id)
                return

            # Log episode start (non-fatal)
            try:
                await self.evaluation_manager.log_episode_start(session_id, episode_id, task_id)
            except Exception as e:
                logger.warning(
                    "Failed to log episode start (non-fatal)",
                    extra={
                        "event": "episode_start_logging_failed",
                        "episode_id": episode_id,
                        "error": str(e),
                    },
                )

            # Mark episode as READY
            self.episode_manager.mark_episode_ready(episode_id)

            logger.info(
                "Episode finalization completed - episode ready",
                extra={
                    "event": "episode_finalization_completed",
                    "episode_id": episode_id,
                    "state": "ready",
                },
            )

        except asyncio.CancelledError:
            logger.info(
                "Episode finalization cancelled",
                extra={
                    "event": "episode_finalization_cancelled",
                    "episode_id": episode_id,
                },
            )
            self.episode_manager.mark_episode_failed_creation(
                episode_id, "Episode finalization cancelled during shutdown"
            )
            await self._cleanup_failed_episode_environment(episode_id)
            raise
        except Exception as e:
            logger.error(
                "Episode finalization failed with unexpected error",
                extra={
                    "event": "episode_finalization_unexpected_error",
                    "episode_id": episode_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            self.episode_manager.mark_episode_failed_creation(episode_id, f"Unexpected finalization error: {e}")
            await self._cleanup_failed_episode_environment(episode_id)

    async def _cleanup_failed_episode_environment(self, episode_id: str) -> None:
        """
        Best-effort teardown when finalization fails.

        Args:
            episode_id: Episode identifier
        """
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if episode:
            session_id = episode.session_id
            session = self.active_sessions.get(session_id)
            if session:
                session.remove_active_episode(episode_id)
                logger.info(
                    "Removed failed episode from session active list",
                    extra={
                        "event": "failed_episode_removed_from_session",
                        "episode_id": episode_id,
                        "session_id": session_id,
                    },
                )

        try:
            await self.execution_manager.cleanup_episode(episode_id)
            logger.info(
                "Successfully cleaned up failed episode environment",
                extra={
                    "event": "episode_failure_cleanup_success",
                    "episode_id": episode_id,
                },
            )
        except Exception as exc:
            # HIGH FIX: Escalate to error (not warning) since this leaves orphaned resources
            logger.error(
                "Episode teardown after failure encountered errors - resources may be orphaned",
                extra={
                    "event": "episode_failure_cleanup_error",
                    "episode_id": episode_id,
                    "error": str(exc),
                },
            )

            # Fallback: Try force cleanup via direct docker compose down
            try:
                import subprocess

                # Get the sandbox environment manager to find the compose project name
                if (
                    hasattr(self.execution_manager, "_sandbox_environment_manager")
                    and self.execution_manager._sandbox_environment_manager
                ):
                    project_name = f"saber-episode-{episode_id}"
                    logger.info(
                        "Attempting fallback cleanup with docker compose down",
                        extra={
                            "event": "episode_fallback_cleanup_attempt",
                            "episode_id": episode_id,
                            "project_name": project_name,
                        },
                    )
                    result = await asyncio.to_thread(
                        subprocess.run,
                        ["docker", "compose", "-p", project_name, "down", "-v", "--remove-orphans", "-t", "1"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        logger.info(
                            "Fallback cleanup succeeded",
                            extra={
                                "event": "episode_fallback_cleanup_success",
                                "episode_id": episode_id,
                            },
                        )
                    else:
                        logger.error(
                            "Fallback cleanup failed",
                            extra={
                                "event": "episode_fallback_cleanup_failed",
                                "episode_id": episode_id,
                                "stderr": result.stderr,
                            },
                        )
            except Exception as fallback_exc:
                logger.error(
                    "Fallback cleanup also failed - manual intervention required",
                    extra={
                        "event": "episode_fallback_cleanup_exception",
                        "episode_id": episode_id,
                        "error": str(fallback_exc),
                    },
                )

    def get_episode_status(self, episode_id: str) -> Episode | None:
        """
        Get episode for status checking.

        Args:
            episode_id: Episode identifier

        Returns:
            Episode object or None if not found
        """
        return self.episode_manager.get_episode_by_id(episode_id)

    def get_benchmark_info(self) -> BenchmarkInfo:
        """
        Get complete benchmark information for client-side orchestration.

        Returns:
            BenchmarkInfo object with all tasks and their episode attempt configurations
        """
        return self.benchmark_manager.get_benchmark_info()

    async def end_episode(
        self,
        session_id: str,
        episode_id: str,
        reason: str = EpisodeTerminationReason.COMPLETED,
        submission: EvalSubmission | None = None,
        cascade_end_attached_episodes: bool = False,
    ) -> EpisodeEndResponse:
        """
        End a specific episode for a session with evaluation.

        This method is IDEMPOTENT - it can be safely called multiple times for the same episode.
        If the episode is already completed, it returns a success response with the existing state.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode to end
            reason: Reason for episode termination (use EpisodeTerminationReason enum values)
            submission: Required submission for evaluation
            cascade_end_attached_episodes: If True, also end episodes that this episode is attached to

        Returns:
            EpisodeEndResponse with evaluation result

        Raises:
            HTTPException: If episode not found or session invalid
            EvaluationError: If evaluation fails
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Get episode first to check its state
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

        # IDEMPOTENT HANDLING: Check if episode is already complete
        if episode.is_complete:
            logger.info(
                "Episode already ended - returning existing state (idempotent)",
                extra={
                    "event": "episode_end_idempotent",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "existing_state": episode.state.value,
                    "existing_reason": episode.completion_reason,
                    "requested_reason": reason,
                },
            )

            # Return success response with existing episode state
            success = episode.state == EpisodeState.COMPLETED
            return EpisodeEndResponse(
                episode_ended=True,
                episode_id=episode_id,
                success=success,
                reason=episode.completion_reason or reason,
                previous_task_id=episode.task_id,
                active_episodes_remaining=len(session.active_episode_ids),
                evaluation_result=None,  # Client-side evaluation
            )

        # Validate episode is active in this session (only if not already complete)
        if episode_id not in session.active_episode_ids:
            # Check if it's in the history (completed but we didn't catch it above)
            if episode_id in session.episode_history:
                logger.info(
                    "Episode in history but not marked complete in manager - treating as already ended",
                    extra={
                        "event": "episode_end_in_history",
                        "session_id": session_id,
                        "episode_id": episode_id,
                    },
                )
                return EpisodeEndResponse(
                    episode_ended=True,
                    episode_id=episode_id,
                    success=True,
                    reason="previously_completed",
                    previous_task_id=episode.task_id,
                    active_episodes_remaining=len(session.active_episode_ids),
                    evaluation_result=None,
                )

            raise HTTPException(
                status_code=400,
                detail=f"Episode {episode_id} is not active in session {session_id}. "
                f"Active episodes: {session.active_episode_ids}",
            )

        task = self.benchmark_manager.get_task(episode.task_id)

        # Handle submission - it's now optional even for successful completion
        # Some workflows may end episodes without explicit submissions
        if submission is not None:
            # Store the EvalSubmission object for rich evaluation data
            episode.eval_submission = submission
            # Extract the submission text for the episode.submission field
            episode.submission = submission.submission
        else:
            # No submission provided - set default based on reason
            episode.eval_submission = None
            if reason in ["completed", "agent_completed", "success"]:
                episode.submission = "Episode completed without explicit submission"
            else:
                episode.submission = f"Episode ended: {reason}"

        # Move episode from active to history IMMEDIATELY to prevent session termination override
        session.complete_episode(episode_id)
        logger.info(
            "Episode removed from active_episode_ids (client end_episode call)",
            extra={
                "event": "episode_removed_from_active_client_end",
                "session_id": session_id,
                "episode_id": episode_id,
                "reason": reason,
                "remaining_active_episodes": len(session.active_episode_ids),
                "active_episode_ids": session.active_episode_ids,
            },
        )

        # End episode through episode manager (pass submission text, not EvalSubmission object)
        submission_text = submission.submission if submission else None
        completed_episode = await self.episode_manager.end_episode(episode_id, reason, submission_text)

        # CLIENT-SIDE EVALUATION: Server does not evaluate episodes
        # Evaluation will be submitted separately by the client via:
        # POST /api/v1/session/{session_id}/episodes/{episode_id}/evaluation

        logger.info(
            "Episode ended (evaluation pending client submission)",
            extra={
                "event": "episode_ended_pending_evaluation",
                "session_id": session_id,
                "episode_id": episode_id,
                "task_id": task.task_id,
                "reason": reason,
            },
        )

        # Cleanup episode containers immediately when episode ends
        try:
            logger.info(
                "Starting episode cleanup",
                extra={
                    "event": "episode_cleanup_start",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "termination_reason": reason,
                },
            )
            try:
                episode_cleanup = await self.execution_manager.cleanup_episode(
                    episode_id, {"episode_end_reason": reason}
                )
                if episode_cleanup:
                    logger.info(
                        "Episode cleanup completed",
                        extra={
                            "event": "episode_cleanup_complete",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )
                else:
                    logger.warning(
                        "Episode cleanup reported failure",
                        extra={
                            "event": "episode_cleanup_failed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )
            except Exception as e:
                # Don't let compose cleanup failure prevent network cleanup
                logger.error(
                    "Episode compose cleanup error, continuing with network cleanup",
                    extra={
                        "event": "episode_compose_cleanup_error",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

            # Additional cleanup: ensure episode network is removed to prevent subnet pool exhaustion
            # This MUST run even if compose cleanup fails above
            # Use retry logic to handle transient failures
            await self._cleanup_episode_network_with_retry(episode_id)
        except Exception as e:
            logger.error(
                "Episode cleanup error",
                extra={
                    "event": "episode_cleanup_error",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

        logger.info(
            "Episode ended",
            extra={
                "event": "episode_ended",
                "session_id": session_id,
                "episode_id": episode_id,
                "reason": reason,
            },
        )

        # Handle cascade termination of attached episodes if requested
        if cascade_end_attached_episodes and completed_episode.attached_to_episode_id:
            attached_episode_id = completed_episode.attached_to_episode_id
            logger.info(
                "Cascade termination requested",
                extra={
                    "event": "episode_cascade_termination_requested",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "attached_episode_id": attached_episode_id,
                },
            )

            try:
                # Check if the attached episode is still active
                attached_episode = self.episode_manager.get_episode_by_id(attached_episode_id)
                if attached_episode and not attached_episode.is_complete:
                    # End the attached episode with cascade completion reason
                    cascade_reason = f"cascade_completed_by_{episode_id}"
                    await self.end_episode(
                        session_id=session_id,
                        episode_id=attached_episode_id,
                        reason=cascade_reason,
                        submission=None,  # Attached episodes don't get submissions from dependent episodes
                        cascade_end_attached_episodes=False,  # Prevent infinite recursion
                    )
                    logger.info(
                        "Cascade-ended attached episode",
                        extra={
                            "event": "episode_cascade_termination_complete",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "attached_episode_id": attached_episode_id,
                            "cascade_reason": cascade_reason,
                        },
                    )
                else:
                    logger.info(
                        "Attached episode already complete; skipping cascade termination",
                        extra={
                            "event": "episode_cascade_skip",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "attached_episode_id": attached_episode_id,
                        },
                    )
            except Exception as e:
                logger.error(
                    "Failed to cascade-end attached episode",
                    extra={
                        "event": "episode_cascade_termination_failed",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "attached_episode_id": attached_episode_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                # Don't fail the main episode end operation due to cascade failures

        # Build response - success based on completion reason (client will submit evaluation separately)
        # Success is True only for explicit success/completion reasons
        success = reason in ["completed", "agent_completed", "success"]

        return EpisodeEndResponse(
            episode_ended=True,
            episode_id=episode_id,
            success=success,
            reason=reason,
            previous_task_id=episode.task_id,
            active_episodes_remaining=len(session.active_episode_ids),
            evaluation_result=None,  # Client-side evaluation - will be submitted separately
        )

    async def execute_action(self, session_id: str, episode_id: str, action: Action) -> CommandResult:
        """
        Execute action for MCP integration.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode to execute action against
            action: Action object containing command and parameters

        Returns:
            CommandResult with execution results

        Raises:
            HTTPException: If episode is not active (404) or not in correct state
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Validate episode is active in this session - raise HTTP 404 for proper error handling
        if episode_id not in session.active_episode_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Episode {episode_id} is not active in session {session_id}. "
                f"Active episodes: {session.active_episode_ids}",
            )

        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found in episode manager")

        if episode.state == EpisodeState.CREATING:
            raise HTTPException(
                status_code=409,  # Conflict - resource exists but not ready
                detail=f"Episode {episode_id} is still being created (background finalization in progress). "
                "Please retry in a few seconds.",
            )
        elif episode.state == EpisodeState.FAILED_CREATION:
            raise HTTPException(
                status_code=500,  # Internal server error - episode creation failed
                detail=f"Episode {episode_id} failed creation: {episode.creation_error or 'unknown error'}",
            )
        elif episode.state != EpisodeState.READY and episode.state != EpisodeState.ACTIVE:
            raise HTTPException(
                status_code=409,  # Conflict - episode in wrong state
                detail=f"Episode {episode_id} is not ready (state: {episode.state.value})",
            )

        try:
            log_operation_start(logger, "Action execution", session_id, tool=action.tool_name, episode=episode_id)

            context = {"session_id": session_id, "episode_id": episode_id}
            command_result = await self.execution_manager.step(action, context)

            if command_result.success:
                log_operation_success(logger, "Action execution", session_id, tool=action.tool_name)
            else:
                log_operation_failure(logger, "Action execution", command_result.error or "Unknown error", session_id)

            step_result = self.episode_manager.step(episode_id, action, command_result)

            # Log action with evaluation manager (ignore failures)
            try:
                await self.evaluation_manager.log_action(
                    session_id=session_id, episode_id=episode_id, action=action, result=command_result
                )
            except Exception as e:
                logger.warning(
                    "Failed to log action telemetry",
                    extra={
                        "event": "episode_action_logging_failed",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "tool": action.tool_name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

            # End episode if step indicates completion OR if EpisodeManager indicates termination
            if step_result.step.done:
                logger.info(
                    "Episode auto-terminating due to step.done=True",
                    extra={
                        "event": "episode_auto_terminate_step_done",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "step_number": len(episode.steps),
                    },
                )
                await self.episode_manager.end_episode(episode_id, EpisodeTerminationReason.COMPLETED)
                session.complete_episode(episode_id)
                logger.info(
                    "Episode removed from active_episode_ids (step.done)",
                    extra={
                        "event": "episode_removed_from_active_step_done",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "remaining_active_episodes": len(session.active_episode_ids),
                    },
                )
                # Add termination metadata to command result
                if not hasattr(command_result, "metadata") or command_result.metadata is None:
                    command_result.metadata = {}
                command_result.metadata["episode_terminated"] = True
                command_result.metadata["termination_reason"] = EpisodeTerminationReason.COMPLETED
                try:
                    await self.evaluation_manager.log_episode_end(session_id, EpisodeTerminationReason.COMPLETED)
                except Exception as e:
                    logger.warning(
                        "Failed to log episode completion",
                        extra={
                            "event": "episode_end_logging_failed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "termination_reason": EpisodeTerminationReason.COMPLETED,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
            elif step_result.should_terminate:
                termination_reason = step_result.termination_reason or EpisodeTerminationReason.TERMINATED
                logger.info(
                    "Episode auto-terminating due to should_terminate=True",
                    extra={
                        "event": "episode_auto_terminate_should_terminate",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "termination_reason": termination_reason,
                        "step_number": len(episode.steps),
                    },
                )
                await self.episode_manager.end_episode(episode_id, termination_reason)
                session.complete_episode(episode_id)
                logger.info(
                    "Episode removed from active_episode_ids (should_terminate)",
                    extra={
                        "event": "episode_removed_from_active_should_terminate",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "termination_reason": termination_reason,
                        "remaining_active_episodes": len(session.active_episode_ids),
                    },
                )
                # Add termination metadata to command result
                if not hasattr(command_result, "metadata") or command_result.metadata is None:
                    command_result.metadata = {}
                command_result.metadata["episode_terminated"] = True
                command_result.metadata["termination_reason"] = termination_reason
                try:
                    await self.evaluation_manager.log_episode_end(session_id, termination_reason)
                except Exception as e:
                    logger.warning(
                        "Failed to log episode termination",
                        extra={
                            "event": "episode_end_logging_failed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "termination_reason": termination_reason,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                logger.info(
                    "Episode ended due to termination condition",
                    extra={
                        "event": "episode_terminated_by_condition",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "termination_reason": termination_reason,
                    },
                )

            return command_result

        except Exception as e:
            logger.error(
                "Command execution failed",
                extra={
                    "event": "episode_command_execution_failed",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "tool": action.tool_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            # Remove episode tracking and trigger immediate cleanup on any error
            try:
                self.episode_manager.remove_episode_on_error(episode_id, e)
                session.remove_active_episode(episode_id)
                logger.info(
                    "Episode removed from tracking due to error",
                    extra={
                        "event": "episode_removed_after_error",
                        "session_id": session_id,
                        "episode_id": episode_id,
                    },
                )

                # Trigger immediate container cleanup through ExecutionManager
                cleanup_success = self.execution_manager.cleanup_session(
                    session_id, {"error": str(e), "error_type": type(e).__name__}
                )
                if cleanup_success:
                    logger.info(
                        "Error-triggered container cleanup completed",
                        extra={
                            "event": "session_error_cleanup_complete",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )
                else:
                    logger.warning(
                        "Error-triggered container cleanup reported failure",
                        extra={
                            "event": "session_error_cleanup_failed",
                            "session_id": session_id,
                            "episode_id": episode_id,
                        },
                    )

            except Exception as cleanup_error:
                logger.error(
                    "Failed to handle error cleanup",
                    extra={
                        "event": "session_error_cleanup_exception",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "error": str(cleanup_error),
                        "error_type": type(cleanup_error).__name__,
                    },
                )

            return CommandResult.error_result(error=str(e))

    async def execute_episode_action(self, session_id: str, episode_id: str, action: Action) -> CommandResult:
        """
        Execute action in context of specific episode.

        This is an alias for execute_action to match the implementation plan naming.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode to execute action against
            action: Action object containing command and parameters

        Returns:
            CommandResult with execution results
        """
        return await self.execute_action(session_id, episode_id, action)

    async def get_current_task(self, session_id: str, episode_id: str) -> Task:
        """
        Get task object for a specific episode.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode

        Returns:
            Task object for the specified episode
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Validate episode is active in this session
        if episode_id not in session.active_episode_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Episode {episode_id} is not active in session {session_id}. "
                f"Active episodes: {session.active_episode_ids}",
            )

        # Get episode and task info from episode manager
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            raise HTTPException(status_code=400, detail=f"Episode {episode_id} not found")

        task = self.benchmark_manager.get_task(episode.task_id)
        return task

    def get_policy(self, session_id: str, episode_id: str) -> PolicyDocument:
        """
        Get domain policy document for a specific episode.

        Args:
            session_id: ID of the client session
            episode_id: ID of the episode to get policy for

        Returns:
            PolicyDocument for the domain configured for the episode's task
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Validate episode exists and belongs to session
        if episode_id not in session.active_episode_ids and episode_id not in session.episode_history:
            raise ValueError(f"Episode {episode_id} not found in session {session_id}")

        # Get episode-specific policy that was configured during start_episode
        return self.policy_manager.get_policy(episode_id)

    async def list_session_episodes(self, session_id: str, status_filter: str | None = None) -> list[Episode]:
        """
        List all episodes for a session with optional status filtering.

        Args:
            session_id: ID of the client session
            status_filter: Optional filter by episode status

        Returns:
            List of episode objects for the session
        """
        session = self._get_session(session_id)
        session.update_activity()

        episodes = self.episode_manager.get_session_episodes(session_id)

        if status_filter:
            episodes = [ep for ep in episodes if ep.state.value == status_filter]

        return episodes

    async def get_episode_details(self, session_id: str, episode_id: str) -> Episode:
        """
        Get detailed information about a specific episode.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode

        Returns:
            Episode object with full details

        Raises:
            HTTPException: If episode not found or doesn't belong to session
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Validate episode belongs to session
        if episode_id not in session.active_episode_ids and episode_id not in session.episode_history:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found in session {session_id}")

        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

        return episode

    def get_evaluation_service(self) -> SessionEvaluationService:
        """
        Get evaluation service instance for retrieval operations.

        Returns:
            SessionEvaluationService instance
        """
        return self.evaluation_service

    # ============================================================================
    # CLIENT-SIDE EVALUATION: get_evaluation_criteria() REMOVED
    # Replaced by separate endpoints:
    # - get_episode_submission() - for submission data
    # - get_episode_steps() - for step history
    # - get_submission_evaluation_criteria() - for submission eval config
    # - get_step_evaluation_criteria() - for step eval config
    # Templates are now served as raw files, not rendered server-side
    # ============================================================================

    async def save_evaluation_file(self, session_id: str, file: UploadFile) -> int:
        """
        Save external evaluation file (.eval) to session directory.

        Args:
            session_id: ID of the client session
            file: UploadFile containing the .eval file

        Returns:
            int: File size in bytes

        Raises:
            HTTPException: If session doesn't exist (404) or file save fails (500)
        """
        logger.info(
            "Saving evaluation file",
            extra={
                "event": "evaluation_file_save_start",
                "session_id": session_id,
                "filename": file.filename,
            },
        )

        # Validate session exists (raises HTTPException if not found)
        self._get_session(session_id)

        try:
            # Get the evaluation store base path
            # Type-safe access to base_path - check if it's a file-based store
            if hasattr(self.evaluation_manager.store, "base_path"):
                store_base_path = Path(self.evaluation_manager.store.base_path)
            else:
                # Fallback for stores without base_path (though none currently exist)
                raise ValueError("Evaluation store does not support file uploads")
            session_dir = store_base_path / session_id

            # Create session directory if it doesn't exist
            session_dir.mkdir(parents=True, exist_ok=True)

            # Save file to session directory
            filename = file.filename
            if filename is None:
                raise ValueError("File must have a filename")
            file_path = session_dir / filename

            # Read file content
            file_content = await file.read()
            file_size = len(file_content)

            # Write file to disk
            with open(file_path, "wb") as f:
                f.write(file_content)

            logger.info(
                "Evaluation file saved",
                extra={
                    "event": "evaluation_file_save_complete",
                    "session_id": session_id,
                    "filename": file.filename,
                    "file_size_bytes": file_size,
                    "path": str(file_path),
                },
            )
            return file_size

        except Exception as e:
            logger.error(
                "Failed to save evaluation file",
                extra={
                    "event": "evaluation_file_save_failed",
                    "session_id": session_id,
                    "filename": file.filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise RuntimeError(f"Failed to save evaluation file: {str(e)}") from e

    # ============================================================================
    # CLIENT-SIDE EVALUATION: override_episode_evaluation() REMOVED
    # Evaluation results are now submitted via POST /api/v1/session/{session_id}/episodes/{episode_id}/evaluation
    # Server stores results without validation or override logic
    # ============================================================================

    def get_episode_by_id(self, episode_id: str) -> Episode | None:
        """
        Get episode by ID - used by MCP API for episode context.

        Args:
            episode_id: ID of the episode to retrieve

        Returns:
            Episode object if found, None otherwise
        """
        return self.episode_manager.get_episode_by_id(episode_id)

    def _get_session(self, session_id: str) -> ClientSession:
        """
        Get session by ID with validation.

        Args:
            session_id: ID of the session to retrieve

        Returns:
            ClientSession object

        Raises:
            HTTPException: If session not found or inactive
        """
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = self.active_sessions[session_id]
        if not session.is_active:
            raise HTTPException(status_code=400, detail="Session is not active")

        return session

    async def _session_cleanup_loop(self) -> None:
        """Periodic cleanup of inactive sessions."""
        logger.info(
            "Starting session cleanup loop",
            extra={
                "event": "session_cleanup_loop_start",
                "session_timeout_minutes": self.session_timeout_minutes,
                "cleanup_interval_minutes": self.cleanup_interval_minutes,
            },
        )

        while not self.shutdown_event.is_set():
            try:
                await self._cleanup_inactive_sessions()

                # Wait for cleanup interval or shutdown signal
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=self.cleanup_interval_minutes * 60)
                    # If shutdown event is set, exit the loop
                    break
                except TimeoutError:
                    # Timeout is expected - continue with next cleanup cycle
                    continue

            except Exception as e:
                logger.error(
                    "Session cleanup loop error",
                    extra={
                        "event": "session_cleanup_loop_error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                # Wait a bit before retrying to avoid tight error loops
                await asyncio.sleep(30)

    async def _cleanup_inactive_sessions(self) -> None:
        """Check for and cleanup inactive sessions."""
        current_time = self._time_source.now()
        timeout_threshold = timedelta(minutes=self.session_timeout_minutes)
        sessions_to_cleanup = []

        for session_id, session in self.active_sessions.items():
            time_since_activity = current_time - session.last_activity

            if time_since_activity > timeout_threshold:
                logger.info(
                    "Session timed out due to inactivity",
                    extra={
                        "event": "session_timeout_detected",
                        "session_id": session_id,
                        "inactivity_seconds": round(time_since_activity.total_seconds(), 1),
                        "timeout_threshold_minutes": self.session_timeout_minutes,
                    },
                )
                sessions_to_cleanup.append(session_id)

        # Cleanup identified sessions
        for session_id in sessions_to_cleanup:
            try:
                cleanup_logger.info(
                    "Cleaning up inactive session due to timeout",
                    extra={
                        "event": "inactive_session_cleanup",
                        "session_id": session_id,
                        "timeout_minutes": self.session_timeout_minutes,
                    },
                )
                await self.terminate_session(session_id)
                logger.info(
                    "Inactive session cleanup complete",
                    extra={
                        "event": "session_cleanup_complete",
                        "session_id": session_id,
                    },
                )
            except Exception as e:
                logger.error(
                    "Error cleaning up inactive session",
                    extra={
                        "event": "session_cleanup_error",
                        "session_id": session_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics about active sessions."""
        current_time = self._time_source.now()
        stats: dict[str, Any] = {
            "total_sessions": len(self.active_sessions),
            "timeout_minutes": self.session_timeout_minutes,
            "cleanup_interval_minutes": self.cleanup_interval_minutes,
            "sessions": [],
        }

        for session_id, session in self.active_sessions.items():
            time_since_activity = current_time - session.last_activity
            episode_counts = session.get_episode_count()
            stats["sessions"].append(
                {
                    "session_id": session_id,
                    "client_id": session.client_id,
                    "active_episode_ids": session.active_episode_ids,
                    "episode_history": session.episode_history,
                    "episode_counts": episode_counts,
                    "task_queue": session.task_queue,
                    "uptime_seconds": (current_time - session.created_at).total_seconds(),
                    "time_since_activity_seconds": time_since_activity.total_seconds(),
                    "is_active": session.is_active,
                }
            )

        return stats

    async def _start_permanent_environment_async(self) -> None:
        """Start permanent environment in background, tracking status for health endpoint."""
        permanent_env_name = self.benchmark_manager.config_loader.get_permanent_environment()
        if not permanent_env_name:
            logger.info(
                "No permanent environment configured",
                extra={"event": "permanent_environment_not_configured"},
            )
            self._permanent_env_startup_complete = True
            return

        try:
            logger.info(
                "Starting permanent environment (background)",
                extra={
                    "event": "permanent_environment_start",
                    "environment": permanent_env_name,
                },
            )
            print(f"[SERVER] Starting permanent environment '{permanent_env_name}' in background...", flush=True)

            # Build path to permanent environment compose file
            from pathlib import Path

            permanent_compose_path = (
                Path(self.config_dir) / "environments" / "permanent" / f"{permanent_env_name}.compose.yml"
            )

            if not permanent_compose_path.exists():
                raise RuntimeError(f"Permanent environment compose file not found: {permanent_compose_path}")

            # Start permanent environment directly through PermanentEnvironmentManager
            # This is a blocking call that waits for all services to be healthy
            if not self.execution_manager._permanent_environment_manager:
                raise RuntimeError("Permanent environment manager is not initialized")

            # Run the blocking compose startup in a thread pool to avoid blocking the event loop
            import asyncio

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.execution_manager._permanent_environment_manager.start_permanent_environment_from_file,
                permanent_compose_path,
            )

            self._permanent_env_startup_complete = True
            print(f"[SERVER] ✓ Permanent environment '{permanent_env_name}' is healthy", flush=True)
            logger.info(
                "Permanent environment started",
                extra={
                    "event": "permanent_environment_started",
                    "environment": permanent_env_name,
                    "compose_path": str(permanent_compose_path),
                },
            )

        except Exception as e:
            self._permanent_env_startup_error = str(e)
            self._permanent_env_startup_complete = True  # Mark complete (with error)
            print(f"[SERVER] ✗ Permanent environment startup failed: {e}", flush=True)
            logger.error(
                "Failed to start permanent environment",
                extra={
                    "event": "permanent_environment_start_failed",
                    "environment": permanent_env_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Don't re-raise - let health endpoint report the error

    async def _start_permanent_environment(self) -> None:
        """Legacy sync wrapper - deprecated, use _start_permanent_environment_async."""
        await self._start_permanent_environment_async()
