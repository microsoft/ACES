"""
SessionManager implementation for SABER domain server.

The SessionManager serves as the central orchestrator for managing client sessions
and coordinating all server components. REST API functionality is handled by SessionAPI.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from ..models import BenchmarkInfo, EpisodeEndResponse, EvalSubmission
from ..models.rest.evaluation import (
    EvaluationCriteriaResponse,
    EvaluationOverrideRequest,
    JudgeMessages,
    TaskEvaluationContext,
)
from .api.session_mcp_api import SessionMCPAPI
from .api.session_rest_api import SessionRestAPI
from .base import Action, CommandResult, Episode
from .benchmarks.benchmark_manager import BenchmarkManager
from .benchmarks.task import Task
from .episodes.constants import EpisodeTerminationReason
from .episodes.episode_manager import EpisodeManager
from .evaluation.evaluation_manager import EvaluationManager
from .evaluation.models import EvaluationResult
from .evaluation.session_evaluation_service import SessionEvaluationService

# CleanupReason removed - using direct component cleanup
from .execution.execution_manager import ExecutionManager
from .policy.policy_manager import PolicyDocument, PolicyManager

logger = get_session_manager_logger(__name__)
cleanup_logger = get_cleanup_logger(__name__)


class ClientSession(BaseModel):
    """Represents an active client session with support for multiple episodes."""

    model_config = ConfigDict()

    session_id: str = Field(..., description="Unique session identifier")
    client_id: str = Field(..., description="Client identifier")

    # Multi-episode support
    active_episode_ids: List[str] = Field(default_factory=list, description="Currently active episode IDs")
    episode_history: List[str] = Field(
        default_factory=list, description="All completed episode IDs in chronological order"
    )

    # Task orchestration support
    task_queue: List[str] = Field(default_factory=list, description="Queued tasks for orchestration")

    # Session metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    is_active: bool = Field(default=True, description="Whether session is active")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context data")

    @field_serializer("created_at", "last_activity")
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime fields to ISO format."""
        return value.isoformat()

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.utcnow()

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

    def get_next_task(self) -> Optional[str]:
        """Get and remove the next task from the queue."""
        if self.task_queue:
            task_id = self.task_queue.pop(0)
            self.update_activity()
            return task_id
        return None

    def has_active_episodes(self) -> bool:
        """Check if session has any active episodes."""
        return len(self.active_episode_ids) > 0

    def get_episode_count(self) -> Dict[str, int]:
        """Get episode counts for analytics."""
        return {
            "active": len(self.active_episode_ids),
            "completed": len(self.episode_history),
            "total": len(self.active_episode_ids) + len(self.episode_history),
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
        manifest: Optional[Dict[str, Any]] = None,
        manifest_path: Optional[str] = None,
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
        """
        self.domain_name = domain_name
        self.config_dir = config_dir
        self.host = host
        self.port = port
        self.mcp_host = mcp_host
        self.mcp_port = mcp_port
        self.session_timeout_minutes = session_timeout_minutes
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.active_sessions: Dict[str, ClientSession] = {}
        self.cleanup_task: Optional[asyncio.Task[None]] = None
        self.shutdown_event = asyncio.Event()

        # Manifest information for health endpoints
        self.manifest = manifest or {}
        self.manifest_path = manifest_path

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
        self.episode_manager = EpisodeManager()

        # Initialize execution manager first
        self.execution_manager = ExecutionManager(config_dir)

        # Initialize permanent environment manager through ExecutionManager for unified container lifecycle
        permanent_config = {
            "domain": domain_name,
            "config_dir": config_dir,
            "logs_dir": str(Path(config_dir).parent / "logs"),  # Logs go to server/logs, not server/config/logs
            "enable_logging": True,
        }
        self.execution_manager.initialize_permanent_environment_manager(permanent_config)

        self.policy_manager = PolicyManager(domain_name)
        self.evaluation_manager = EvaluationManager()

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

    def get_health_metadata(self) -> Dict[str, Any]:
        """Get health metadata including manifest information."""
        import hashlib
        import os
        from pathlib import Path

        metadata: Dict[str, Any] = {
            "status": "healthy",
            "domain": self.domain_name,
        }

        # Add manifest information if available
        if self.manifest:
            domain_info = self.manifest.get("domain", {})
            manifest_data: Dict[str, Any] = {
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
        build_metadata: Dict[str, str] = {}
        for env_var in ["GIT_SHA", "BUILD_TIMESTAMP", "IMAGE_TAG"]:
            value = os.getenv(env_var)
            if value:
                build_metadata[env_var.lower()] = value

        if build_metadata:
            metadata["build_metadata"] = build_metadata

        # Check permanent environment health if configured
        perm_env_health = self._check_permanent_environment_health()
        if not perm_env_health["healthy"]:
            metadata["status"] = "unhealthy"
            metadata["permanent_environment_error"] = perm_env_health["error"]

        metadata["permanent_environment"] = perm_env_health

        return metadata

    def _check_permanent_environment_health(self) -> Dict[str, Any]:
        """Check permanent environment health using ComposeHealthChecker.

        Returns:
            Dict with health status and details
        """
        # Check if permanent environment is configured
        permanent_env_name = self.benchmark_manager.config_loader.get_permanent_environment()
        if not permanent_env_name:
            return {"healthy": True, "status": "not_configured", "message": "No permanent environment configured"}

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

    async def start_server(self) -> None:
        """Start both REST and MCP servers concurrently with session cleanup."""
        import asyncio

        # Start permanent environment if configured
        await self._start_permanent_environment()

        # Start the session cleanup task
        self.cleanup_task = asyncio.create_task(self._session_cleanup_loop())

        # Start both servers concurrently using asyncio tasks
        rest_task = asyncio.create_task(self.rest_api.start_server())
        mcp_task = asyncio.create_task(self.mcp_api.start_mcp_server())

        # Wait for both to complete (they run indefinitely)
        await asyncio.gather(rest_task, mcp_task, self.cleanup_task)

    async def shutdown(self) -> None:
        """Shutdown the SessionManager and cleanup resources."""
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
                self.execution_manager.stop_permanent_environment()
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

        # Clean up SABER episode networks before shutting down MCP server
        await self._cleanup_saber_episode_networks()

        # Shutdown MCP server first
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

    async def terminate_session(self, session_id: str) -> None:
        """
        Terminate a client session and cleanup resources.

        Args:
            session_id: ID of the session to terminate
        """
        log_session_end(logger, session_id, "SessionManager termination")
        session = self._get_session(session_id)

        # End all active episodes (this will trigger individual episode cleanup)
        if session.has_active_episodes():
            try:
                logger.info(
                    "Terminating active episodes during session shutdown",
                    extra={
                        "event": "session_active_episodes_terminating",
                        "session_id": session_id,
                        "active_episode_count": len(session.active_episode_ids),
                        "active_episode_ids": list(session.active_episode_ids),
                    },
                )
                # End all active episodes - each will trigger its own cleanup
                for episode_id in session.active_episode_ids.copy():  # Copy to avoid modification during iteration
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
                        self.episode_manager.end_episode(episode_id, EpisodeTerminationReason.SESSION_TERMINATED, None)
                        logger.info(
                            "Episode terminated during session shutdown",
                            extra={
                                "event": "session_episode_terminated",
                                "session_id": session_id,
                                "episode_id": episode_id,
                                "termination_reason": EpisodeTerminationReason.SESSION_TERMINATED,
                            },
                        )
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

                # Clear remaining state
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

        # Check for any orphaned episodes that might still need cleanup
        try:
            log_operation_start(logger, "Orphaned episode cleanup check", session_id)
            episodes_to_cleanup = list(session.episode_history)
            logger.info(
                "Checking for orphaned episodes",
                extra={
                    "event": "orphaned_episode_cleanup_check",
                    "session_id": session_id,
                    "episode_ids": episodes_to_cleanup,
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

        Args:
            session_id: ID of the client session
            task_id: ID of the task to start

        Returns:
            Episode object for the started episode
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Get the task object to access its configuration
        task = self.benchmark_manager.get_task(task_id)

        # Start episode through episode manager (pass task for dependency tracking)
        episode = self.episode_manager.start_episode(
            session_id=session_id, task_id=task_id, initial_context=task.initial_context.copy(), task=task
        )

        # Handle automatic dependency resolution if task specifies depends_on_task_id
        effective_attach_to_episode_id = None
        if task.depends_on_task_id:
            logger.info(
                "Task dependency detected",
                extra={
                    "event": "task_dependency_detected",
                    "session_id": session_id,
                    "episode_id": episode.episode_id,
                    "task_id": task_id,
                    "depends_on_task_id": task.depends_on_task_id,
                },
            )

            # Get dependency configuration from benchmark manager
            dependency_config = self.benchmark_manager.get_dependency_config()

            # Find available episode with the required task_id (with retry logic)
            try:
                available_episode_id = await self.episode_manager.find_available_episode_for_dependency_with_retry(
                    session_id=session_id,
                    target_task_id=task.depends_on_task_id,
                    dependent_task_id=task_id,
                    max_wait_seconds=dependency_config["wait_seconds"],
                    retry_interval=dependency_config["retry_interval"],
                    max_retry_interval=dependency_config["max_retry_interval"],
                )
            except ValueError as e:
                # Circular dependency or other validation error
                dependency_error = ValueError(f"Dependency validation failed: {e}")
                self.episode_manager.remove_episode_on_error(episode.episode_id, dependency_error)
                raise ValueError(f"Cannot create episode for task {task_id}: {e}")

            if available_episode_id:
                effective_attach_to_episode_id = available_episode_id
                # Attach the episodes at the episode manager level
                self.episode_manager.attach_episode_to_episode(episode.episode_id, available_episode_id)
                logger.info(
                    "Episode attached to dependency",
                    extra={
                        "event": "episode_dependency_attached",
                        "session_id": session_id,
                        "episode_id": episode.episode_id,
                        "dependency_episode_id": available_episode_id,
                        "task_id": task_id,
                        "depends_on_task_id": task.depends_on_task_id,
                    },
                )
            else:
                # Fail after retry period - required dependency not available
                dependency_error = ValueError(
                    f"No available episodes with required dependency task_id {task.depends_on_task_id} "
                    f"(waited {dependency_config['wait_seconds']}s)"
                )
                self.episode_manager.remove_episode_on_error(episode.episode_id, dependency_error)
                raise ValueError(
                    f"Cannot create episode for task {task_id}: no available episodes with required dependency "
                    f"task_id {task.depends_on_task_id} after waiting {dependency_config['wait_seconds']}s"
                )

        # Configure execution manager with task object, episode ID, and attachment (automatic only)
        # This includes Docker Compose environment creation and health checks
        try:
            self.execution_manager.configure_for_task(
                episode.episode_id, task, session_id=session_id, target_episode_id=effective_attach_to_episode_id
            )
        except Exception as e:
            logger.error(
                "Failed to configure execution environment",
                extra={
                    "event": "episode_environment_configuration_failed",
                    "session_id": session_id,
                    "episode_id": episode.episode_id,
                    "task_id": task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Cleanup episode since execution environment configuration failed (including health checks)
            self.episode_manager.remove_episode_on_error(episode.episode_id, e)

            # Create specific error message for health check failures vs other failures
            if "ComposeHealthCheckError" in str(type(e)) or "health" in str(e).lower():
                error_detail = (
                    f"Episode creation failed due to Docker environment health check failure for task {task_id}. "
                    f"All containers must be healthy before episode creation can proceed. "
                    f"Details: {str(e)}"
                )
            else:
                error_detail = (
                    f"Failed to create execution environment for task {task_id}. "
                    f"This may be due to Docker Compose health check failures or other environment issues: {str(e)}"
                )

            raise HTTPException(status_code=500, detail=error_detail)

        # Generate & store all three prompts (fail-fast if misconfigured)
        rendered_prompts = self.benchmark_manager.prompt_generator.render_agent_prompts_for_task(task)
        # For now, use the instruction prompt as the policy prompt (backward compatibility)
        # TODO: Update PolicyManager to handle all three prompts in structured format
        instruction_prompt = rendered_prompts["instruction"]
        self.policy_manager.set_episode_policy(episode.episode_id, session_id, instruction_prompt)

        # Configure episode manager with task object
        self.episode_manager.configure_for_task(episode.episode_id, task)

        # Configure evaluation manager for task (fail fast if invalid)
        try:
            self.evaluation_manager.configure_for_task(task)
        except Exception as e:
            logger.error(
                "Failed to configure evaluation",
                extra={
                    "event": "episode_evaluation_configuration_failed",
                    "session_id": session_id,
                    "episode_id": episode.episode_id,
                    "task_id": task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Cleanup episode since evaluation configuration failed
            self.episode_manager.remove_episode_on_error(episode.episode_id, e)
            session.remove_active_episode(episode.episode_id)
            raise HTTPException(
                status_code=400, detail=f"Task {task_id} has invalid evaluation configuration: {str(e)}"
            )

        # Add episode to session's active episodes
        session.add_active_episode(episode.episode_id)

        # Log episode start (non-fatal on failure)
        try:
            await self.evaluation_manager.log_episode_start(session_id, episode.episode_id, task_id)
        except Exception as e:
            logger.warning(
                "Failed to log episode start",
                extra={
                    "event": "episode_start_logging_failed",
                    "session_id": session_id,
                    "episode_id": episode.episode_id,
                    "task_id": task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

        logger.info(
            "Episode started",
            extra={
                "event": "episode_started",
                "session_id": session_id,
                "episode_id": episode.episode_id,
                "task_id": task_id,
                "attached_episode_id": effective_attach_to_episode_id,
            },
        )
        return episode

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
        submission: Optional[EvalSubmission] = None,
        cascade_end_attached_episodes: bool = False,
    ) -> EpisodeEndResponse:
        """
        End a specific episode for a session with evaluation.

        Args:
            session_id: ID of the client session
            episode_id: ID of the specific episode to end
            reason: Reason for episode termination (use EpisodeTerminationReason enum values)
            submission: Required submission for evaluation
            cascade_end_attached_episodes: If True, also end episodes that this episode is attached to

        Returns:
            EpisodeEndResponse with evaluation result

        Raises:
            HTTPException: If episode not active or submission missing
            EvaluationError: If evaluation fails
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

        # Get episode and task for evaluation
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

        task = self.benchmark_manager.get_task(episode.task_id)

        # Handle submission based on reason - only require submission for successful completion
        if reason in ["completed", "agent_completed", "success"] and submission is None:
            raise HTTPException(
                status_code=400, detail="Successful episode completion requires a submission for evaluation"
            )

        # Extract submission text and store EvalSubmission object if provided
        if submission is not None:
            # Store the EvalSubmission object for rich evaluation data
            episode.eval_submission = submission
            # Extract the submission text for the episode.submission field
            episode.submission = submission.submission
        else:
            # No submission provided (error case)
            episode.eval_submission = None
            episode.submission = "Episode failed - no submission"

        # Move episode from active to history IMMEDIATELY to prevent session termination override
        session.complete_episode(episode_id)

        # End episode through episode manager (pass submission text, not EvalSubmission object)
        submission_text = submission.submission if submission else None
        completed_episode = self.episode_manager.end_episode(episode_id, reason, submission_text)

        # Evaluate episode using the new evaluation system
        try:
            evaluation_result = await self.evaluation_manager.evaluate_episode(completed_episode, task)
            logger.info(
                "Episode evaluated",
                extra={
                    "event": "episode_evaluated",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "task_id": task.task_id,
                    "score": evaluation_result.score,
                    "max_score": evaluation_result.max_score,
                    "success": evaluation_result.success,
                },
            )
        except Exception as e:
            logger.error(
                "Episode evaluation failed",
                extra={
                    "event": "episode_evaluation_failed",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "task_id": task.task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Evaluation failure means episode failure - fail fast
            raise HTTPException(status_code=500, detail=f"Episode evaluation failed: {str(e)}")

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
            episode_cleanup = self.execution_manager.cleanup_episode(episode_id, {"episode_end_reason": reason})
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

            # Additional cleanup: ensure episode network is removed to prevent subnet pool exhaustion
            await self._cleanup_episode_network(episode_id)
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

        # Build response with evaluation result - success derived from evaluation (evaluation_result always present)
        return EpisodeEndResponse(
            episode_ended=True,
            episode_id=episode_id,
            success=evaluation_result.success,
            reason=reason,
            previous_task_id=episode.task_id,
            active_episodes_remaining=len(session.active_episode_ids),
            evaluation_result=evaluation_result.dict(),
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
        """
        session = self._get_session(session_id)
        session.update_activity()

        # Validate episode is active in this session
        if episode_id not in session.active_episode_ids:
            return CommandResult.error_result(
                error=f"Episode {episode_id} is not active in session {session_id}. "
                f"Active episodes: {session.active_episode_ids}"
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
                self.episode_manager.end_episode(episode_id, EpisodeTerminationReason.COMPLETED)
                session.complete_episode(episode_id)
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
                self.episode_manager.end_episode(episode_id, termination_reason)
                session.complete_episode(episode_id)
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

    async def list_session_episodes(self, session_id: str, status_filter: Optional[str] = None) -> List[Episode]:
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

    async def get_evaluation_criteria(self, session_id: str, episode_id: str) -> EvaluationCriteriaResponse:
        """
        Get complete evaluation criteria package for client-side evaluation.

        This method returns evaluation criteria for both complete and incomplete episodes.
        For incomplete episodes, the submission will be None and judge messages will
        not be rendered (since they require a submission).

        Args:
            session_id: ID of the client session (for context only)
            episode_id: ID of the specific episode

        Returns:
            EvaluationCriteriaResponse containing evaluation criteria package

        Raises:
            HTTPException: If episode or task not found
            RuntimeError: If complete episode is missing submission
        """
        logger.info(
            "Fetching evaluation criteria",
            extra={
                "event": "evaluation_criteria_requested",
                "session_id": session_id,
                "episode_id": episode_id,
            },
        )

        # Get episode data - no session validation needed since evaluation criteria
        # should be available for any episode (complete or incomplete) regardless of session state
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            logger.error(
                "Episode not found for evaluation criteria",
                extra={
                    "event": "evaluation_criteria_episode_missing",
                    "session_id": session_id,
                    "episode_id": episode_id,
                },
            )
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

        logger.debug(
            "Episode located for evaluation criteria",
            extra={
                "event": "evaluation_criteria_episode_found",
                "session_id": session_id,
                "episode_id": episode_id,
                "episode_state": episode.state.value,
                "task_id": episode.task_id,
                "step_count": len(episode.steps),
            },
        )

        # Check if episode has submission (for complete episodes)
        submission_value = getattr(episode, "submission", None)
        logger.debug(
            "Episode submission value inspected",
            extra={
                "event": "evaluation_criteria_submission_value",
                "session_id": session_id,
                "episode_id": episode_id,
                "submission_present": submission_value is not None,
                "submission_type": type(submission_value).__name__,
            },
        )

        has_submission = hasattr(episode, "submission") and episode.submission is not None and episode.submission != ""
        if episode.is_complete and not has_submission:
            logger.error(
                "Completed episode missing submission for evaluation",
                extra={
                    "event": "evaluation_criteria_missing_submission",
                    "session_id": session_id,
                    "episode_id": episode_id,
                },
            )
            raise RuntimeError(f"Complete episode {episode_id} missing required submission for evaluation")

        if has_submission:
            submission_preview = episode.submission[:100] if episode.submission else ""
            logger.debug(
                "Episode submission available",
                extra={
                    "event": "evaluation_criteria_submission_available",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "submission_preview": submission_preview,
                },
            )
        else:
            logger.debug(
                "Episode missing submission",
                extra={
                    "event": "evaluation_criteria_submission_missing",
                    "session_id": session_id,
                    "episode_id": episode_id,
                },
            )

        # Get task information
        task = self.benchmark_manager.get_task(episode.task_id)
        if not task:
            logger.error(
                "Task not found for evaluation criteria",
                extra={
                    "event": "evaluation_criteria_task_missing",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "task_id": episode.task_id,
                },
            )
            raise HTTPException(status_code=404, detail=f"Task {episode.task_id} not found")

        logger.debug(
            "Task located for evaluation criteria",
            extra={
                "event": "evaluation_criteria_task_found",
                "session_id": session_id,
                "episode_id": episode_id,
                "task_id": task.task_id,
                "task_title": task.title,
            },
        )

        # Ensure task has evaluation configuration
        if not task.evaluation_config:
            logger.error(
                "Task missing evaluation configuration",
                extra={
                    "event": "evaluation_criteria_missing_config",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "task_id": task.task_id,
                },
            )
            raise RuntimeError(f"Task {task.task_id} missing evaluation configuration")

        logger.debug(
            "Task evaluation configuration inspected",
            extra={
                "event": "evaluation_criteria_config_inspected",
                "session_id": session_id,
                "episode_id": episode_id,
                "task_id": task.task_id,
                "strategy": task.evaluation_config.get("strategy"),
            },
        )

        # Build task context with subtasks for step-level evaluation
        subtasks_data = [
            {
                "subtask_id": subtask.subtask_id,
                "title": subtask.title,
                "description": subtask.description,
                "objective": subtask.objective,
                "max_score": subtask.max_score,  # Include optional max_score for aggregation
            }
            for subtask in task.subtasks
        ]

        task_context = TaskEvaluationContext(
            task_id=task.task_id,
            title=task.title,
            description=task.description,
            domain=task.domain,
            subtasks=subtasks_data,
        )

        # Initialize judge messages as None
        judge_messages = None

        # Add judge messages for LLM evaluation if needed and submission is available
        if task.evaluation_config.get("strategy") == "llm_judge" and has_submission:
            try:
                logger.debug(
                    "Rendering judge prompts",
                    extra={
                        "event": "evaluation_criteria_rendering_judge_prompts",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "task_id": task.task_id,
                    },
                )
                # Use PromptGenerator to render judge prompts
                judge_payload = self.benchmark_manager.prompt_generator.render_judge_prompt_for_episode(task, episode)

                # Extract system and user prompts from messages array
                system_prompt = None
                user_prompt = None

                for message in judge_payload.messages:
                    if message["role"] == "system":
                        system_prompt = message["content"]
                    elif message["role"] == "user":
                        user_prompt = message["content"]

                # Validate that we have both system and user prompts
                if system_prompt is None or user_prompt is None:
                    raise ValueError(
                        f"Missing required prompts: system_prompt={'present' if system_prompt else 'missing'}, "
                        f"user_prompt={'present' if user_prompt else 'missing'}"
                    )

                # Build JudgeMessages object
                judge_messages = JudgeMessages(
                    system_message=system_prompt, user_message=user_prompt, model=judge_payload.model
                )

                logger.debug(
                    "Rendered judge messages",
                    extra={
                        "event": "evaluation_criteria_judge_render_success",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "task_id": task.task_id,
                        "model": judge_payload.model,
                    },
                )

            except Exception as e:
                logger.error(
                    "Failed to render judge messages",
                    extra={
                        "event": "evaluation_criteria_judge_render_failed",
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "task_id": task.task_id,
                        "strategy": task.evaluation_config.get("strategy"),
                        "episode_state": episode.state.value,
                        "step_count": len(episode.steps),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                if hasattr(e, "__traceback__"):
                    import traceback

                    logger.error(
                        "Judge message rendering stack trace",
                        extra={
                            "event": "evaluation_criteria_judge_render_trace",
                            "session_id": session_id,
                            "episode_id": episode_id,
                            "task_id": task.task_id,
                            "traceback": traceback.format_exc(),
                        },
                    )
                # Fall back to None - client will need to handle this case
                judge_messages = None

        # Build and return evaluation criteria response
        final_submission = episode.submission if has_submission else None
        logger.debug(
            "Final submission prepared for response",
            extra={
                "event": "evaluation_criteria_submission_prepared",
                "session_id": session_id,
                "episode_id": episode_id,
                "submission_present": final_submission is not None,
                "submission_type": type(final_submission).__name__ if final_submission is not None else "NoneType",
            },
        )

        evaluation_criteria = EvaluationCriteriaResponse(
            session_id=session_id,
            episode_id=episode_id,
            task_id=task.task_id,
            submission=final_submission,
            task_context=task_context,
            evaluation_config=task.evaluation_config,
            judge_messages=judge_messages,
        )

        logger.info(
            "Evaluation criteria retrieved",
            extra={
                "event": "evaluation_criteria_retrieved",
                "session_id": session_id,
                "episode_id": episode_id,
                "task_id": task.task_id,
            },
        )
        return evaluation_criteria

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

    async def override_episode_evaluation(
        self,
        session_id: str,
        episode_id: str,
        override_request: EvaluationOverrideRequest,
    ) -> EvaluationResult:
        """
        Override evaluation result for an episode with external evaluation data.

        Args:
            session_id: ID of the client session
            episode_id: ID of the episode to override evaluation for
            override_request: Typed override request containing all evaluation data

        Returns:
            EvaluationResult: The overridden evaluation result

        Raises:
            HTTPException: If session or episode not found
            InvalidEvaluationRequestError: If evaluation data is invalid
        """
        logger.info(
            "Overriding evaluation",
            extra={
                "event": "evaluation_override_requested",
                "session_id": session_id,
                "episode_id": episode_id,
                "strategy": override_request.strategy,
            },
        )

        # Validate session exists
        session = self._get_session(session_id)
        if not session:
            logger.error(
                "Session not found for evaluation override",
                extra={
                    "event": "evaluation_override_session_missing",
                    "session_id": session_id,
                    "episode_id": episode_id,
                },
            )
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Validate episode exists
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            logger.error(
                "Episode not found for evaluation override",
                extra={
                    "event": "evaluation_override_episode_missing",
                    "session_id": session_id,
                    "episode_id": episode_id,
                },
            )
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

        logger.debug(
            "Session and episode validated for evaluation override",
            extra={
                "event": "evaluation_override_context_valid",
                "session_id": session_id,
                "episode_id": episode_id,
            },
        )

        # Convert evaluation_data dict to EpisodeEvaluationData object
        from .evaluation.models import EpisodeEvaluationData

        try:
            episode_eval_data = EpisodeEvaluationData(**override_request.evaluation_data)
        except Exception as e:
            logger.error(
                "Failed to parse override evaluation data",
                extra={
                    "event": "evaluation_override_parse_failed",
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise HTTPException(status_code=422, detail=f"Invalid evaluation data: {e}")

        # Delegate to evaluation manager for override
        evaluation_result = await self.evaluation_manager.override_evaluation_result(
            session_id=session_id,
            episode_id=episode_id,
            evaluation_data=episode_eval_data,
            strategy=override_request.strategy,
            raw_score=override_request.raw_score,
            max_score=override_request.max_score,
            score=override_request.score,
            success=override_request.success,
            details=override_request.details,
        )

        logger.info(
            "Evaluation override applied",
            extra={
                "event": "evaluation_override_applied",
                "session_id": session_id,
                "episode_id": episode_id,
                "strategy": override_request.strategy,
                "score": override_request.score,
                "max_score": override_request.max_score,
                "success": override_request.success,
            },
        )
        return evaluation_result

    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
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
                except asyncio.TimeoutError:
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
        current_time = datetime.utcnow()
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

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions."""
        current_time = datetime.utcnow()
        stats: Dict[str, Any] = {
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

    async def _start_permanent_environment(self) -> None:
        """Start permanent environment if configured through ExecutionManager lifecycle management."""
        permanent_env_name = self.benchmark_manager.config_loader.get_permanent_environment()
        if not permanent_env_name:
            logger.info(
                "No permanent environment configured",
                extra={"event": "permanent_environment_not_configured"},
            )
            return

        try:
            logger.info(
                "Starting permanent environment",
                extra={
                    "event": "permanent_environment_start",
                    "environment": permanent_env_name,
                },
            )

            # Build path to permanent environment compose file
            from pathlib import Path

            permanent_compose_path = (
                Path(self.config_dir) / "environments" / "permanent" / f"{permanent_env_name}.compose.yml"
            )

            if not permanent_compose_path.exists():
                raise RuntimeError(f"Permanent environment compose file not found: {permanent_compose_path}")

            # Start permanent environment directly through PermanentEnvironmentManager
            if not self.execution_manager._permanent_environment_manager:
                raise RuntimeError("Permanent environment manager is not initialized")

            self.execution_manager._permanent_environment_manager.start_permanent_environment_from_file(
                permanent_compose_path
            )

            logger.info(
                "Permanent environment started",
                extra={
                    "event": "permanent_environment_started",
                    "environment": permanent_env_name,
                    "compose_path": str(permanent_compose_path),
                },
            )

        except Exception as e:
            logger.error(
                "Failed to start permanent environment",
                extra={
                    "event": "permanent_environment_start_failed",
                    "environment": permanent_env_name,
                    "compose_path": str(permanent_compose_path),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise
