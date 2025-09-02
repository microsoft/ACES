"""
Container Log Manager

Manages persistent logging for client-side containers with session organization,
log rotation, and retention policies.
"""

import asyncio
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import docker.models.containers

from .config import ClientLoggingConfig, SessionLoggingContext
from .log_stream_collector import LogStreamCollector

logger = logging.getLogger(__name__)


class ContainerLogManager:
    """Manages persistent logging for client-side containers."""

    def __init__(
        self,
        config: ClientLoggingConfig,
        session_context: SessionLoggingContext,
        session_log_dir: Optional[Path] = None,
    ):
        """Initialize container log manager."""
        self.config = config
        self.session_context = session_context

        # Use provided session log directory or create one based on session ID
        if session_log_dir:
            self.session_log_dir = session_log_dir
            self._provided_session_log_dir = True
        else:
            base_dir = Path(config.base_log_dir) if isinstance(config.base_log_dir, str) else config.base_log_dir
            self.session_log_dir = base_dir / session_context.session_id
            self._provided_session_log_dir = False

        # Log stream collector
        self.log_collector = LogStreamCollector(config, session_context)

        # Track active logging containers
        self._active_loggers: Dict[str, asyncio.Task] = {}

        # Ensure session directory exists
        self._setup_session_directory()

    def _setup_session_directory(self) -> None:
        """Set up the session logging directory structure."""
        try:
            # If no session_log_dir was provided (backward compatibility mode),
            # create timestamped directory to prevent overwrites
            if not self._provided_session_log_dir:
                timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
                base_dir = (
                    Path(self.config.base_log_dir)
                    if isinstance(self.config.base_log_dir, str)
                    else self.config.base_log_dir
                )
                timestamped_session_dir = base_dir / f"{timestamp}_{self.session_context.session_id}"
                self.session_log_dir = timestamped_session_dir

            # Ensure the session log directory exists (may already be created by harness)
            self.session_log_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories following server pattern
            subdirs = [
                "container-logs/client-containers",
                "container-logs/sidecar-containers",
                "container-logs/agent-containers",
                "container-events",
            ]

            for subdir in subdirs:
                (self.session_log_dir / subdir).mkdir(parents=True, exist_ok=True)

            # Write session metadata
            self._write_session_metadata()

            logger.info(f"📁 Session log directory: {self.session_log_dir}")

        except Exception as e:
            logger.error(f"❌ Failed to setup session directory: {e}")
            raise

    def _write_session_metadata(self) -> None:
        """Write session metadata to meta.json."""
        try:
            import json

            metadata = {
                "session_id": self.session_context.session_id,
                "client_id": self.session_context.client_id,
                "task_id": self.session_context.task_id,
                "episode_id": self.session_context.episode_id,
                "start_time": datetime.utcnow().isoformat() + "Z",
                "config": {
                    "log_level": self.config.log_level,
                    "log_format": self.config.log_format,
                    "max_log_size_mb": self.config.max_log_size_mb,
                    "retention_days": self.config.retention_days,
                },
            }

            meta_file = self.session_log_dir / "meta.json"
            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.warning(f"⚠️ Failed to write session metadata: {e}")

    async def start_logging_container(
        self, container: docker.models.containers.Container, log_file_name: str, component_type: str = "unknown"
    ) -> bool:
        """
        Start streaming logs from container to file with timestamped naming.

        Args:
            container: Docker container to log
            log_file_name: Base name for log file (will be timestamped)
            component_type: Type of component (client, sidecar, agent)

        Returns:
            bool: True if logging started successfully
        """
        if not self.config.enable_container_logging:
            logger.debug("📝 Container logging disabled")
            return False

        container_id = container.id[:12]

        try:
            # Check if already logging this container
            if container_id in self._active_loggers:
                logger.warning(f"⚠️ Already logging container {container_id}")
                return True

            # Create timestamped log file name following server pattern
            timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
            container_name = container.name or container_id

            # Determine subdirectory and filename based on component type
            if component_type == "sidecar":
                subdir = "container-logs/sidecar-containers"
                timestamped_filename = f"{timestamp}_sidecar_{container_name}.log"
            elif component_type == "agent":
                subdir = "container-logs/agent-containers"
                # Extract task/episode info from log_file_name if available
                base_name = log_file_name.replace(".log", "").replace("agent_containers/", "")
                timestamped_filename = f"{timestamp}_agent_{base_name}_{container_name}.log"
            elif component_type == "client":
                subdir = "container-logs/client-containers"
                timestamped_filename = f"{timestamp}_client_{container_name}.log"
            else:
                subdir = "container-logs"
                timestamped_filename = f"{timestamp}_{component_type}_{container_name}.log"

            log_file_path = self.session_log_dir / subdir / timestamped_filename

            # Start logging task
            logging_task = asyncio.create_task(
                self.log_collector.stream_container_logs(
                    container=container,
                    output_file=log_file_path,
                    container_name=container_name,
                    component_type=component_type,
                )
            )

            # Track the task
            self._active_loggers[container_id] = logging_task

            logger.info(f"📝 Started logging {container_name} ({component_type}) -> {subdir}/{timestamped_filename}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start logging for {container_id}: {e}")
            return False

    async def stop_logging_container(self, container_id: str) -> bool:
        """
        Stop logging and ensure final logs are captured.

        Args:
            container_id: Container ID (can be short form)

        Returns:
            bool: True if logging stopped successfully
        """
        try:
            # Find the container by short ID
            matching_loggers = [
                (cid, task)
                for cid, task in self._active_loggers.items()
                if cid.startswith(container_id) or container_id.startswith(cid)
            ]

            if not matching_loggers:
                logger.debug(f"📝 No active logger found for container {container_id}")
                return True

            # Stop all matching loggers
            for cid, task in matching_loggers:
                if not task.done():
                    logger.debug(f"📝 Stopping log stream for {cid}")
                    task.cancel()

                    try:
                        await asyncio.wait_for(task, timeout=self.config.log_collection_timeout)
                    except asyncio.TimeoutError:
                        logger.warning(f"⚠️ Log collection timeout for {cid}")
                    except asyncio.CancelledError:
                        pass  # Expected

                # Remove from tracking
                self._active_loggers.pop(cid, None)

            logger.info(f"📝 Stopped logging for container {container_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to stop logging for {container_id}: {e}")
            return False

    def get_container_logs(self, log_file_name: str) -> Optional[str]:
        """
        Retrieve logs for debugging.

        Args:
            log_file_name: Name of log file to read

        Returns:
            Log content as string, or None if not found
        """
        try:
            log_file_path = self.session_log_dir / log_file_name

            if not log_file_path.exists():
                logger.warning(f"⚠️ Log file not found: {log_file_path}")
                return None

            with open(log_file_path, "r", encoding="utf-8") as f:
                return f.read()

        except Exception as e:
            logger.error(f"❌ Failed to read log file {log_file_name}: {e}")
            return None

    def list_log_files(self) -> list[str]:
        """List all log files in the session directory."""
        try:
            log_files = []

            if self.session_log_dir.exists():
                for file_path in self.session_log_dir.rglob("*.log"):
                    # Get relative path from session directory
                    relative_path = file_path.relative_to(self.session_log_dir)
                    log_files.append(str(relative_path))

            return sorted(log_files)

        except Exception as e:
            logger.error(f"❌ Failed to list log files: {e}")
            return []

    async def cleanup_and_finalize(self) -> None:
        """Clean up active loggers and finalize session."""
        try:
            logger.info("🧹 Finalizing container logging session")

            # Stop all active loggers
            await self.log_collector.stop_all_streams()
            self._active_loggers.clear()

            # Update session metadata with end time
            self._finalize_session_metadata()

            # Apply retention and rotation policies
            await self._apply_retention_policies()

            logger.info("✅ Container logging session finalized")

        except Exception as e:
            logger.error(f"❌ Failed to finalize logging session: {e}")

    def _finalize_session_metadata(self) -> None:
        """Update session metadata with completion information."""
        try:
            import json

            meta_file = self.session_log_dir / "meta.json"

            if meta_file.exists():
                with open(meta_file, "r") as f:
                    metadata = json.load(f)

                metadata["end_time"] = datetime.utcnow().isoformat() + "Z"
                metadata["log_files"] = self.list_log_files()

                with open(meta_file, "w") as f:
                    json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.warning(f"⚠️ Failed to finalize session metadata: {e}")

    async def _apply_retention_policies(self) -> None:
        """Apply log retention and rotation policies."""
        try:
            # Clean up old sessions
            await self._cleanup_old_sessions()

            # Rotate large log files
            await self._rotate_large_files()

        except Exception as e:
            logger.warning(f"⚠️ Failed to apply retention policies: {e}")

    async def _cleanup_old_sessions(self) -> None:
        """Remove sessions older than retention period."""
        try:
            base_dir = (
                Path(self.config.base_log_dir)
                if isinstance(self.config.base_log_dir, str)
                else self.config.base_log_dir
            )
            if not base_dir.exists():
                return

            cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)

            for session_dir in base_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                # Check if session is old enough for cleanup
                if session_dir.stat().st_mtime < cutoff_date.timestamp():
                    logger.info(f"🗑️ Cleaning up old session: {session_dir.name}")
                    shutil.rmtree(session_dir)

        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup old sessions: {e}")

    async def _rotate_large_files(self) -> None:
        """Rotate log files that exceed size limit."""
        try:
            max_size_bytes = self.config.max_log_size_mb * 1024 * 1024

            for log_file in self.session_log_dir.rglob("*.log"):
                if log_file.stat().st_size > max_size_bytes:
                    logger.info(f"🔄 Rotating large log file: {log_file.name}")

                    # Create rotated filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rotated_name = f"{log_file.stem}_{timestamp}.log"
                    rotated_path = log_file.parent / rotated_name

                    # Move file
                    log_file.rename(rotated_path)

                    # Compress if configured
                    if self.config.compress_old_logs:
                        await self._compress_log_file(rotated_path)

        except Exception as e:
            logger.warning(f"⚠️ Failed to rotate large files: {e}")

    async def _compress_log_file(self, log_file: Path) -> None:
        """Compress a log file."""
        try:
            import gzip

            compressed_path = log_file.with_suffix(log_file.suffix + ".gz")

            with open(log_file, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove original file
            log_file.unlink()

            logger.debug(f"🗜️ Compressed log file: {compressed_path.name}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to compress log file {log_file}: {e}")

    def get_active_logger_count(self) -> int:
        """Get count of active loggers."""
        return len(self._active_loggers)

    def log_container_lifecycle_event(
        self, event_type: str, container_info: Dict[str, Any], additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log container lifecycle events (start, stop, crash, etc.).

        Args:
            event_type: Type of event (start, stop, crash, error)
            container_info: Information about the container
            additional_data: Additional event data
        """
        if not self.config.enable_container_logging:
            return

        try:
            timestamp = datetime.utcnow().isoformat() + "Z"

            event_data = {
                "timestamp": timestamp,
                "session_id": self.session_context.session_id,
                "client_id": self.session_context.client_id,
                "event_type": event_type,
                "container_info": container_info,
                "additional_data": additional_data or {},
            }

            # Log to a daily events file
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            events_file = self.session_log_dir / "container-events" / f"container-events-{date_str}.jsonl"
            events_file.parent.mkdir(parents=True, exist_ok=True)

            with open(events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data) + "\n")

            logger.debug(f"📝 Logged container event: {event_type} for {container_info.get('name', 'unknown')}")

        except Exception as e:
            logger.error(f"❌ Failed to log container lifecycle event: {e}")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if self._active_loggers:
            logger.warning("⚠️ ContainerLogManager deleted with active loggers")
