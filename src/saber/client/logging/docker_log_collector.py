"""
Docker Log Collector

Simplified approach that uses Docker's native logging drivers and collects logs
after container completion. This eliminates complex real-time streaming and
provides more reliable log capture.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import docker

from .config import ClientLoggingConfig, SessionLoggingContext

logger = logging.getLogger(__name__)


class DockerLogCollector:
    """Collects container logs using Docker's native logging and post-processing."""

    def __init__(self, config: ClientLoggingConfig, session_context: SessionLoggingContext):
        """Initialize Docker log collector."""
        self.config = config
        self.session_context = session_context
        self.docker_client = docker.from_env()  # type: ignore[attr-defined]

    def collect_container_logs(
        self,
        container: Any,
        output_file: Path,
        component_type: str = "unknown",
    ) -> bool:
        """
        Collect all logs from a container using Docker's logging system.

        This method should be called after the container has finished execution.
        It uses Docker's native log collection which is more reliable than streaming.

        Args:
            container: Docker container to collect logs from
            output_file: File to write logs to
            component_type: Type of component (client, sidecar, agent)

        Returns:
            bool: True if logs were collected successfully
        """
        try:
            logger.info("📝 Collecting logs for %s -> %s", container.name, output_file)

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Get all logs from container (Docker handles the buffering and streaming)
            logs = container.logs(stdout=True, stderr=True, timestamps=True, tail="all")  # Get all logs

            if not logs:
                logger.warning("⚠️ No logs found for container %s", container.name)
                # Create empty file with metadata header
                with open(output_file, "w", encoding="utf-8") as f:
                    self._write_session_header(f, container, component_type)
                    f.write("# No logs captured from this container\n")
                return True

            # Write logs to file with metadata
            with open(output_file, "w", encoding="utf-8") as f:
                # Write session metadata header
                self._write_session_header(f, container, component_type)

                # Process and write logs
                log_text = logs.decode("utf-8", errors="ignore")
                for line in log_text.split("\n"):
                    if line.strip():
                        formatted_line = self._format_log_entry(line.strip(), container.name, component_type)
                        f.write(formatted_line + "\n")

            # Get file size for reporting
            file_size = output_file.stat().st_size
            logger.info("✅ Collected %d bytes of logs for %s", file_size, container.name)
            return True

        except Exception as e:
            logger.error("❌ Failed to collect logs for %s: %s", container.name, e)
            return False

    def collect_logs_from_docker_path(
        self,
        container_id: str,
        output_file: Path,
        component_type: str = "unknown",
    ) -> bool:
        """
        Collect logs directly from Docker's internal log files.

        This is a backup method if the container is no longer accessible
        but the log files still exist in /var/lib/docker/containers.

        Args:
            container_id: Full container ID
            output_file: File to write logs to
            component_type: Type of component

        Returns:
            bool: True if logs were collected successfully
        """
        try:
            # Docker stores logs in /var/lib/docker/containers/<id>/<id>-json.log
            docker_log_path = Path(f"/var/lib/docker/containers/{container_id}/{container_id}-json.log")

            if not docker_log_path.exists():
                logger.debug(
                    "📝 Docker internal log file not found: %s (container likely removed)",
                    docker_log_path,
                )
                logger.info(
                    "ℹ️  Log collection via Docker internal path unavailable for removed container %s",
                    container_id[:12],
                )
                return False

            logger.info("📝 Collecting logs from Docker path: %s -> %s", docker_log_path, output_file)

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Read and process Docker's JSON log format
            with open(output_file, "w", encoding="utf-8") as out_f:
                # Write basic header
                self._write_basic_header(out_f, container_id, component_type)

                with open(docker_log_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        if line.strip():
                            try:
                                # Docker logs are in JSON format:
                                # {"log": "message", "stream": "stdout", "time": "2023-..."}
                                log_entry = json.loads(line.strip())
                                timestamp = log_entry.get("time", "")
                                message = log_entry.get("log", "").rstrip("\n\r")
                                stream = log_entry.get("stream", "stdout")

                                if message:
                                    # Format: [timestamp] [stream] message
                                    formatted_line = f"[{timestamp}] [{stream}] {message}"
                                    out_f.write(formatted_line + "\n")

                            except json.JSONDecodeError:
                                # If not JSON, write as-is
                                out_f.write(line)

            file_size = output_file.stat().st_size
            logger.info("✅ Collected %d bytes from Docker log path", file_size)
            return True

        except Exception as e:
            logger.error("❌ Failed to collect logs from Docker path: %s", e)
            return False

    def _write_session_header(self, f: TextIO, container: Any, component_type: str) -> None:
        """Write session metadata header to log file."""
        if not self.config.include_metadata:
            return

        timestamp = datetime.utcnow().isoformat() + "Z"

        # Get image name safely
        image_name = "unknown"
        try:
            if container.image and container.image.tags:
                image_name = container.image.tags[0]
        except Exception:
            pass

        if self.config.log_format == "json":
            header = {
                "type": "session_header",
                "log_session_start": timestamp,
                "session_id": self.session_context.session_id,
                "client_id": self.session_context.client_id,
                "task_id": self.session_context.task_id,
                "episode_id": self.session_context.episode_id,
                "container_id": container.id,
                "container_name": container.name,
                "component_type": component_type,
                "image": image_name,
                "log_format": self.config.log_format,
                "log_level": self.config.log_level,
            }
            f.write(json.dumps(header) + "\n")
        else:
            f.write(f"# Container Logs for {container.name}\n")
            f.write(f"# Component Type: {component_type}\n")
            f.write(f"# Session ID: {self.session_context.session_id}\n")
            f.write(f"# Client ID: {self.session_context.client_id}\n")
            f.write(f"# Container ID: {container.id}\n")
            f.write(f"# Image: {image_name}\n")
            f.write(f"# Log Collected: {timestamp}\n")
            f.write("# " + "=" * 60 + "\n\n")

    def _write_basic_header(self, f: TextIO, container_id: str, component_type: str) -> None:
        """Write basic header when container object is not available."""
        if not self.config.include_metadata:
            return

        timestamp = datetime.utcnow().isoformat() + "Z"

        if self.config.log_format == "json":
            header = {
                "type": "session_header",
                "log_session_start": timestamp,
                "session_id": self.session_context.session_id,
                "client_id": self.session_context.client_id,
                "task_id": self.session_context.task_id,
                "episode_id": self.session_context.episode_id,
                "container_id": container_id,
                "component_type": component_type,
                "log_format": self.config.log_format,
                "log_level": self.config.log_level,
            }
            f.write(json.dumps(header) + "\n")
        else:
            f.write("# Container Logs (collected from Docker path)\n")
            f.write(f"# Component Type: {component_type}\n")
            f.write(f"# Session ID: {self.session_context.session_id}\n")
            f.write(f"# Container ID: {container_id}\n")
            f.write(f"# Log Collected: {timestamp}\n")
            f.write("# " + "=" * 60 + "\n\n")

    def _format_log_entry(self, log_line: str, container_name: str, component_type: str) -> str:
        """Format log entry with minimal processing."""
        # For sidecar components, use plain format to preserve the 🚨🚨🚨 banners
        if component_type == "sidecar" or self.config.log_format != "json":
            return log_line  # Return the log line as-is

        # For JSON format (non-sidecar), add some metadata
        if self.config.log_format == "json":
            # Parse Docker timestamp if present
            timestamp = datetime.utcnow().isoformat() + "Z"
            content = log_line

            # Extract Docker timestamp if present (format: 2023-09-01T12:00:00.000000000Z message)
            if log_line and len(log_line) > 30 and log_line[30:31] in [" ", "\t"]:
                try:
                    timestamp_part = log_line[:30]
                    content = log_line[31:]
                    # Validate timestamp format
                    datetime.fromisoformat(timestamp_part.replace("Z", "+00:00"))
                    timestamp = timestamp_part
                except (ValueError, IndexError):
                    # Not a valid timestamp, use the whole line as content
                    pass

            return json.dumps(
                {
                    "timestamp": timestamp,
                    "container_name": container_name,
                    "component_type": component_type,
                    "session_id": self.session_context.session_id,
                    "content": content,
                }
            )

        return log_line
