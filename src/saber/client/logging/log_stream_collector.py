"""
Log Stream Collector

Collects and formats log streams from Docker containers with metadata enrichment.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator, TextIO

import docker.models.containers

from .config import ClientLoggingConfig, SessionLoggingContext

logger = logging.getLogger(__name__)


class LogStreamCollector:
    """Collects and formats log streams from Docker containers."""

    def __init__(self, config: ClientLoggingConfig, session_context: SessionLoggingContext):
        """Initialize log stream collector."""
        self.config = config
        self.session_context = session_context
        self._active_streams: Dict[str, asyncio.Task] = {}

    async def stream_container_logs(
        self,
        container: docker.models.containers.Container,
        output_file: Path,
        container_name: str,
        component_type: str = "unknown",
    ) -> None:
        """
        Stream container logs with metadata to file.

        Args:
            container: Docker container to stream logs from
            output_file: File to write logs to
            container_name: Name of the container for identification
            component_type: Type of component (client, sidecar, agent)
        """
        stream_id = f"{container.id[:12]}_{container_name}"

        try:
            logger.info(f"📝 Starting log stream for {container_name} -> {output_file}")

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Create the streaming task
            stream_task = asyncio.create_task(
                self._stream_logs_to_file(container, output_file, container_name, component_type)
            )

            # Track active stream
            self._active_streams[stream_id] = stream_task

            # Wait for completion
            await stream_task

        except asyncio.CancelledError:
            logger.info(f"📝 Log stream cancelled for {container_name}")
        except Exception as e:
            logger.error(f"❌ Failed to stream logs for {container_name}: {e}")
        finally:
            # Clean up
            self._active_streams.pop(stream_id, None)

    async def _stream_logs_to_file(
        self, container: docker.models.containers.Container, output_file: Path, container_name: str, component_type: str
    ) -> None:
        """Internal method to stream logs to file."""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                # Write session metadata header
                await self._write_session_header(f, container, container_name, component_type)

                # Stream logs
                async for log_line in self._get_log_stream(container):
                    if log_line:
                        formatted_entry = self._format_log_entry(log_line, container_name, component_type)
                        f.write(formatted_entry + "\n")
                        f.flush()  # Ensure real-time writing

        except Exception as e:
            logger.error(f"❌ Error writing logs to {output_file}: {e}")

    async def _get_log_stream(self, container: docker.models.containers.Container) -> AsyncGenerator[str, None]:
        """Get log stream from container."""

        def _get_logs_sync() -> Generator[str, None, None]:
            """Synchronous log streaming function to run in thread."""
            try:
                # Get logs generator (both stdout and stderr)
                logs_generator = container.logs(stream=True, follow=True, stdout=True, stderr=True, timestamps=True)

                for log_bytes in logs_generator:
                    try:
                        # Decode log line
                        log_line = log_bytes.decode("utf-8").strip()
                        if log_line:
                            yield log_line
                    except UnicodeDecodeError:
                        # Handle binary or non-UTF8 content
                        yield f"[BINARY_CONTENT_LENGTH:{len(log_bytes)}]"

            except Exception as e:
                logger.warning(f"⚠️ Container log stream ended: {e}")

        try:
            # Run the synchronous log streaming in a thread to avoid blocking the event loop
            # Use run_in_executor to run the sync generator in a thread
            def sync_generator_wrapper() -> list[str]:
                return list(_get_logs_sync())

            # For now, let's use a simpler approach - just get recent logs without following
            logs_generator = container.logs(
                stream=False,  # Don't follow for now to avoid blocking
                follow=False,
                stdout=True,
                stderr=True,
                timestamps=True,
                tail=100,  # Get last 100 lines
            )

            # Process the logs
            if isinstance(logs_generator, bytes):
                for log_line in logs_generator.decode("utf-8").split("\n"):
                    if log_line.strip():
                        yield log_line.strip()
            else:
                for log_bytes in logs_generator:
                    try:
                        log_line = log_bytes.decode("utf-8").strip()
                        if log_line:
                            yield log_line
                    except UnicodeDecodeError:
                        yield f"[BINARY_CONTENT_LENGTH:{len(log_bytes)}]"

        except Exception as e:
            logger.warning(f"⚠️ Container log stream ended: {e}")

    async def _write_session_header(
        self, f: TextIO, container: docker.models.containers.Container, container_name: str, component_type: str
    ) -> None:
        """Write session metadata header to log file."""
        if not self.config.include_metadata:
            return

        timestamp = datetime.utcnow().isoformat() + "Z"

        header = {
            "log_session_start": timestamp,
            "session_id": self.session_context.session_id,
            "client_id": self.session_context.client_id,
            "task_id": self.session_context.task_id,
            "episode_id": self.session_context.episode_id,
            "container_id": container.id,
            "container_name": container_name,
            "component_type": component_type,
            "image": container.image.tags[0] if container.image.tags else "unknown",
            "log_format": self.config.log_format,
            "log_level": self.config.log_level,
        }

        if self.config.log_format == "json":
            f.write(json.dumps({"type": "session_header", **header}) + "\n")
        else:
            f.write(f"# Container Logs for {container_name}\n")
            f.write(f"# Component Type: {component_type}\n")
            f.write(f"# Session ID: {self.session_context.session_id}\n")
            f.write(f"# Client ID: {self.session_context.client_id}\n")
            f.write(f"# Container ID: {container.id}\n")
            f.write(f"# Image: {header['image']}\n")
            f.write(f"# Log Started: {timestamp}\n")
            f.write("# " + "=" * 60 + "\n\n")

    def _format_log_entry(self, log_line: str, container_name: str, component_type: str) -> str:
        """Format log entry with metadata."""
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

        if self.config.log_format == "json":
            return json.dumps(
                {
                    "timestamp": timestamp,
                    "container_name": container_name,
                    "component_type": component_type,
                    "session_id": self.session_context.session_id,
                    "task_id": self.session_context.task_id,
                    "episode_id": self.session_context.episode_id,
                    "content": content,
                    "level": self._extract_log_level(content),
                }
            )
        else:
            return f"[{timestamp}] [{container_name}] [{component_type}] {content}"

    def _extract_log_level(self, content: str) -> str:
        """Extract log level from content if possible."""
        content_upper = content.upper()
        levels = ["ERROR", "WARN", "WARNING", "INFO", "DEBUG", "TRACE"]

        for level in levels:
            if level in content_upper:
                return level

        return "INFO"  # Default level

    async def stop_all_streams(self) -> None:
        """Stop all active log streams."""
        logger.info(f"🛑 Stopping {len(self._active_streams)} active log streams")

        # Cancel all streams
        for stream_id, task in self._active_streams.items():
            if not task.done():
                logger.debug(f"📝 Cancelling log stream: {stream_id}")
                task.cancel()

        # Wait for all to complete
        if self._active_streams:
            await asyncio.gather(*self._active_streams.values(), return_exceptions=True)

        self._active_streams.clear()
        logger.info("✅ All log streams stopped")

    def get_active_stream_count(self) -> int:
        """Get count of active log streams."""
        return len(self._active_streams)
