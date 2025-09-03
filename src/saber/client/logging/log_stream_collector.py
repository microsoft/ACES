"""
Log Stream Collector

Collects and formats log streams from Docker containers with metadata enrichment.
"""

import asyncio
import json
import logging
import queue
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, TextIO, Tuple, Union

from .config import ClientLoggingConfig, SessionLoggingContext

logger = logging.getLogger(__name__)


async def asyncio_stream_wrapper(
    blocking_generator_func: Callable[[], Generator[Any, None, None]], loop: asyncio.AbstractEventLoop
) -> AsyncGenerator[Any, None]:
    """
    Wrap a blocking generator function to work with asyncio for real-time streaming.

    Args:
        blocking_generator_func: Function that returns a generator
        loop: Current event loop

    Yields:
        Items from the blocking generator in real-time
    """
    import threading

    # Create a queue for communication between threads
    log_queue: queue.Queue[Tuple[str, Any]] = queue.Queue()
    exception_holder: List[Union[Exception, None]] = [None]
    finished: List[bool] = [False]

    def producer() -> None:
        """Producer function to run in separate thread."""
        try:
            for item in blocking_generator_func():
                log_queue.put(("item", item))
        except Exception as e:
            exception_holder[0] = e
            log_queue.put(("error", e))
        finally:
            finished[0] = True
            log_queue.put(("done", None))

    # Start producer thread
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    try:
        while True:
            # Check if we have items in queue
            try:
                # Non-blocking get with timeout
                msg_type, item = log_queue.get_nowait()

                if msg_type == "item":
                    yield item
                elif msg_type == "error":
                    raise item
                elif msg_type == "done":
                    break

            except queue.Empty:
                # No items available, yield control briefly
                await asyncio.sleep(0.01)  # 10ms delay

                # Check if thread finished
                if finished[0] and log_queue.empty():
                    break

                continue

    finally:
        # Wait for thread to complete (with timeout)
        thread.join(timeout=1.0)

    # Re-raise any exception that occurred
    if exception_holder[0]:
        raise exception_holder[0]


class LogStreamCollector:
    """Collects and formats log streams from Docker containers."""

    def __init__(self, config: ClientLoggingConfig, session_context: SessionLoggingContext):
        """Initialize log stream collector."""
        self.config = config
        self.session_context = session_context
        self._active_streams: Dict[str, asyncio.Task] = {}

    async def stream_container_logs(
        self,
        container: Any,
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
        self, container: Any, output_file: Path, container_name: str, component_type: str
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

    async def _get_log_stream(self, container: Any) -> AsyncGenerator[str, None]:
        """Get real-time log stream from container - proper streaming approach."""

        def get_blocking_log_stream() -> Generator[str, None, None]:
            """Get blocking log stream in thread executor."""
            try:
                # Use proper streaming with follow=True for real-time logs
                log_stream = container.logs(
                    stream=True,  # Enable streaming
                    follow=True,  # Follow new logs as they appear
                    stdout=True,
                    stderr=True,
                    timestamps=True,
                    tail="all",  # Start with existing logs, then stream new ones
                )

                # Process streaming logs
                for log_chunk in log_stream:
                    if log_chunk:
                        log_text = log_chunk.decode("utf-8", errors="ignore")
                        for line in log_text.split("\n"):
                            if line.strip():
                                yield line.strip()

            except Exception as e:
                logger.error(f"❌ Container log stream failed: {e}")
                raise

        try:
            # Run the blocking stream in a thread executor to avoid blocking asyncio
            loop = asyncio.get_event_loop()

            # Stream logs from the blocking generator
            async for line in asyncio_stream_wrapper(get_blocking_log_stream, loop):
                yield line

        except Exception as e:
            logger.error(f"❌ Container log stream failed: {e}")
            raise

    async def _write_session_header(self, f: TextIO, container: Any, container_name: str, component_type: str) -> None:
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

        # Use plain format for sidecar components to match agent logs
        if component_type == "sidecar" or self.config.log_format != "json":
            return content  # Just return the clean log content without extra formatting

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
