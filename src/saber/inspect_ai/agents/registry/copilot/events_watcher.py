"""Events file watcher for real-time events.jsonl monitoring.

This module provides components for watching and incrementally parsing
the Copilot SDK's events.jsonl file during agent execution.

Key components:
- ParserState: Immutable snapshot of parser state
- IncrementalEventParser: Stateful parser for incremental file reads
- EventsFileWatcher: Async file watcher using watchfiles

Usage:
    parser = IncrementalEventParser(events_path)
    events = parser.parse_new_events()  # First call reads all existing events
    # ... file grows ...
    new_events = parser.parse_new_events()  # Only returns newly appended events
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .....logging_config import LogCategory, get_saber_logger
from .events import CopilotEvent, parse_event

logger = get_saber_logger(LogCategory.AGENT, __name__)


# =============================================================================
# Parser State Model
# =============================================================================


class ParserState(BaseModel):
    """Immutable state snapshot of the incremental parser.

    This model captures the current position in the events.jsonl file
    and any buffered partial line content.
    """

    model_config = ConfigDict(frozen=True)

    byte_offset: int
    """Current byte position in the file."""

    partial_line: str
    """Buffered partial line content (incomplete JSON at EOF)."""

    events_parsed: int
    """Total number of events successfully parsed."""


# =============================================================================
# Incremental Event Parser
# =============================================================================


class IncrementalEventParser:
    """Stateful parser for incrementally reading events.jsonl.

    Tracks file position and buffers partial lines to enable
    efficient incremental parsing of appended events.

    The parser handles:
    - Incremental reads from byte offset
    - Partial line buffering at EOF
    - File truncation detection and auto-reset
    - Corrupted/invalid JSON line skipping

    Example:
        parser = IncrementalEventParser(Path("events.jsonl"))
        events = parser.parse_new_events()  # Read all existing events
        # ... file grows ...
        new_events = parser.parse_new_events()  # Only new events
    """

    def __init__(self, events_path: Path) -> None:
        """Initialize the parser.

        Args:
            events_path: Path to the events.jsonl file to parse
        """
        self._events_path = events_path
        self._byte_offset: int = 0
        self._partial_line: str = ""
        self._events_parsed: int = 0

    @property
    def events_path(self) -> Path:
        """Get the path to the events file."""
        return self._events_path

    @property
    def state(self) -> ParserState:
        """Get current parser state snapshot.

        Returns:
            Immutable ParserState with current byte offset, partial line, and count
        """
        return ParserState(
            byte_offset=self._byte_offset,
            partial_line=self._partial_line,
            events_parsed=self._events_parsed,
        )

    def parse_new_events(self) -> list[CopilotEvent]:
        """Read and parse any new events since last call.

        Reads from the current byte offset to end of file, parsing
        complete JSON lines and buffering any partial line at EOF.

        Handles file truncation by auto-resetting if file size is
        smaller than the current byte offset.

        Returns:
            List of newly parsed events (may be empty if no new complete lines)
        """
        if not self._events_path.exists():
            logger.debug(f"Events file does not exist: {self._events_path}")
            return []

        # Check for file truncation
        file_size = self._events_path.stat().st_size
        if file_size < self._byte_offset:
            logger.warning(f"File truncation detected: size {file_size} < offset {self._byte_offset}, resetting parser")
            self.reset()

        events: list[CopilotEvent] = []

        try:
            with open(self._events_path, encoding="utf-8") as f:
                # Seek to current position
                f.seek(self._byte_offset)

                # Read remaining content (new bytes since last read)
                new_content = f.read()

                if not new_content and not self._partial_line:
                    return []

                # Prepend any buffered partial line from previous read
                content = self._partial_line + new_content
                self._partial_line = ""

                # Split into lines
                lines = content.split("\n")

                # Last element may be partial (if no trailing newline)
                # Check if content ends with newline
                if content.endswith("\n"):
                    # All lines are complete, last element is empty string
                    lines = lines[:-1]  # Remove empty trailing element
                else:
                    # Last line is partial, buffer it
                    self._partial_line = lines[-1]
                    lines = lines[:-1]

                # Parse each complete line
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event_dict = json.loads(line)
                        event = parse_event(event_dict)
                        events.append(event)
                        self._events_parsed += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping corrupted JSON line: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Skipping unparseable event: {e}")
                        continue

                # Update byte offset to end of newly read content
                # (file_size is where we've read up to)
                self._byte_offset = file_size

        except OSError as e:
            logger.error(f"Error reading events file: {e}")
            return []

        if events:
            logger.debug(f"Parsed {len(events)} new events, total: {self._events_parsed}")

        return events

    def reset(self) -> None:
        """Reset parser state.

        Clears byte offset, partial line buffer, and events count.
        Use for testing or recovery from file rotation.
        """
        self._byte_offset = 0
        self._partial_line = ""
        self._events_parsed = 0
        logger.debug("Parser state reset")


# =============================================================================
# Event Callback Type
# =============================================================================

# Type alias for event callback - receives a batch of events
EventCallback = Callable[[list[CopilotEvent]], Coroutine[Any, Any, None]]


# =============================================================================
# Events File Watcher
# =============================================================================


class EventsFileWatcher:
    """Async watcher for Copilot events.jsonl file.

    Uses watchfiles.awatch() to monitor file changes and emits
    parsed events via callback or queue.

    The watcher:
    - Emits existing events on start (initial read)
    - Monitors for file changes using watchfiles
    - Parses new events incrementally via IncrementalEventParser
    - Supports clean shutdown via stop signal

    Example:
        async def on_events(events):
            for event in events:
                print(event)

        watcher = EventsFileWatcher(events_path, on_events=on_events)
        task = asyncio.create_task(watcher.start())
        # ... later ...
        watcher.stop()
        await task
    """

    def __init__(
        self,
        events_path: Path,
        on_events: EventCallback | None = None,
        event_queue: asyncio.Queue[CopilotEvent] | None = None,
    ) -> None:
        """Initialize the watcher.

        Args:
            events_path: Path to events.jsonl file to watch
            on_events: Async callback invoked with new events (batch)
            event_queue: Queue to put events into (alternative to callback)

        Raises:
            ValueError: If neither on_events nor event_queue is provided
        """
        if on_events is None and event_queue is None:
            raise ValueError("Either on_events or event_queue must be provided")

        self._events_path = events_path
        self._on_events = on_events
        self._event_queue = event_queue
        self._parser = IncrementalEventParser(events_path)
        self._stop_event = asyncio.Event()
        self._is_running = False

    @property
    def events_path(self) -> Path:
        """Get the path to the events file."""
        return self._events_path

    @property
    def is_running(self) -> bool:
        """Check if watcher is currently running."""
        return self._is_running

    async def start(self) -> None:
        """Start watching for file changes.

        Blocks until stop() is called or an error occurs.
        Emits existing events on start, then watches for new events.
        """
        self._is_running = True
        self._stop_event.clear()

        try:
            # Do initial read of existing events
            await self._read_and_emit_events()

            # Import watchfiles here to make it optional for testing
            try:
                from watchfiles import awatch
            except ImportError:
                logger.warning("watchfiles not installed, falling back to polling mode")
                await self._poll_mode()
                return

            # Watch the parent directory (file may not exist yet)
            watch_path = self._events_path.parent
            if not watch_path.exists():
                watch_path.mkdir(parents=True, exist_ok=True)

            # Use awatch with stop_event
            async for changes in awatch(
                watch_path,
                stop_event=self._stop_event,
                debounce=50,  # 50ms debounce
                step=100,  # 100ms step between checks
            ):
                if self._stop_event.is_set():
                    break

                # Check if our file was modified
                for _change_type, path in changes:
                    if Path(path) == self._events_path:
                        await self._read_and_emit_events()
                        break

        except asyncio.CancelledError:
            logger.debug("Watcher cancelled")
        except Exception as e:
            logger.error(f"Watcher error: {e}")
        finally:
            self._is_running = False
            logger.debug("Watcher stopped")

    async def _poll_mode(self) -> None:
        """Fallback polling mode when watchfiles unavailable."""
        poll_interval = 0.1  # 100ms

        while not self._stop_event.is_set():
            await self._read_and_emit_events()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=poll_interval,
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                continue  # Keep polling

    async def _read_and_emit_events(self) -> None:
        """Read new events and emit via callback or queue."""
        try:
            events = self._parser.parse_new_events()

            if not events:
                return

            # Emit via callback
            if self._on_events:
                try:
                    await self._on_events(events)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            # Put in queue
            if self._event_queue:
                for event in events:
                    await self._event_queue.put(event)

        except Exception as e:
            logger.error(f"Error reading/emitting events: {e}")

    def stop(self) -> None:
        """Signal the watcher to stop.

        Does not block; watcher will stop on next iteration.
        """
        self._stop_event.set()
        logger.debug("Watcher stop signal sent")


__all__ = [
    "ParserState",
    "IncrementalEventParser",
    "EventsFileWatcher",
    "EventCallback",
]
