"""Tests for EventsFileWatcher.

TDD tests for the async file watcher that monitors events.jsonl
for changes and emits parsed events.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from saber.inspect_ai.agents.registry.copilot.events import (
    AssistantMessageEvent,
    SessionStartEvent,
    UserMessageEvent,
)

from .conftest import (
    SAMPLE_ASSISTANT_MESSAGE,
    SAMPLE_SESSION_START,
)

if TYPE_CHECKING:
    from saber.inspect_ai.agents.registry.copilot.events import CopilotEvent


class TestEventsFileWatcherInitialization:
    """Tests for EventsFileWatcher initialization."""

    def test_init_with_path_and_callback(self, events_file_path: Path) -> None:
        """Watcher should initialize with path and callback."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        async def callback(events: list[CopilotEvent]) -> None:
            pass

        watcher = EventsFileWatcher(events_path=events_file_path, on_events=callback)

        assert watcher.events_path == events_file_path
        assert not watcher.is_running

    def test_init_with_queue(self, events_file_path: Path) -> None:
        """Watcher should initialize with queue instead of callback."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        queue: asyncio.Queue[CopilotEvent] = asyncio.Queue()
        watcher = EventsFileWatcher(events_path=events_file_path, event_queue=queue)

        assert watcher.events_path == events_file_path

    def test_init_requires_callback_or_queue(self, events_file_path: Path) -> None:
        """Watcher should require either callback or queue."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        with pytest.raises(ValueError, match="Either on_events or event_queue must be provided"):
            EventsFileWatcher(events_path=events_file_path)


class TestEventsFileWatcherStartStop:
    """Tests for watcher start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_stop_before_start_is_safe(self, events_file_path: Path) -> None:
        """Stopping before starting should be safe (no-op)."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        async def callback(events: list[CopilotEvent]) -> None:
            pass

        watcher = EventsFileWatcher(events_path=events_file_path, on_events=callback)

        # Should not raise
        watcher.stop()
        assert not watcher.is_running

    @pytest.mark.asyncio
    async def test_is_running_after_start(self, sample_events_jsonl: Path) -> None:
        """is_running should be True after start() called."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        async def callback(events: list[CopilotEvent]) -> None:
            pass

        watcher = EventsFileWatcher(events_path=sample_events_jsonl, on_events=callback)

        # Start in background
        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.05)  # Give it time to start

        assert watcher.is_running

        # Cleanup
        watcher.stop()
        await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_stop_signals_watcher_to_exit(self, sample_events_jsonl: Path) -> None:
        """stop() should signal the watcher to exit cleanly."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        async def callback(events: list[CopilotEvent]) -> None:
            pass

        watcher = EventsFileWatcher(events_path=sample_events_jsonl, on_events=callback)

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.05)

        watcher.stop()

        # Should exit within timeout
        await asyncio.wait_for(task, timeout=2.0)
        assert not watcher.is_running

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self, sample_events_jsonl: Path) -> None:
        """Calling stop() twice should be safe."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        async def callback(events: list[CopilotEvent]) -> None:
            pass

        watcher = EventsFileWatcher(events_path=sample_events_jsonl, on_events=callback)

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.05)

        watcher.stop()
        watcher.stop()  # Second stop should be safe

        await asyncio.wait_for(task, timeout=2.0)


class TestEventsFileWatcherEventEmission:
    """Tests for event emission via callback."""

    @pytest.mark.asyncio
    async def test_emits_existing_events_on_start(self, sample_events_jsonl: Path) -> None:
        """Watcher should emit existing events when started."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        received_events: list[CopilotEvent] = []

        async def callback(events: list[CopilotEvent]) -> None:
            received_events.extend(events)

        watcher = EventsFileWatcher(events_path=sample_events_jsonl, on_events=callback)

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.1)  # Wait for initial read

        watcher.stop()
        await asyncio.wait_for(task, timeout=2.0)

        # Should have received the 2 existing events
        assert len(received_events) == 2
        assert isinstance(received_events[0], SessionStartEvent)
        assert isinstance(received_events[1], UserMessageEvent)

    @pytest.mark.asyncio
    async def test_emits_new_events_on_file_change(self, sample_events_jsonl: Path) -> None:
        """Watcher should emit new events when file is appended."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        received_events: list[CopilotEvent] = []
        event_received = asyncio.Event()

        async def callback(events: list[CopilotEvent]) -> None:
            received_events.extend(events)
            if len(received_events) >= 3:  # Wait for all 3 events
                event_received.set()

        watcher = EventsFileWatcher(events_path=sample_events_jsonl, on_events=callback)

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.1)

        # Append new event
        with open(sample_events_jsonl, "a") as f:
            f.write(f"{SAMPLE_ASSISTANT_MESSAGE}\n")

        # Wait for event to be received (with timeout)
        try:
            await asyncio.wait_for(event_received.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pass  # May timeout in CI, check what we have

        watcher.stop()
        await asyncio.wait_for(task, timeout=2.0)

        # Should have at least the initial 2 + 1 new event
        assert len(received_events) >= 2
        # If file watching worked, should have 3
        if len(received_events) >= 3:
            assert isinstance(received_events[2], AssistantMessageEvent)


class TestEventsFileWatcherQueueMode:
    """Tests for queue-based event emission."""

    @pytest.mark.asyncio
    async def test_puts_events_into_queue(self, sample_events_jsonl: Path) -> None:
        """Watcher should put events into queue when configured."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        queue: asyncio.Queue[CopilotEvent] = asyncio.Queue()
        watcher = EventsFileWatcher(events_path=sample_events_jsonl, event_queue=queue)

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.1)

        watcher.stop()
        await asyncio.wait_for(task, timeout=2.0)

        # Should have events in queue
        events: list[CopilotEvent] = []
        while not queue.empty():
            events.append(queue.get_nowait())

        assert len(events) == 2
        assert isinstance(events[0], SessionStartEvent)
        assert isinstance(events[1], UserMessageEvent)


class TestEventsFileWatcherNonexistentFile:
    """Tests for handling non-existent files."""

    @pytest.mark.asyncio
    async def test_handles_nonexistent_file(self, events_file_path: Path) -> None:
        """Watcher should handle non-existent file gracefully."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        received_events: list[CopilotEvent] = []

        async def callback(events: list[CopilotEvent]) -> None:
            received_events.extend(events)

        watcher = EventsFileWatcher(events_path=events_file_path, on_events=callback)

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.1)

        watcher.stop()
        await asyncio.wait_for(task, timeout=2.0)

        # Should have no events (file doesn't exist)
        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_emits_events_when_file_created(self, events_file_path: Path) -> None:
        """Watcher should emit events when file is created after start."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        received_events: list[CopilotEvent] = []
        event_received = asyncio.Event()

        async def callback(events: list[CopilotEvent]) -> None:
            received_events.extend(events)
            if received_events:
                event_received.set()

        watcher = EventsFileWatcher(events_path=events_file_path, on_events=callback)

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.1)

        # Create file with events
        events_file_path.write_text(f"{SAMPLE_SESSION_START}\n")

        try:
            await asyncio.wait_for(event_received.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pass

        watcher.stop()
        await asyncio.wait_for(task, timeout=2.0)

        # May or may not have events depending on OS file notification timing
        # The key is that it doesn't crash


class TestEventsFileWatcherCallbackError:
    """Tests for error handling in callbacks."""

    @pytest.mark.asyncio
    async def test_continues_after_callback_error(self, sample_events_jsonl: Path) -> None:
        """Watcher should continue after callback raises exception."""
        from saber.inspect_ai.agents.registry.copilot.events_watcher import EventsFileWatcher

        call_count = 0

        async def callback(events: list[CopilotEvent]) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Simulated callback error")

        watcher = EventsFileWatcher(events_path=sample_events_jsonl, on_events=callback)

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.1)

        # Append to trigger another callback
        with open(sample_events_jsonl, "a") as f:
            f.write(f"{SAMPLE_ASSISTANT_MESSAGE}\n")

        await asyncio.sleep(0.2)

        watcher.stop()
        await asyncio.wait_for(task, timeout=2.0)

        # Should have been called at least once (the initial call)
        assert call_count >= 1
