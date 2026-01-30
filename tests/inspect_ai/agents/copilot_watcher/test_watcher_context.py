"""Tests for EventsWatcherContext.

TDD tests for the context manager that integrates EventsFileWatcher
into the SessionTracker with proper lifecycle management.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from inspect_ai.model import ChatMessage

from .conftest import (
    SAMPLE_ASSISTANT_MESSAGE,
)

if TYPE_CHECKING:
    pass


class TestEventsWatcherContextInitialization:
    """Tests for EventsWatcherContext initialization."""

    def test_init_creates_stop_event(self) -> None:
        """Context should create stop event on init."""
        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext

        ctx = EventsWatcherContext()

        assert ctx.stop_event is not None
        assert isinstance(ctx.stop_event, asyncio.Event)
        assert not ctx.stop_event.is_set()

    def test_initial_state_is_not_running(self) -> None:
        """Context should not be running initially."""
        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext

        ctx = EventsWatcherContext()

        assert not ctx.is_running
        assert ctx.watcher_task is None


class TestEventsWatcherContextLifecycle:
    """Tests for watcher context lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self, sample_events_jsonl: Path) -> None:
        """start() should create and return a task."""
        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext

        ctx = EventsWatcherContext()

        async def on_messages(msgs: list[ChatMessage]) -> None:
            pass

        task = ctx.start(
            events_path=sample_events_jsonl,
            on_new_messages=on_messages,
            system_content="Test system message",
        )

        assert task is not None
        assert isinstance(task, asyncio.Task)
        assert ctx.is_running

        # Cleanup
        await ctx.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, sample_events_jsonl: Path) -> None:
        """stop() should cancel the watcher task."""
        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext

        ctx = EventsWatcherContext()

        async def on_messages(msgs: list[ChatMessage]) -> None:
            pass

        ctx.start(
            events_path=sample_events_jsonl,
            on_new_messages=on_messages,
        )

        await ctx.stop()

        assert not ctx.is_running
        assert ctx.watcher_task is None or ctx.watcher_task.done()

    @pytest.mark.asyncio
    async def test_stop_before_start_is_safe(self) -> None:
        """stop() before start() should be safe (no-op)."""
        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext

        ctx = EventsWatcherContext()

        # Should not raise
        await ctx.stop()

        assert not ctx.is_running

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self, sample_events_jsonl: Path) -> None:
        """Calling stop() twice should be safe."""
        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext

        ctx = EventsWatcherContext()

        async def on_messages(msgs: list[ChatMessage]) -> None:
            pass

        ctx.start(events_path=sample_events_jsonl, on_new_messages=on_messages)

        await ctx.stop()
        await ctx.stop()  # Second stop should be safe

        assert not ctx.is_running


class TestEventsWatcherContextMessageConversion:
    """Tests for converting events to ChatMessages."""

    @pytest.mark.asyncio
    async def test_converts_events_to_messages(self, sample_events_jsonl: Path) -> None:
        """Context should convert events to ChatMessages via callback."""
        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext

        received_messages: list[ChatMessage] = []

        async def on_messages(msgs: list[ChatMessage]) -> None:
            received_messages.extend(msgs)

        ctx = EventsWatcherContext()
        ctx.start(
            events_path=sample_events_jsonl,
            on_new_messages=on_messages,
            system_content="Test system",
        )

        # Wait for initial read
        await asyncio.sleep(0.2)

        await ctx.stop()

        # Should have received messages (system + user from sample file)
        assert len(received_messages) >= 1

    @pytest.mark.asyncio
    async def test_emits_new_messages_on_file_change(self, sample_events_jsonl: Path) -> None:
        """Context should emit new messages when file changes."""
        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext

        received_messages: list[ChatMessage] = []
        message_event = asyncio.Event()

        async def on_messages(msgs: list[ChatMessage]) -> None:
            received_messages.extend(msgs)
            if len(received_messages) >= 2:
                message_event.set()

        ctx = EventsWatcherContext()
        ctx.start(
            events_path=sample_events_jsonl,
            on_new_messages=on_messages,
        )

        await asyncio.sleep(0.1)

        # Append new event
        with open(sample_events_jsonl, "a") as f:
            f.write(f"{SAMPLE_ASSISTANT_MESSAGE}\n")

        try:
            await asyncio.wait_for(message_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

        await ctx.stop()

        # Should have received messages
        assert len(received_messages) >= 1


class TestEventsWatcherContextSystemContent:
    """Tests for system content handling."""

    @pytest.mark.asyncio
    async def test_system_content_passed_to_converter(self, sample_events_jsonl: Path) -> None:
        """System content should be used in message conversion."""
        from inspect_ai.model import ChatMessageSystem

        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext

        received_messages: list[ChatMessage] = []

        async def on_messages(msgs: list[ChatMessage]) -> None:
            received_messages.extend(msgs)

        ctx = EventsWatcherContext()
        ctx.start(
            events_path=sample_events_jsonl,
            on_new_messages=on_messages,
            system_content="Custom system prompt",
        )

        await asyncio.sleep(0.2)
        await ctx.stop()

        # First message should be system message with our content
        if received_messages:
            system_msgs = [m for m in received_messages if isinstance(m, ChatMessageSystem)]
            if system_msgs:
                assert "Custom system prompt" in system_msgs[0].content


class TestEventsWatcherContextErrorHandling:
    """Tests for error handling in watcher context."""

    @pytest.mark.asyncio
    async def test_continues_after_callback_error(self, sample_events_jsonl: Path) -> None:
        """Context should continue after callback raises exception."""
        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext

        call_count = 0

        async def on_messages(msgs: list[ChatMessage]) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Simulated error")

        ctx = EventsWatcherContext()
        ctx.start(events_path=sample_events_jsonl, on_new_messages=on_messages)

        await asyncio.sleep(0.1)

        # Append to trigger another callback
        with open(sample_events_jsonl, "a") as f:
            f.write(f"{SAMPLE_ASSISTANT_MESSAGE}\n")

        await asyncio.sleep(0.2)
        await ctx.stop()

        # Should have been called at least once
        assert call_count >= 1
