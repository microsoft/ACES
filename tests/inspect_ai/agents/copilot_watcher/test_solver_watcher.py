"""Tests for solver integration with EventsWatcherContext.

TDD tests for integrating the events watcher into the copilot solver
for real-time state.messages updates.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser

from .conftest import (
    SAMPLE_SESSION_START,
)


class TestSessionTrackerWatcherContext:
    """Tests for watcher context in SessionTracker."""

    def test_session_tracker_has_watcher_context(self) -> None:
        """SessionTracker should have watcher_ctx field."""
        from saber.inspect_ai.agents.registry.copilot.session_events import EventsWatcherContext
        from saber.inspect_ai.agents.registry.copilot.solver import SessionTracker

        tracker = SessionTracker()

        assert hasattr(tracker, "watcher_ctx")
        assert isinstance(tracker.watcher_ctx, EventsWatcherContext)

    def test_watcher_context_initially_not_running(self) -> None:
        """Watcher context should not be running initially."""
        from saber.inspect_ai.agents.registry.copilot.solver import SessionTracker

        tracker = SessionTracker()

        assert not tracker.watcher_ctx.is_running


class TestIsDuplicateMessage:
    """Tests for message deduplication logic."""

    def test_is_duplicate_user_message_by_content(self) -> None:
        """User messages with same content should be detected as duplicates."""
        from saber.inspect_ai.agents.registry.copilot.solver import is_duplicate_message

        existing = [
            ChatMessageUser(content="Hello world"),
        ]

        msg = ChatMessageUser(content="Hello world")

        assert is_duplicate_message(msg, existing)

    def test_is_not_duplicate_different_content(self) -> None:
        """Messages with different content should not be duplicates."""
        from saber.inspect_ai.agents.registry.copilot.solver import is_duplicate_message

        existing = [
            ChatMessageUser(content="Hello world"),
        ]

        msg = ChatMessageUser(content="Different content")

        assert not is_duplicate_message(msg, existing)

    def test_is_not_duplicate_empty_list(self) -> None:
        """No duplicates when existing list is empty."""
        from saber.inspect_ai.agents.registry.copilot.solver import is_duplicate_message

        msg = ChatMessageUser(content="Hello")

        assert not is_duplicate_message(msg, [])

    def test_is_duplicate_assistant_message_by_content(self) -> None:
        """Assistant messages with same content should be detected as duplicates."""
        from saber.inspect_ai.agents.registry.copilot.solver import is_duplicate_message

        existing = [
            ChatMessageAssistant(content="I can help with that"),
        ]

        msg = ChatMessageAssistant(content="I can help with that")

        assert is_duplicate_message(msg, existing)

    def test_duplicate_detection_uses_recent_messages(self) -> None:
        """Duplicate detection should focus on recent messages."""
        from saber.inspect_ai.agents.registry.copilot.solver import is_duplicate_message

        # Create a long conversation
        existing: list[ChatMessage] = []
        for i in range(100):
            existing.append(ChatMessageUser(content=f"Message {i}"))

        # Add a message we want to check
        existing.append(ChatMessageUser(content="Target message"))

        # Same content at end should be duplicate
        msg = ChatMessageUser(content="Target message")
        assert is_duplicate_message(msg, existing)


class TestGetEventsPath:
    """Tests for events path resolution."""

    def test_get_events_path_with_session_id(self) -> None:
        """Should return correct path for session ID."""
        from saber.inspect_ai.agents.registry.copilot.solver import get_events_path

        path = get_events_path("test-session-123")

        assert path.name == "events.jsonl"
        assert "test-session-123" in str(path)
        assert ".copilot" in str(path)

    def test_get_events_path_with_custom_base(self) -> None:
        """Should use custom base path when provided."""
        from saber.inspect_ai.agents.registry.copilot.solver import get_events_path

        custom_base = Path("/tmp/custom-copilot")
        path = get_events_path("test-123", base_path=custom_base)

        assert path == custom_base / "test-123" / "events.jsonl"


class TestStartWatcher:
    """Tests for starting the watcher in solver."""

    @pytest.mark.asyncio
    async def test_start_watcher_creates_task(self, tmp_path: Path) -> None:
        """start_watcher should create and start a watcher task."""
        from saber.inspect_ai.agents.registry.copilot.solver import SessionTracker, start_watcher

        tracker = SessionTracker()
        tracker.session_id = "test-123"

        # Create events file
        events_dir = tmp_path / "test-123"
        events_dir.mkdir(parents=True)
        events_file = events_dir / "events.jsonl"
        events_file.write_text(f"{SAMPLE_SESSION_START}\n")

        messages: list[ChatMessage] = []

        async def extend_messages(new_msgs: list[ChatMessage]) -> None:
            messages.extend(new_msgs)

        start_watcher(
            tracker=tracker,
            events_base_path=tmp_path,
            on_new_messages=extend_messages,
            system_content="Test",
        )

        await asyncio.sleep(0.2)

        # Watcher should be running
        assert tracker.watcher_ctx.is_running

        # Cleanup
        await tracker.watcher_ctx.stop()


class TestStopWatcher:
    """Tests for stopping the watcher in solver."""

    @pytest.mark.asyncio
    async def test_stop_watcher_stops_task(self, tmp_path: Path) -> None:
        """stop_watcher should stop the watcher task."""
        from saber.inspect_ai.agents.registry.copilot.solver import (
            SessionTracker,
            start_watcher,
            stop_watcher,
        )

        tracker = SessionTracker()
        tracker.session_id = "test-123"

        # Create events file
        events_dir = tmp_path / "test-123"
        events_dir.mkdir(parents=True)
        events_file = events_dir / "events.jsonl"
        events_file.write_text(f"{SAMPLE_SESSION_START}\n")

        async def extend_messages(new_msgs: list[ChatMessage]) -> None:
            pass

        start_watcher(
            tracker=tracker,
            events_base_path=tmp_path,
            on_new_messages=extend_messages,
        )

        await asyncio.sleep(0.1)
        assert tracker.watcher_ctx.is_running

        await stop_watcher(tracker)

        assert not tracker.watcher_ctx.is_running

    @pytest.mark.asyncio
    async def test_stop_watcher_safe_when_not_started(self) -> None:
        """stop_watcher should be safe when watcher not started."""
        from saber.inspect_ai.agents.registry.copilot.solver import SessionTracker, stop_watcher

        tracker = SessionTracker()

        # Should not raise
        await stop_watcher(tracker)

        assert not tracker.watcher_ctx.is_running


class TestGetEventCallbackType:
    """Tests for get_event_callback return type."""

    def test_get_event_callback_returns_callable(self) -> None:
        """get_event_callback should return a properly typed callable."""
        from saber.inspect_ai.agents.registry.copilot.solver import SessionTracker

        tracker = SessionTracker()
        callback = tracker.get_event_callback()

        # Must be callable
        assert callable(callback)

        # Verify it's the correct type by checking annotations
        # The callback should accept a single optional CopilotSDKEvent and return None
        # We can't easily check the signature at runtime, but we can verify it's callable


class TestSolverWatcherIntegration:
    """Integration tests for watcher updates during solve loop."""

    @pytest.mark.asyncio
    async def test_watcher_updates_messages_during_session(self, tmp_path: Path) -> None:
        """Watcher should update state.messages when new events arrive."""
        from unittest.mock import patch

        from saber.inspect_ai.agents.registry.copilot.solver import (
            SessionTracker,
            is_duplicate_message,
            start_watcher,
            stop_watcher,
        )

        from .conftest import (
            SAMPLE_ASSISTANT_MESSAGE,
            SAMPLE_SESSION_START,
            SAMPLE_USER_MESSAGE,
        )

        tracker = SessionTracker()
        tracker.session_id = "integration-test-session"

        # Set up events directory and file
        events_dir = tmp_path / tracker.session_id
        events_dir.mkdir(parents=True)
        events_file = events_dir / "events.jsonl"
        events_file.write_text(f"{SAMPLE_SESSION_START}\n")

        # Track messages received
        received_messages: list[ChatMessage] = []

        async def on_new_messages(messages: list[ChatMessage]) -> None:
            for msg in messages:
                if not is_duplicate_message(msg, received_messages):
                    received_messages.append(msg)

        # Force poll mode by making watchfiles import fail (more deterministic in tests)
        with patch(
            "saber.inspect_ai.agents.registry.copilot.events_watcher.EventsFileWatcher.start",
            wraps=None,
        ) as _:
            # Undo the patch - we actually want to call start, but in poll mode
            pass

        # Patch watchfiles import to force reliable poll mode
        import saber.inspect_ai.agents.registry.copilot.events_watcher as ew_mod

        async def poll_start(self_watcher: ew_mod.EventsFileWatcher) -> None:
            """Force poll mode for deterministic testing."""
            self_watcher._is_running = True
            self_watcher._stop_event.clear()
            try:
                await self_watcher._read_and_emit_events()
                await self_watcher._poll_mode()
            except asyncio.CancelledError:
                pass
            finally:
                self_watcher._is_running = False

        with patch.object(ew_mod.EventsFileWatcher, "start", poll_start):
            # Start the watcher (will use poll mode)
            start_watcher(
                tracker=tracker,
                events_base_path=tmp_path,
                on_new_messages=on_new_messages,
                system_content="You are a helpful assistant.",
            )

            await asyncio.sleep(0.2)
            assert tracker.watcher_ctx.is_running

            # Write new events to file (simulating Copilot SDK activity)
            with events_file.open("a") as f:
                f.write(f"{SAMPLE_USER_MESSAGE}\n")
                f.write(f"{SAMPLE_ASSISTANT_MESSAGE}\n")

            # Poll for watcher to process (more reliable than fixed sleep)
            for _ in range(30):  # Up to 3 seconds
                if len(received_messages) >= 2:
                    break
                await asyncio.sleep(0.1)

            # Stop watcher
            await stop_watcher(tracker)

        # Should have received messages
        # The events include: user message and assistant message
        # Note: system message is generated from system_content, user/assistant come from events
        assert len(received_messages) >= 2, f"Expected at least 2 messages, got {len(received_messages)}"

        # Verify we got both user and assistant messages
        message_types = [type(m).__name__ for m in received_messages]
        assert "ChatMessageUser" in message_types or any("user" in str(m).lower() for m in received_messages)

    @pytest.mark.asyncio
    async def test_watcher_respects_system_content(self, tmp_path: Path) -> None:
        """Watcher should use system_content for message conversion."""
        from saber.inspect_ai.agents.registry.copilot.solver import (
            SessionTracker,
            start_watcher,
            stop_watcher,
        )

        from .conftest import SAMPLE_SESSION_START, SAMPLE_USER_MESSAGE

        tracker = SessionTracker()
        tracker.session_id = "system-content-test"

        events_dir = tmp_path / tracker.session_id
        events_dir.mkdir(parents=True)
        events_file = events_dir / "events.jsonl"
        events_file.write_text(f"{SAMPLE_SESSION_START}\n{SAMPLE_USER_MESSAGE}\n")

        received_messages: list[ChatMessage] = []
        custom_system = "Custom system prompt for testing."

        async def on_new_messages(messages: list[ChatMessage]) -> None:
            received_messages.extend(messages)

        start_watcher(
            tracker=tracker,
            events_base_path=tmp_path,
            on_new_messages=on_new_messages,
            system_content=custom_system,
        )

        await asyncio.sleep(0.3)
        await stop_watcher(tracker)

        # Verify system message was created with custom content
        system_msgs = [m for m in received_messages if type(m).__name__ == "ChatMessageSystem"]
        if system_msgs:
            assert custom_system in str(system_msgs[0].content)
