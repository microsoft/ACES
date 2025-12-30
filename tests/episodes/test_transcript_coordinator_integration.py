"""Integration tests for TranscriptCoordinator with state machine."""

import asyncio
import pytest
from saber.models.constants import MetadataKeys
from saber.server.episodes.episode_manager import EpisodeManager
from saber.server.episodes.connection_manager import ConnectionManager
from saber.server.episodes.transcript_coordinator import TranscriptCoordinator
from saber.server.base import EpisodeState


@pytest.mark.asyncio
class TestCoordinatorStateMachineIntegration:
    """Test integration between coordinator and state machine."""

    async def test_notify_modification_triggers_auto_continue(self):
        """Red team injection should trigger auto-continue when reaching WAITING_FOR_USER."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Create episode with auto-continue enabled
        episode = episode_manager.start_episode("session-1", "task-1")
        episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = [
            {"role": "user", "content": "Hello"}
        ]
        episode.context[MetadataKeys.TRANSCRIPT_VERSION] = 1
        episode.context[MetadataKeys.AUTO_CONTINUE_ENABLED] = True

        # Red team injects assistant response (no tool_calls)
        modified_transcript = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        await coordinator.notify_modification(
            episode_id=episode.episode_id,
            modified_transcript=modified_transcript,
            operation="append",
            injected_by="red-team",
        )

        # Wait for async auto-continue to complete
        await asyncio.sleep(0.1)

        # Should have 3 messages: user, assistant, auto-continue user
        final_transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(final_transcript) == 3
        assert final_transcript[2]["role"] == "user"
        assert "Continue" in final_transcript[2]["content"]

    async def test_lifecycle_hooks_prevent_memory_leaks(self):
        """Episode termination should clean up state machine resources."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Create and terminate episode
        episode = episode_manager.start_episode("session-1", "task-1")
        episode_id = episode.episode_id

        # Wait for async lifecycle hook to complete
        await asyncio.sleep(0.01)

        # Check state machine initialized locks
        assert episode_id in coordinator._state_machine._episode_locks

        # Terminate episode
        await episode_manager.end_episode(episode_id, "test_cleanup")

        # State machine should clean up
        assert episode_id not in coordinator._state_machine._episode_locks

    async def test_state_included_in_websocket_event(self):
        """WebSocket events should include state information."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Mock connection to capture events
        events_captured = []

        async def mock_broadcast(episode_id, message):
            events_captured.append({
                "episode_id": episode_id,
                "message": message
            })

        connection_manager.broadcast_to_episode = mock_broadcast

        # Create episode and inject modification
        episode = episode_manager.start_episode("session-1", "task-1")
        episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = []
        episode.context[MetadataKeys.AUTO_CONTINUE_ENABLED] = False

        modified_transcript = [{"role": "user", "content": "Hello"}]

        await coordinator.notify_modification(
            episode_id=episode.episode_id,
            modified_transcript=modified_transcript,
            operation="append",
            injected_by="test",
        )

        # Check event format
        assert len(events_captured) == 1
        event = events_captured[0]
        assert event["message"].type == "is_waiting_on_assistant"
        assert event["message"].data.state == "WAITING_FOR_ASSISTANT"

    async def test_auto_continue_does_not_trigger_on_tool_calls(self):
        """Auto-continue should NOT trigger when assistant has tool_calls."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Create episode with auto-continue enabled
        episode = episode_manager.start_episode("session-1", "task-1")
        episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = [
            {"role": "user", "content": "Hello"}
        ]
        episode.context[MetadataKeys.TRANSCRIPT_VERSION] = 1
        episode.context[MetadataKeys.AUTO_CONTINUE_ENABLED] = True

        # Red team injects assistant response with tool_calls
        modified_transcript = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Let me help", "tool_calls": [{"id": "1", "function": {"name": "test"}}]}
        ]

        await coordinator.notify_modification(
            episode_id=episode.episode_id,
            modified_transcript=modified_transcript,
            operation="append",
            injected_by="red-team",
        )

        # Wait for any async operations
        await asyncio.sleep(0.1)

        # Should still have 2 messages (no auto-continue triggered)
        final_transcript = episode.context[MetadataKeys.CLIENT_TRANSCRIPT]
        assert len(final_transcript) == 2

    async def test_websocket_event_type_changes_with_state(self):
        """Event type should match the current state (is_waiting_on_*)."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        events_captured = []

        async def mock_broadcast(episode_id, message):
            events_captured.append(message)

        connection_manager.broadcast_to_episode = mock_broadcast

        # Create episode
        episode = episode_manager.start_episode("session-1", "task-1")
        episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = []
        episode.context[MetadataKeys.AUTO_CONTINUE_ENABLED] = False

        # Scenario 1: User message → WAITING_FOR_ASSISTANT
        await coordinator.notify_modification(
            episode_id=episode.episode_id,
            modified_transcript=[{"role": "user", "content": "Hello"}],
            operation="append",
            injected_by="test",
        )

        assert events_captured[-1].type == "is_waiting_on_assistant"

        # Scenario 2: Assistant with tool_calls → WAITING_FOR_TOOLS
        await coordinator.notify_modification(
            episode_id=episode.episode_id,
            modified_transcript=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Let me check", "tool_calls": [{"id": "1"}]}
            ],
            operation="append",
            injected_by="test",
        )

        assert events_captured[-1].type == "is_waiting_on_tools"

        # Scenario 3: Tool response → WAITING_FOR_USER (assistant needs to respond)
        await coordinator.notify_modification(
            episode_id=episode.episode_id,
            modified_transcript=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Let me check", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "result", "tool_call_id": "1"}
            ],
            operation="append",
            injected_by="test",
        )

        # Tool message means waiting for user (who can continue the conversation)
        assert events_captured[-1].type == "is_waiting_on_user"

    async def test_stuck_state_monitor_integration(self):
        """End-to-end: stuck episodes trigger error events."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        # Use fast check interval for testing
        coordinator = TranscriptCoordinator(
            episode_manager,
            connection_manager,
            stuck_check_interval=0.05,
            stuck_threshold=0.1
        )

        # Set up event capture BEFORE starting monitor
        events = []
        async def capture(episode_id, message):
            events.append(message)

        connection_manager.broadcast_to_episode = capture

        # Start monitor
        await coordinator.start_monitor()

        try:
            # Create episode that will become stuck
            episode = episode_manager.start_episode("session-1", "task-1")
            episode.context[MetadataKeys.CLIENT_TRANSCRIPT] = [
                {"role": "user", "content": "test"}
            ]

            await asyncio.sleep(0.02)

            # Wait for stuck detection (threshold + 2 check intervals + buffer)
            await asyncio.sleep(0.3)

            # Verify error event
            errors = [e for e in events if e.type == "transcript_error"]
            assert len(errors) > 0
            assert errors[0].data.error == "stuck_state"

        finally:
            await coordinator.stop_monitor()
