"""Unit tests for TranscriptStateMachine core state detection."""

import asyncio
import pytest
import time
from saber.models.constants import MetadataKeys
from saber.server.episodes.transcript_state_machine import (
    TranscriptState,
    TranscriptStateMachine,
)
from saber.server.base import Episode, EpisodeState


class TestTranscriptStateDetection:
    """Test state detection from transcript messages."""

    def test_empty_transcript_is_waiting_for_user(self):
        """Empty transcript should be WAITING_FOR_USER."""
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={MetadataKeys.CLIENT_TRANSCRIPT: []}
        )

        state_machine = TranscriptStateMachine()
        state = state_machine.get_state(episode)

        assert state == TranscriptState.WAITING_FOR_USER

    def test_user_message_is_waiting_for_assistant(self):
        """Last message from user → WAITING_FOR_ASSISTANT."""
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"}
                ]
            }
        )

        state_machine = TranscriptStateMachine()
        state = state_machine.get_state(episode)

        assert state == TranscriptState.WAITING_FOR_ASSISTANT

    def test_assistant_without_tool_calls_is_waiting_for_user(self):
        """Assistant message without tool_calls → WAITING_FOR_USER."""
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        )

        state_machine = TranscriptStateMachine()
        state = state_machine.get_state(episode)

        assert state == TranscriptState.WAITING_FOR_USER

    def test_assistant_with_tool_calls_is_waiting_for_tools(self):
        """Assistant message with tool_calls → WAITING_FOR_TOOLS."""
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "user", "content": "List files"},
                    {
                        "role": "assistant",
                        "content": "I'll list the files",
                        "tool_calls": [
                            {"id": "1", "type": "function", "function": {"name": "bash"}}
                        ]
                    }
                ]
            }
        )

        state_machine = TranscriptStateMachine()
        state = state_machine.get_state(episode)

        assert state == TranscriptState.WAITING_FOR_TOOLS

    def test_tool_result_is_waiting_for_user(self):
        """Tool result message → WAITING_FOR_USER."""
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": "1", "type": "function"}]
                    },
                    {"role": "tool", "tool_call_id": "1", "content": "file.txt"}
                ]
            }
        )

        state_machine = TranscriptStateMachine()
        state = state_machine.get_state(episode)

        assert state == TranscriptState.WAITING_FOR_USER

    def test_system_messages_are_skipped(self):
        """System messages don't affect state detection."""
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                    {"role": "system", "content": "Injected system message"}
                ]
            }
        )

        state_machine = TranscriptStateMachine()
        state = state_machine.get_state(episode)

        # Should ignore system and look at assistant
        assert state == TranscriptState.WAITING_FOR_USER

    def test_only_system_messages(self):
        """Transcript with only system messages → WAITING_FOR_USER."""
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "System message 1"},
                    {"role": "system", "content": "System message 2"}
                ]
            }
        )

        state_machine = TranscriptStateMachine()
        state = state_machine.get_state(episode)

        # Should return default state when only system messages
        assert state == TranscriptState.WAITING_FOR_USER


class TestTranscriptValidation:
    """Test transcript structure validation."""

    def test_malformed_transcript_missing_role(self):
        """Transcript with missing role → ERROR_MALFORMED."""
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"content": "Hello"}  # Missing role
                ]
            }
        )

        state_machine = TranscriptStateMachine()
        state = state_machine.get_state(episode)

        assert state == TranscriptState.ERROR_MALFORMED

    def test_malformed_transcript_invalid_role(self):
        """Transcript with invalid role → ERROR_MALFORMED."""
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "invalid_role", "content": "Hello"}
                ]
            }
        )

        state_machine = TranscriptStateMachine()
        state = state_machine.get_state(episode)

        assert state == TranscriptState.ERROR_MALFORMED

    def test_malformed_transcript_invalid_tool_calls(self):
        """Transcript with malformed tool_calls → ERROR_MALFORMED."""
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {
                        "role": "assistant",
                        "content": "Calling tool",
                        "tool_calls": "not a list"  # Should be list
                    }
                ]
            }
        )

        state_machine = TranscriptStateMachine()
        state = state_machine.get_state(episode)

        assert state == TranscriptState.ERROR_MALFORMED


class TestStateTransitionValidation:
    """Test state transition validation logic."""

    def test_valid_transition_user_to_assistant(self):
        """WAITING_FOR_USER → WAITING_FOR_ASSISTANT is valid."""
        state_machine = TranscriptStateMachine()

        is_valid = state_machine.is_valid_transition(
            TranscriptState.WAITING_FOR_USER,
            TranscriptState.WAITING_FOR_ASSISTANT
        )

        assert is_valid is True

    def test_valid_transition_assistant_to_tools(self):
        """WAITING_FOR_ASSISTANT → WAITING_FOR_TOOLS is valid."""
        state_machine = TranscriptStateMachine()

        is_valid = state_machine.is_valid_transition(
            TranscriptState.WAITING_FOR_ASSISTANT,
            TranscriptState.WAITING_FOR_TOOLS
        )

        assert is_valid is True

    def test_valid_transition_assistant_to_user(self):
        """WAITING_FOR_ASSISTANT → WAITING_FOR_USER is valid."""
        state_machine = TranscriptStateMachine()

        is_valid = state_machine.is_valid_transition(
            TranscriptState.WAITING_FOR_ASSISTANT,
            TranscriptState.WAITING_FOR_USER
        )

        assert is_valid is True

    def test_valid_transition_tools_to_user(self):
        """WAITING_FOR_TOOLS → WAITING_FOR_USER is valid."""
        state_machine = TranscriptStateMachine()

        is_valid = state_machine.is_valid_transition(
            TranscriptState.WAITING_FOR_TOOLS,
            TranscriptState.WAITING_FOR_USER
        )

        assert is_valid is True

    def test_invalid_transition_tools_to_tools(self):
        """WAITING_FOR_TOOLS → WAITING_FOR_TOOLS is invalid (self-transition)."""
        state_machine = TranscriptStateMachine()

        is_valid = state_machine.is_valid_transition(
            TranscriptState.WAITING_FOR_TOOLS,
            TranscriptState.WAITING_FOR_TOOLS
        )

        # Same state is always valid (no-op)
        assert is_valid is True

    def test_invalid_transition_user_to_tools(self):
        """WAITING_FOR_USER → WAITING_FOR_TOOLS is invalid."""
        state_machine = TranscriptStateMachine()

        is_valid = state_machine.is_valid_transition(
            TranscriptState.WAITING_FOR_USER,
            TranscriptState.WAITING_FOR_TOOLS
        )

        assert is_valid is False

    def test_invalid_transition_tools_to_assistant(self):
        """WAITING_FOR_TOOLS → WAITING_FOR_ASSISTANT is invalid."""
        state_machine = TranscriptStateMachine()

        is_valid = state_machine.is_valid_transition(
            TranscriptState.WAITING_FOR_TOOLS,
            TranscriptState.WAITING_FOR_ASSISTANT
        )

        assert is_valid is False


class TestEventTypeMapping:
    """Test state to event type conversion."""

    def test_waiting_for_user_maps_to_is_waiting_on_user(self):
        """WAITING_FOR_USER → is_waiting_on_user."""
        event_type = TranscriptStateMachine.state_to_event_type(
            TranscriptState.WAITING_FOR_USER
        )
        assert event_type == "is_waiting_on_user"

    def test_waiting_for_assistant_maps_to_is_waiting_on_assistant(self):
        """WAITING_FOR_ASSISTANT → is_waiting_on_assistant."""
        event_type = TranscriptStateMachine.state_to_event_type(
            TranscriptState.WAITING_FOR_ASSISTANT
        )
        assert event_type == "is_waiting_on_assistant"

    def test_waiting_for_tools_maps_to_is_waiting_on_tools(self):
        """WAITING_FOR_TOOLS → is_waiting_on_tools."""
        event_type = TranscriptStateMachine.state_to_event_type(
            TranscriptState.WAITING_FOR_TOOLS
        )
        assert event_type == "is_waiting_on_tools"

    def test_error_stuck_maps_to_transcript_error(self):
        """ERROR_STUCK → transcript_error."""
        event_type = TranscriptStateMachine.state_to_event_type(
            TranscriptState.ERROR_STUCK
        )
        assert event_type == "transcript_error"

    def test_error_malformed_maps_to_transcript_error(self):
        """ERROR_MALFORMED → transcript_error."""
        event_type = TranscriptStateMachine.state_to_event_type(
            TranscriptState.ERROR_MALFORMED
        )
        assert event_type == "transcript_error"


class TestStateTimestampTracking:
    """Test state timestamp tracking and stuck detection."""

    @pytest.mark.asyncio
    async def test_state_timestamp_updated(self):
        """update_state_timestamp should update timestamp."""
        state_machine = TranscriptStateMachine()

        # Initialize episode
        await state_machine.on_episode_created("ep-1")

        # Get initial timestamp
        initial_time = state_machine._state_timestamps["ep-1"]

        # Wait a bit
        await asyncio.sleep(0.05)

        # Update timestamp
        state_machine.update_state_timestamp("ep-1")

        # Timestamp should be updated
        new_time = state_machine._state_timestamps["ep-1"]
        assert new_time > initial_time

    @pytest.mark.asyncio
    async def test_is_episode_stuck_below_threshold(self):
        """Episodes below threshold should not be stuck."""
        state_machine = TranscriptStateMachine()

        # Initialize episode
        await state_machine.on_episode_created("ep-1")

        # Check immediately - should not be stuck
        is_stuck = state_machine.is_episode_stuck("ep-1", threshold_seconds=1.0)
        assert is_stuck is False

    @pytest.mark.asyncio
    async def test_is_episode_stuck_above_threshold(self):
        """Episodes above threshold should be stuck."""
        state_machine = TranscriptStateMachine()

        # Initialize episode
        await state_machine.on_episode_created("ep-1")

        # Wait past threshold
        await asyncio.sleep(0.15)

        # Check with short threshold - should be stuck
        is_stuck = state_machine.is_episode_stuck("ep-1", threshold_seconds=0.1)
        assert is_stuck is True

    @pytest.mark.asyncio
    async def test_get_state_duration(self):
        """Should return accurate duration in current state."""
        state_machine = TranscriptStateMachine()

        # Initialize episode
        await state_machine.on_episode_created("ep-1")

        # Wait a bit
        await asyncio.sleep(0.1)

        # Get duration
        duration = state_machine.get_state_duration("ep-1")

        # Should be approximately 0.1 seconds
        assert duration >= 0.1
        assert duration < 0.2

    @pytest.mark.asyncio
    async def test_get_state_duration_untracked_episode(self):
        """Should return 0.0 for untracked episodes."""
        state_machine = TranscriptStateMachine()

        # Get duration for non-existent episode
        duration = state_machine.get_state_duration("ep-999")

        assert duration == 0.0

    @pytest.mark.asyncio
    async def test_is_episode_stuck_untracked_episode(self):
        """Should return False for untracked episodes."""
        state_machine = TranscriptStateMachine()

        # Check non-existent episode
        is_stuck = state_machine.is_episode_stuck("ep-999", threshold_seconds=1.0)

        assert is_stuck is False
