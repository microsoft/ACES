"""Unit tests for TranscriptCoordinator with WebSocket notifications.

Tests cover Phase 2:
- Version tracking (monotonic sequence numbers)
- SHA256 checksum validation (restart detection)
- Differential sync (delta vs full transcript)
- WebSocket notification broadcasting
- Last write wins semantics
- Episode isolation
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.models.constants import MetadataKeys
from saber.models.transcript import (
    SyncStrategy,
    TranscriptSyncRequest,
    TranscriptSyncResponse,
    TranscriptVersion,
    compute_checksum,
)
from saber.server.base import Episode, EpisodeState
from saber.server.episodes.transcript_coordinator import TranscriptCoordinator


class MockConnectionManager:
    """Mock ConnectionManager for testing."""

    def __init__(self):
        self.broadcasts = []

    async def broadcast_to_episode(self, episode_id: str, message: dict):
        """Record broadcasts for verification."""
        self.broadcasts.append({"episode_id": episode_id, "message": message})

    async def cleanup_episode(self, episode_id: str):
        """Mock cleanup."""
        pass


class MockEpisodeManager:
    """Mock EpisodeManager for testing."""

    def __init__(self):
        self.episodes = {}

    def get_episode_by_id(self, episode_id: str):
        return self.episodes.get(episode_id)

    def start_episode(self, session_id: str, task_id: str, **kwargs):
        """Mock start_episode to support lifecycle hooks."""
        episode_id = f"{session_id}-{task_id}"
        episode = Episode(
            episode_id=episode_id,
            session_id=session_id,
            task_id=task_id,
        )
        self.episodes[episode_id] = episode
        return episode

    def end_episode(self, episode_id: str, **kwargs):
        """Mock end_episode."""
        if episode_id in self.episodes:
            del self.episodes[episode_id]


class TestTranscriptCoordinatorInit:
    """Test TranscriptCoordinator initialization."""

    def test_init_with_dependencies(self):
        """Test coordinator initializes with episode_manager and connection_manager."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()

        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        assert coordinator.episode_manager is episode_manager
        assert coordinator.connection_manager is connection_manager
        assert isinstance(coordinator._coordination_configs, dict)
        assert isinstance(coordinator._lock, asyncio.Lock)


class TestTranscriptCoordinatorChecksumComputation:
    """Test SHA256 checksum computation for restart/modification detection."""

    def test_compute_checksum_empty_transcript(self):
        """Test checksum computation for empty transcript."""
        checksum = compute_checksum([])

        # SHA256 of empty JSON array "[]"
        expected = hashlib.sha256(json.dumps([], sort_keys=True).encode()).hexdigest()
        assert checksum == expected
        assert len(checksum) == 64  # Full SHA256 is 64 hex chars

    def test_compute_checksum_single_message(self):
        """Test checksum computation for single message."""
        messages = [{"role": "user", "content": "Hello"}]
        checksum = compute_checksum(messages)

        expected = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        assert checksum == expected

    def test_compute_checksum_stable_with_reordering(self):
        """Test checksum is stable when keys are reordered (sort_keys=True)."""
        msg1 = [{"content": "Hello", "role": "user"}]
        msg2 = [{"role": "user", "content": "Hello"}]

        checksum1 = compute_checksum(msg1)
        checksum2 = compute_checksum(msg2)

        assert checksum1 == checksum2  # Same despite key order

    def test_compute_checksum_different_for_different_content(self):
        """Test checksum changes when content changes."""
        msg1 = [{"role": "user", "content": "Hello"}]
        msg2 = [{"role": "user", "content": "Hi"}]

        checksum1 = compute_checksum(msg1)
        checksum2 = compute_checksum(msg2)

        assert checksum1 != checksum2


class TestTranscriptCoordinatorGetCurrentVersion:
    """Test getting current version from episode context."""

    @pytest.mark.asyncio
    async def test_get_current_version_empty_episode(self):
        """Test getting version from episode with no transcript."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        episode = Episode(
            episode_id="ep-123",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={},
        )

        version = coordinator._get_current_version(episode)

        assert version.sequence == 0
        assert version.message_count == 0
        assert version.last_operation == "append"
        assert len(version.checksum) == 64

    @pytest.mark.asyncio
    async def test_get_current_version_with_transcript(self):
        """Test getting version from episode with existing transcript."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        episode = Episode(
            episode_id="ep-123",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: messages,
                MetadataKeys.TRANSCRIPT_VERSION: 2,
                MetadataKeys.TRANSCRIPT_LAST_OPERATION: "append",
            },
        )

        version = coordinator._get_current_version(episode)

        assert version.sequence == 2
        assert version.message_count == 2
        assert version.last_operation == "append"
        assert version.checksum == compute_checksum(messages)


class TestTranscriptCoordinatorSync:
    """Test sync() method with differential sync logic."""

    @pytest.mark.asyncio
    async def test_sync_push_new_messages_increments_version(self):
        """Test pushing new messages increments version (last write wins)."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Setup episode with initial transcript
        episode = Episode(
            episode_id="ep-blue-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [{"role": "system", "content": "Initial"}],
                MetadataKeys.TRANSCRIPT_VERSION: 1,
                MetadataKeys.TRANSCRIPT_LAST_OPERATION: "append",
            },
        )
        episode_manager.episodes["ep-blue-1"] = episode

        # Push new messages
        request = TranscriptSyncRequest(
            episode_id="ep-blue-1",
            since_version=1,
            messages_to_push=[{"role": "user", "content": "New message"}],
        )

        response = await coordinator.sync(request)

        # Version should increment
        assert response.current_version.sequence == 2
        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == 2
        assert len(episode.context[MetadataKeys.CLIENT_TRANSCRIPT]) == 2

    @pytest.mark.asyncio
    async def test_sync_no_change_returns_empty_delta(self):
        """Test sync with no changes returns empty delta."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        messages = [{"role": "system", "content": "Test"}]
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: messages,
                MetadataKeys.TRANSCRIPT_VERSION: 1,
            },
        )
        episode_manager.episodes["ep-1"] = episode

        request = TranscriptSyncRequest(episode_id="ep-1", since_version=1)
        response = await coordinator.sync(request)

        assert response.sync_mode == "no_change"
        assert response.delta == []
        assert response.full_transcript is None
        assert not response.modified

    @pytest.mark.asyncio
    async def test_sync_delta_mode_returns_new_messages_only(self):
        """Test delta mode returns only new messages since client version."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        messages = [
            {"role": "system", "content": "Msg1"},
            {"role": "user", "content": "Msg2"},
            {"role": "assistant", "content": "Msg3"},
        ]
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: messages,
                MetadataKeys.TRANSCRIPT_VERSION: 3,
            },
        )
        episode_manager.episodes["ep-1"] = episode

        # Client has version 1 (first message only)
        request = TranscriptSyncRequest(episode_id="ep-1", since_version=1)
        response = await coordinator.sync(request)

        assert response.sync_mode == "delta"
        assert response.delta == messages[1:]  # Last 2 messages
        assert response.full_transcript is None
        assert response.modified

    @pytest.mark.asyncio
    async def test_sync_restart_detected_returns_full_transcript(self):
        """Test restart detection triggers full transcript sync.

        When a restart operation occurs, the client's checksum won't match
        the server's transcript, triggering a full sync instead of delta.
        """
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        current_messages = [
            {"role": "system", "content": "After restart"},
        ]
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: current_messages,
                MetadataKeys.TRANSCRIPT_VERSION: 2,
                MetadataKeys.TRANSCRIPT_LAST_OPERATION: "restart",
            },
        )
        episode_manager.episodes["ep-1"] = episode

        # Client has wrong checksum (different content from before restart)
        wrong_checksum = compute_checksum([{"role": "system", "content": "ORIGINAL"}])
        request = TranscriptSyncRequest(
            episode_id="ep-1",
            since_version=1,
            client_checksum=wrong_checksum,
        )
        response = await coordinator.sync(request)

        assert response.sync_mode == "full"
        assert response.full_transcript == current_messages
        assert response.delta is None
        assert response.modified

    @pytest.mark.asyncio
    async def test_sync_checksum_validation(self):
        """Test checksum validation detects invalid client state."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        messages = [
            {"role": "system", "content": "Test1"},
            {"role": "user", "content": "Test2"},
        ]
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: messages,
                MetadataKeys.TRANSCRIPT_VERSION: 2,
            },
        )
        episode_manager.episodes["ep-1"] = episode

        # Valid checksum should allow delta sync
        valid_checksum = compute_checksum(messages[:1])
        request = TranscriptSyncRequest(
            episode_id="ep-1",
            since_version=1,
            client_checksum=valid_checksum,
        )
        response = await coordinator.sync(request)

        assert response.sync_mode == "delta"


class TestTranscriptCoordinatorNotifyModification:
    """Test notify_modification() method with WebSocket broadcasting."""

    @pytest.mark.asyncio
    async def test_notify_modification_increments_version(self):
        """Test notify_modification increments version monotonically."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Setup episode
        episode = Episode(
            episode_id="ep-blue-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [{"role": "system", "content": "Initial"}],
                MetadataKeys.TRANSCRIPT_VERSION: 1,
                MetadataKeys.TRANSCRIPT_LAST_OPERATION: "append",
            },
        )
        episode_manager.episodes["ep-blue-1"] = episode

        # Notify modification
        modified_transcript = [
            {"role": "system", "content": "Initial"},
            {"role": "system", "content": "Injected!"},
        ]
        await coordinator.notify_modification(
            episode_id="ep-blue-1",
            modified_transcript=modified_transcript,
            operation="append",
            injected_by="ep-red-2",
        )

        # Version should increment from 1 to 2
        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == 2
        assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == "append"

    @pytest.mark.asyncio
    async def test_notify_modification_broadcasts_websocket_event(self):
        """Test notify_modification broadcasts WebSocket event."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Setup episode
        episode = Episode(
            episode_id="ep-blue-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [],
                MetadataKeys.TRANSCRIPT_VERSION: 0,
            },
        )
        episode_manager.episodes["ep-blue-1"] = episode

        # Notify modification
        await coordinator.notify_modification(
            episode_id="ep-blue-1",
            modified_transcript=[{"role": "system", "content": "Injected"}],
            operation="append",
            injected_by="ep-red-2",
        )

        # Verify WebSocket broadcast
        assert len(connection_manager.broadcasts) == 1
        broadcast = connection_manager.broadcasts[0]
        assert broadcast["episode_id"] == "ep-blue-1"
        # Message is now a Pydantic model with state-aware event type
        # After adding a system message to an empty transcript, state is WAITING_FOR_USER
        assert broadcast["message"].type == "is_waiting_on_user"
        assert broadcast["message"].data.version == 1
        assert broadcast["message"].data.operation.value == "append"
        assert broadcast["message"].data.injected_by == "ep-red-2"

    @pytest.mark.asyncio
    async def test_notify_modification_sets_last_operation(self):
        """Test notify_modification records operation type."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [],
                MetadataKeys.TRANSCRIPT_VERSION: 0,
            },
        )
        episode_manager.episodes["ep-1"] = episode

        # Test different operation types (only append and restart are supported now)
        for operation in ["append", "restart"]:
            await coordinator.notify_modification(
                episode_id="ep-1",
                modified_transcript=[{"role": "system", "content": f"Op: {operation}"}],
                operation=operation,
                injected_by="ep-red",
            )

            assert episode.context[MetadataKeys.TRANSCRIPT_LAST_OPERATION] == operation

    @pytest.mark.asyncio
    async def test_notify_modification_increments_modification_count(self):
        """Test modification counter increments."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [],
                MetadataKeys.TRANSCRIPT_VERSION: 0,
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )
        episode_manager.episodes["ep-1"] = episode

        # Multiple modifications
        for i in range(3):
            await coordinator.notify_modification(
                episode_id="ep-1",
                modified_transcript=[{"role": "system", "content": f"Mod {i}"}],
                operation="append",
                injected_by="ep-red",
            )

        assert episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 3

    @pytest.mark.asyncio
    async def test_notify_modification_validates_version(self):
        """Test that notify_modification validates expected_base_version."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        # Setup episode at version 5
        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "Msg1"},
                    {"role": "user", "content": "Msg2"},
                ],
                MetadataKeys.TRANSCRIPT_VERSION: 5,
            },
        )
        episode_manager.episodes["ep-1"] = episode

        # Valid version should succeed
        await coordinator.notify_modification(
            episode_id="ep-1",
            modified_transcript=[{"role": "system", "content": "Msg1"}],
            operation="restart",
            injected_by="red",
            expected_base_version=5,
        )

        # Invalid (stale) version should raise
        with pytest.raises(ValueError, match="Modification based on stale version"):
            await coordinator.notify_modification(
                episode_id="ep-1",
                modified_transcript=[{"role": "system", "content": "Msg1"}],
                operation="append",
                injected_by="red",
                expected_base_version=3,  # Stale - episode is at version 6 now
            )

    @pytest.mark.asyncio
    async def test_notify_modification_validates_checksum(self):
        """Test that notify_modification validates expected_base_checksum."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        messages = [
            {"role": "system", "content": "Test1"},
            {"role": "user", "content": "Test2"},
        ]

        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: messages,
                MetadataKeys.TRANSCRIPT_VERSION: 2,
            },
        )
        episode_manager.episodes["ep-1"] = episode

        # Valid checksum should succeed
        valid_checksum = compute_checksum(messages)
        await coordinator.notify_modification(
            episode_id="ep-1",
            modified_transcript=messages + [{"role": "assistant", "content": "New"}],
            operation="append",
            injected_by="red",
            expected_base_checksum=valid_checksum,
        )

        # Invalid checksum should raise
        wrong_checksum = compute_checksum([{"role": "system", "content": "DIFFERENT"}])
        with pytest.raises(ValueError, match="Modification checksum mismatch"):
            await coordinator.notify_modification(
                episode_id="ep-1",
                modified_transcript=messages,
                operation="rewrite",
                injected_by="red",
                expected_base_checksum=wrong_checksum,
            )

    @pytest.mark.asyncio
    async def test_notify_modification_validation_optional(self):
        """Test that validation parameters are optional (backward compatible)."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [],
                MetadataKeys.TRANSCRIPT_VERSION: 0,
            },
        )
        episode_manager.episodes["ep-1"] = episode

        # Should work without validation parameters (old behavior)
        await coordinator.notify_modification(
            episode_id="ep-1",
            modified_transcript=[{"role": "system", "content": "Test"}],
            operation="append",
            injected_by="red",
            # No expected_base_version or expected_base_checksum
        )

        # Should succeed
        assert episode.context[MetadataKeys.TRANSCRIPT_VERSION] == 1


class TestTranscriptCoordinatorEpisodeIsolation:
    """Test episode-scoped isolation of transcript coordination."""

    @pytest.mark.asyncio
    async def test_modifications_isolated_by_episode(self):
        """Test modifications to one episode don't affect another."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Create two episodes
        episode1 = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [{"role": "user", "content": "Ep1"}],
                "_transcript_version": 1,
            },
        )
        episode2 = Episode(
            episode_id="ep-2",
            task_id="task-2",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [{"role": "user", "content": "Ep2"}],
                "_transcript_version": 1,
            },
        )
        episode_manager.episodes["ep-1"] = episode1
        episode_manager.episodes["ep-2"] = episode2

        # Verify isolation (will test with actual notify_modification once implemented)
        version1 = coordinator._get_current_version(episode1)
        version2 = coordinator._get_current_version(episode2)

        assert version1.checksum != version2.checksum

    @pytest.mark.asyncio
    async def test_websocket_events_routed_to_correct_episode(self):
        """Test WebSocket events are routed only to target episode."""
        # Will be implemented once notify_modification() exists
        pass


class TestTranscriptCoordinatorLastWriteWins:
    """Test last write wins semantics with monotonic versioning."""

    @pytest.mark.asyncio
    async def test_concurrent_modifications_last_write_wins(self):
        """Test concurrent modifications use last write wins."""
        # Scenario:
        # T=0: version 5
        # T=1: Red team modifies (version 6)
        # T=2: Blue team pushes based on version 5
        # T=3: Blue push succeeds (version 7), red's modification at version 6

        # Will be tested once sync() and notify_modification() are implemented
        pass

    @pytest.mark.asyncio
    async def test_version_always_increments_monotonically(self):
        """Test version counter always increments (never decrements)."""
        # Will be tested once methods are implemented
        pass


class TestTranscriptCoordinatorCleanup:
    """Test cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_episode_removes_coordination_state(self):
        """Test cleanup_episode removes episode coordination state."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Setup coordination state
        coordinator._coordination_configs["ep-1"] = {"some": "config"}

        # This will test cleanup once method exists
        # Expected: _coordination_configs["ep-1"] removed


class TestTranscriptCoordinatorIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_complete_flow_red_team_inject_blue_team_sync(self):
        """Test complete flow: red injects → WebSocket event → blue syncs delta."""
        # This will be a comprehensive integration test once all methods exist
        # Flow:
        # 1. Blue team has version 5 (5 messages)
        # 2. Red team calls notify_modification (append 1 message)
        # 3. Version increments to 6
        # 4. WebSocket event broadcast
        # 5. Blue team syncs with version 5
        # 6. Blue receives delta (1 new message)
        # 7. Blue updates local state to version 6
        pass

    @pytest.mark.asyncio
    async def test_rewrite_detection_and_full_sync(self):
        """Test rewrite detection triggers full transcript sync."""
        # This will test the rewrite scenario once sync() exists
        pass


class TestTranscriptCoordinatorErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_sync_episode_not_found(self):
        """Test sync with non-existent episode."""
        episode_manager = MockEpisodeManager()
        coordinator = TranscriptCoordinator(episode_manager, MockConnectionManager())

        request = TranscriptSyncRequest(episode_id="nonexistent")

        # Should raise ValueError once sync() is implemented
        # with pytest.raises(ValueError, match="Episode .* not found"):
        #     await coordinator.sync(request)

    @pytest.mark.asyncio
    async def test_notify_modification_episode_not_found(self):
        """Test notify_modification with non-existent episode."""
        # Will be tested once notify_modification() is implemented
        pass


# Fixtures for common test setup
@pytest.fixture
def sample_transcript():
    """Sample transcript messages."""
    return [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]


@pytest.fixture
def blue_episode(sample_transcript):
    """Blue team episode with transcript."""
    return Episode(
        episode_id="ep-blue-123",
        task_id="task-1",
        session_id="session-1",
        state=EpisodeState.ACTIVE,
        context={
            MetadataKeys.CLIENT_TRANSCRIPT: sample_transcript.copy(),
            "_transcript_version": 3,
            "_transcript_last_operation": "append",
            MetadataKeys.ORCHESTRATION_ROLE: "blue_team",
        },
    )


@pytest.fixture
def red_episode():
    """Red team episode."""
    return Episode(
        episode_id="ep-red-456",
        task_id="task-red",
        session_id="session-1",
        state=EpisodeState.ACTIVE,
        context={
            MetadataKeys.ORCHESTRATION_ROLE: "red_team",
            MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-123"],
        },
    )


@pytest.fixture
def coordinator_with_episodes(blue_episode, red_episode):
    """Coordinator with blue and red episodes."""
    episode_manager = MockEpisodeManager()
    episode_manager.episodes[blue_episode.episode_id] = blue_episode
    episode_manager.episodes[red_episode.episode_id] = red_episode

    connection_manager = MockConnectionManager()
    coordinator = TranscriptCoordinator(episode_manager, connection_manager)

    return coordinator, episode_manager, connection_manager, blue_episode, red_episode


# =============================================================================
# Observer Mode Tests
# =============================================================================


class TestTranscriptCoordinatorObserverMode:
    """Test observer mode sync functionality."""

    @pytest.mark.asyncio
    async def test_sync_observer_mode_returns_full_transcript(self, coordinator_with_episodes):
        """Test observer mode returns full transcript."""
        coordinator, _, _, blue_episode, _ = coordinator_with_episodes

        request = TranscriptSyncRequest(
            episode_id=blue_episode.episode_id,
            since_version=0,
            is_observer=True,
            hide_system_prompt=False,
        )

        response = await coordinator.sync(request)

        assert response.sync_mode.value == "full"
        assert response.full_transcript is not None
        assert len(response.full_transcript) == 3  # system, user, assistant
        assert response.current_version.sequence == 3

    @pytest.mark.asyncio
    async def test_sync_observer_hides_system_prompt(self, coordinator_with_episodes):
        """Test observer mode with hide_system_prompt filters system messages."""
        coordinator, _, _, blue_episode, _ = coordinator_with_episodes

        request = TranscriptSyncRequest(
            episode_id=blue_episode.episode_id,
            since_version=0,
            is_observer=True,
            hide_system_prompt=True,
        )

        response = await coordinator.sync(request)

        # Should only return messages starting from first assistant
        assert response.full_transcript is not None
        assert len(response.full_transcript) == 1  # Only assistant message
        assert response.full_transcript[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_sync_observer_delta_mode(self, coordinator_with_episodes):
        """Test observer mode with delta retrieval."""
        coordinator, _, _, blue_episode, _ = coordinator_with_episodes

        request = TranscriptSyncRequest(
            episode_id=blue_episode.episode_id,
            since_version=2,
            is_observer=True,
            hide_system_prompt=False,
            retrieval_mode="delta",
        )

        response = await coordinator.sync(request)

        assert response.sync_mode.value == "delta"
        assert response.delta is not None
        assert len(response.delta) == 1  # One new message since version 2

    @pytest.mark.asyncio
    async def test_sync_observer_no_change_when_up_to_date(self, coordinator_with_episodes):
        """Test observer mode returns no_change when client is up to date."""
        coordinator, _, _, blue_episode, _ = coordinator_with_episodes

        request = TranscriptSyncRequest(
            episode_id=blue_episode.episode_id,
            since_version=3,  # Already at version 3
            is_observer=True,
            hide_system_prompt=False,
            retrieval_mode="delta",
        )

        response = await coordinator.sync(request)

        assert response.sync_mode.value == "no_change"
        assert response.modified is False

    @pytest.mark.asyncio
    async def test_sync_observer_tail_mode(self, coordinator_with_episodes):
        """Test observer mode with tail retrieval."""
        coordinator, _, _, blue_episode, _ = coordinator_with_episodes

        request = TranscriptSyncRequest(
            episode_id=blue_episode.episode_id,
            since_version=0,
            is_observer=True,
            hide_system_prompt=False,
            retrieval_mode="tail",
            tail_count=2,
        )

        response = await coordinator.sync(request)

        assert response.sync_mode.value == "full"
        assert response.full_transcript is not None
        assert len(response.full_transcript) == 2  # Last 2 messages

    @pytest.mark.asyncio
    async def test_sync_observer_no_checksum_tracking(self, coordinator_with_episodes):
        """Test observer mode computes checksum on filtered view."""
        coordinator, _, _, blue_episode, _ = coordinator_with_episodes

        request = TranscriptSyncRequest(
            episode_id=blue_episode.episode_id,
            since_version=0,
            is_observer=True,
        )

        response = await coordinator.sync(request)

        # Observer mode computes checksum on their filtered view
        assert len(response.current_version.checksum) == 64  # Valid SHA256


class TestFilterForObserver:
    """Test _filter_for_observer helper."""

    def test_filter_removes_system_messages(self):
        """Test filtering removes messages before first assistant."""
        coordinator = TranscriptCoordinator(MockEpisodeManager(), MockConnectionManager())

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Thanks"},
        ]

        filtered = coordinator._filter_for_observer(messages, hide_system_prompt=True)

        assert len(filtered) == 2
        assert filtered[0]["role"] == "assistant"
        assert filtered[1]["role"] == "user"

    def test_filter_returns_all_when_disabled(self):
        """Test filtering returns all messages when hide_system_prompt=False."""
        coordinator = TranscriptCoordinator(MockEpisodeManager(), MockConnectionManager())

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        filtered = coordinator._filter_for_observer(messages, hide_system_prompt=False)

        assert len(filtered) == 2
        assert filtered[0]["role"] == "system"

    def test_filter_handles_empty_list(self):
        """Test filtering handles empty message list."""
        coordinator = TranscriptCoordinator(MockEpisodeManager(), MockConnectionManager())

        filtered = coordinator._filter_for_observer([], hide_system_prompt=True)

        assert filtered == []

    def test_filter_handles_no_assistant_messages(self):
        """Test filtering with no assistant messages returns empty."""
        coordinator = TranscriptCoordinator(MockEpisodeManager(), MockConnectionManager())

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        filtered = coordinator._filter_for_observer(messages, hide_system_prompt=True)

        # No assistant message found, so nothing is returned
        assert len(filtered) == 0

class TestRestartOperationWithInitialTranscript:
    """Test restart operation and INITIAL_TRANSCRIPT preservation.

    These tests verify that:
    1. INITIAL_TRANSCRIPT is captured after the first assistant response
    2. Restart operation uses INITIAL_TRANSCRIPT to reset the conversation
    3. Blue team's opening response is preserved through restart
    """

    @pytest.mark.asyncio
    async def test_initial_transcript_captured_after_first_assistant_response(self):
        """Test that INITIAL_TRANSCRIPT is updated when first assistant message is added."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Setup episode with initial system/user messages (no assistant yet)
        initial_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        episode = Episode(
            episode_id="ep-blue-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: initial_messages,
                MetadataKeys.INITIAL_TRANSCRIPT: list(initial_messages),  # Copy, no assistant
                MetadataKeys.TRANSCRIPT_VERSION: 1,
            },
        )
        episode_manager.episodes["ep-blue-1"] = episode

        # Push first assistant response (simulating agent's first generate())
        request = TranscriptSyncRequest(
            episode_id="ep-blue-1",
            operation="append",
            messages_to_push=[{"role": "assistant", "content": "Hello! How can I help you today?"}],
        )
        await coordinator.sync(request)

        # Verify INITIAL_TRANSCRIPT now includes the assistant response
        initial = episode.context.get(MetadataKeys.INITIAL_TRANSCRIPT, [])
        assert len(initial) == 3
        assert initial[0]["role"] == "system"
        assert initial[1]["role"] == "user"
        assert initial[2]["role"] == "assistant"
        assert initial[2]["content"] == "Hello! How can I help you today?"

    @pytest.mark.asyncio
    async def test_initial_transcript_not_updated_on_subsequent_assistant_messages(self):
        """Test that INITIAL_TRANSCRIPT is not updated after first assistant response."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Setup episode that already has first assistant response captured
        initial_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "First response"},
        ]
        episode = Episode(
            episode_id="ep-blue-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: initial_messages,
                MetadataKeys.INITIAL_TRANSCRIPT: list(initial_messages),  # Has assistant
                MetadataKeys.TRANSCRIPT_VERSION: 2,
            },
        )
        episode_manager.episodes["ep-blue-1"] = episode

        # Push another user and assistant message
        request = TranscriptSyncRequest(
            episode_id="ep-blue-1",
            operation="append",
            messages_to_push=[
                {"role": "user", "content": "Follow up question"},
                {"role": "assistant", "content": "Second response"},
            ],
        )
        await coordinator.sync(request)

        # Verify INITIAL_TRANSCRIPT was NOT updated
        initial = episode.context.get(MetadataKeys.INITIAL_TRANSCRIPT, [])
        assert len(initial) == 3  # Still only original 3 messages
        assert initial[2]["content"] == "First response"  # Original assistant response

    @pytest.mark.asyncio
    async def test_restart_operation_resets_to_initial_transcript(self):
        """Test restart operation resets transcript to INITIAL_TRANSCRIPT state."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Setup episode with extended conversation
        initial_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        current_messages = initial_messages + [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I don't have weather data."},
            {"role": "user", "content": "Then what can you do?"},
        ]
        episode = Episode(
            episode_id="ep-blue-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: current_messages,
                MetadataKeys.INITIAL_TRANSCRIPT: list(initial_messages),  # Original state
                MetadataKeys.TRANSCRIPT_VERSION: 5,
            },
        )
        episode_manager.episodes["ep-blue-1"] = episode

        # Red team performs restart with a new injection
        request = TranscriptSyncRequest(
            episode_id="ep-blue-1",
            operation="restart",
            messages_to_push=[{"role": "user", "content": "New attack prompt"}],
        )
        response = await coordinator.sync(request)

        # Verify transcript was reset to initial + new message
        transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
        assert len(transcript) == 4  # system, user, assistant, new_user
        assert transcript[0]["content"] == "You are a helpful assistant"
        assert transcript[1]["content"] == "Hello"
        assert transcript[2]["content"] == "Hello! How can I help?"  # Blue's first response preserved!
        assert transcript[3]["content"] == "New attack prompt"  # Red's new injection

    @pytest.mark.asyncio
    async def test_restart_without_initial_transcript_falls_back_to_append(self):
        """Test restart falls back to append if INITIAL_TRANSCRIPT is missing."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Setup episode without INITIAL_TRANSCRIPT set
        current_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        episode = Episode(
            episode_id="ep-blue-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: current_messages,
                # No INITIAL_TRANSCRIPT!
                MetadataKeys.TRANSCRIPT_VERSION: 3,
            },
        )
        episode_manager.episodes["ep-blue-1"] = episode

        # Attempt restart
        request = TranscriptSyncRequest(
            episode_id="ep-blue-1",
            operation="restart",
            messages_to_push=[{"role": "user", "content": "New message"}],
        )
        await coordinator.sync(request)

        # Should fall back to append
        transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
        assert len(transcript) == 4  # Original 3 + new message
        assert transcript[3]["content"] == "New message"

    @pytest.mark.asyncio
    async def test_multiple_restarts_always_use_same_initial_transcript(self):
        """Test multiple restarts always reset to same INITIAL_TRANSCRIPT."""
        episode_manager = MockEpisodeManager()
        connection_manager = MockConnectionManager()
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Setup episode with initial state
        initial_messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "First assistant response"},
        ]
        episode = Episode(
            episode_id="ep-blue-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: list(initial_messages),
                MetadataKeys.INITIAL_TRANSCRIPT: list(initial_messages),
                MetadataKeys.TRANSCRIPT_VERSION: 2,
            },
        )
        episode_manager.episodes["ep-blue-1"] = episode

        # First restart
        request1 = TranscriptSyncRequest(
            episode_id="ep-blue-1",
            operation="restart",
            messages_to_push=[{"role": "user", "content": "Attack 1"}],
        )
        await coordinator.sync(request1)

        # Simulate some conversation after restart
        request2 = TranscriptSyncRequest(
            episode_id="ep-blue-1",
            operation="append",
            messages_to_push=[
                {"role": "assistant", "content": "Response to attack 1"},
                {"role": "user", "content": "Follow up"},
            ],
        )
        await coordinator.sync(request2)

        # Second restart - should go back to original initial
        request3 = TranscriptSyncRequest(
            episode_id="ep-blue-1",
            operation="restart",
            messages_to_push=[{"role": "user", "content": "Attack 2"}],
        )
        await coordinator.sync(request3)

        # Verify reset to initial transcript
        transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
        assert len(transcript) == 4
        assert transcript[0]["content"] == "System prompt"
        assert transcript[1]["content"] == "First user message"
        assert transcript[2]["content"] == "First assistant response"
        assert transcript[3]["content"] == "Attack 2"
