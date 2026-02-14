"""Tests for stuck state monitor."""

import asyncio

import pytest

from saber.models.constants import MetadataKeys
from saber.server.episodes.connection_manager import ConnectionManager
from saber.server.episodes.episode_manager import EpisodeManager
from saber.server.episodes.transcript.coordinator import TranscriptCoordinator
from saber.server.episodes.transcript.stuck_state_monitor import StuckStateMonitor


@pytest.mark.asyncio
class TestStuckStateMonitor:
    """Test stuck state detection and monitoring."""

    async def test_monitor_starts_and_stops_cleanly(self):
        """Monitor should start/stop without errors."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        monitor = StuckStateMonitor(
            coordinator=coordinator,
            episode_manager=episode_manager,
            check_interval=0.1,  # Fast for testing
            stuck_threshold=1.0,  # 1 second
        )

        # Start monitor
        await monitor.start()

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop cleanly
        await monitor.stop()

    async def test_detects_stuck_episode(self):
        """Monitor should detect episodes stuck >threshold."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Create episode
        episode_manager.start_episode("session-1", "task-1")

        # Wait for lifecycle hook
        await asyncio.sleep(0.01)

        # Create monitor with short threshold
        events_captured = []

        async def mock_broadcast(episode_id, message):
            events_captured.append(message)

        connection_manager.broadcast_to_episode = mock_broadcast

        monitor = StuckStateMonitor(
            coordinator=coordinator,
            episode_manager=episode_manager,
            check_interval=0.1,
            stuck_threshold=0.2,  # 200ms
        )

        await monitor.start()

        # Wait for threshold to be exceeded
        await asyncio.sleep(0.4)

        # Should have detected stuck state
        assert len(events_captured) > 0
        assert any(e.type == "transcript_error" for e in events_captured)

        await monitor.stop()

    async def test_does_not_flag_active_episodes(self):
        """Episodes with recent state changes should not be flagged."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        episode = episode_manager.start_episode("session-1", "task-1")

        await asyncio.sleep(0.01)

        events_captured = []

        async def mock_broadcast(episode_id, message):
            events_captured.append(message)

        connection_manager.broadcast_to_episode = mock_broadcast

        monitor = StuckStateMonitor(
            coordinator=coordinator,
            episode_manager=episode_manager,
            check_interval=0.1,
            stuck_threshold=1.0,  # 1 second
        )

        await monitor.start()

        # Keep updating state timestamps (simulating activity)
        for _i in range(3):
            await asyncio.sleep(0.15)
            coordinator._state_machine.update_state_timestamp(episode.episode_id)

        # Should not have error events
        error_events = [e for e in events_captured if e.type == "transcript_error"]
        assert len(error_events) == 0

        await monitor.stop()

    async def test_respects_per_episode_threshold(self):
        """Episodes can have custom stuck thresholds."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Episode with custom threshold
        episode = episode_manager.start_episode("session-1", "task-1")
        episode.context[MetadataKeys.STUCK_STATE_THRESHOLD] = 0.5  # 500ms

        await asyncio.sleep(0.01)

        events_captured = []

        async def mock_broadcast(episode_id, message):
            events_captured.append(message)

        connection_manager.broadcast_to_episode = mock_broadcast

        monitor = StuckStateMonitor(
            coordinator=coordinator,
            episode_manager=episode_manager,
            check_interval=0.1,
            stuck_threshold=10.0,  # High default
        )

        await monitor.start()

        # Wait past custom threshold but below default
        await asyncio.sleep(0.7)

        # Should trigger based on episode's custom threshold
        error_events = [e for e in events_captured if e.type == "transcript_error"]
        assert len(error_events) > 0

        await monitor.stop()

    async def test_skips_terminated_episodes(self):
        """Monitor should skip episodes that are terminated/completed."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        # Create and immediately end episode
        episode = episode_manager.start_episode("session-1", "task-1")

        await asyncio.sleep(0.01)

        # End the episode
        await episode_manager.end_episode(episode.episode_id, "test_cleanup")

        events_captured = []

        async def mock_broadcast(episode_id, message):
            events_captured.append(message)

        connection_manager.broadcast_to_episode = mock_broadcast

        monitor = StuckStateMonitor(
            coordinator=coordinator,
            episode_manager=episode_manager,
            check_interval=0.1,
            stuck_threshold=0.1,  # Short threshold
        )

        await monitor.start()
        await asyncio.sleep(0.3)

        # Should not have any error events (episode is terminated)
        error_events = [e for e in events_captured if e.type == "transcript_error"]
        assert len(error_events) == 0

        await monitor.stop()

    async def test_only_alerts_once_per_stuck_episode(self):
        """Monitor should not spam alerts for the same stuck episode."""
        episode_manager = EpisodeManager()
        connection_manager = ConnectionManager(episode_manager)
        coordinator = TranscriptCoordinator(episode_manager, connection_manager)

        episode_manager.start_episode("session-1", "task-1")

        await asyncio.sleep(0.01)

        events_captured = []

        async def mock_broadcast(episode_id, message):
            events_captured.append(message)

        connection_manager.broadcast_to_episode = mock_broadcast

        monitor = StuckStateMonitor(
            coordinator=coordinator,
            episode_manager=episode_manager,
            check_interval=0.1,
            stuck_threshold=0.2,
        )

        await monitor.start()

        # Wait for multiple check cycles
        await asyncio.sleep(0.5)

        await monitor.stop()

        # Should only have one error event (idempotent)
        error_events = [e for e in events_captured if e.type == "transcript_error"]
        assert len(error_events) == 1
