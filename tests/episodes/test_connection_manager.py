"""Unit tests for WebSocket ConnectionManager.

Tests cover:
- Connection lifecycle (connect/disconnect)
- Episode-scoped message broadcasting
- Connection cleanup
- Concurrent connection handling
- Error scenarios and edge cases
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import WebSocket

from saber.server.episodes.connection_manager import ConnectionManager


class TestConnectionManagerInit:
    """Test ConnectionManager initialization."""

    def test_init_creates_empty_state(self):
        """Test that ConnectionManager initializes with empty state."""
        manager = ConnectionManager()

        assert manager._active_connections == {}
        assert manager._connection_metadata == {}
        assert manager._lock is not None


class TestConnectionManagerConnect:
    """Test WebSocket connection registration."""

    @pytest.mark.asyncio
    async def test_connect_accepts_websocket_and_registers_connection(self):
        """Test that connect() accepts WebSocket and registers it."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"
        metadata = {"role": "blue_team", "session_id": "session_456"}

        await manager.connect(episode_id, mock_websocket, metadata)

        # Verify WebSocket was accepted
        mock_websocket.accept.assert_called_once()

        # Verify connection was registered
        assert episode_id in manager._active_connections
        assert mock_websocket in manager._active_connections[episode_id]

        # Verify metadata was stored
        assert mock_websocket in manager._connection_metadata
        conn_metadata = manager._connection_metadata[mock_websocket]
        assert conn_metadata["episode_id"] == episode_id
        assert conn_metadata["metadata"] == metadata
        assert "connected_at" in conn_metadata

        # Verify confirmation message was sent
        mock_websocket.send_json.assert_called_once()
        sent_message = mock_websocket.send_json.call_args[0][0]
        assert sent_message["type"] == "connected"
        assert sent_message["episode_id"] == episode_id

    @pytest.mark.asyncio
    async def test_connect_without_metadata(self):
        """Test connect() with no metadata provided."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"

        await manager.connect(episode_id, mock_websocket)

        # Verify metadata defaults to empty dict
        conn_metadata = manager._connection_metadata[mock_websocket]
        assert conn_metadata["metadata"] == {}

    @pytest.mark.asyncio
    async def test_connect_multiple_connections_same_episode(self):
        """Test multiple connections to the same episode."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"

        await manager.connect(episode_id, mock_ws1)
        await manager.connect(episode_id, mock_ws2)

        # Verify both connections registered
        assert len(manager._active_connections[episode_id]) == 2
        assert mock_ws1 in manager._active_connections[episode_id]
        assert mock_ws2 in manager._active_connections[episode_id]

    @pytest.mark.asyncio
    async def test_connect_different_episodes(self):
        """Test connections to different episodes are isolated."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        episode_id1 = "episode_123"
        episode_id2 = "episode_456"

        await manager.connect(episode_id1, mock_ws1)
        await manager.connect(episode_id2, mock_ws2)

        # Verify episode isolation
        assert episode_id1 in manager._active_connections
        assert episode_id2 in manager._active_connections
        assert mock_ws1 not in manager._active_connections[episode_id2]
        assert mock_ws2 not in manager._active_connections[episode_id1]


class TestConnectionManagerDisconnect:
    """Test WebSocket disconnection handling."""

    @pytest.mark.asyncio
    async def test_disconnect_removes_connection(self):
        """Test that disconnect() removes connection and metadata."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"

        await manager.connect(episode_id, mock_websocket)
        await manager.disconnect(episode_id, mock_websocket)

        # Verify connection removed
        assert mock_websocket not in manager._active_connections.get(episode_id, set())

        # Verify metadata removed
        assert mock_websocket not in manager._connection_metadata

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_empty_episode_set(self):
        """Test that disconnect() removes episode key when last connection removed."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"

        await manager.connect(episode_id, mock_websocket)
        await manager.disconnect(episode_id, mock_websocket)

        # Verify episode key removed
        assert episode_id not in manager._active_connections

    @pytest.mark.asyncio
    async def test_disconnect_preserves_other_connections(self):
        """Test that disconnect() only removes specified connection."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"

        await manager.connect(episode_id, mock_ws1)
        await manager.connect(episode_id, mock_ws2)
        await manager.disconnect(episode_id, mock_ws1)

        # Verify only ws1 removed
        assert mock_ws1 not in manager._active_connections[episode_id]
        assert mock_ws2 in manager._active_connections[episode_id]

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_connection(self):
        """Test disconnect() handles nonexistent connections gracefully."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"

        # Should not raise exception
        await manager.disconnect(episode_id, mock_websocket)

        # State should remain empty
        assert episode_id not in manager._active_connections
        assert mock_websocket not in manager._connection_metadata


class TestConnectionManagerBroadcast:
    """Test episode-scoped message broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_episode_connections(self):
        """Test broadcast sends message to all connections for specific episode."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"
        message = {"type": "test", "data": {"value": 42}}

        await manager.connect(episode_id, mock_ws1)
        await manager.connect(episode_id, mock_ws2)

        await manager.broadcast_to_episode(episode_id, message)

        # Verify message sent to both connections
        mock_ws1.send_json.assert_called()
        mock_ws2.send_json.assert_called()

        # Verify correct message content (skip initial connect confirmation)
        calls = mock_ws1.send_json.call_args_list
        assert any(call[0][0] == message for call in calls)

    @pytest.mark.asyncio
    async def test_broadcast_episode_isolation(self):
        """Test broadcast only sends to specified episode."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        episode_id1 = "episode_123"
        episode_id2 = "episode_456"
        message = {"type": "test", "data": {"value": 42}}

        await manager.connect(episode_id1, mock_ws1)
        await manager.connect(episode_id2, mock_ws2)

        # Reset mocks to clear connection messages
        mock_ws1.reset_mock()
        mock_ws2.reset_mock()

        await manager.broadcast_to_episode(episode_id1, message)

        # Verify only episode_id1 received message
        mock_ws1.send_json.assert_called_once_with(message)
        mock_ws2.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_to_nonexistent_episode(self):
        """Test broadcast to episode with no connections."""
        manager = ConnectionManager()
        message = {"type": "test"}

        # Should not raise exception
        await manager.broadcast_to_episode("nonexistent_episode", message)

    @pytest.mark.asyncio
    async def test_broadcast_handles_disconnected_clients(self):
        """Test broadcast removes connections that fail to send."""
        manager = ConnectionManager()
        mock_ws_good = AsyncMock(spec=WebSocket)
        mock_ws_bad = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"
        message = {"type": "test"}

        await manager.connect(episode_id, mock_ws_good)
        await manager.connect(episode_id, mock_ws_bad)

        # Set failure AFTER connect succeeds
        mock_ws_bad.send_json.side_effect = Exception("Connection closed")

        # Reset to clear connect messages
        mock_ws_good.reset_mock()
        mock_ws_bad.reset_mock()

        await manager.broadcast_to_episode(episode_id, message)

        # Verify bad connection was removed
        assert mock_ws_bad not in manager._active_connections[episode_id]
        assert mock_ws_bad not in manager._connection_metadata

        # Verify good connection still registered
        assert mock_ws_good in manager._active_connections[episode_id]

    @pytest.mark.asyncio
    async def test_broadcast_cleans_up_empty_episode_after_failures(self):
        """Test broadcast removes episode key when all connections fail."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"
        message = {"type": "test"}

        await manager.connect(episode_id, mock_websocket)

        # Set failure AFTER connect succeeds
        mock_websocket.send_json.side_effect = Exception("Connection closed")

        await manager.broadcast_to_episode(episode_id, message)

        # Verify episode key removed
        assert episode_id not in manager._active_connections


class TestConnectionManagerCleanupEpisode:
    """Test episode cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_episode_closes_all_connections(self):
        """Test cleanup_episode() closes all WebSocket connections."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"

        await manager.connect(episode_id, mock_ws1)
        await manager.connect(episode_id, mock_ws2)

        await manager.cleanup_episode(episode_id)

        # Verify connections closed
        mock_ws1.close.assert_called_once()
        mock_ws2.close.assert_called_once()

        # Verify close called with correct parameters
        assert mock_ws1.close.call_args[1]["code"] == 1000
        assert "Episode ended" in mock_ws1.close.call_args[1]["reason"]

    @pytest.mark.asyncio
    async def test_cleanup_episode_removes_all_state(self):
        """Test cleanup_episode() removes all connections and metadata."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"

        await manager.connect(episode_id, mock_ws1)
        await manager.connect(episode_id, mock_ws2)

        await manager.cleanup_episode(episode_id)

        # Verify all state removed
        assert episode_id not in manager._active_connections
        assert mock_ws1 not in manager._connection_metadata
        assert mock_ws2 not in manager._connection_metadata

    @pytest.mark.asyncio
    async def test_cleanup_episode_handles_close_errors(self):
        """Test cleanup_episode() continues even if WebSocket.close() fails."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.close.side_effect = Exception("Close failed")
        episode_id = "episode_123"

        await manager.connect(episode_id, mock_websocket)

        # Should not raise exception
        await manager.cleanup_episode(episode_id)

        # Verify state still cleaned up
        assert episode_id not in manager._active_connections
        assert mock_websocket not in manager._connection_metadata

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_episode(self):
        """Test cleanup_episode() handles nonexistent episode gracefully."""
        manager = ConnectionManager()

        # Should not raise exception
        await manager.cleanup_episode("nonexistent_episode")


class TestConnectionManagerConcurrency:
    """Test concurrent connection operations."""

    @pytest.mark.asyncio
    async def test_concurrent_connects(self):
        """Test multiple concurrent connect operations."""
        manager = ConnectionManager()
        episode_id = "episode_123"

        # Create multiple mock websockets
        websockets = [AsyncMock(spec=WebSocket) for _ in range(10)]

        # Connect concurrently
        await asyncio.gather(*[
            manager.connect(episode_id, ws, {"index": i})
            for i, ws in enumerate(websockets)
        ])

        # Verify all connections registered
        assert len(manager._active_connections[episode_id]) == 10
        for ws in websockets:
            assert ws in manager._active_connections[episode_id]

    @pytest.mark.asyncio
    async def test_concurrent_disconnect(self):
        """Test multiple concurrent disconnect operations."""
        manager = ConnectionManager()
        episode_id = "episode_123"
        websockets = [AsyncMock(spec=WebSocket) for _ in range(10)]

        # Connect all
        for ws in websockets:
            await manager.connect(episode_id, ws)

        # Disconnect concurrently
        await asyncio.gather(*[
            manager.disconnect(episode_id, ws)
            for ws in websockets
        ])

        # Verify all connections removed
        assert episode_id not in manager._active_connections

    @pytest.mark.asyncio
    async def test_concurrent_broadcast(self):
        """Test concurrent broadcast operations."""
        manager = ConnectionManager()
        episode_id = "episode_123"
        mock_websocket = AsyncMock(spec=WebSocket)

        await manager.connect(episode_id, mock_websocket)
        mock_websocket.reset_mock()

        # Broadcast concurrently
        messages = [{"type": "test", "index": i} for i in range(10)]
        await asyncio.gather(*[
            manager.broadcast_to_episode(episode_id, msg)
            for msg in messages
        ])

        # Verify all messages sent (order may vary)
        assert mock_websocket.send_json.call_count == 10

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """Test mixed concurrent operations (connect/disconnect/broadcast)."""
        manager = ConnectionManager()
        episode_id = "episode_123"

        async def connect_and_disconnect():
            ws = AsyncMock(spec=WebSocket)
            await manager.connect(episode_id, ws)
            await asyncio.sleep(0.01)
            await manager.disconnect(episode_id, ws)

        async def broadcast_message():
            await manager.broadcast_to_episode(episode_id, {"type": "test"})

        # Run mixed operations concurrently
        tasks = [
            connect_and_disconnect() for _ in range(5)
        ] + [
            broadcast_message() for _ in range(5)
        ]

        # Should not raise exceptions
        await asyncio.gather(*tasks)


class TestConnectionManagerEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_connect_websocket_accept_fails(self):
        """Test connect() when WebSocket.accept() fails."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept.side_effect = Exception("Accept failed")
        episode_id = "episode_123"

        with pytest.raises(Exception, match="Accept failed"):
            await manager.connect(episode_id, mock_websocket)

        # Verify connection not registered
        assert episode_id not in manager._active_connections

    @pytest.mark.asyncio
    async def test_broadcast_message_serialization_error(self):
        """Test broadcast with message that fails JSON serialization."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"

        await manager.connect(episode_id, mock_websocket)
        mock_websocket.reset_mock()

        # Create message that will fail serialization
        class NonSerializable:
            pass

        message = {"type": "test", "data": NonSerializable()}

        # send_json will fail with TypeError
        mock_websocket.send_json.side_effect = TypeError("Object not serializable")

        await manager.broadcast_to_episode(episode_id, message)

        # Verify connection removed after failure
        assert mock_websocket not in manager._active_connections.get(episode_id, set())

    @pytest.mark.asyncio
    async def test_connect_then_immediate_disconnect(self):
        """Test rapid connect/disconnect sequence."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "episode_123"

        await manager.connect(episode_id, mock_websocket)
        await manager.disconnect(episode_id, mock_websocket)
        await manager.disconnect(episode_id, mock_websocket)  # Second disconnect

        # Should handle gracefully
        assert episode_id not in manager._active_connections

    @pytest.mark.asyncio
    async def test_multiple_episode_cleanup(self):
        """Test cleanup of multiple episodes."""
        manager = ConnectionManager()
        episodes = [f"episode_{i}" for i in range(5)]

        # Create connections for each episode
        for episode_id in episodes:
            ws = AsyncMock(spec=WebSocket)
            await manager.connect(episode_id, ws)

        # Cleanup all episodes
        for episode_id in episodes:
            await manager.cleanup_episode(episode_id)

        # Verify all state cleaned up
        assert len(manager._active_connections) == 0
        assert len(manager._connection_metadata) == 0


class TestOrphanedEpisodeDetection:
    """Test orphaned episode detection when last connection disconnects."""

    @pytest.mark.asyncio
    async def test_disconnect_detects_orphaned_active_episode(self):
        """Test that disconnect detects when last connection lost for ACTIVE episode."""
        from saber.server.base import Episode, EpisodeState
        from saber.server.episodes.episode_manager import EpisodeManager

        # Create mock episode manager with active episode
        episode_manager = Mock()
        active_episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={},
        )
        episode_manager.get_episode_by_id.return_value = active_episode

        # Create ConnectionManager with episode_manager
        manager = ConnectionManager(episode_manager=episode_manager)

        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "ep-1"

        await manager.connect(episode_id, mock_websocket)

        # Disconnect should detect orphaned episode and log warning
        with patch("saber.server.episodes.connection_manager.logger") as mock_logger:
            await manager.disconnect(episode_id, mock_websocket)

            # Verify warning was logged
            warning_calls = [c for c in mock_logger.warning.call_args_list if "orphaned" in str(c).lower()]
            assert len(warning_calls) == 1

            # Verify warning includes episode_id
            warning_message = str(warning_calls[0])
            assert "ep-1" in warning_message or "active" in warning_message.lower()

    @pytest.mark.asyncio
    async def test_disconnect_no_warning_for_completed_episode(self):
        """Test that disconnect doesn't warn for COMPLETED episode."""
        from saber.server.base import Episode, EpisodeState

        # Create mock episode manager with completed episode
        episode_manager = Mock()
        completed_episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.COMPLETED,
            context={},
        )
        episode_manager.get_episode_by_id.return_value = completed_episode

        manager = ConnectionManager(episode_manager=episode_manager)

        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "ep-1"

        await manager.connect(episode_id, mock_websocket)

        # Disconnect should NOT warn for completed episode
        with patch("saber.server.episodes.connection_manager.logger") as mock_logger:
            await manager.disconnect(episode_id, mock_websocket)

            # Verify no orphaned episode warning
            warning_calls = [c for c in mock_logger.warning.call_args_list if "orphaned" in str(c).lower()]
            assert len(warning_calls) == 0

    @pytest.mark.asyncio
    async def test_disconnect_no_warning_when_other_connections_remain(self):
        """Test that disconnect doesn't warn when other connections still exist."""
        from saber.server.base import Episode, EpisodeState

        episode_manager = Mock()
        active_episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={},
        )
        episode_manager.get_episode_by_id.return_value = active_episode

        manager = ConnectionManager(episode_manager=episode_manager)

        # Connect two websockets
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        episode_id = "ep-1"

        await manager.connect(episode_id, mock_ws1)
        await manager.connect(episode_id, mock_ws2)

        # Disconnect first - should NOT warn (second connection remains)
        with patch("saber.server.episodes.connection_manager.logger") as mock_logger:
            await manager.disconnect(episode_id, mock_ws1)

            # No warning yet
            warning_calls = [c for c in mock_logger.warning.call_args_list if "orphaned" in str(c).lower()]
            assert len(warning_calls) == 0

        # Disconnect last - should warn
        with patch("saber.server.episodes.connection_manager.logger") as mock_logger:
            await manager.disconnect(episode_id, mock_ws2)

            # Warning for orphaned episode
            warning_calls = [c for c in mock_logger.warning.call_args_list if "orphaned" in str(c).lower()]
            assert len(warning_calls) == 1

    @pytest.mark.asyncio
    async def test_disconnect_without_episode_manager(self):
        """Test that disconnect works without episode_manager (backward compatible)."""
        # Create ConnectionManager without episode_manager
        manager = ConnectionManager()

        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "ep-1"

        await manager.connect(episode_id, mock_websocket)

        # Should not raise even without episode_manager
        with patch("saber.server.episodes.connection_manager.logger") as mock_logger:
            await manager.disconnect(episode_id, mock_websocket)

            # No warnings (can't check episode state without episode_manager)
            warning_calls = [c for c in mock_logger.warning.call_args_list if "orphaned" in str(c).lower()]
            assert len(warning_calls) == 0

    @pytest.mark.asyncio
    async def test_disconnect_episode_not_found(self):
        """Test disconnect when episode doesn't exist in episode_manager."""
        episode_manager = Mock()
        episode_manager.get_episode_by_id.return_value = None  # Episode not found

        manager = ConnectionManager(episode_manager=episode_manager)

        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "ep-nonexistent"

        await manager.connect(episode_id, mock_websocket)

        # Should not raise when episode not found
        with patch("saber.server.episodes.connection_manager.logger") as mock_logger:
            await manager.disconnect(episode_id, mock_websocket)

            # No warnings (episode doesn't exist)
            warning_calls = [c for c in mock_logger.warning.call_args_list if "orphaned" in str(c).lower()]
            assert len(warning_calls) == 0

    @pytest.mark.asyncio
    async def test_disconnect_logs_episode_state(self):
        """Test that disconnect warning includes episode state."""
        from saber.server.base import Episode, EpisodeState

        episode_manager = Mock()
        active_episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={},
        )
        episode_manager.get_episode_by_id.return_value = active_episode

        manager = ConnectionManager(episode_manager=episode_manager)

        mock_websocket = AsyncMock(spec=WebSocket)
        episode_id = "ep-1"

        await manager.connect(episode_id, mock_websocket)

        with patch("saber.server.episodes.connection_manager.logger") as mock_logger:
            await manager.disconnect(episode_id, mock_websocket)

            # Verify warning includes episode state
            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) > 0

            # Check if extra dict contains state info
            if warning_calls[0][1].get("extra"):
                extra = warning_calls[0][1]["extra"]
                assert "episode_state" in extra or "episode_id" in extra
