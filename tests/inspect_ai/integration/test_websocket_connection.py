"""Tests for WebSocket connection management.

Tests cover:
- WebSocket URL building from REST URL
- Connection establishment
- Reconnection with exponential backoff
- Connection state tracking
- Cleanup and resource release
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.client.transcript import WebSocketConnectionManager
from saber.models.rest.websocket_config import WebSocketConfig


class TestWebSocketUrlBuilding:
    """Tests for WebSocket URL construction."""

    def test_build_websocket_url_basic(self):
        """Test WebSocket URL construction from basic HTTP URL."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        expected = "ws://localhost:8000/api/v1/episodes/episode-456/ws"
        assert manager.ws_url == expected

    def test_build_websocket_url_https(self):
        """Test WebSocket URL construction from HTTPS URL."""
        manager = WebSocketConnectionManager(
            episode_id="episode-789",
            rest_url="https://example.com",
        )

        expected = "wss://example.com/api/v1/episodes/episode-789/ws"
        assert manager.ws_url == expected

    def test_build_websocket_url_with_path(self):
        """Test WebSocket URL construction preserves base path (for proxies)."""
        manager = WebSocketConnectionManager(
            episode_id="episode-abc",
            rest_url="http://proxy.com/saber",
        )

        expected = "ws://proxy.com/saber/api/v1/episodes/episode-abc/ws"
        assert manager.ws_url == expected

    def test_build_websocket_url_with_port(self):
        """Test WebSocket URL construction with custom port."""
        manager = WebSocketConnectionManager(
            episode_id="episode-def",
            rest_url="http://localhost:9000",
        )

        expected = "ws://localhost:9000/api/v1/episodes/episode-def/ws"
        assert manager.ws_url == expected

    def test_build_websocket_url_with_trailing_slash(self):
        """Test WebSocket URL construction handles trailing slash."""
        manager = WebSocketConnectionManager(
            episode_id="episode-ghi",
            rest_url="http://localhost:8000/",
        )

        expected = "ws://localhost:8000/api/v1/episodes/episode-ghi/ws"
        assert manager.ws_url == expected

    def test_build_websocket_url_with_nested_path(self):
        """Test WebSocket URL construction with nested base path."""
        manager = WebSocketConnectionManager(
            episode_id="episode-jkl",
            rest_url="https://api.example.com/v2/saber",
        )

        expected = "wss://api.example.com/v2/saber/api/v1/episodes/episode-jkl/ws"
        assert manager.ws_url == expected


class TestConnectionState:
    """Tests for connection state tracking."""

    def test_initial_state_not_connected(self):
        """Test that initial state is not connected."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        assert manager.is_connected() is False
        assert manager.websocket is None

    def test_is_connected_with_open_websocket(self):
        """Test is_connected returns True with open WebSocket."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock an open WebSocket
        mock_ws = MagicMock()
        mock_ws.closed = False
        manager._websocket = mock_ws

        assert manager.is_connected() is True

    def test_is_connected_with_closed_websocket(self):
        """Test is_connected returns False with closed WebSocket."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock a closed WebSocket - need to remove state attr since MagicMock adds it
        mock_ws = MagicMock(spec=["closed"])
        mock_ws.closed = True
        manager._websocket = mock_ws

        assert manager.is_connected() is False


class TestEnsureConnected:
    """Tests for ensure_connected method."""

    @pytest.mark.asyncio
    async def test_ensure_connected_already_connected(self):
        """Test ensure_connected returns existing connection if open."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock an existing open WebSocket
        mock_ws = AsyncMock()
        mock_ws.closed = False
        manager._websocket = mock_ws

        result = await manager.ensure_connected()

        assert result is mock_ws

    @pytest.mark.asyncio
    async def test_ensure_connected_establishes_connection(self):
        """Test ensure_connected establishes new connection."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Mock websockets.connect
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.recv = AsyncMock(return_value='{"type": "connected"}')

        with patch("websockets.connect", return_value=AsyncMock(return_value=mock_ws)):
            with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
                mock_connect.return_value = mock_ws

                result = await manager.ensure_connected()

                assert result is mock_ws
                assert manager.is_connected()


class TestClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_close_websocket(self):
        """Test close closes the WebSocket."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Set up mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.close = AsyncMock()
        manager._websocket = mock_ws

        await manager.close()

        mock_ws.close.assert_called_once()
        assert manager._websocket is None

    @pytest.mark.asyncio
    async def test_close_cancels_listener_task(self):
        """Test close cancels listener task."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Create a dummy listener task
        async def dummy_listener():
            await asyncio.sleep(10)

        manager._listener_task = asyncio.create_task(dummy_listener())

        # Set up mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.close = AsyncMock()
        manager._websocket = mock_ws

        await manager.close()

        assert manager._listener_task is None

    @pytest.mark.asyncio
    async def test_close_no_op_when_not_connected(self):
        """Test close is safe when not connected."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Should not raise
        await manager.close()
        assert manager._websocket is None


class TestForceReconnect:
    """Tests for force_reconnect method."""

    @pytest.mark.asyncio
    async def test_force_reconnect_closes_existing(self):
        """Test force_reconnect closes existing connection before reconnecting."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        # Set up existing connection
        old_ws = AsyncMock()
        old_ws.closed = False
        old_ws.close = AsyncMock()
        manager._websocket = old_ws

        # Mock new connection
        new_ws = AsyncMock()
        new_ws.closed = False
        new_ws.recv = AsyncMock(return_value='{"type": "connected"}')

        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = new_ws

            result = await manager.force_reconnect()

            # Old connection should have been closed
            old_ws.close.assert_called_once()
            # New connection should be established
            assert result is new_ws


class TestProperties:
    """Tests for property accessors."""

    def test_episode_id_property(self):
        """Test episode_id property."""
        manager = WebSocketConnectionManager(
            episode_id="episode-test",
            rest_url="http://localhost:8000",
        )

        assert manager.episode_id == "episode-test"

    def test_config_property(self):
        """Test config property returns the configuration."""
        config = WebSocketConfig()
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
            ws_config=config,
        )

        assert manager.config is config

    def test_default_config(self):
        """Test that default config is created if not provided."""
        manager = WebSocketConnectionManager(
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        assert manager.config is not None
        assert isinstance(manager.config, WebSocketConfig)
