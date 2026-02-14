"""Unit tests for WebSocket message handlers.

Tests the handler classes extracted from the monolithic WebSocket endpoint:
- PingHandler: Keepalive responses
- PushMessageHandler: Message push with cross-episode support (push-only protocol)
- WebSocketMessageRouter: Message routing
"""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from saber.models.rest.websocket_messages import WebSocketMessageType
from saber.server.api.websocket_handlers import (
    PingHandler,
    PushMessageHandler,
    WebSocketMessageRouter,
    get_message_router,
)


class MockWebSocket:
    """Mock WebSocket for testing handlers."""

    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []

    async def send_json(self, data: dict[str, Any]) -> None:
        """Record sent messages."""
        self.sent_messages.append(data)

    def get_last_message(self) -> dict[str, Any] | None:
        """Get the most recently sent message."""
        return self.sent_messages[-1] if self.sent_messages else None


class MockConnectionManager:
    """Mock ConnectionManager for testing."""

    def __init__(self) -> None:
        self.broadcasts: list[dict[str, Any]] = []

    async def broadcast_to_episode(self, episode_id: str, message: dict[str, Any]) -> None:
        """Record broadcasts for verification."""
        self.broadcasts.append({"episode_id": episode_id, "message": message})


class MockEpisode:
    """Lightweight mock episode with session_id."""

    def __init__(self, session_id: str = "session-001") -> None:
        self.session_id = session_id


class MockCoordinator:
    """Mock TranscriptCoordinator for testing handlers (push-only protocol)."""

    def __init__(self) -> None:
        self.push_calls: list[dict[str, Any]] = []
        self.push_response_sequence: int = 1
        self.connection_manager = MockConnectionManager()
        self.episode_manager = MagicMock()

        # Default: episode_manager returns a mock episode
        self.episode_manager.get_episode_by_id = MagicMock(
            return_value=MockEpisode(session_id="session-001"),
        )

    async def push_message(
        self,
        episode_id: str,
        session_id: str,
        message: dict[str, Any],
        operation: str = "append",
    ) -> int:
        """Record push calls and return configured sequence number."""
        self.push_calls.append(
            {
                "episode_id": episode_id,
                "session_id": session_id,
                "message": message,
                "operation": operation,
            }
        )
        return self.push_response_sequence


# =============================================================================
# PingHandler Tests
# =============================================================================


class TestPingHandler:
    """Tests for PingHandler."""

    @pytest.mark.asyncio
    async def test_ping_responds_with_pong(self):
        """Test that ping message receives pong response."""
        handler = PingHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        await handler.handle(
            data={"type": "ping", "id": "ping-123"},
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        assert len(websocket.sent_messages) == 1
        response = websocket.get_last_message()
        assert response["type"] == "pong"
        assert "timestamp" in response

    @pytest.mark.asyncio
    async def test_ping_timestamp_is_iso_format(self):
        """Test that pong timestamp is valid ISO format."""
        handler = PingHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        await handler.handle(
            data={"type": "ping"},
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        response = websocket.get_last_message()
        # Should parse without error
        datetime.fromisoformat(response["timestamp"])


# =============================================================================
# PushMessageHandler Tests
# =============================================================================


class TestPushMessageHandler:
    """Tests for PushMessageHandler (push-only protocol)."""

    @pytest.mark.asyncio
    async def test_push_message_same_episode(self) -> None:
        """Test normal push to own episode — coordinator.push_message called, ack sent."""
        handler = PushMessageHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()
        coordinator.push_response_sequence = 3

        await handler.handle(
            data={
                "type": "push_message",
                "id": "push-001",
                "data": {
                    "message": {"role": "assistant", "content": "Hello"},
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        # Verify coordinator.push_message was called correctly
        assert len(coordinator.push_calls) == 1
        call = coordinator.push_calls[0]
        assert call["episode_id"] == "ep-001"
        assert call["session_id"] == "session-001"
        assert call["message"] == {"role": "assistant", "content": "Hello"}
        assert call["operation"] == "append"

        # Verify ack sent with correct sequence
        response = websocket.get_last_message()
        assert response is not None
        assert response["type"] == "push_ack"
        assert response["id"] == "push-001"
        assert response["data"]["sequence"] == 3
        assert response["data"]["target_episode_id"] is None

    @pytest.mark.asyncio
    async def test_push_message_cross_episode(self) -> None:
        """Test push to different episode — target_episode_id used, ack has target info."""
        handler = PushMessageHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()
        coordinator.push_response_sequence = 7

        # Set up target episode with a different session_id
        coordinator.episode_manager.get_episode_by_id = MagicMock(
            return_value=MockEpisode(session_id="session-blue"),
        )

        await handler.handle(
            data={
                "type": "push_message",
                "id": "push-002",
                "data": {
                    "message": {"role": "system", "content": "Injected prompt"},
                    "target_episode_id": "ep-blue",
                    "strategy": "append",
                },
            },
            websocket=websocket,
            episode_id="ep-red",
            coordinator=coordinator,
        )

        # Verify push was directed at target episode
        assert len(coordinator.push_calls) == 1
        call = coordinator.push_calls[0]
        assert call["episode_id"] == "ep-blue"
        assert call["session_id"] == "session-blue"

        # Verify ack includes target_episode_id
        response = websocket.get_last_message()
        assert response is not None
        assert response["data"]["target_episode_id"] == "ep-blue"
        assert response["data"]["sequence"] == 7

    @pytest.mark.asyncio
    async def test_push_message_episode_not_found(self) -> None:
        """Test graceful handling when target episode doesn't exist — sends error response."""
        handler = PushMessageHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        # Episode not found
        coordinator.episode_manager.get_episode_by_id = MagicMock(return_value=None)

        await handler.handle(
            data={
                "type": "push_message",
                "id": "push-003",
                "data": {
                    "message": {"role": "user", "content": "Hello"},
                },
            },
            websocket=websocket,
            episode_id="ep-missing",
            coordinator=coordinator,
        )

        # No push should have been called
        assert len(coordinator.push_calls) == 0
        # Error response should have been sent
        assert len(websocket.sent_messages) == 1
        error_msg = websocket.get_last_message()
        assert error_msg is not None
        assert error_msg["type"] == "transcript_error"
        assert error_msg["data"]["error"] == "push_failed"
        assert "ep-missing" in error_msg["data"]["message"]

    @pytest.mark.asyncio
    async def test_push_message_coordinator_error(self) -> None:
        """Test that coordinator exceptions send error response instead of crashing."""
        handler = PushMessageHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        # Make push_message raise ValueError
        async def _raise_value_error(**kwargs: object) -> int:
            raise ValueError("Invalid message format")

        coordinator.push_message = _raise_value_error  # type: ignore[assignment]

        await handler.handle(
            data={
                "type": "push_message",
                "id": "push-err",
                "data": {
                    "message": {"role": "user", "content": "Boom"},
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        # Error response should have been sent (not a crash)
        assert len(websocket.sent_messages) == 1
        error_msg = websocket.get_last_message()
        assert error_msg is not None
        assert error_msg["type"] == "transcript_error"
        assert error_msg["data"]["error"] == "push_failed"
        assert "Invalid message format" in error_msg["data"]["message"]

    @pytest.mark.asyncio
    async def test_push_message_with_restart_strategy(self) -> None:
        """Test push with restart strategy passes operation correctly."""
        handler = PushMessageHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        await handler.handle(
            data={
                "type": "push_message",
                "id": "push-004",
                "data": {
                    "message": {"role": "user", "content": "Fresh start"},
                    "strategy": "restart",
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        call = coordinator.push_calls[0]
        assert call["operation"] == "restart"

    @pytest.mark.asyncio
    async def test_push_message_no_broadcast_by_handler(self) -> None:
        """Handler should NOT broadcast state events — coordinator does that internally."""
        handler = PushMessageHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        await handler.handle(
            data={
                "type": "push_message",
                "id": "push-005",
                "data": {
                    "message": {"role": "assistant", "content": "Reply"},
                    "target_episode_id": "ep-blue",
                },
            },
            websocket=websocket,
            episode_id="ep-red",
            coordinator=coordinator,
        )

        # Handler should NOT have called broadcast — coordinator handles it
        assert len(coordinator.connection_manager.broadcasts) == 0


# =============================================================================
# WebSocketMessageRouter Tests
# =============================================================================


class TestWebSocketMessageRouter:
    """Tests for WebSocketMessageRouter."""

    @pytest.mark.asyncio
    async def test_route_ping_message(self) -> None:
        """Test routing ping message to PingHandler."""
        router = WebSocketMessageRouter()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        result = await router.route(
            data={"type": "ping"},
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        assert result is True
        response = websocket.get_last_message()
        assert response is not None
        assert response["type"] == "pong"

    @pytest.mark.asyncio
    async def test_route_push_message(self) -> None:
        """Test routing push_message to PushMessageHandler."""
        router = WebSocketMessageRouter()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        result = await router.route(
            data={
                "type": "push_message",
                "id": "push-001",
                "data": {
                    "message": {"role": "user", "content": "Hi"},
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        assert result is True
        response = websocket.get_last_message()
        assert response is not None
        assert response["type"] == "push_ack"

    @pytest.mark.asyncio
    async def test_route_sync_request_is_unknown(self) -> None:
        """SYNC_REQUEST handler was removed — should return False (unhandled)."""
        router = WebSocketMessageRouter()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        result = await router.route(
            data={"type": "sync_request", "id": "sync-001", "data": {"since_version": 0}},
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        assert result is False
        assert len(websocket.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_route_unknown_message_type(self) -> None:
        """Test routing unknown message type returns False."""
        router = WebSocketMessageRouter()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        result = await router.route(
            data={"type": "unknown_type"},
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        assert result is False
        assert len(websocket.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_route_missing_type(self) -> None:
        """Test routing message without type returns False."""
        router = WebSocketMessageRouter()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        result = await router.route(
            data={"id": "msg-001"},  # No type field
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        assert result is False


class TestGetMessageRouter:
    """Tests for singleton router access."""

    def test_get_message_router_returns_singleton(self) -> None:
        """Test that get_message_router returns the same instance."""
        router1 = get_message_router()
        router2 = get_message_router()

        assert router1 is router2

    def test_router_has_all_handlers(self) -> None:
        """Test that router has handlers for expected message types (no SYNC_REQUEST)."""
        router = get_message_router()

        assert WebSocketMessageType.PING in router._handlers
        assert WebSocketMessageType.PUSH_MESSAGE in router._handlers
