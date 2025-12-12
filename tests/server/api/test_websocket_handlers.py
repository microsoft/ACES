"""Unit tests for WebSocket message handlers.

Tests the handler classes extracted from the monolithic WebSocket endpoint:
- PingHandler: Keepalive responses
- SyncRequestHandler: Transcript sync (owner and observer modes)
- PushMessageHandler: Message push with cross-episode injection support
- WebSocketMessageRouter: Message routing
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.models.constants import MetadataKeys
from saber.models.rest.websocket_constants import WebSocketMessageType
from saber.models.rest.websocket_messages import SyncMode, TranscriptOperation
from saber.models.transcript import TranscriptSyncResponse, TranscriptVersion
from saber.server.api.websocket_handlers import (
    PingHandler,
    PushMessageHandler,
    SyncRequestHandler,
    WebSocketMessageRouter,
    get_message_router,
)
from saber.server.base import Episode, EpisodeState


class MockWebSocket:
    """Mock WebSocket for testing handlers."""

    def __init__(self):
        self.sent_messages: List[Dict[str, Any]] = []

    async def send_json(self, data: Dict[str, Any]) -> None:
        """Record sent messages."""
        self.sent_messages.append(data)

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the most recently sent message."""
        return self.sent_messages[-1] if self.sent_messages else None


class MockConnectionManager:
    """Mock ConnectionManager for testing."""

    def __init__(self):
        self.broadcasts: List[Dict[str, Any]] = []

    async def broadcast_to_episode(self, episode_id: str, message: Dict[str, Any]) -> None:
        """Record broadcasts for verification."""
        self.broadcasts.append({"episode_id": episode_id, "message": message})


class MockCoordinator:
    """Mock TranscriptCoordinator for testing handlers."""

    # Valid SHA256 checksum (64 hex chars)
    DEFAULT_CHECKSUM = "a" * 64

    def __init__(self):
        self.sync_calls: List[Any] = []
        self.sync_response: Optional[TranscriptSyncResponse] = None
        self.connection_manager = MockConnectionManager()
        self.episode_manager = MagicMock()
        self._state_machine = MagicMock()

    async def sync(self, request) -> TranscriptSyncResponse:
        """Record sync calls and return configured response."""
        self.sync_calls.append(request)
        if self.sync_response:
            return self.sync_response
        # Default response
        return TranscriptSyncResponse(
            current_version=TranscriptVersion(
                sequence=1,
                checksum=self.DEFAULT_CHECKSUM,
                message_count=1,
                last_operation="append",
            ),
            delta=None,
            full_transcript=[{"role": "user", "content": "Hello"}],
            sync_mode=SyncMode.FULL,
            modified=True,
            blocked=False,
            wait_time_seconds=0.001,
        )


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
# SyncRequestHandler Tests
# =============================================================================


class TestSyncRequestHandler:
    """Tests for SyncRequestHandler."""

    @pytest.mark.asyncio
    async def test_sync_request_same_episode(self):
        """Test sync request for same episode (owner mode)."""
        handler = SyncRequestHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        await handler.handle(
            data={
                "type": "sync_request",
                "id": "sync-001",
                "data": {
                    "since_version": 0,
                    "client_checksum": None,
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        # Verify coordinator was called with correct request
        assert len(coordinator.sync_calls) == 1
        request = coordinator.sync_calls[0]
        assert request.episode_id == "ep-001"
        assert request.is_observer is False

        # Verify response sent
        response = websocket.get_last_message()
        assert response["type"] == "sync_response"
        assert response["id"] == "sync-001"
        assert "data" in response

    @pytest.mark.asyncio
    async def test_sync_request_cross_episode_observer(self):
        """Test sync request for different episode (observer mode)."""
        handler = SyncRequestHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        await handler.handle(
            data={
                "type": "sync_request",
                "id": "sync-002",
                "data": {
                    "since_version": 0,
                    "target_episode_id": "ep-blue-team",  # Different from connection
                    "hide_system_prompt": True,
                    "retrieval_mode": "full",
                },
            },
            websocket=websocket,
            episode_id="ep-red-team",  # Connection owner
            coordinator=coordinator,
        )

        # Verify observer mode was set
        request = coordinator.sync_calls[0]
        assert request.episode_id == "ep-blue-team"
        assert request.is_observer is True
        assert request.hide_system_prompt is True

    @pytest.mark.asyncio
    async def test_sync_request_with_delta_mode(self):
        """Test sync request returns delta when client is behind."""
        handler = SyncRequestHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        # Configure coordinator to return delta
        coordinator.sync_response = TranscriptSyncResponse(
            current_version=TranscriptVersion(
                sequence=5,
                checksum=MockCoordinator.DEFAULT_CHECKSUM,
                message_count=5,
                last_operation="append",
            ),
            delta=[{"role": "assistant", "content": "New message"}],
            full_transcript=None,
            sync_mode=SyncMode.DELTA,
            modified=True,
            blocked=False,
            wait_time_seconds=0.001,
        )

        await handler.handle(
            data={
                "type": "sync_request",
                "id": "sync-003",
                "data": {
                    "since_version": 4,
                    "client_checksum": MockCoordinator.DEFAULT_CHECKSUM,
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        response = websocket.get_last_message()
        assert response["data"]["sync_mode"] == "delta"
        assert response["data"]["delta"] is not None
        assert response["data"]["full_transcript"] is None

    @pytest.mark.asyncio
    async def test_sync_request_no_change(self):
        """Test sync request when client is up to date."""
        handler = SyncRequestHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        coordinator.sync_response = TranscriptSyncResponse(
            current_version=TranscriptVersion(
                sequence=5,
                checksum=MockCoordinator.DEFAULT_CHECKSUM,
                message_count=5,
                last_operation="append",
            ),
            delta=[],
            full_transcript=None,
            sync_mode=SyncMode.NO_CHANGE,
            modified=False,
            blocked=False,
            wait_time_seconds=0.001,
        )

        await handler.handle(
            data={
                "type": "sync_request",
                "id": "sync-004",
                "data": {
                    "since_version": 5,
                    "client_checksum": MockCoordinator.DEFAULT_CHECKSUM,
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        response = websocket.get_last_message()
        assert response["data"]["sync_mode"] == "no_change"
        assert response["data"]["modified"] is False

    @pytest.mark.asyncio
    async def test_sync_request_tail_mode(self):
        """Test sync request with tail retrieval mode."""
        handler = SyncRequestHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        await handler.handle(
            data={
                "type": "sync_request",
                "id": "sync-005",
                "data": {
                    "since_version": 0,
                    "target_episode_id": "ep-other",
                    "retrieval_mode": "tail",
                    "tail_count": 5,
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        request = coordinator.sync_calls[0]
        assert request.retrieval_mode == "tail"
        assert request.tail_count == 5


# =============================================================================
# PushMessageHandler Tests
# =============================================================================


class TestPushMessageHandler:
    """Tests for PushMessageHandler."""

    @pytest.mark.asyncio
    async def test_push_message_same_episode(self):
        """Test normal push to own episode."""
        handler = PushMessageHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        await handler.handle(
            data={
                "type": "push_message",
                "id": "push-001",
                "data": {
                    "message": {"role": "assistant", "content": "Hello"},
                    "since_version": 0,
                    "client_checksum": None,
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        # Verify push was processed
        assert len(coordinator.sync_calls) == 1
        request = coordinator.sync_calls[0]
        assert request.messages_to_push == [{"role": "assistant", "content": "Hello"}]

        # Verify ack sent
        response = websocket.get_last_message()
        assert response["type"] == "push_ack"
        assert response["data"]["version"] == 1

    @pytest.mark.asyncio
    async def test_push_message_cross_episode_injection(self):
        """Test push to different episode (injection mode)."""
        handler = PushMessageHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        # Setup target episode as a mock to allow attribute setting
        target_episode = MagicMock()
        target_episode.context = {MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0}
        target_episode.update_context_atomic = AsyncMock()
        coordinator.episode_manager.get_episode_by_id = MagicMock(return_value=target_episode)

        # Setup state machine mock
        from saber.server.episodes.transcript_state_machine import TranscriptState

        coordinator._state_machine.get_state = MagicMock(return_value=TranscriptState.WAITING_FOR_ASSISTANT)

        await handler.handle(
            data={
                "type": "push_message",
                "id": "push-002",
                "data": {
                    "message": {"role": "system", "content": "Injected prompt"},
                    "since_version": 0,
                    "target_episode_id": "ep-blue",  # Different from connection
                    "strategy": "append",
                },
            },
            websocket=websocket,
            episode_id="ep-red",  # Red team injecting to blue
            coordinator=coordinator,
        )

        # Verify cross-episode push
        request = coordinator.sync_calls[0]
        assert request.episode_id == "ep-blue"

        # Verify ack includes injection metadata
        response = websocket.get_last_message()
        assert response["data"]["target_episode_id"] == "ep-blue"
        assert response["data"]["modification_count"] == 1

        # Verify state event broadcast to target
        assert len(coordinator.connection_manager.broadcasts) == 1
        broadcast = coordinator.connection_manager.broadcasts[0]
        assert broadcast["episode_id"] == "ep-blue"
        # The message is a StateEventMessage Pydantic model
        state_event = broadcast["message"]
        assert state_event.data.injected_by == "ep-red"

    @pytest.mark.asyncio
    async def test_push_message_with_restart_strategy(self):
        """Test push with restart strategy.

        Restart resets the transcript to the initial state (system->user->assistant)
        and then appends the new message.
        """
        handler = PushMessageHandler()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        await handler.handle(
            data={
                "type": "push_message",
                "id": "push-003",
                "data": {
                    "message": {"role": "user", "content": "Fresh start"},
                    "since_version": 10,
                    "strategy": "restart",
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        request = coordinator.sync_calls[0]
        assert request.operation == "restart"


# =============================================================================
# WebSocketMessageRouter Tests
# =============================================================================


class TestWebSocketMessageRouter:
    """Tests for WebSocketMessageRouter."""

    @pytest.mark.asyncio
    async def test_route_ping_message(self):
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
        assert response["type"] == "pong"

    @pytest.mark.asyncio
    async def test_route_sync_request(self):
        """Test routing sync_request to SyncRequestHandler."""
        router = WebSocketMessageRouter()
        websocket = MockWebSocket()
        coordinator = MockCoordinator()

        result = await router.route(
            data={
                "type": "sync_request",
                "id": "sync-001",
                "data": {"since_version": 0},
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        assert result is True
        response = websocket.get_last_message()
        assert response["type"] == "sync_response"

    @pytest.mark.asyncio
    async def test_route_push_message(self):
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
                    "since_version": 0,
                },
            },
            websocket=websocket,
            episode_id="ep-001",
            coordinator=coordinator,
        )

        assert result is True
        response = websocket.get_last_message()
        assert response["type"] == "push_ack"

    @pytest.mark.asyncio
    async def test_route_unknown_message_type(self):
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
    async def test_route_missing_type(self):
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

    def test_get_message_router_returns_singleton(self):
        """Test that get_message_router returns the same instance."""
        router1 = get_message_router()
        router2 = get_message_router()

        assert router1 is router2

    def test_router_has_all_handlers(self):
        """Test that router has handlers for all message types."""
        router = get_message_router()

        assert WebSocketMessageType.PING in router._handlers
        assert WebSocketMessageType.SYNC_REQUEST in router._handlers
        assert WebSocketMessageType.PUSH_MESSAGE in router._handlers
