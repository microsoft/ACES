"""WebSocket message handlers for transcript synchronization.

This module extracts WebSocket message handling logic from session_rest_api.py
into dedicated handler classes for better separation of concerns.

Each handler implements a single message type (PING, SYNC_REQUEST, PUSH_MESSAGE)
and encapsulates the business logic for that operation.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from fastapi import WebSocket

from ...logging_config import get_api_logger
from ...models.constants import MetadataKeys
from ...models.rest.websocket_constants import WebSocketMessageType
from ...models.rest.websocket_messages import (
    PongMessage,
    PushAckData,
    PushAckMessage,
    StateEventData,
    StateEventMessage,
    SyncMode,
    SyncRequestData,
    SyncResponseData,
    SyncResponseMessage,
    TranscriptOperation,
    TranscriptVersion,
)
from ...models.transcript import TranscriptSyncRequest

if TYPE_CHECKING:
    from ..episodes.transcript_coordinator import TranscriptCoordinator

logger = get_api_logger(__name__)


class WebSocketHandlerProtocol(Protocol):
    """Protocol for WebSocket message handlers."""

    async def handle(
        self,
        data: dict[str, Any],
        websocket: WebSocket,
        episode_id: str,
        coordinator: "TranscriptCoordinator",
    ) -> None:
        """Handle a WebSocket message.

        Args:
            data: Parsed JSON message data
            websocket: WebSocket connection
            episode_id: Current episode ID (connection owner)
            coordinator: TranscriptCoordinator instance
        """
        ...


class BaseWebSocketHandler(ABC):
    """Base class for WebSocket message handlers."""

    @abstractmethod
    async def handle(
        self,
        data: dict[str, Any],
        websocket: WebSocket,
        episode_id: str,
        coordinator: "TranscriptCoordinator",
    ) -> None:
        """Handle a WebSocket message."""
        pass


class PingHandler(BaseWebSocketHandler):
    """Handler for PING keepalive messages."""

    async def handle(
        self,
        data: dict[str, Any],
        websocket: WebSocket,
        episode_id: str,
        coordinator: "TranscriptCoordinator",
    ) -> None:
        """Respond to ping with pong."""
        pong = PongMessage(timestamp=datetime.utcnow().isoformat())
        await websocket.send_json(pong.model_dump())


class SyncRequestHandler(BaseWebSocketHandler):
    """Handler for SYNC_REQUEST messages.

    Supports both same-episode sync and cross-episode observer access.
    The coordinator handles the distinction internally.
    """

    async def handle(
        self,
        data: dict[str, Any],
        websocket: WebSocket,
        episode_id: str,
        coordinator: "TranscriptCoordinator",
    ) -> None:
        """Handle transcript sync request.

        Two modes (handled by coordinator):
        - Same-episode (is_observer=False): Full bidirectional sync with checksum validation
        - Cross-episode (is_observer=True): Read-only observer access with security filtering
        """
        request_data = SyncRequestData(**data.get("data", {}))

        # Determine target episode and observer mode
        target_episode_id = request_data.target_episode_id or episode_id
        is_observer = target_episode_id != episode_id

        # Build unified sync request - coordinator handles observer mode internally
        sync_request = TranscriptSyncRequest(
            episode_id=target_episode_id,
            since_version=request_data.since_version or 0,
            client_checksum=request_data.client_checksum,
            is_observer=is_observer,
            hide_system_prompt=request_data.hide_system_prompt,  # None = use default
            retrieval_mode=request_data.retrieval_mode or "full",
            tail_count=request_data.tail_count or 10,
        )

        # Single code path - coordinator handles owner vs observer mode
        sync_response = await coordinator.sync(sync_request)

        # Build response from sync response
        # sync_mode is guaranteed to be SyncMode after TranscriptSyncResponse.__post_init__
        sync_mode = (
            sync_response.sync_mode
            if isinstance(sync_response.sync_mode, SyncMode)
            else SyncMode(sync_response.sync_mode)
        )
        response_data = SyncResponseData(
            current_version=TranscriptVersion(
                sequence=sync_response.current_version.sequence,
                checksum=sync_response.current_version.checksum,
                message_count=sync_response.current_version.message_count,
                last_operation=sync_response.current_version.last_operation,
            ),
            delta=sync_response.delta,
            full_transcript=sync_response.full_transcript,
            sync_mode=sync_mode,
            modified=sync_response.modified,
        )

        message = SyncResponseMessage(
            data=response_data,
            id=data.get("id", ""),
            timestamp=datetime.utcnow().isoformat(),
        )
        await websocket.send_json(message.model_dump())


class PushMessageHandler(BaseWebSocketHandler):
    """Handler for PUSH_MESSAGE messages.

    Supports both normal pushes and cross-episode injection.
    """

    async def handle(
        self,
        data: dict[str, Any],
        websocket: WebSocket,
        episode_id: str,
        coordinator: "TranscriptCoordinator",
    ) -> None:
        """Handle message push (normal or injection mode).

        For cross-episode pushes (injections), broadcasts state event
        to the target episode after successful push.
        """
        from ...models.rest.websocket_messages import PushMessageRequestData

        push_data = PushMessageRequestData(**data.get("data", {}))

        # Support cross-episode pushes (red team targeting blue team)
        target_episode_id = push_data.target_episode_id or episode_id
        is_cross_episode = target_episode_id != episode_id

        logger.info(
            "[RESTART_DEBUG] PushMessageHandler.handle() called",
            extra={
                "source_episode_id": episode_id,
                "target_episode_id": target_episode_id,
                "is_cross_episode": is_cross_episode,
                "strategy": push_data.strategy,
                "message_role": push_data.message.get("role") if push_data.message else None,
            },
        )

        # Execute the push via coordinator
        sync_request = TranscriptSyncRequest(
            episode_id=target_episode_id,
            since_version=push_data.since_version,
            client_checksum=push_data.client_checksum,
            messages_to_push=[push_data.message],
            operation=push_data.strategy,
        )

        sync_response = await coordinator.sync(sync_request)

        logger.info(
            "[RESTART_DEBUG] Push sync completed",
            extra={
                "target_episode_id": target_episode_id,
                "new_version": sync_response.current_version.sequence,
                "sync_mode": (
                    sync_response.sync_mode.value
                    if sync_response.sync_mode and hasattr(sync_response.sync_mode, "value")
                    else sync_response.sync_mode
                ),
            },
        )

        # Build response
        ack_data, state_event = await self._build_push_response(
            coordinator=coordinator,
            sync_response=sync_response,
            episode_id=episode_id,
            target_episode_id=target_episode_id,
            is_cross_episode=is_cross_episode,
            strategy=push_data.strategy,
        )

        # Send push acknowledgment to the pusher
        ack_message = PushAckMessage(
            data=ack_data,
            id=data.get("id", ""),
            timestamp=datetime.utcnow().isoformat(),
        )
        await websocket.send_json(ack_message.model_dump())

        logger.info(
            "[RESTART_DEBUG] Push ack sent to pusher",
            extra={
                "source_episode_id": episode_id,
                "target_episode_id": target_episode_id,
                "ack_version": ack_data.version,
            },
        )

        # Broadcast state event to target episode ONLY for cross-episode injections.
        # For same-episode pushes, skip broadcast - the client already knows what
        # happened (it initiated the push), and broadcasting to self just fills up
        # the client's event queue with unneeded state events.
        if state_event and is_cross_episode:
            logger.info(
                "[RESTART_DEBUG] Broadcasting state event to target episode (cross-episode)",
                extra={
                    "target_episode_id": target_episode_id,
                    "event_type": state_event.type,
                    "event_version": state_event.data.version if state_event.data else None,
                    "event_state": state_event.data.state if state_event.data else None,
                },
            )
            await coordinator.connection_manager.broadcast_to_episode(
                episode_id=target_episode_id,
                message=state_event,
            )
            logger.info(
                "[RESTART_DEBUG] State event broadcast completed",
                extra={"target_episode_id": target_episode_id},
            )
        elif state_event:
            logger.debug(
                "[RESTART_DEBUG] Skipping state event broadcast for same-episode push",
                extra={
                    "episode_id": episode_id,
                    "event_type": state_event.type,
                },
            )

    async def _build_push_response(
        self,
        coordinator: "TranscriptCoordinator",
        sync_response: Any,
        episode_id: str,
        target_episode_id: str,
        is_cross_episode: bool,
        strategy: str,
    ) -> tuple[PushAckData, StateEventMessage | None]:
        """Build push acknowledgment and optional state event.

        Returns:
            Tuple of (PushAckData, Optional[StateEventMessage])
        """
        target_episode = coordinator.episode_manager.get_episode_by_id(target_episode_id)

        modification_count: int | None = None
        state_event: StateEventMessage | None = None

        if is_cross_episode and target_episode:
            # Cross-episode push (injection): update modification count
            modification_count = target_episode.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0) + 1

            await target_episode.update_context_atomic({MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: modification_count})

            # Get current state for the event
            state_machine = coordinator._state_machine
            target_state = state_machine.get_state(target_episode)

            # Convert strategy to TranscriptOperation
            try:
                operation = TranscriptOperation(strategy)
            except ValueError:
                operation = TranscriptOperation.APPEND

            # Build state event for broadcast
            from ..episodes.transcript_state_machine import TranscriptStateMachine

            event_type = TranscriptStateMachine.state_to_event_type(target_state)

            state_event = StateEventMessage(
                type=event_type,
                data=StateEventData(
                    version=sync_response.current_version.sequence,
                    operation=operation,
                    modification_count=modification_count,
                    injected_by=episode_id,
                    state=target_state.value,
                ),
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat(),
            )

        ack_data = PushAckData(
            version=sync_response.current_version.sequence,
            checksum=sync_response.current_version.checksum,
            modification_count=modification_count,
            target_episode_id=target_episode_id if is_cross_episode else None,
        )

        return ack_data, state_event


class WebSocketMessageRouter:
    """Routes WebSocket messages to appropriate handlers.

    Provides a clean interface for the WebSocket endpoint to delegate
    message handling without caring about message type details.
    """

    def __init__(self) -> None:
        """Initialize the router with default handlers."""
        self._handlers: dict[str, BaseWebSocketHandler] = {
            WebSocketMessageType.PING: PingHandler(),
            WebSocketMessageType.SYNC_REQUEST: SyncRequestHandler(),
            WebSocketMessageType.PUSH_MESSAGE: PushMessageHandler(),
        }

    async def route(
        self,
        data: dict[str, Any],
        websocket: WebSocket,
        episode_id: str,
        coordinator: "TranscriptCoordinator",
    ) -> bool:
        """Route a message to its handler.

        Args:
            data: Parsed JSON message data
            websocket: WebSocket connection
            episode_id: Current episode ID
            coordinator: TranscriptCoordinator instance

        Returns:
            True if message was handled, False if unknown message type
        """
        message_type = data.get("type")
        if message_type is None:
            logger.warning(
                "WebSocket message missing type field",
                extra={"episode_id": episode_id},
            )
            return False

        handler = self._handlers.get(message_type)

        if handler:
            await handler.handle(data, websocket, episode_id, coordinator)
            return True
        else:
            logger.warning(
                "Unknown WebSocket message type",
                extra={"episode_id": episode_id, "type": message_type},
            )
            return False


# Singleton router instance for use across all WebSocket connections
_message_router = WebSocketMessageRouter()


def get_message_router() -> WebSocketMessageRouter:
    """Get the singleton message router instance."""
    return _message_router


__all__ = [
    "BaseWebSocketHandler",
    "PingHandler",
    "SyncRequestHandler",
    "PushMessageHandler",
    "WebSocketMessageRouter",
    "get_message_router",
]
