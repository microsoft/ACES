"""WebSocket message handlers for transcript push-only protocol.

This module extracts WebSocket message handling logic from session_rest_api.py
into dedicated handler classes for better separation of concerns.

Each handler implements a single message type (PING, PUSH_MESSAGE)
and encapsulates the business logic for that operation.
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket

from ...logging_config import get_api_logger
from ...models.rest.websocket_messages import (
    PongMessage,
    PushAckData,
    PushAckMessage,
    PushMessageRequestData,
    TranscriptErrorData,
    TranscriptErrorMessage,
    TranscriptErrorType,
    WebSocketMessageType,
)

if TYPE_CHECKING:
    from ..episodes.transcript.coordinator import TranscriptCoordinator

logger = get_api_logger(__name__)


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
        pong = PongMessage(timestamp=datetime.now(UTC).isoformat())
        await websocket.send_json(pong.model_dump())


class PushMessageHandler(BaseWebSocketHandler):
    """Handler for PUSH_MESSAGE messages.

    Supports both normal pushes and cross-episode injection.
    The coordinator handles DB storage and state broadcasting internally.
    """

    async def handle(
        self,
        data: dict[str, Any],
        websocket: WebSocket,
        episode_id: str,
        coordinator: "TranscriptCoordinator",
    ) -> None:
        """Handle message push (normal or cross-episode).

        Flow:
        1. Parse push data
        2. Look up target episode to get session_id
        3. Call coordinator.push_message() (handles DB + broadcast)
        4. Send PushAckMessage back to pusher
        """
        push_data = PushMessageRequestData(**data.get("data", {}))
        target_episode_id = push_data.target_episode_id or episode_id

        # Get session_id from episode
        episode = coordinator.episode_manager.get_episode_by_id(target_episode_id)
        if not episode:
            logger.warning(
                "Push target episode not found",
                extra={"target_episode_id": target_episode_id},
            )
            error_msg = TranscriptErrorMessage(
                data=TranscriptErrorData(
                    error=TranscriptErrorType.PUSH_FAILED,
                    message=f"Episode {target_episode_id} not found",
                ),
                timestamp=datetime.now(UTC).isoformat(),
            )
            await websocket.send_json(error_msg.model_dump())
            return

        try:
            # Push via coordinator (handles DB storage + state broadcast)
            sequence = await coordinator.push_message(
                episode_id=target_episode_id,
                session_id=episode.session_id,
                message=push_data.message,
                operation=push_data.strategy,
            )
        except (ValueError, RuntimeError) as e:
            logger.error(
                "Push message failed",
                extra={
                    "target_episode_id": target_episode_id,
                    "error": str(e),
                },
            )
            error_msg = TranscriptErrorMessage(
                data=TranscriptErrorData(
                    error=TranscriptErrorType.PUSH_FAILED,
                    message=str(e),
                ),
                timestamp=datetime.now(UTC).isoformat(),
            )
            await websocket.send_json(error_msg.model_dump())
            return
        except Exception as e:
            logger.error(
                "Unexpected error during push message",
                extra={
                    "target_episode_id": target_episode_id,
                    "error": str(e),
                },
                exc_info=True,
            )
            error_msg = TranscriptErrorMessage(
                data=TranscriptErrorData(
                    error=TranscriptErrorType.PUSH_FAILED,
                    message="Internal error during push",
                ),
                timestamp=datetime.now(UTC).isoformat(),
            )
            await websocket.send_json(error_msg.model_dump())
            return

        # Send ack to pusher
        is_cross_episode = target_episode_id != episode_id
        ack_data = PushAckData(
            sequence=sequence,
            target_episode_id=target_episode_id if is_cross_episode else None,
        )
        ack_message = PushAckMessage(
            data=ack_data,
            id=data.get("id", ""),
            timestamp=datetime.now(UTC).isoformat(),
        )
        await websocket.send_json(ack_message.model_dump())


class WebSocketMessageRouter:
    """Routes WebSocket messages to appropriate handlers.

    Provides a clean interface for the WebSocket endpoint to delegate
    message handling without caring about message type details.
    """

    def __init__(self) -> None:
        """Initialize the router with default handlers."""
        self._handlers: dict[str, BaseWebSocketHandler] = {
            WebSocketMessageType.PING: PingHandler(),
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
    "PushMessageHandler",
    "WebSocketMessageRouter",
    "get_message_router",
]
