"""Protocol definitions for episode management dependencies.

This module provides Protocol (structural typing) definitions that allow
components to declare their dependencies without circular imports.

Protocols enable duck typing with type checking - any object that implements
the required methods can be used, without requiring inheritance.
"""

from typing import Protocol

from ...models.rest.websocket_messages import WebSocketServerMessage
from ..base import Episode


class EpisodeManagerProtocol(Protocol):
    """Protocol defining the required interface for EpisodeManager.

    This protocol allows TranscriptCoordinator and other components to
    declare their dependency on EpisodeManager without importing the
    concrete class (avoiding circular imports).
    """

    def get_episode_by_id(self, episode_id: str) -> Episode | None:
        """Get episode by ID.

        Args:
            episode_id: Episode identifier

        Returns:
            Episode instance if found, None otherwise
        """
        ...


class ConnectionManagerProtocol(Protocol):
    """Protocol defining the required interface for ConnectionManager.

    This protocol allows TranscriptCoordinator and other components to
    declare their dependency on ConnectionManager without importing the
    concrete class.
    """

    async def broadcast_to_episode(
        self,
        episode_id: str,
        message: WebSocketServerMessage,
    ) -> None:
        """Broadcast message to all WebSocket connections for an episode.

        Args:
            episode_id: Target episode identifier
            message: Typed WebSocket message
        """
        ...

    async def cleanup_episode(self, episode_id: str) -> None:
        """Cleanup all WebSocket connections for an episode.

        Args:
            episode_id: Episode identifier to cleanup
        """
        ...


__all__ = ["EpisodeManagerProtocol", "ConnectionManagerProtocol"]
