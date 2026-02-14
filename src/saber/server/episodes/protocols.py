"""Protocol definitions for episode management dependencies.

This module provides Protocol (structural typing) definitions that allow
components to declare their dependencies without circular imports.

Protocols enable duck typing with type checking - any object that implements
the required methods can be used, without requiring inheritance.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from ...models.rest.websocket_messages import WebSocketServerMessage
from ..base import Episode

# Type aliases for lifecycle callbacks
EpisodeCreatedCallback = Callable[[str], Awaitable[None]]  # (episode_id) -> None
EpisodeEndedCallback = Callable[[str], Awaitable[None]]  # (episode_id) -> None


class EpisodeManagerProtocol(Protocol):
    """Protocol defining the required interface for EpisodeManager.

    This protocol allows TranscriptCoordinator and other components to
    declare their dependency on EpisodeManager without importing the
    concrete class (avoiding circular imports).
    """

    @property
    def episodes(self) -> dict[str, Episode]:
        """Active episodes dictionary (episode_id -> Episode)."""
        ...

    def get_episode_by_id(self, episode_id: str) -> Episode | None:
        """Get episode by ID.

        Args:
            episode_id: Episode identifier

        Returns:
            Episode instance if found, None otherwise
        """
        ...

    def register_on_episode_created(self, callback: EpisodeCreatedCallback) -> None:
        """Register a callback for episode creation events.

        Args:
            callback: Async callable invoked with episode_id after creation
        """
        ...

    def register_on_episode_ended(self, callback: EpisodeEndedCallback) -> None:
        """Register a callback for episode termination events.

        Args:
            callback: Async callable invoked with episode_id before termination
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


__all__ = [
    "ConnectionManagerProtocol",
    "EpisodeCreatedCallback",
    "EpisodeEndedCallback",
    "EpisodeManagerProtocol",
]
