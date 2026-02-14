"""Transcript repository - thin wrapper around EpisodeEventRepository.

Provides transcript-specific operations while delegating to the unified
event storage layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...db.event_repository import EpisodeEventRepository


class TranscriptRepository:
    """Repository for transcript message operations.

    Wraps EpisodeEventRepository to provide a transcript-specific interface.
    All data is stored in Redis via the underlying event repository.
    """

    def __init__(self, event_repo: EpisodeEventRepository) -> None:
        """Initialize with event repository.

        Args:
            event_repo: Underlying event storage repository
        """
        self._event_repo = event_repo

    async def append_message(
        self,
        episode_id: str,
        session_id: str,
        # Any: pass-through JSONB message content from clients
        message: dict[str, Any],
    ) -> int:
        """Append a message to the transcript.

        Args:
            episode_id: Episode UUID
            session_id: Session ID
            message: Message dict with role, content, etc. (Any: JSONB pass-through)

        Returns:
            The sequence number assigned to this message
        """
        role = message.get("role", "user")
        tool_call_id = message.get("tool_call_id")

        sequence = await self._event_repo.insert_message(
            episode_id=episode_id,
            session_id=session_id,
            role=role,
            content=message,
            tool_call_id=tool_call_id,
        )

        return sequence

    async def get_transcript(
        self,
        episode_id: str,
        since_sequence: int = 0,
    ) -> list[dict[str, Any]]:
        """Get transcript messages.

        Args:
            episode_id: Episode UUID
            since_sequence: Only return messages after this sequence (0 = all)

        Returns:
            List of message dicts ordered by sequence
            (Any: deserialized JSONB, structure varies by message role)
        """
        if since_sequence > 0:
            # Go directly to the filtered path — no need to fetch all messages first
            events = await self._event_repo.get_episode_events(episode_id, event_types=["message"])
            filtered = []
            for event in events:
                if event.get("sequence_num", 0) > since_sequence:
                    content = event.get("content")
                    if content is not None:
                        filtered.append(content)
            return filtered

        return await self._event_repo.get_transcript_messages(episode_id)

    async def get_message_count(self, episode_id: str) -> int:
        """Get count of messages in transcript.

        Args:
            episode_id: Episode UUID

        Returns:
            Number of messages
        """
        return await self._event_repo.get_transcript_count(episode_id)

    async def get_latest_sequence(self, episode_id: str) -> int:
        """Get the latest sequence number.

        Args:
            episode_id: Episode UUID

        Returns:
            Latest sequence number, or -1 if empty
        """
        return await self._event_repo.get_latest_sequence(episode_id)


__all__ = ["TranscriptRepository"]
