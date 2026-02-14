"""Repository for episode event storage and retrieval using Redis."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ...logging_config import LogCategory, get_saber_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_saber_logger(LogCategory.DATABASE, __name__)

_DEFAULT_TTL_SECONDS = 86400  # 24 hours — max episode lifetime from first event


class EpisodeEventRepository:
    """Repository for unified episode event storage backed by Redis.

    This repository provides CRUD operations for episode events using
    Redis sorted sets and counters.

    Key patterns:
        - ``episode:{episode_id}:events`` — Sorted Set of JSON event blobs
        - ``episode:{episode_id}:seq`` — atomic sequence counter
        - ``episode:{episode_id}:msg_count`` — message counter
        - ``episode:{episode_id}:step_count`` — completed-action counter

    Event Types:
        - 'message': Transcript message (user, assistant, system, tool)
        - 'action_start': Tool execution started
        - 'action_end': Tool execution completed

    Usage:
        client = redis_manager.client
        repo = EpisodeEventRepository(client)

        seq = await repo.insert_message(
            episode_id="...",
            session_id="...",
            role="user",
            content={"content": "Hello"},
        )

        events = await repo.get_episode_events(episode_id)
    """

    def __init__(self, client: Redis) -> None:
        """Initialize repository with a redis.asyncio client.

        Args:
            client: redis.asyncio.Redis client from RedisManager
        """
        self._client = client
        self._ttl_set: set[str] = set()

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _events_key(episode_id: str) -> str:
        return f"episode:{episode_id}:events"

    @staticmethod
    def _seq_key(episode_id: str) -> str:
        return f"episode:{episode_id}:seq"

    @staticmethod
    def _msg_count_key(episode_id: str) -> str:
        return f"episode:{episode_id}:msg_count"

    @staticmethod
    def _step_count_key(episode_id: str) -> str:
        return f"episode:{episode_id}:step_count"

    # ------------------------------------------------------------------
    # TTL helper
    # ------------------------------------------------------------------

    async def _ensure_ttl(self, episode_id: str) -> None:
        """Set TTL on all Redis keys for an episode, once per episode lifetime.

        TTL is only applied on the first insert for each episode (tracked via
        ``_ttl_set``). This means every episode has a **24-hour maximum
        lifetime** measured from its first stored event, regardless of
        subsequent activity.

        Design notes:
            * **Performance**: Skipping repeated ``EXPIRE`` calls on every
              write keeps the hot path fast.
            * **Crash edge-case**: If the process crashes after the data write
              but before the ``EXPIRE`` calls complete, the affected keys will
              persist in Redis without an expiry.  This is an accepted trade-off
              — a periodic Redis key-scan or operator intervention can clean up
              orphaned keys if necessary.

        Args:
            episode_id: Episode UUID whose keys should receive a TTL.
        """
        if episode_id in self._ttl_set:
            return
        self._ttl_set.add(episode_id)
        ttl = _DEFAULT_TTL_SECONDS
        for key in (
            self._events_key(episode_id),
            self._seq_key(episode_id),
            self._msg_count_key(episode_id),
            self._step_count_key(episode_id),
        ):
            await self._client.expire(key, ttl)

    # ------------------------------------------------------------------
    # Insert methods
    # ------------------------------------------------------------------

    async def insert_message(
        self,
        episode_id: str,
        session_id: str,
        role: str,
        # JSON content — structure varies by event_type
        content: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> int:
        """Insert a transcript message event.

        Args:
            episode_id: Episode UUID
            session_id: Session ID
            role: Message role (user, assistant, system, tool)
            content: Full message content as dict (Any: JSON blob, schema varies by role)
            tool_call_id: For tool messages, the tool_call_id being responded to

        Returns:
            The sequence number assigned to this event
        """
        seq: int = int(await self._client.incr(self._seq_key(episode_id)))

        event = {
            "episode_id": episode_id,
            "session_id": session_id,
            "event_type": "message",
            "sequence_num": seq,
            "role": role,
            "content": content,
            "tool_call_id": tool_call_id,
        }

        await self._client.zadd(self._events_key(episode_id), {json.dumps(event): seq})
        await self._client.incr(self._msg_count_key(episode_id))
        await self._ensure_ttl(episode_id)

        logger.debug(
            "Inserted message event",
            extra={
                "event": "db_message_inserted",
                "episode_id": episode_id,
                "sequence_num": seq,
                "role": role,
            },
        )

        return seq

    async def insert_action_start(
        self,
        episode_id: str,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        # Any: tool parameters are arbitrary JSON from different tool schemas
        parameters: dict[str, Any],
    ) -> int:
        """Insert an action_start event (tool execution beginning).

        Args:
            episode_id: Episode UUID
            session_id: Session ID
            tool_call_id: Links to transcript's tool_call
            tool_name: Name of tool being executed
            parameters: Tool parameters (Any: arbitrary JSON from tool schemas)

        Returns:
            The sequence number assigned to this event
        """
        seq: int = int(await self._client.incr(self._seq_key(episode_id)))

        content = {
            "tool_name": tool_name,
            "parameters": parameters,
        }

        event = {
            "episode_id": episode_id,
            "session_id": session_id,
            "event_type": "action_start",
            "sequence_num": seq,
            "tool_call_id": tool_call_id,
            "content": content,
        }

        await self._client.zadd(self._events_key(episode_id), {json.dumps(event): seq})
        await self._ensure_ttl(episode_id)

        logger.debug(
            "Inserted action_start event",
            extra={
                "event": "db_action_start_inserted",
                "episode_id": episode_id,
                "sequence_num": seq,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
            },
        )

        return seq

    async def insert_action_end(
        self,
        episode_id: str,
        session_id: str,
        tool_call_id: str,
        # Any: CommandResult serialized as JSON — structure varies by tool
        result: dict[str, Any],
        execution_time_ms: int,
        success: bool,
    ) -> int:
        """Insert an action_end event (tool execution completed).

        Args:
            episode_id: Episode UUID
            session_id: Session ID
            tool_call_id: Links to action_start and transcript
            result: CommandResult as dict (Any: JSON, schema varies by tool)
            execution_time_ms: Execution time in milliseconds
            success: Whether execution succeeded

        Returns:
            The sequence number assigned to this event
        """
        seq: int = int(await self._client.incr(self._seq_key(episode_id)))

        event = {
            "episode_id": episode_id,
            "session_id": session_id,
            "event_type": "action_end",
            "sequence_num": seq,
            "tool_call_id": tool_call_id,
            "content": result,
            "execution_time_ms": execution_time_ms,
            "success": success,
        }

        await self._client.zadd(self._events_key(episode_id), {json.dumps(event): seq})
        await self._client.incr(self._step_count_key(episode_id))
        await self._ensure_ttl(episode_id)

        logger.debug(
            "Inserted action_end event",
            extra={
                "event": "db_action_end_inserted",
                "episode_id": episode_id,
                "sequence_num": seq,
                "tool_call_id": tool_call_id,
                "success": success,
                "execution_time_ms": execution_time_ms,
            },
        )

        return seq

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    async def get_episode_events(
        self,
        episode_id: str,
        event_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all events for an episode, ordered by sequence.

        Args:
            episode_id: Episode UUID
            event_types: Optional filter for event types

        Returns:
            List of event dicts ordered by sequence_num
        """
        raw: list[bytes] = await self._client.zrangebyscore(self._events_key(episode_id), "-inf", "+inf")

        events: list[dict[str, Any]] = [json.loads(r) for r in raw]

        if event_types:
            events = [e for e in events if e.get("event_type") in event_types]

        return events

    async def get_transcript_messages(self, episode_id: str) -> list[dict[str, Any]]:
        """Get only message content for an episode.

        Args:
            episode_id: Episode UUID

        Returns:
            List of message content dicts ordered by sequence
        """
        events = await self.get_episode_events(episode_id, event_types=["message"])
        return [e.get("content", {}) for e in events]

    async def get_transcript_count(self, episode_id: str) -> int:
        """Get count of message events for an episode.

        Args:
            episode_id: Episode UUID

        Returns:
            Number of message events in the transcript
        """
        val = await self._client.get(self._msg_count_key(episode_id))
        return int(val) if val else 0

    async def get_step_count(self, episode_id: str) -> int:
        """Get count of completed actions (steps) for an episode.

        Args:
            episode_id: Episode UUID

        Returns:
            Number of action_end events
        """
        val = await self._client.get(self._step_count_key(episode_id))
        return int(val) if val else 0

    async def get_latest_sequence(self, episode_id: str) -> int:
        """Get the latest sequence number for an episode.

        Args:
            episode_id: Episode UUID

        Returns:
            Current sequence number, or -1 if no events exist
        """
        val = await self._client.get(self._seq_key(episode_id))
        return int(val) if val else -1

    async def delete_episode_events(self, episode_id: str) -> int:
        """Delete all events for an episode (cleanup).

        Args:
            episode_id: Episode UUID

        Returns:
            Number of events deleted
        """
        count: int = int(await self._client.zcard(self._events_key(episode_id)))

        await self._client.delete(
            self._events_key(episode_id),
            self._seq_key(episode_id),
            self._msg_count_key(episode_id),
            self._step_count_key(episode_id),
        )

        # Clear TTL tracking for this episode
        self._ttl_set.discard(episode_id)

        logger.info(
            "Deleted episode events",
            extra={
                "event": "db_episode_deleted",
                "episode_id": episode_id,
                "count": count,
            },
        )

        return count
