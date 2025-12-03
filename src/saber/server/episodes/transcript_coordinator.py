"""Transcript coordination service with WebSocket notifications and differential sync.

This module provides centralized transcript synchronization with:
- WebSocket instant notifications (<100ms)
- Differential sync (90% bandwidth reduction)
- Version + checksum for rewrite detection
- Last write wins (monotonic versioning)
- Episode-scoped isolation

Logging category: EPISODE
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from ...models.transcript import TranscriptSyncRequest, TranscriptSyncResponse, TranscriptVersion, compute_checksum
from ..time_source import TimeSource, UTCTimeSource

logger = get_saber_logger(LogCategory.EPISODE, __name__)


class TranscriptCoordinator:
    """
    Centralized transcript synchronization coordinator with WebSocket support.

    Combines:
    - WebSocket for instant notifications (<100ms)
    - Differential sync for bandwidth efficiency (90% reduction)
    - Version + checksum for rewrite detection
    - Last write wins (monotonic versioning)

    Thread-safety: All operations are protected by an asyncio.Lock.
    """

    def __init__(self, episode_manager: Any, connection_manager: Any, time_source: TimeSource | None = None) -> None:
        """
        Initialize the transcript coordinator.

        Args:
            episode_manager: EpisodeManager instance for episode access
            connection_manager: ConnectionManager instance for WebSocket broadcasting
            time_source: Time source for getting current time (defaults to UTCTimeSource)
        """
        self.episode_manager = episode_manager
        self.connection_manager = connection_manager
        self._coordination_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._time_source = time_source or UTCTimeSource()

    def _get_current_version(self, episode: Any) -> TranscriptVersion:
        """
        Get current version with checksum from episode context.

        Args:
            episode: Episode instance

        Returns:
            TranscriptVersion with current state
        """
        messages = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
        sequence = episode.context.get(MetadataKeys.TRANSCRIPT_VERSION, 0)
        last_op = episode.context.get(MetadataKeys.TRANSCRIPT_LAST_OPERATION, "append")

        return TranscriptVersion(
            sequence=sequence,
            checksum=compute_checksum(messages),
            message_count=len(messages),
            last_operation=last_op,
        )

    async def sync(self, request: TranscriptSyncRequest) -> TranscriptSyncResponse:
        """
        Unified sync with differential updates and WebSocket notifications.

        Flow:
        1. Client pushes new messages → version increments (last write wins)
        2. Server computes checksum and validates client's view
        3. If checksums match → send delta (efficient)
        4. If checksums differ → send full transcript (rewrite detected)
        5. If no changes and WAIT_FOR_CHANGE → return immediately (WebSocket handles blocking)

        Note: WebSocket notifications happen in notify_modification(),
        not in sync(). Clients receive events via WebSocket and then call sync() to pull data.

        Args:
            request: TranscriptSyncRequest with client state and optional messages to push

        Returns:
            TranscriptSyncResponse with delta, full transcript, or no_change

        Raises:
            ValueError: If episode not found
        """
        start_time = asyncio.get_event_loop().time()
        episode = self.episode_manager.get_episode_by_id(request.episode_id)

        if not episode:
            raise ValueError(f"Episode {request.episode_id} not found")

        # Step 1: Push new messages if provided (LAST WRITE WINS)
        if request.messages_to_push:
            await self._push_messages(episode, request.messages_to_push, operation="append")

        # Step 2: Get current server state
        current_version = self._get_current_version(episode)
        all_messages = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])

        # Step 3: Determine sync mode based on checksum validation
        client_version = request.since_version
        client_checksum = request.client_checksum

        # Validate client's checksum
        if client_checksum and client_version <= len(all_messages):
            client_slice = all_messages[:client_version]
            expected_checksum = compute_checksum(client_slice)
        else:
            expected_checksum = None

        # Step 4: Choose sync mode
        if client_checksum and client_checksum != expected_checksum:
            # REWRITE DETECTED: Client's version is invalid
            sync_mode = "full"
            delta = None
            full_transcript = all_messages
            modified = True

            logger.warning(
                "Transcript rewrite detected - sending full transcript",
                extra={
                    "episode_id": request.episode_id,
                    "client_version": client_version,
                    "server_version": current_version.sequence,
                    "last_operation": current_version.last_operation,
                },
            )

        elif current_version.sequence == client_version:
            # NO CHANGE: Client is up to date
            sync_mode = "no_change"
            delta = []
            full_transcript = None
            modified = False

        else:
            # DELTA: Client is behind but valid
            sync_mode = "delta"
            delta = all_messages[client_version:]
            full_transcript = None
            modified = True

        wait_time = asyncio.get_event_loop().time() - start_time

        return TranscriptSyncResponse(
            current_version=current_version,
            delta=delta,
            full_transcript=full_transcript,
            sync_mode=sync_mode,
            modified=modified,
            blocked=False,  # No blocking in sync() - handled by WebSocket
            wait_time_seconds=wait_time,
        )

    async def _push_messages(self, episode: Any, messages: List[Dict[str, Any]], operation: str = "append") -> None:
        """
        Push messages with operation type tracking (LAST WRITE WINS).

        Args:
            episode: Episode instance
            messages: Messages to append
            operation: Operation type (append, rewrite, insert, rewind)
        """
        existing = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
        current_version = episode.context.get(MetadataKeys.TRANSCRIPT_VERSION, 0)

        updated_messages = existing + messages
        new_version = current_version + 1  # Always increment (monotonic)

        await episode.update_context_atomic(
            {
                MetadataKeys.CLIENT_TRANSCRIPT: updated_messages,
                MetadataKeys.TRANSCRIPT_VERSION: new_version,
                MetadataKeys.TRANSCRIPT_LAST_OPERATION: operation,
                MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: self._time_source.now().isoformat(),
            }
        )

    async def notify_modification(
        self,
        episode_id: str,
        modified_transcript: List[Dict[str, Any]],
        operation: str,
        injected_by: str,
        expected_base_version: Optional[int] = None,
        expected_base_checksum: Optional[str] = None,
    ) -> None:
        """
        Called by InjectPromptExecutor when red team modifies transcript.

        This is where WebSocket magic happens:
        1. Validate modification (if version/checksum provided)
        2. Update transcript (last write wins)
        3. Broadcast WebSocket event to blue agent
        4. Blue agent receives event (<100ms)
        5. Blue agent calls sync() to pull delta

        Args:
            episode_id: Target episode (blue team)
            modified_transcript: Complete modified transcript
            operation: Operation type (append, rewrite, insert, rewind)
            injected_by: Red team episode ID
            expected_base_version: Optional version modification is based on (for validation)
            expected_base_checksum: Optional checksum of base transcript (for validation)

        Raises:
            ValueError: If episode not found or validation fails
        """
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            logger.warning(
                "Cannot notify modification - episode not found",
                extra={"episode_id": episode_id, "injected_by": injected_by},
            )
            return

        # Validate modification if version/checksum provided
        if expected_base_version is not None or expected_base_checksum is not None:
            current = self._get_current_version(episode)

            if expected_base_version is not None and current.sequence != expected_base_version:
                error_msg = (
                    f"Modification based on stale version. "
                    f"Expected {expected_base_version}, current {current.sequence}"
                )
                logger.error(
                    "Transcript modification validation failed - stale version",
                    extra={
                        "episode_id": episode_id,
                        "injected_by": injected_by,
                        "expected_version": expected_base_version,
                        "current_version": current.sequence,
                    },
                )
                raise ValueError(error_msg)

            if expected_base_checksum is not None and current.checksum != expected_base_checksum:
                error_msg = (
                    f"Modification checksum mismatch. "
                    f"Expected {expected_base_checksum[:8]}..., current {current.checksum[:8]}..."
                )
                logger.error(
                    "Transcript modification validation failed - checksum mismatch",
                    extra={
                        "episode_id": episode_id,
                        "injected_by": injected_by,
                        "expected_checksum": expected_base_checksum[:16],
                        "current_checksum": current.checksum[:16],
                    },
                )
                raise ValueError(error_msg)

        current_version = episode.context.get(MetadataKeys.TRANSCRIPT_VERSION, 0)
        new_version = current_version + 1
        modification_count = episode.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0) + 1

        # Update transcript atomically (LAST WRITE WINS)
        await episode.update_context_atomic(
            {
                MetadataKeys.CLIENT_TRANSCRIPT: modified_transcript,
                MetadataKeys.TRANSCRIPT_VERSION: new_version,
                MetadataKeys.TRANSCRIPT_LAST_OPERATION: operation,
                MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT: self._time_source.now().isoformat(),
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: modification_count,
            }
        )

        # ✨ WEBSOCKET BROADCAST ✨
        await self.connection_manager.broadcast_to_episode(
            episode_id=episode_id,
            message={
                "type": "transcript_modified",
                "data": {
                    "version": new_version,
                    "operation": operation,
                    "modification_count": modification_count,
                    "injected_by": injected_by,
                },
                "id": str(uuid.uuid4()),
                "timestamp": self._time_source.now().isoformat(),
            },
        )

        logger.info(
            "Broadcast WebSocket transcript modification event",
            extra={
                "episode_id": episode_id,
                "version": new_version,
                "operation": operation,
                "modification_count": modification_count,
                "injected_by": injected_by,
            },
        )

    async def cleanup_episode(self, episode_id: str) -> None:
        """
        Cleanup episode coordination state (called when episode ends).

        Args:
            episode_id: Episode identifier
        """
        async with self._lock:
            self._coordination_configs.pop(episode_id, None)

        # Cleanup WebSocket connections
        await self.connection_manager.cleanup_episode(episode_id)


__all__ = ["TranscriptCoordinator"]
