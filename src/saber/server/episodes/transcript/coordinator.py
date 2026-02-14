"""Simplified transcript coordination service - push-only with DB storage.

This module provides centralized transcript coordination with:
- Push-only message storage via TranscriptRepository
- Redis storage as the source of truth
- WebSocket notifications for state changes
- State machine integration
- Auto-continue functionality
- Stuck state monitoring

Logging category: EPISODE
"""

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from ....logging_config import LogCategory, get_saber_logger
from ....models.constants import MetadataKeys
from ....models.rest.websocket_messages import (
    StateEventData,
    StateEventMessage,
    TranscriptOperation,
)
from ...time_source import TimeSource, UTCTimeSource
from ..protocols import ConnectionManagerProtocol, EpisodeManagerProtocol
from .auto_continue_manager import AutoContinueManager
from .repository import TranscriptRepository
from .state_machine import TranscriptState, TranscriptStateMachine
from .stuck_state_monitor import StuckStateMonitor

if TYPE_CHECKING:
    from ...db.event_repository import EpisodeEventRepository

logger = get_saber_logger(LogCategory.EPISODE, __name__)


class TranscriptCoordinator:
    """
    Simplified transcript coordinator with push-only storage.

    Uses Redis (via TranscriptRepository) as the source of truth.
    Read methods (get_transcript, get_message_count, get_latest_sequence)
    return safe defaults (empty lists, zero counts) when event_repository
    is not configured. Write methods (push_message) raise ValueError if
    the repository is missing — DB is required for writes.

    Flow:
    1. Client pushes message via WebSocket
    2. Server stores in DB, returns sequence number
    3. Server broadcasts state event to observers
    4. Clients read transcript via REST/get_transcript()

    Thread-safety: All operations are protected by an asyncio.Lock.
    """

    def __init__(
        self,
        episode_manager: EpisodeManagerProtocol,
        connection_manager: ConnectionManagerProtocol,
        event_repository: "EpisodeEventRepository | None" = None,
        time_source: TimeSource | None = None,
        stuck_check_interval: float = 30.0,
        stuck_threshold: float = 300.0,
    ) -> None:
        """
        Initialize the transcript coordinator.

        Args:
            episode_manager: EpisodeManager instance for episode access
            connection_manager: ConnectionManager instance for WebSocket broadcasting
            event_repository: Redis-backed repository for transcript storage (optional for backward compat)
            time_source: Time source for getting current time (defaults to UTCTimeSource)
            stuck_check_interval: Seconds between stuck state checks (default 30s)
            stuck_threshold: Default stuck threshold in seconds (default 5min)
        """
        self.episode_manager = episode_manager
        self.connection_manager = connection_manager
        self._repo = TranscriptRepository(event_repository) if event_repository else None
        self._state_machine = TranscriptStateMachine()
        self._lock = asyncio.Lock()
        self._time_source = time_source or UTCTimeSource()

        # Initialize auto-continue manager
        self._auto_continue = AutoContinueManager(episode_manager, self)

        # Initialize stuck state monitor
        self._stuck_monitor = StuckStateMonitor(
            coordinator=self,
            episode_manager=episode_manager,
            check_interval=stuck_check_interval,
            stuck_threshold=stuck_threshold,
        )

        # Register lifecycle hooks via callback registration (no monkey-patching)
        self._register_lifecycle_hooks()

    def _register_lifecycle_hooks(self) -> None:
        """Register state machine lifecycle hooks with episode manager."""
        self.episode_manager.register_on_episode_created(self._on_episode_created)
        self.episode_manager.register_on_episode_ended(self._on_episode_ended)

    def set_repository(self, event_repository: "EpisodeEventRepository") -> None:
        """Set the event repository for DB-backed transcript storage.

        Used for late-binding after async DB connection is established.

        Args:
            event_repository: Connected EpisodeEventRepository instance
        """
        self._repo = TranscriptRepository(event_repository)

    async def seed_initial_transcript(
        self,
        episode_id: str,
        session_id: str,
        messages: list[dict[str, str | list[str] | None]],
    ) -> None:
        """Seed the initial transcript (system + user messages) into Redis.

        Called after episode creation to persist the initial prompts.
        Must be called before any push_message() calls.

        Args:
            episode_id: Target episode ID
            session_id: Session ID
            messages: List of initial message dicts (system, user)
        """
        if not self._repo:
            logger.debug(
                "Skipping transcript seed (no repo configured)",
                extra={"episode_id": episode_id},
            )
            return

        for message in messages:
            await self._repo.append_message(
                episode_id=episode_id,
                session_id=session_id,
                message=message,
            )

        logger.debug(
            "Seeded initial transcript into Redis",
            extra={
                "episode_id": episode_id,
                "message_count": len(messages),
            },
        )

    async def _on_episode_created(self, episode_id: str) -> None:
        """Handle episode creation: init state machine and seed transcript in Redis."""
        await self._state_machine.on_episode_created(episode_id)

        # Seed initial transcript from episode context into Redis
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if episode:
            initial_messages = episode.context.get(MetadataKeys.INITIAL_TRANSCRIPT, [])
            if initial_messages and self._repo:
                await self.seed_initial_transcript(
                    episode_id=episode_id,
                    session_id=episode.session_id,
                    messages=initial_messages,
                )

    async def _on_episode_ended(self, episode_id: str) -> None:
        """Handle episode termination event."""
        await self._state_machine.on_episode_terminated(episode_id)

    async def push_message(
        self,
        episode_id: str,
        session_id: str,
        message: dict[str, str | list[str] | None],
        operation: str = "append",
    ) -> int:
        """
        Push a message to the transcript.

        Stores in DB and broadcasts state event. Triggers auto-continue if needed.

        Args:
            episode_id: Target episode ID
            session_id: Session ID
            message: Message dict with role, content, etc.
            operation: Operation type (default "append")

        Returns:
            Sequence number assigned to this message

        Raises:
            ValueError: If episode not found or DB not configured
        """
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            raise ValueError(f"Episode {episode_id} not found")

        if not self._repo:
            raise ValueError("Event repository not configured - DB storage required")

        async with self._lock:
            # Store in DB
            sequence = await self._repo.append_message(
                episode_id=episode_id,
                session_id=session_id,
                message=message,
            )

            logger.debug(
                "Pushed message to transcript",
                extra={
                    "episode_id": episode_id,
                    "sequence": sequence,
                    "role": message.get("role"),
                },
            )

            # Broadcast state event and get new state for auto-continue
            new_state = await self._broadcast_state_event(
                episode_id=episode_id,
                sequence=sequence,
                operation=operation,
                message=message,
            )

            # Trigger auto-continue if needed
            if new_state:

                async def _safe_auto_continue(ep_id: str, state: TranscriptState) -> None:
                    try:
                        await self._auto_continue.handle_auto_continue(ep_id, state)
                    except Exception:
                        logger.exception("Auto-continue failed", extra={"episode_id": ep_id})

                asyncio.create_task(_safe_auto_continue(episode_id, new_state))

            return sequence

    async def get_transcript(
        self,
        episode_id: str,
        since_sequence: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Get the transcript for an episode.

        Args:
            episode_id: Episode ID
            since_sequence: Only return messages after this sequence (0 = all)

        Returns:
            List of message dicts ordered by sequence
        """
        if not self._repo:
            return []
        return await self._repo.get_transcript(episode_id, since_sequence)

    async def get_initial_state_event(
        self,
        episode_id: str,
    ) -> StateEventMessage | None:
        """
        Get state event for initial WebSocket connection.

        Args:
            episode_id: Episode ID

        Returns:
            StateEventMessage for current state, or None if episode not found
        """
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            return None

        if not self._repo:
            # No DB - return basic state event
            return StateEventMessage(
                type="is_waiting_on_user",
                data=StateEventData(
                    version=0,
                    operation=TranscriptOperation.INIT,
                    modification_count=0,
                    state=TranscriptState.WAITING_FOR_USER.value,
                ),
                id=str(uuid.uuid4()),
                timestamp=self._time_source.now().isoformat(),
            )

        # Fetch transcript from DB
        transcript = await self._repo.get_transcript(episode_id)
        sequence = await self._repo.get_latest_sequence(episode_id)

        # Compute state
        current_state = self._state_machine.compute_state_from_transcript(transcript)
        event_type = TranscriptStateMachine.state_to_event_type(current_state)

        return StateEventMessage(
            type=event_type,
            data=StateEventData(
                version=max(0, sequence),
                operation=TranscriptOperation.INIT,
                modification_count=0,
                state=current_state.value,
            ),
            id=str(uuid.uuid4()),
            timestamp=self._time_source.now().isoformat(),
        )

    async def get_message_count(self, episode_id: str) -> int:
        """Get message count for lightweight polling."""
        if not self._repo:
            return 0
        return await self._repo.get_message_count(episode_id)

    async def get_latest_sequence(self, episode_id: str) -> int:
        """Get latest sequence number."""
        if not self._repo:
            return 0
        return await self._repo.get_latest_sequence(episode_id)

    async def _broadcast_state_event(
        self,
        episode_id: str,
        sequence: int,
        operation: str,
        message: dict[str, str | list[str] | None],
    ) -> TranscriptState | None:
        """Broadcast state event after push.

        Computes state from the full transcript to ensure correct state
        during multi-tool-call flows.

        Args:
            episode_id: Episode ID
            sequence: Current sequence number
            operation: Operation type (append, etc.)
            message: The pushed message dict (fallback for state computation)

        Returns:
            The new transcript state for auto-continue handling
        """
        # Compute state from the full transcript (not just the latest message)
        # to handle multi-tool-call flows correctly
        if self._repo:
            transcript = await self._repo.get_transcript(episode_id)
        else:
            transcript = [message]
        new_state = self._state_machine.compute_state_from_transcript(transcript)

        # Update timestamp on state change
        self._state_machine.update_state_timestamp(episode_id)

        event_type = TranscriptStateMachine.state_to_event_type(new_state)

        try:
            operation_enum = TranscriptOperation(operation)
        except ValueError:
            operation_enum = TranscriptOperation.APPEND

        state_event = StateEventMessage(
            type=event_type,
            data=StateEventData(
                version=sequence,
                operation=operation_enum,
                modification_count=0,
                state=new_state.value,
            ),
            id=str(uuid.uuid4()),
            timestamp=self._time_source.now().isoformat(),
        )

        await self.connection_manager.broadcast_to_episode(
            episode_id=episode_id,
            message=state_event,
        )

        return new_state

    async def start_monitor(self) -> None:
        """Start background monitors (stuck state detection)."""
        await self._stuck_monitor.start()

    async def stop_monitor(self) -> None:
        """Stop background monitors."""
        await self._stuck_monitor.stop()

    async def cleanup_episode(self, episode_id: str) -> None:
        """Cleanup episode coordination state.

        Args:
            episode_id: Episode to cleanup
        """
        await self.connection_manager.cleanup_episode(episode_id)


__all__ = ["TranscriptCoordinator"]
