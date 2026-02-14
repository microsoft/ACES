"""Background monitor for detecting stuck episode states."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from ....logging_config import LogCategory, get_saber_logger
from ....models.constants import MetadataKeys

if TYPE_CHECKING:
    from ...base import Episode
    from ..protocols import EpisodeManagerProtocol
    from .coordinator import TranscriptCoordinator

logger = get_saber_logger(LogCategory.EPISODE, __name__)


class StuckStateMonitor:
    """Background monitor for detecting stuck episodes.

    Periodically checks all active episodes and detects those stuck
    in the same state for too long. Broadcasts ERROR_STUCK events
    and optionally triggers remediation.
    """

    def __init__(
        self,
        coordinator: TranscriptCoordinator,
        episode_manager: EpisodeManagerProtocol,
        check_interval: float = 30.0,
        stuck_threshold: float = 300.0,
    ) -> None:
        """Initialize stuck state monitor.

        Args:
            coordinator: TranscriptCoordinator instance
            episode_manager: EpisodeManager instance
            check_interval: Seconds between checks (default 30s)
            stuck_threshold: Default stuck threshold in seconds (default 5min)
        """
        self.coordinator = coordinator
        self.episode_manager = episode_manager
        self.check_interval = check_interval
        self.stuck_threshold = stuck_threshold
        self._monitor_task: asyncio.Task | None = None
        self._running = False
        self._alerted_episodes: set[str] = set()  # Track episodes we've already alerted for

    async def start(self) -> None:
        """Start the background monitor."""
        if self._running:
            logger.warning("Stuck state monitor already running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info(
            "Started stuck state monitor",
            extra={
                "check_interval": self.check_interval,
                "stuck_threshold": self.stuck_threshold,
            },
        )

    async def stop(self) -> None:
        """Stop the background monitor."""
        if not self._running:
            return

        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped stuck state monitor")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_episodes()
            except Exception as e:
                logger.error("Error in stuck state monitor", extra={"error": str(e)}, exc_info=True)

            # Wait before next check
            await asyncio.sleep(self.check_interval)

    async def _check_all_episodes(self) -> None:
        """Check all active episodes for stuck states."""
        # Get all active episodes from the episodes dictionary
        all_episodes = list(self.episode_manager.episodes.values())

        for episode in all_episodes:
            # Skip terminated/completed episodes
            if episode.state.value in ("completed", "error", "terminated"):
                continue

            # Skip episodes we've already alerted for (idempotency)
            if episode.episode_id in self._alerted_episodes:
                continue

            # Get episode-specific threshold or use default
            threshold = episode.context.get(MetadataKeys.STUCK_STATE_THRESHOLD, self.stuck_threshold)

            # Check if stuck
            if self.coordinator._state_machine.is_episode_stuck(episode.episode_id, threshold):
                await self._handle_stuck_episode(episode, threshold)

    async def _handle_stuck_episode(self, episode: Episode, threshold: float) -> None:
        """Handle detection of stuck episode.

        Args:
            episode: Stuck episode
            threshold: Threshold that was exceeded
        """
        # Mark as alerted to avoid duplicate alerts
        self._alerted_episodes.add(episode.episode_id)

        duration = self.coordinator._state_machine.get_state_duration(episode.episode_id)

        current_state = episode.context.get(MetadataKeys.CURRENT_TRANSCRIPT_STATE, "UNKNOWN")

        logger.error(
            "Episode stuck in state",
            extra={
                "episode_id": episode.episode_id,
                "state": current_state,
                "duration_seconds": duration,
                "threshold_seconds": threshold,
            },
        )

        # Broadcast error event
        from ....models.rest.websocket_messages import TranscriptErrorData, TranscriptErrorMessage, TranscriptErrorType

        error_message = TranscriptErrorMessage(
            data=TranscriptErrorData(
                error=TranscriptErrorType.STUCK_STATE,
                state=current_state,
                duration_seconds=duration,
                threshold_seconds=threshold,
            ),
            timestamp=str(time.time()),
        )

        await self.coordinator.connection_manager.broadcast_to_episode(
            episode_id=episode.episode_id,
            message=error_message,
        )

        # Update episode context with stuck flag
        await episode.update_context_atomic(
            {
                MetadataKeys.EPISODE_STUCK: True,
                MetadataKeys.STUCK_SINCE: time.time(),
            }
        )


__all__ = ["StuckStateMonitor"]
