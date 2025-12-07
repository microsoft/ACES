"""Auto-continue manager for transcript coordination."""

from typing import Any

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from .transcript_state_machine import TranscriptState

logger = get_saber_logger(LogCategory.EPISODE, __name__)


class AutoContinueManager:
    """Manages automatic continuation of agent execution.

    Injects "continue" messages when transcript reaches WAITING_FOR_USER state
    and auto-continue is enabled. Tracks idempotency to prevent duplicate injections.
    """

    DEFAULT_CONTINUE_PROMPT = "Continue solving the task."

    def __init__(self, episode_manager: Any, transcript_coordinator: Any):
        """Initialize auto-continue manager.

        Args:
            episode_manager: Episode manager for retrieving episodes
            transcript_coordinator: Transcript coordinator for injecting messages
        """
        self.episode_manager = episode_manager
        self.transcript_coordinator = transcript_coordinator

    async def handle_auto_continue(self, episode_id: str, new_state: TranscriptState) -> None:
        """Handle auto-continue logic when state changes.

        Args:
            episode_id: Episode identifier
            new_state: New transcript state
        """
        # Only trigger on WAITING_FOR_USER state
        if new_state != TranscriptState.WAITING_FOR_USER:
            return

        # Get episode
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            logger.warning("Episode not found for auto-continue", extra={"episode_id": episode_id})
            return

        # Check if auto-continue is enabled
        if not episode.context.get(MetadataKeys.AUTO_CONTINUE_ENABLED, False):
            logger.debug("Auto-continue disabled", extra={"episode_id": episode_id})
            return

        # Check idempotency - prevent duplicate injections for same version
        current_version = episode.context.get(MetadataKeys.TRANSCRIPT_VERSION, 0)
        last_injected_version = episode.context.get(MetadataKeys.LAST_AUTO_CONTINUE_VERSION)

        if last_injected_version == current_version:
            logger.debug(
                "Auto-continue already injected for this version",
                extra={"episode_id": episode_id, "version": current_version},
            )
            return

        # Get continue prompt (custom or default)
        continue_prompt = episode.context.get(MetadataKeys.CONTINUE_PROMPT, self.DEFAULT_CONTINUE_PROMPT)

        # Inject continue message
        transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
        modified_transcript = transcript + [{"role": "user", "content": continue_prompt}]

        logger.info(
            "Injecting auto-continue message",
            extra={"episode_id": episode_id, "version": current_version, "prompt": continue_prompt},
        )

        # Notify coordinator of modification
        await self.transcript_coordinator.notify_modification(
            episode_id=episode_id,
            modified_transcript=modified_transcript,
            operation="auto_continue",
            injected_by="system",
        )


__all__ = ["AutoContinueManager"]
