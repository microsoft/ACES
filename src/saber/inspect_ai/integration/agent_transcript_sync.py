"""Manual transcript sync for agents that bypass model.generate().

This module provides a helper class for agents that use custom inference
mechanisms (like the Copilot SDK) instead of Inspect AI's model.generate().

The standard WebSocketTranscriptSyncingModelWrapper wraps model.generate()
to automatically sync transcripts. For agents that bypass this (e.g., using
external SDKs), this module provides manual sync capabilities.

Usage:
    from saber.inspect_ai.integration.agent_transcript_sync import AgentTranscriptSync

    # In your solver:
    transcript_sync = AgentTranscriptSync(state)
    await transcript_sync.initialize()

    # After each turn:
    await transcript_sync.sync_state_messages(state)

Logging category: AGENT.
"""

from typing import TYPE_CHECKING, Any

from inspect_ai.model import ChatMessage
from inspect_ai.solver import TaskState

from ...logging_config import LogCategory, get_saber_logger
from ..constants import InspectStoreKeys

if TYPE_CHECKING:
    pass

logger = get_saber_logger(LogCategory.AGENT, __name__)


class AgentTranscriptSync:
    """Helper class to sync agent transcripts to SABER server.

    This class provides manual transcript synchronization for agents that
    bypass the normal model.generate() flow. It uses the
    WebSocketTranscriptSyncingModelWrapper created by solver_factory.

    The wrapper is retrieved from state.store where it's placed by
    solver_factory._wrap_model_for_transcript_sync().

    Example:
        ```python
        async def solve(state: TaskState) -> TaskState:
            # Initialize transcript sync
            transcript_sync = AgentTranscriptSync(state)
            await transcript_sync.initialize()

            # Your agent loop
            while not done:
                # ... agent logic that adds messages to state.messages ...

                # Sync after each turn
                await transcript_sync.sync_state_messages(state)

            return state
        ```

    Attributes:
        is_enabled: Whether transcript sync is active
    """

    def __init__(self, state: TaskState) -> None:
        """Initialize transcript sync helper.

        Args:
            state: TaskState containing the model wrapper in store
        """
        self._state = state
        self._wrapper: Any = None
        self._last_synced_count = 0
        self._enabled = False

    async def initialize(self) -> bool:
        """Initialize the transcript sync connection.

        Retrieves the WebSocketTranscriptSyncingModelWrapper from state.store
        and establishes the WebSocket connection.

        Returns:
            True if sync is enabled and initialized, False otherwise
        """
        # Import here to avoid circular imports
        from .model_wrapper import WebSocketTranscriptSyncingModelWrapper

        # Get the WebSocket wrapper from state store (created by solver_factory)
        self._wrapper = self._state.store.get(InspectStoreKeys.MODEL_WRAPPER)

        if self._wrapper is None:
            logger.warning(
                "No WebSocket model wrapper found in state.store - transcript sync disabled",
                extra={
                    "store_keys": (
                        list(self._state.store._store.keys()) if hasattr(self._state.store, "_store") else []
                    )
                },
            )
            return False

        if not isinstance(self._wrapper, WebSocketTranscriptSyncingModelWrapper):
            logger.warning(
                f"Model wrapper is not WebSocketTranscriptSyncingModelWrapper: {type(self._wrapper).__name__}",
            )
            return False

        try:
            # Ensure WebSocket connection is established
            await self._wrapper._client.ensure_connected()
            self._enabled = True
            logger.info(
                "Agent transcript sync initialized",
                extra={"episode_id": self._wrapper._episode_id},
            )
            return True
        except Exception as e:
            logger.warning(
                f"Failed to initialize transcript sync: {e}",
                extra={"error_type": type(e).__name__},
            )
            return False

    async def push_messages(self, messages: list[ChatMessage]) -> bool:
        """Push new messages to the SABER server.

        Uses the wrapper's TranscriptSyncClient.push_message() which handles
        connection management, retry logic, and reconnection internally.

        Args:
            messages: List of ChatMessage objects to push

        Returns:
            True if all messages were pushed successfully, False otherwise
        """
        if not self._enabled or not self._wrapper:
            return False

        if not messages:
            return True

        success = True
        for msg in messages:
            try:
                pushed = await self._wrapper._client.push_message(
                    message=msg,
                    context="agent_manual_push",
                )
                if not pushed:
                    success = False
                    logger.warning(
                        f"Failed to push message with role={msg.role}",
                        extra={"episode_id": self._wrapper._episode_id},
                    )
            except Exception as e:
                logger.warning(
                    f"Error pushing message: {e}",
                    extra={"role": msg.role, "error_type": type(e).__name__},
                )
                success = False

        if success:
            logger.debug(
                f"Pushed {len(messages)} messages to SABER server",
                extra={
                    "episode_id": self._wrapper._episode_id,
                    "message_count": len(messages),
                    "local_message_count": len(self._wrapper._client.local_messages),
                },
            )

        return success

    async def sync_state_messages(self, state: TaskState) -> bool:
        """Sync all new messages from state since last sync.

        This method tracks which messages have been synced and only
        pushes new ones (differential sync).

        Args:
            state: TaskState with current messages

        Returns:
            True if sync succeeded, False otherwise
        """
        if not self._enabled:
            return False

        current_count = len(state.messages)
        if current_count <= self._last_synced_count:
            return True  # Nothing new to sync

        new_messages = state.messages[self._last_synced_count :]
        success = await self.push_messages(new_messages)

        if success:
            self._last_synced_count = current_count

        return success

    def reset_sync_state(self) -> None:
        """Reset the sync state to re-sync all messages.

        Call this if you need to re-push the entire transcript.
        """
        self._last_synced_count = 0

    @property
    def is_enabled(self) -> bool:
        """Check if transcript sync is enabled."""
        return self._enabled

    @property
    def synced_message_count(self) -> int:
        """Get the number of messages that have been synced."""
        return self._last_synced_count


__all__ = ["AgentTranscriptSync"]
