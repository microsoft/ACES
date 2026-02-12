"""Transcript state machine for coordinating agent execution."""

import asyncio
import time
from enum import Enum
from typing import Any, Literal

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from ..base import Episode

logger = get_saber_logger(LogCategory.EPISODE, __name__)

# Type alias for valid state event types that can be used in StateEventMessage
StateEventType = Literal[
    "transcript_modified",
    "is_waiting_on_user",
    "is_waiting_on_assistant",
    "is_waiting_on_tools",
]


class TranscriptState(str, Enum):
    """Transcript states for agent coordination."""

    WAITING_FOR_USER = "WAITING_FOR_USER"
    WAITING_FOR_ASSISTANT = "WAITING_FOR_ASSISTANT"
    WAITING_FOR_TOOLS = "WAITING_FOR_TOOLS"
    ERROR_STUCK = "ERROR_STUCK"
    ERROR_MALFORMED = "ERROR_MALFORMED"


class TranscriptStateMachine:
    """State machine for transcript coordination.

    Manages the three-state lifecycle of agent-server transcript synchronization:
    - WAITING_FOR_USER: Waiting for user input
    - WAITING_FOR_ASSISTANT: Waiting for assistant response
    - WAITING_FOR_TOOLS: Waiting for tool execution results

    Also detects error states:
    - ERROR_MALFORMED: Invalid transcript structure
    - ERROR_STUCK: Transcript stuck in same state (detected by monitor)
    """

    # Valid state transitions
    VALID_TRANSITIONS: dict[TranscriptState, set[TranscriptState]] = {
        TranscriptState.WAITING_FOR_USER: {TranscriptState.WAITING_FOR_ASSISTANT},
        TranscriptState.WAITING_FOR_ASSISTANT: {TranscriptState.WAITING_FOR_TOOLS, TranscriptState.WAITING_FOR_USER},
        TranscriptState.WAITING_FOR_TOOLS: {TranscriptState.WAITING_FOR_USER},
    }

    def __init__(self) -> None:
        """Initialize state machine with lifecycle management."""
        self._episode_locks: dict[str, asyncio.Lock] = {}
        self._state_timestamps: dict[str, float] = {}

    async def on_episode_created(self, episode_id: str) -> None:
        """Lifecycle hook called when episode is created.

        Initializes episode-level resources.

        Args:
            episode_id: Episode identifier
        """
        self._episode_locks[episode_id] = asyncio.Lock()
        self._state_timestamps[episode_id] = time.time()

        logger.debug("Episode lifecycle initialized", extra={"episode_id": episode_id})

    async def on_episode_terminated(self, episode_id: str) -> None:
        """Lifecycle hook called when episode terminates.

        Cleans up episode-level resources to prevent memory leaks.

        Args:
            episode_id: Episode identifier
        """
        self._episode_locks.pop(episode_id, None)
        self._state_timestamps.pop(episode_id, None)

        logger.debug("Episode lifecycle cleaned up", extra={"episode_id": episode_id})

    def update_state_timestamp(self, episode_id: str) -> None:
        """Update timestamp when state changes.

        Should be called after state transitions.

        Args:
            episode_id: Episode identifier
        """
        self._state_timestamps[episode_id] = time.time()

    def is_episode_stuck(self, episode_id: str, threshold_seconds: float = 300.0) -> bool:
        """Check if episode has been in same state too long.

        Args:
            episode_id: Episode to check
            threshold_seconds: Max time in same state (default 5 minutes)

        Returns:
            True if stuck, False otherwise
        """
        if episode_id not in self._state_timestamps:
            return False

        elapsed = time.time() - self._state_timestamps[episode_id]
        return elapsed > threshold_seconds

    def get_state_duration(self, episode_id: str) -> float:
        """Get time spent in current state.

        Args:
            episode_id: Episode identifier

        Returns:
            Seconds in current state, or 0.0 if not tracked
        """
        if episode_id not in self._state_timestamps:
            return 0.0

        return time.time() - self._state_timestamps[episode_id]

    def get_state(self, episode: Episode) -> TranscriptState:
        """Detect current state from episode transcript.

        Args:
            episode: Episode to analyze

        Returns:
            Current transcript state
        """
        transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])

        # Validate transcript structure
        if transcript and not self._is_valid_transcript(transcript):
            logger.error("Malformed transcript detected", extra={"episode_id": episode.episode_id})
            return TranscriptState.ERROR_MALFORMED

        return self._compute_state(transcript)

    def _compute_state(self, transcript: list[dict[str, Any]]) -> TranscriptState:
        """Compute state from transcript messages.

        Args:
            transcript: List of transcript messages

        Returns:
            Computed transcript state
        """
        if not transcript:
            return TranscriptState.WAITING_FOR_USER

        # Find the last assistant message with tool_calls and check if all are responded to
        # This is more accurate than just looking at the last message
        last_assistant_with_tools_idx = -1
        pending_tool_call_ids: set[str] = set()

        for i, msg in enumerate(transcript):
            role = msg.get("role")
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    last_assistant_with_tools_idx = i
                    # Reset pending set to this assistant's tool_calls
                    pending_tool_call_ids = {tc["id"] for tc in tool_calls if isinstance(tc, dict) and tc.get("id")}
            elif role == "tool" and last_assistant_with_tools_idx >= 0:
                # Tool response - remove from pending
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id:
                    pending_tool_call_ids.discard(tool_call_id)

        # If there are still pending tool_calls, we're waiting for tools
        if pending_tool_call_ids:
            return TranscriptState.WAITING_FOR_TOOLS

        # No pending tool_calls - check the last non-system message
        for msg in reversed(transcript):
            role = msg.get("role")

            if role == "system":
                continue  # Skip system messages

            if role == "user":
                return TranscriptState.WAITING_FOR_ASSISTANT

            if role == "assistant":
                # No pending tool_calls (checked above), so waiting for user
                return TranscriptState.WAITING_FOR_USER

            if role == "tool":
                # All tool_calls responded to (checked above), so waiting for user
                return TranscriptState.WAITING_FOR_USER

            # Unknown role - stop processing
            break

        return TranscriptState.WAITING_FOR_USER

    def _is_valid_transcript(self, transcript: list[dict[str, Any]]) -> bool:
        """Validate transcript message structure.

        Args:
            transcript: List of transcript messages

        Returns:
            True if transcript is valid, False otherwise
        """
        for msg in transcript:
            # Check basic structure
            if not isinstance(msg, dict) or "role" not in msg:
                return False

            # Validate role
            role = msg.get("role")
            if role not in ("user", "assistant", "tool", "system"):
                return False

            # Validate tool_calls if present and non-None
            # Note: Pydantic's model_dump() includes tool_calls=None for assistant
            # messages without tool calls - this is valid and should not be rejected.
            if role == "assistant" and "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                if tool_calls is not None and not isinstance(tool_calls, list):
                    return False

        return True

    def is_valid_transition(self, old_state: TranscriptState, new_state: TranscriptState) -> bool:
        """Check if state transition is valid.

        Args:
            old_state: Current state
            new_state: Target state

        Returns:
            True if transition is valid, False otherwise
        """
        # Same state is always valid (no-op)
        if old_state == new_state:
            return True

        # Check transition table
        valid_targets = self.VALID_TRANSITIONS.get(old_state, set())
        return new_state in valid_targets

    @staticmethod
    def state_to_event_type(state: TranscriptState) -> StateEventType:
        """Convert state to WebSocket event type.

        Uses is_waiting_on_<role> format for events.
        Error states return "transcript_modified" as a safe default since
        TranscriptErrorMessage should be used for actual error notifications.

        Args:
            state: TranscriptState

        Returns:
            Event type literal (e.g., "is_waiting_on_assistant")
        """
        event_map: dict[TranscriptState, StateEventType] = {
            TranscriptState.WAITING_FOR_USER: "is_waiting_on_user",
            TranscriptState.WAITING_FOR_ASSISTANT: "is_waiting_on_assistant",
            TranscriptState.WAITING_FOR_TOOLS: "is_waiting_on_tools",
            # Error states use transcript_modified as default
            # Actual error notifications should use TranscriptErrorMessage
            TranscriptState.ERROR_STUCK: "transcript_modified",
            TranscriptState.ERROR_MALFORMED: "transcript_modified",
        }
        return event_map.get(state, "transcript_modified")


__all__ = [
    "StateEventType",
    "TranscriptState",
    "TranscriptStateMachine",
]
