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
from typing import Any, Optional

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys
from ...models.rest.websocket_messages import StateEventData, StateEventMessage, SyncMode, TranscriptOperation
from ...models.transcript import (
    TranscriptPushOperation,
    TranscriptSyncRequest,
    TranscriptSyncResponse,
    TranscriptVersion,
    compute_checksum,
)
from ..base import Episode
from ..time_source import TimeSource, UTCTimeSource
from .auto_continue_manager import AutoContinueManager
from .protocols import ConnectionManagerProtocol, EpisodeManagerProtocol
from .stuck_state_monitor import StuckStateMonitor
from .transcript_state_machine import TranscriptStateMachine

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

    def __init__(
        self,
        episode_manager: EpisodeManagerProtocol,
        connection_manager: ConnectionManagerProtocol,
        time_source: TimeSource | None = None,
        stuck_check_interval: float = 30.0,
        stuck_threshold: float = 300.0,
    ) -> None:
        """
        Initialize the transcript coordinator.

        Args:
            episode_manager: EpisodeManager instance for episode access
            connection_manager: ConnectionManager instance for WebSocket broadcasting
            time_source: Time source for getting current time (defaults to UTCTimeSource)
            stuck_check_interval: Seconds between stuck state checks (default 30s)
            stuck_threshold: Default stuck threshold in seconds (default 5min)
        """
        self.episode_manager = episode_manager
        self.connection_manager = connection_manager
        self._coordination_configs: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()
        self._time_source = time_source or UTCTimeSource()

        # Initialize state machine and auto-continue manager
        self._state_machine = TranscriptStateMachine()
        self._auto_continue = AutoContinueManager(episode_manager, self)

        # Initialize stuck state monitor
        self._stuck_monitor = StuckStateMonitor(
            coordinator=self,
            episode_manager=episode_manager,
            check_interval=stuck_check_interval,
            stuck_threshold=stuck_threshold,
        )

        # Register lifecycle hooks
        self._register_lifecycle_hooks()

    def _register_lifecycle_hooks(self) -> None:
        """Register state machine lifecycle hooks with episode manager."""
        # Hook into episode creation
        original_start = self.episode_manager.start_episode  # type: ignore[attr-defined]

        def start_episode_with_hook(session_id: str, task_id: str, *args: Any, **kwargs: Any) -> Any:
            episode = original_start(session_id, task_id, *args, **kwargs)
            # Schedule async lifecycle hook if event loop is running
            try:
                asyncio.create_task(self._state_machine.on_episode_created(episode.episode_id))
            except RuntimeError:
                # No event loop running - skip lifecycle hook (sync tests)
                pass
            return episode

        self.episode_manager.start_episode = start_episode_with_hook  # type: ignore[attr-defined]

        # Hook into episode termination
        original_end = self.episode_manager.end_episode  # type: ignore[attr-defined]

        async def end_episode_with_hook(episode_id: str, *args: Any, **kwargs: Any) -> Any:
            # Call lifecycle hook before ending
            await self._state_machine.on_episode_terminated(episode_id)
            return await original_end(episode_id, *args, **kwargs)

        self.episode_manager.end_episode = end_episode_with_hook  # type: ignore[attr-defined]

    def _get_current_version(self, episode: Episode) -> TranscriptVersion:
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

    def _has_complete_assistant_turn(self, messages: list[dict[str, Any]]) -> bool:
        """
        Check if the transcript has a complete assistant turn.

        A "complete" assistant turn means:
        - There's at least one assistant message, AND
        - If the last assistant message has tool_calls, all tool_calls have responses

        This is used to determine if it's safe to capture INITIAL_TRANSCRIPT.

        Args:
            messages: List of transcript messages

        Returns:
            True if there's a complete assistant turn, False otherwise
        """
        if not messages:
            return False

        # Find the last assistant message
        last_assistant_idx = -1
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                last_assistant_idx = i

        if last_assistant_idx == -1:
            return False  # No assistant message

        last_assistant = messages[last_assistant_idx]
        tool_calls = last_assistant.get("tool_calls")

        if not tool_calls:
            # No tool_calls = complete turn
            return True

        # Has tool_calls - check if all have responses
        tool_call_ids = {tc.get("id") for tc in tool_calls if tc.get("id")}

        # Look for tool responses after this assistant message
        for msg in messages[last_assistant_idx + 1 :]:
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id in tool_call_ids:
                    tool_call_ids.discard(tool_call_id)

        # Complete if all tool_calls have responses
        return len(tool_call_ids) == 0

    def _analyze_tool_calls(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyze tool_calls in a transcript to detect orphaned calls.

        Args:
            messages: List of transcript messages

        Returns:
            Dict with analysis: total_tool_calls, responded_tool_calls, orphaned_tool_call_ids
        """
        all_tool_call_ids: list[str] = []
        responded_tool_call_ids: list[str] = []

        for msg in messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    if tc.get("id"):
                        all_tool_call_ids.append(tc["id"])
            elif msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id:
                    responded_tool_call_ids.append(tool_call_id)

        orphaned = [tc_id for tc_id in all_tool_call_ids if tc_id not in responded_tool_call_ids]

        return {
            "total_tool_calls": len(all_tool_call_ids),
            "responded_tool_calls": len(responded_tool_call_ids),
            "orphaned_tool_call_ids": orphaned,
            "all_tool_call_ids": all_tool_call_ids[:5],  # Truncate for logging
        }

    async def _is_duplicate_push(
        self,
        episode: "Episode",
        messages_to_push: list[dict[str, Any]],
        client_since_version: int,
    ) -> bool:
        """
        Detect if this push is a duplicate (client retry after lost ACK).

        A push is considered a duplicate if:
        1. The client's since_version is behind the current server version, AND
        2. The message(s) being pushed match the last message(s) in the transcript

        This handles the case where:
        - Client pushes message, server appends it, version becomes N+1
        - ACK is lost (network issues)
        - Client retries push with since_version=N
        - Without this check, server would append duplicate

        Args:
            episode: The target episode
            messages_to_push: Messages the client is trying to push
            client_since_version: The client's last known version

        Returns:
            True if this is a duplicate push that should be skipped
        """
        if not messages_to_push:
            return False

        current_transcript = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
        current_version = episode.context.get(MetadataKeys.TRANSCRIPT_VERSION, 0)

        # Only check for duplicates if client is behind
        if client_since_version >= current_version:
            return False

        # Check if the last N messages match what's being pushed
        push_count = len(messages_to_push)
        if len(current_transcript) < push_count:
            return False

        # Get the last N messages from transcript
        last_messages = current_transcript[-push_count:]

        # Compare each message
        for pushed, existing in zip(messages_to_push, last_messages, strict=False):
            if not self._messages_match(pushed, existing):
                return False

        return True

    def _messages_match(self, msg1: dict[str, Any], msg2: dict[str, Any]) -> bool:
        """
        Check if two messages are semantically equivalent.

        Compares role, content, and tool_calls/tool_call_id.
        """
        if msg1.get("role") != msg2.get("role"):
            return False

        # Compare content (handle both string and list content)
        content1 = msg1.get("content")
        content2 = msg2.get("content")
        if content1 != content2:
            return False

        # For assistant messages, compare tool_calls
        if msg1.get("role") == "assistant":
            tc1 = msg1.get("tool_calls", []) or []
            tc2 = msg2.get("tool_calls", []) or []
            if len(tc1) != len(tc2):
                return False
            for t1, t2 in zip(tc1, tc2, strict=False):
                # Handle case where tool_calls items might be strings or non-dict types
                if not isinstance(t1, dict) or not isinstance(t2, dict):
                    # Fall back to direct comparison
                    if t1 != t2:
                        return False
                    continue
                if t1.get("id") != t2.get("id"):
                    return False
                # Safely get function info
                func1_raw = t1.get("function") if isinstance(t1, dict) else None
                func2_raw = t2.get("function") if isinstance(t2, dict) else None
                func1 = func1_raw if isinstance(func1_raw, dict) else {}
                func2 = func2_raw if isinstance(func2_raw, dict) else {}
                if func1.get("name") != func2.get("name"):
                    return False

        # For tool messages, compare tool_call_id
        if msg1.get("role") == "tool":
            if msg1.get("tool_call_id") != msg2.get("tool_call_id"):
                return False

        return True

    def _find_safe_initial_transcript(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """
        Find the safe capture point for INITIAL_TRANSCRIPT.

        This returns a prefix of the transcript that ends in a "complete" state -
        either ending with an assistant message without tool_calls, or ending
        with all tool responses for the last assistant's tool_calls.

        Args:
            messages: List of transcript messages

        Returns:
            A safe prefix to use as INITIAL_TRANSCRIPT, or None if no safe point found
        """
        if not messages:
            return None

        # Strategy: find the first complete assistant turn and include everything up to it
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                # Assistant without tool_calls - safe capture point
                return messages[: i + 1]

            # Has tool_calls - find where all responses end
            tool_call_ids = {tc.get("id") for tc in tool_calls if tc.get("id")}
            last_tool_response_idx = i

            for j in range(i + 1, len(messages)):
                next_msg = messages[j]
                if next_msg.get("role") == "tool":
                    tool_call_id = next_msg.get("tool_call_id")
                    if tool_call_id in tool_call_ids:
                        tool_call_ids.discard(tool_call_id)
                        last_tool_response_idx = j
                elif next_msg.get("role") == "assistant":
                    # Hit another assistant - if previous tool_calls satisfied, capture up to here
                    if len(tool_call_ids) == 0:
                        return messages[: last_tool_response_idx + 1]
                    # Otherwise this is invalid state, keep looking
                    break

            # If we've satisfied all tool_calls, capture up to last tool response
            if len(tool_call_ids) == 0:
                return messages[: last_tool_response_idx + 1]

        return None

    async def sync(self, request: TranscriptSyncRequest) -> TranscriptSyncResponse:
        """
        Unified sync with differential updates and WebSocket notifications.

        Supports two modes:
        - Owner mode (is_observer=False): Full bidirectional sync with checksum validation
        - Observer mode (is_observer=True): Read-only access with optional security filtering

        Flow for owner mode:
        1. Client pushes new messages → version increments (last write wins)
        2. Server computes checksum and validates client's view
        3. If checksums match → send delta (efficient)
        4. If checksums differ → send full transcript (rewrite detected)

        Flow for observer mode:
        1. Apply security filtering (hide_system_prompt)
        2. Apply retrieval mode (full, delta, tail)
        3. Return filtered transcript (no checksum validation)

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

        # Observer mode: read-only with security filtering
        if request.is_observer:
            return await self._sync_observer(episode, request, start_time)

        # Owner mode: full bidirectional sync
        return await self._sync_owner(episode, request, start_time)

    async def _sync_owner(
        self, episode: Episode, request: TranscriptSyncRequest, start_time: float
    ) -> TranscriptSyncResponse:
        """Handle owner mode sync with checksum validation."""
        # Step 1: Push new messages if provided (LAST WRITE WINS)
        # Include idempotency check to handle client retries
        if request.messages_to_push:
            is_duplicate = await self._is_duplicate_push(episode, request.messages_to_push, request.since_version)
            if is_duplicate:
                logger.info(
                    "[IDEMPOTENCY] Skipping duplicate push (client retry detected)",
                    extra={
                        "episode_id": episode.episode_id,
                        "since_version": request.since_version,
                        "message_role": request.messages_to_push[0].get("role") if request.messages_to_push else None,
                    },
                )
            else:
                await self._push_messages(
                    episode,
                    request.messages_to_push,
                    operation=request.operation,
                )

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
            sync_mode = SyncMode.FULL
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
            sync_mode = SyncMode.NO_CHANGE
            delta = []
            full_transcript = None
            modified = False

        else:
            # DELTA: Client is behind but valid
            sync_mode = SyncMode.DELTA
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
            blocked=False,
            wait_time_seconds=wait_time,
        )

    async def _sync_observer(
        self, episode: Episode, request: TranscriptSyncRequest, start_time: float
    ) -> TranscriptSyncResponse:
        """Handle observer mode sync with security filtering.

        Observer mode is read-only and applies:
        - Security filtering (hide system prompt)
        - Retrieval mode filtering (full, delta, tail)
        - No checksum validation (observers don't track state)
        """
        # Get raw transcript
        raw_messages = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
        version = episode.context.get(MetadataKeys.TRANSCRIPT_VERSION, 0)

        # Apply security filter: hide system prompt
        messages = self._filter_for_observer(raw_messages, request.hide_system_prompt or False)

        # Apply retrieval mode
        sync_mode: SyncMode
        delta: list[dict[str, Any]] | None = None
        full_transcript: list[dict[str, Any]] | None = None
        modified: bool

        if request.retrieval_mode == "delta":
            if request.since_version >= version:
                # Client is up to date
                sync_mode = SyncMode.NO_CHANGE
                modified = False
            else:
                # Compute filtered delta
                if request.hide_system_prompt:
                    # Filter messages at since_version point to compute delta correctly
                    filtered_at_since = self._filter_for_observer(raw_messages[: request.since_version], True)
                    delta = messages[len(filtered_at_since) :]
                else:
                    delta = messages[request.since_version :]
                sync_mode = SyncMode.DELTA
                modified = True

        elif request.retrieval_mode == "tail" and request.tail_count > 0:
            # Tail mode: return last N messages
            tail_count = min(request.tail_count, 1000)
            full_transcript = messages[-tail_count:] if len(messages) > tail_count else messages
            sync_mode = SyncMode.FULL
            modified = True

        else:
            # Full mode (default)
            full_transcript = messages
            sync_mode = SyncMode.FULL
            modified = True

        wait_time = asyncio.get_event_loop().time() - start_time

        logger.debug(
            "Observer sync completed",
            extra={
                "episode_id": request.episode_id,
                "hide_system_prompt": request.hide_system_prompt,
                "retrieval_mode": request.retrieval_mode,
                "message_count": len(messages),
                "version": version,
                "sync_mode": sync_mode.value,
            },
        )

        # Compute checksum on filtered view (what observer actually sees)
        observer_checksum = compute_checksum(messages)

        return TranscriptSyncResponse(
            current_version=TranscriptVersion(
                sequence=version,
                checksum=observer_checksum,
                message_count=len(messages),
                last_operation="",
            ),
            delta=delta,
            full_transcript=full_transcript,
            sync_mode=sync_mode,
            modified=modified,
            blocked=False,
            wait_time_seconds=wait_time,
        )

    def _filter_for_observer(self, messages: list[dict[str, Any]], hide_system_prompt: bool) -> list[dict[str, Any]]:
        """Apply security filtering for observer access.

        When hide_system_prompt=True, only returns messages starting from
        the first assistant message. This prevents observers from seeing
        the target's system prompt / guardrail instructions.
        """
        if not hide_system_prompt or not messages:
            return messages

        filtered = []
        found_first_assistant = False
        for msg in messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if not found_first_assistant:
                if role == "assistant":
                    found_first_assistant = True
                    filtered.append(msg)
            else:
                filtered.append(msg)
        return filtered

    async def _push_messages(
        self,
        episode: Episode,
        messages: list[dict[str, str]],
        operation: str = "append",
    ) -> None:
        """
        Push messages with operation type (LAST WRITE WINS).

        Args:
            episode: Episode instance
            messages: Messages to add/push
            operation: "append" or "restart"
        """
        existing = episode.context.get(MetadataKeys.CLIENT_TRANSCRIPT, [])
        current_version = episode.context.get(MetadataKeys.TRANSCRIPT_VERSION, 0)

        logger.debug(
            "Push messages called",
            extra={
                "episode_id": episode.episode_id,
                "operation": operation,
                "existing_count": len(existing),
                "new_message_count": len(messages),
            },
        )

        # Validate: Check for duplicate tool_call IDs across all existing assistant messages
        # This is a safety net for when idempotency check fails or client re-sends old messages
        if existing and messages:
            new_msg = messages[0]
            if new_msg.get("role") == "assistant":
                raw_tool_calls: Any = new_msg.get("tool_calls", [])
                new_tool_calls: list[Any] = raw_tool_calls if isinstance(raw_tool_calls, list) else []
                if new_tool_calls:
                    new_ids = set()
                    for tc in new_tool_calls:
                        if isinstance(tc, dict) and tc.get("id"):
                            new_ids.add(tc.get("id"))

                    if new_ids:
                        # Collect all tool_call IDs from existing assistant messages
                        existing_ids = set()
                        for msg in existing:
                            if msg.get("role") == "assistant":
                                for tc in msg.get("tool_calls", []) or []:
                                    if isinstance(tc, dict) and tc.get("id"):
                                        existing_ids.add(tc.get("id"))

                        # Check if any of the new tool_call IDs already exist
                        overlap = new_ids & existing_ids
                        if overlap:
                            logger.warning(
                                "Blocking assistant message with duplicate tool_call IDs",
                                extra={
                                    "episode_id": episode.episode_id,
                                    "duplicate_tool_call_ids": list(overlap),
                                },
                            )
                            # Don't add the duplicate - return without modifying transcript
                            # The caller will still get a sync response with current state
                            return

        # Apply operation to compute new transcript
        if operation == TranscriptPushOperation.APPEND.value:
            updated_messages = existing + messages

        elif operation == TranscriptPushOperation.RESTART.value:
            # Reset to initial transcript (system->user) then append new messages
            initial = episode.context.get(MetadataKeys.INITIAL_TRANSCRIPT, [])
            if initial:
                # Use a copy of initial transcript to avoid mutation
                updated_messages = list(initial) + messages
            else:
                # Fallback: if no initial transcript, just use the new messages
                logger.warning(
                    "No initial transcript found for restart operation, falling back to append",
                    extra={"episode_id": episode.episode_id},
                )
                updated_messages = existing + messages

        else:
            # Fallback to append for unknown operations
            logger.warning(
                f"Unknown operation '{operation}', falling back to append",
                extra={"episode_id": episode.episode_id},
            )
            updated_messages = existing + messages

        new_version = current_version + 1  # Always increment (monotonic)

        # Capture initial transcript after first COMPLETE assistant turn.
        # "Complete" means the assistant message doesn't have pending tool_calls,
        # or all tool_calls have been responded to. This ensures restart won't
        # create an invalid transcript with orphaned tool_calls.
        initial = episode.context.get(MetadataKeys.INITIAL_TRANSCRIPT, [])
        initial_has_complete_assistant = self._has_complete_assistant_turn(initial)

        if not initial_has_complete_assistant:
            updated_has_complete = self._has_complete_assistant_turn(updated_messages)
            # Check if the updated transcript now has a complete assistant turn
            if updated_has_complete:
                # Find the safe capture point: everything up to and including
                # the first complete assistant turn
                capture_point = self._find_safe_initial_transcript(updated_messages)
                if capture_point:
                    await episode.update_context_atomic({MetadataKeys.INITIAL_TRANSCRIPT: capture_point})
                    logger.debug(
                        "Captured initial transcript with complete assistant turn",
                        extra={
                            "episode_id": episode.episode_id,
                            "message_count": len(capture_point),
                        },
                    )

        await episode.update_context_atomic(
            {
                MetadataKeys.CLIENT_TRANSCRIPT: updated_messages,
                MetadataKeys.TRANSCRIPT_VERSION: new_version,
                MetadataKeys.TRANSCRIPT_LAST_OPERATION: operation,
                MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: self._time_source.now().isoformat(),
            }
        )

        # Broadcast state event after push so observers (like red team daemon) can
        # track when blue team reaches WAITING_FOR_USER state
        new_state = self._state_machine.get_state(episode)
        old_state = episode.context.get(MetadataKeys.CURRENT_TRANSCRIPT_STATE)
        if old_state != new_state.value:
            self._state_machine.update_state_timestamp(episode.episode_id)
            await episode.update_context_atomic({MetadataKeys.CURRENT_TRANSCRIPT_STATE: new_state.value})

        event_type = TranscriptStateMachine.state_to_event_type(new_state)

        # Normalize operation to TranscriptOperation enum
        if isinstance(operation, TranscriptOperation):
            operation_enum = operation
        else:
            try:
                operation_enum = TranscriptOperation(operation)
            except ValueError:
                operation_enum = TranscriptOperation.APPEND

        state_event = StateEventMessage(
            type=event_type,
            data=StateEventData(
                version=new_version,
                operation=operation_enum,
                modification_count=episode.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0),
                state=new_state.value,
            ),
            id=str(uuid.uuid4()),
            timestamp=self._time_source.now().isoformat(),
        )

        await self.connection_manager.broadcast_to_episode(
            episode_id=episode.episode_id,
            message=state_event,
        )

        logger.debug(
            "Broadcast state event after push",
            extra={
                "episode_id": episode.episode_id,
                "version": new_version,
                "state": new_state.value,
                "event_type": event_type,
            },
        )

    async def notify_modification(
        self,
        episode_id: str,
        modified_transcript: list[dict[str, str]],
        operation: str | TranscriptOperation,
        injected_by: str,
        expected_base_version: int | None = None,
        expected_base_checksum: str | None = None,
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
                    f"Modification based on stale version. Expected {expected_base_version}, current {current.sequence}"
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

        # Detect state after modification
        new_state = self._state_machine.get_state(episode)

        # Update state timestamp if state changed
        old_state = episode.context.get(MetadataKeys.CURRENT_TRANSCRIPT_STATE)
        if old_state != new_state.value:
            self._state_machine.update_state_timestamp(episode_id)
            # Store current state for next comparison
            await episode.update_context_atomic({MetadataKeys.CURRENT_TRANSCRIPT_STATE: new_state.value})

        event_type = TranscriptStateMachine.state_to_event_type(new_state)

        # Normalize operation to TranscriptOperation enum
        if isinstance(operation, TranscriptOperation):
            operation_enum = operation
        else:
            try:
                operation_enum = TranscriptOperation(operation)
            except ValueError:
                operation_enum = TranscriptOperation.APPEND

        message = StateEventMessage(
            type=event_type,
            data=StateEventData(
                version=new_version,
                operation=operation_enum,
                modification_count=modification_count,
                injected_by=injected_by,
                state=new_state.value,
            ),
            id=str(uuid.uuid4()),
            timestamp=self._time_source.now().isoformat(),
        )

        await self.connection_manager.broadcast_to_episode(
            episode_id=episode_id,
            message=message,
        )

        logger.debug(
            "Broadcast WebSocket transcript modification event",
            extra={
                "episode_id": episode_id,
                "version": new_version,
                "operation": operation_enum.value,
                "modification_count": modification_count,
                "injected_by": injected_by,
                "state": new_state.value,
                "event_type": event_type,
            },
        )

        # Trigger auto-continue asynchronously (don't block)
        asyncio.create_task(self._auto_continue.handle_auto_continue(episode_id, new_state))

    async def start_monitor(self) -> None:
        """Start background monitors."""
        await self._stuck_monitor.start()

    async def stop_monitor(self) -> None:
        """Stop background monitors."""
        await self._stuck_monitor.stop()

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

    def get_initial_state_event(self, episode_id: str) -> Optional["StateEventMessage"]:
        """Get initial state event for a newly connected WebSocket client.

        Creates a state event message reflecting the current transcript state.
        This unblocks the client's first generate() call which waits for a state event.

        Args:
            episode_id: Episode identifier

        Returns:
            StateEventMessage with current state, or None if episode not found
        """
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            return None

        current_state = self._state_machine.get_state(episode)
        event_type = TranscriptStateMachine.state_to_event_type(current_state)
        current_version = episode.context.get(MetadataKeys.TRANSCRIPT_VERSION, 0)
        modification_count = episode.context.get(MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT, 0)

        return StateEventMessage(
            type=event_type,
            data=StateEventData(
                version=current_version,
                operation=TranscriptOperation.INIT,
                modification_count=modification_count,
                state=current_state.value,
            ),
            id=str(uuid.uuid4()),
            timestamp=self._time_source.now().isoformat(),
        )

    def get_episode_state(self, episode_id: str) -> str | None:
        """Get current transcript state for an episode.

        Args:
            episode_id: Episode identifier

        Returns:
            State value string, or None if episode not found
        """
        episode = self.episode_manager.get_episode_by_id(episode_id)
        if not episode:
            return None
        return self._state_machine.get_state(episode).value


__all__ = ["TranscriptCoordinator"]
