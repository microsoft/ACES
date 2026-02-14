"""Unit tests for TranscriptCoordinator with mocked dependencies.

Tests cover:
- push_message: stores in DB and broadcasts state event
- push_message: raises ValueError when episode not found
- push_message: raises ValueError when repo not configured
- get_transcript: delegates to repo
- get_transcript: returns [] when repo is None
- get_initial_state_event: returns StateEventMessage for valid episode
- get_initial_state_event: returns None when episode not found
- get_initial_state_event: returns fallback when no repo
- _broadcast_state_event: computes state from message (including tool_calls), broadcasts via connection_manager
- cleanup_episode: delegates to connection_manager
- lifecycle hooks: _on_episode_created / _on_episode_ended delegate to state machine
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.models.constants import MetadataKeys
from saber.models.rest.websocket_messages import (
    StateEventMessage,
    TranscriptOperation,
)
from saber.server.base import Episode
from saber.server.episodes.transcript.coordinator import TranscriptCoordinator

TEST_EPISODE_ID = "12345678-1234-1234-1234-123456789abc"
TEST_SESSION_ID = "session-001"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_episode() -> Episode:
    """Create a test Episode."""
    return Episode(
        episode_id=TEST_EPISODE_ID,
        task_id="task-1",
        session_id=TEST_SESSION_ID,
    )


@pytest.fixture
def mock_episode_manager(mock_episode: Episode) -> MagicMock:
    """Create a mock EpisodeManagerProtocol.

    Includes register_on_episode_created / register_on_episode_ended so
    the coordinator's __init__ succeeds.
    """
    mgr = MagicMock()
    mgr.get_episode_by_id = MagicMock(return_value=mock_episode)
    # Store registered callbacks so tests can inspect them
    mgr._created_cbs: list[Callable[[str], Awaitable[None]]] = []
    mgr._ended_cbs: list[Callable[[str], Awaitable[None]]] = []
    mgr.register_on_episode_created = MagicMock(
        side_effect=lambda cb: mgr._created_cbs.append(cb),
    )
    mgr.register_on_episode_ended = MagicMock(
        side_effect=lambda cb: mgr._ended_cbs.append(cb),
    )
    return mgr


@pytest.fixture
def mock_connection_manager() -> MagicMock:
    """Create a mock ConnectionManagerProtocol."""
    mgr = MagicMock()
    mgr.broadcast_to_episode = AsyncMock()
    mgr.cleanup_episode = AsyncMock()
    return mgr


@pytest.fixture
def mock_event_repo() -> MagicMock:
    """Create a mock EpisodeEventRepository."""
    repo = MagicMock()
    repo.insert_message = AsyncMock(return_value=1)
    repo.get_transcript_messages = AsyncMock(return_value=[])
    repo.get_episode_events = AsyncMock(return_value=[])
    repo.get_transcript_count = AsyncMock(return_value=0)
    repo.get_latest_sequence = AsyncMock(return_value=-1)
    return repo


@pytest.fixture
def coordinator(
    mock_episode_manager: MagicMock,
    mock_connection_manager: MagicMock,
    mock_event_repo: MagicMock,
) -> TranscriptCoordinator:
    """Create a TranscriptCoordinator with mocked dependencies."""
    return TranscriptCoordinator(
        episode_manager=mock_episode_manager,
        connection_manager=mock_connection_manager,
        event_repository=mock_event_repo,
    )


@pytest.fixture
def coordinator_no_repo(
    mock_episode_manager: MagicMock,
    mock_connection_manager: MagicMock,
) -> TranscriptCoordinator:
    """Create a TranscriptCoordinator without an event repository."""
    return TranscriptCoordinator(
        episode_manager=mock_episode_manager,
        connection_manager=mock_connection_manager,
        event_repository=None,
    )


# ---------------------------------------------------------------------------
# push_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPushMessage:
    """Tests for push_message()."""

    async def test_stores_in_db_and_broadcasts(
        self,
        coordinator: TranscriptCoordinator,
        mock_connection_manager: MagicMock,
    ) -> None:
        """push_message stores message in DB and broadcasts a state event."""
        # The coordinator wraps event_repo in a TranscriptRepository;
        # we verify the broadcast side-effect
        seq = await coordinator.push_message(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            message={"role": "user", "content": "hello"},
        )

        assert isinstance(seq, int)
        mock_connection_manager.broadcast_to_episode.assert_called_once()
        call_kwargs = mock_connection_manager.broadcast_to_episode.call_args
        assert call_kwargs[1]["episode_id"] == TEST_EPISODE_ID
        msg = call_kwargs[1]["message"]
        assert isinstance(msg, StateEventMessage)

    async def test_raises_when_episode_not_found(
        self,
        coordinator: TranscriptCoordinator,
        mock_episode_manager: MagicMock,
    ) -> None:
        """push_message raises ValueError when episode is not found."""
        mock_episode_manager.get_episode_by_id.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await coordinator.push_message(
                episode_id="nonexistent",
                session_id=TEST_SESSION_ID,
                message={"role": "user", "content": "hi"},
            )

    async def test_raises_when_repo_not_configured(
        self,
        coordinator_no_repo: TranscriptCoordinator,
    ) -> None:
        """push_message raises ValueError when repo is None."""
        with pytest.raises(ValueError, match="Event repository not configured"):
            await coordinator_no_repo.push_message(
                episode_id=TEST_EPISODE_ID,
                session_id=TEST_SESSION_ID,
                message={"role": "user", "content": "hi"},
            )


# ---------------------------------------------------------------------------
# get_transcript
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetTranscript:
    """Tests for get_transcript()."""

    async def test_delegates_to_repo(
        self,
        coordinator: TranscriptCoordinator,
        mock_event_repo: MagicMock,
    ) -> None:
        """get_transcript delegates to the TranscriptRepository."""
        mock_event_repo.get_transcript_messages = AsyncMock(
            return_value=[{"role": "user", "content": "hi"}],
        )

        result = await coordinator.get_transcript(TEST_EPISODE_ID)

        assert len(result) == 1
        assert result[0]["role"] == "user"

    async def test_returns_empty_when_no_repo(
        self,
        coordinator_no_repo: TranscriptCoordinator,
    ) -> None:
        """get_transcript returns [] when repo is None."""
        result = await coordinator_no_repo.get_transcript(TEST_EPISODE_ID)
        assert result == []


# ---------------------------------------------------------------------------
# get_initial_state_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetInitialStateEvent:
    """Tests for get_initial_state_event()."""

    async def test_returns_state_event_for_valid_episode(
        self,
        coordinator: TranscriptCoordinator,
        mock_event_repo: MagicMock,
    ) -> None:
        """get_initial_state_event returns a StateEventMessage for a valid episode."""
        mock_event_repo.get_transcript_messages = AsyncMock(return_value=[])
        mock_event_repo.get_latest_sequence = AsyncMock(return_value=-1)

        event = await coordinator.get_initial_state_event(TEST_EPISODE_ID)

        assert event is not None
        assert isinstance(event, StateEventMessage)
        assert event.data.operation == TranscriptOperation.INIT

    async def test_returns_none_when_episode_not_found(
        self,
        coordinator: TranscriptCoordinator,
        mock_episode_manager: MagicMock,
    ) -> None:
        """get_initial_state_event returns None when episode is not found."""
        mock_episode_manager.get_episode_by_id.return_value = None

        event = await coordinator.get_initial_state_event("nonexistent")
        assert event is None

    async def test_returns_fallback_when_no_repo(
        self,
        coordinator_no_repo: TranscriptCoordinator,
    ) -> None:
        """get_initial_state_event returns a basic state event when no repo."""
        event = await coordinator_no_repo.get_initial_state_event(TEST_EPISODE_ID)

        assert event is not None
        assert isinstance(event, StateEventMessage)
        assert event.data.operation == TranscriptOperation.INIT
        assert event.type == "is_waiting_on_user"


# ---------------------------------------------------------------------------
# _broadcast_state_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBroadcastStateEvent:
    """Tests for _broadcast_state_event()."""

    async def test_computes_state_and_broadcasts(
        self,
        coordinator: TranscriptCoordinator,
        mock_connection_manager: MagicMock,
    ) -> None:
        """_broadcast_state_event computes state from message and broadcasts."""
        state = await coordinator._broadcast_state_event(
            episode_id=TEST_EPISODE_ID,
            sequence=5,
            operation="append",
            message={"role": "assistant", "content": "hi"},
        )

        assert state is not None
        mock_connection_manager.broadcast_to_episode.assert_called_once()
        call_kwargs = mock_connection_manager.broadcast_to_episode.call_args[1]
        msg = call_kwargs["message"]
        assert isinstance(msg, StateEventMessage)
        assert msg.data.version == 5
        assert msg.data.operation == TranscriptOperation.APPEND

    async def test_assistant_with_tool_calls_detects_waiting_for_tools(
        self,
        coordinator: TranscriptCoordinator,
        mock_connection_manager: MagicMock,
        mock_event_repo: MagicMock,
    ) -> None:
        """_broadcast_state_event detects WAITING_FOR_TOOLS when message has tool_calls."""
        from saber.server.episodes.transcript.state_machine import TranscriptState

        # Simulate that the message was already pushed to the repo (as happens in push_message)
        tool_message = {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash"}}],
        }
        mock_event_repo.get_transcript_messages = AsyncMock(return_value=[tool_message])

        state = await coordinator._broadcast_state_event(
            episode_id=TEST_EPISODE_ID,
            sequence=3,
            operation="append",
            message=tool_message,
        )

        assert state == TranscriptState.WAITING_FOR_TOOLS
        msg = mock_connection_manager.broadcast_to_episode.call_args[1]["message"]
        assert msg.data.state == TranscriptState.WAITING_FOR_TOOLS.value

    async def test_handles_unknown_operation(
        self,
        coordinator: TranscriptCoordinator,
        mock_connection_manager: MagicMock,
    ) -> None:
        """_broadcast_state_event falls back to APPEND for unknown operations."""
        state = await coordinator._broadcast_state_event(
            episode_id=TEST_EPISODE_ID,
            sequence=1,
            operation="unknown_op",
            message={"role": "user", "content": "hello"},
        )

        assert state is not None
        msg = mock_connection_manager.broadcast_to_episode.call_args[1]["message"]
        assert msg.data.operation == TranscriptOperation.APPEND


# ---------------------------------------------------------------------------
# cleanup_episode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCleanupEpisode:
    """Tests for cleanup_episode()."""

    async def test_delegates_to_connection_manager(
        self,
        coordinator: TranscriptCoordinator,
        mock_connection_manager: MagicMock,
    ) -> None:
        """cleanup_episode delegates to connection_manager.cleanup_episode."""
        await coordinator.cleanup_episode(TEST_EPISODE_ID)

        mock_connection_manager.cleanup_episode.assert_called_once_with(TEST_EPISODE_ID)


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLifecycleHooks:
    """Tests for lifecycle hook registration and execution."""

    async def test_registers_hooks_on_init(
        self,
        mock_episode_manager: MagicMock,
        mock_connection_manager: MagicMock,
    ) -> None:
        """__init__ registers created and ended callbacks on the episode manager."""
        # Reset to verify fresh registration
        mock_episode_manager.register_on_episode_created.reset_mock()
        mock_episode_manager.register_on_episode_ended.reset_mock()

        TranscriptCoordinator(
            episode_manager=mock_episode_manager,
            connection_manager=mock_connection_manager,
            event_repository=None,
        )

        mock_episode_manager.register_on_episode_created.assert_called_once()
        mock_episode_manager.register_on_episode_ended.assert_called_once()

    async def test_on_episode_created_delegates_to_state_machine(
        self,
        coordinator: TranscriptCoordinator,
    ) -> None:
        """_on_episode_created delegates to state_machine.on_episode_created."""
        with patch.object(
            coordinator._state_machine, "on_episode_created", new_callable=AsyncMock
        ) as mock_sm:
            await coordinator._on_episode_created(TEST_EPISODE_ID)
            mock_sm.assert_called_once_with(TEST_EPISODE_ID)

    async def test_on_episode_ended_delegates_to_state_machine(
        self,
        coordinator: TranscriptCoordinator,
    ) -> None:
        """_on_episode_ended delegates to state_machine.on_episode_terminated."""
        with patch.object(
            coordinator._state_machine, "on_episode_terminated", new_callable=AsyncMock
        ) as mock_sm:
            await coordinator._on_episode_ended(TEST_EPISODE_ID)
            mock_sm.assert_called_once_with(TEST_EPISODE_ID)


# ---------------------------------------------------------------------------
# get_message_count / get_latest_sequence fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReadFallbacks:
    """Tests for read methods returning safe defaults without repo."""

    async def test_get_message_count_returns_zero_without_repo(
        self,
        coordinator_no_repo: TranscriptCoordinator,
    ) -> None:
        """get_message_count returns 0 when repo is None."""
        count = await coordinator_no_repo.get_message_count(TEST_EPISODE_ID)
        assert count == 0

    async def test_get_latest_sequence_returns_zero_without_repo(
        self,
        coordinator_no_repo: TranscriptCoordinator,
    ) -> None:
        """get_latest_sequence returns 0 when repo is None."""
        seq = await coordinator_no_repo.get_latest_sequence(TEST_EPISODE_ID)
        assert seq == 0


# ---------------------------------------------------------------------------
# seed_initial_transcript
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSeedInitialTranscript:
    """Tests for seed_initial_transcript()."""

    async def test_seeds_messages_into_redis(
        self,
        coordinator: TranscriptCoordinator,
        mock_event_repo: MagicMock,
    ) -> None:
        """seed_initial_transcript stores each message via repo.append_message."""
        messages = [
            {"role": "system", "content": "You are a security agent"},
            {"role": "user", "content": "Solve this challenge"},
        ]

        await coordinator.seed_initial_transcript(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            messages=messages,
        )

        # Verify each message was appended
        assert mock_event_repo.insert_message.call_count == 2

    async def test_skips_when_no_repo(
        self,
        coordinator_no_repo: TranscriptCoordinator,
    ) -> None:
        """seed_initial_transcript is a no-op when repo is None."""
        messages = [
            {"role": "system", "content": "System prompt"},
        ]

        # Should not raise
        await coordinator_no_repo.seed_initial_transcript(
            episode_id=TEST_EPISODE_ID,
            session_id=TEST_SESSION_ID,
            messages=messages,
        )


# ---------------------------------------------------------------------------
# _on_episode_created seeds Redis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnEpisodeCreatedSeedsRedis:
    """Tests for _on_episode_created seeding transcript into Redis."""

    async def test_seeds_initial_transcript_from_context(
        self,
        coordinator: TranscriptCoordinator,
        mock_episode: Episode,
        mock_event_repo: MagicMock,
    ) -> None:
        """_on_episode_created reads INITIAL_TRANSCRIPT and seeds Redis."""
        initial_messages = [
            {"role": "system", "content": "You are a security agent"},
            {"role": "user", "content": "Solve the challenge"},
        ]
        mock_episode.context[MetadataKeys.INITIAL_TRANSCRIPT] = initial_messages

        await coordinator._on_episode_created(TEST_EPISODE_ID)

        # Verify messages were seeded (2 calls to insert_message)
        assert mock_event_repo.insert_message.call_count == 2

    async def test_skips_seeding_when_no_initial_transcript(
        self,
        coordinator: TranscriptCoordinator,
        mock_episode: Episode,
        mock_event_repo: MagicMock,
    ) -> None:
        """_on_episode_created does nothing when INITIAL_TRANSCRIPT is missing."""
        # No INITIAL_TRANSCRIPT in context
        await coordinator._on_episode_created(TEST_EPISODE_ID)

        # No messages should be seeded
        mock_event_repo.insert_message.assert_not_called()

    async def test_skips_seeding_when_episode_not_found(
        self,
        coordinator: TranscriptCoordinator,
        mock_episode_manager: MagicMock,
        mock_event_repo: MagicMock,
    ) -> None:
        """_on_episode_created handles missing episode gracefully."""
        mock_episode_manager.get_episode_by_id.return_value = None

        await coordinator._on_episode_created("nonexistent-episode")

        # No messages should be seeded
        mock_event_repo.insert_message.assert_not_called()
