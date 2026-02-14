"""Tests for auto-continue functionality."""

import pytest

from saber.models.constants import MetadataKeys
from saber.server.base import Episode, EpisodeState
from saber.server.episodes.transcript.auto_continue_manager import AutoContinueManager
from saber.server.episodes.transcript.state_machine import TranscriptState


class MockEpisodeManager:
    """Mock episode manager for testing."""

    def __init__(self):
        self.episodes = {}

    def get_episode_by_id(self, episode_id: str):
        return self.episodes.get(episode_id)


class MockTranscriptCoordinator:
    """Mock transcript coordinator for testing."""

    def __init__(self):
        self.pushed_messages: list[dict[str, str]] = []

    async def push_message(
        self,
        episode_id: str,
        session_id: str,
        message: dict[str, str],
        operation: str = "append",
    ) -> int:
        self.pushed_messages.append({
            "episode_id": episode_id,
            "session_id": session_id,
            "message": message,
            "operation": operation,
        })
        return len(self.pushed_messages)


@pytest.mark.asyncio
class TestAutoContinueBasic:
    """Test basic auto-continue functionality."""

    async def test_auto_continue_injects_message(self):
        """Auto-continue should inject continue message when enabled."""
        episode_manager = MockEpisodeManager()
        coordinator = MockTranscriptCoordinator()

        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.TRANSCRIPT_VERSION: 2,
                MetadataKeys.AUTO_CONTINUE_ENABLED: True,
            }
        )
        episode_manager.episodes["ep-1"] = episode

        manager = AutoContinueManager(episode_manager, coordinator)
        await manager.handle_auto_continue("ep-1", TranscriptState.WAITING_FOR_USER)

        # Should inject message
        assert len(coordinator.pushed_messages) == 1
        msg = coordinator.pushed_messages[0]
        assert msg["operation"] == "auto_continue"
        assert msg["session_id"] == "session-1"
        assert msg["message"]["role"] == "user"

    async def test_auto_continue_disabled_skips_injection(self):
        """Auto-continue should skip when disabled."""
        episode_manager = MockEpisodeManager()
        coordinator = MockTranscriptCoordinator()

        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.AUTO_CONTINUE_ENABLED: False,
            }
        )
        episode_manager.episodes["ep-1"] = episode

        manager = AutoContinueManager(episode_manager, coordinator)
        await manager.handle_auto_continue("ep-1", TranscriptState.WAITING_FOR_USER)

        # Should NOT inject
        assert len(coordinator.pushed_messages) == 0

    async def test_auto_continue_wrong_state_skips(self):
        """Auto-continue should only trigger in WAITING_FOR_USER state."""
        episode_manager = MockEpisodeManager()
        coordinator = MockTranscriptCoordinator()

        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.AUTO_CONTINUE_ENABLED: True,
            }
        )
        episode_manager.episodes["ep-1"] = episode

        manager = AutoContinueManager(episode_manager, coordinator)

        # Wrong states - should not inject
        await manager.handle_auto_continue("ep-1", TranscriptState.WAITING_FOR_ASSISTANT)
        await manager.handle_auto_continue("ep-1", TranscriptState.WAITING_FOR_TOOLS)

        assert len(coordinator.pushed_messages) == 0

    async def test_auto_continue_custom_prompt(self):
        """Auto-continue should use custom prompt if provided."""
        episode_manager = MockEpisodeManager()
        coordinator = MockTranscriptCoordinator()

        custom_prompt = "Keep going with the task!"

        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.AUTO_CONTINUE_ENABLED: True,
                MetadataKeys.CONTINUE_PROMPT: custom_prompt,
            }
        )
        episode_manager.episodes["ep-1"] = episode

        manager = AutoContinueManager(episode_manager, coordinator)
        await manager.handle_auto_continue("ep-1", TranscriptState.WAITING_FOR_USER)

        # Should use custom prompt
        assert len(coordinator.pushed_messages) == 1
        msg = coordinator.pushed_messages[0]
        assert msg["message"]["content"] == custom_prompt


@pytest.mark.asyncio
class TestAutoContinueIdempotency:
    """Test idempotency tracking."""

    async def test_idempotency_prevents_duplicate_injection(self):
        """Auto-continue should not inject twice for same version."""
        episode_manager = MockEpisodeManager()
        coordinator = MockTranscriptCoordinator()

        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.TRANSCRIPT_VERSION: 2,
                MetadataKeys.AUTO_CONTINUE_ENABLED: True,
                MetadataKeys.LAST_AUTO_CONTINUE_VERSION: 2,  # Already injected
            }
        )
        episode_manager.episodes["ep-1"] = episode

        manager = AutoContinueManager(episode_manager, coordinator)
        await manager.handle_auto_continue("ep-1", TranscriptState.WAITING_FOR_USER)

        # Should NOT inject (idempotency)
        assert len(coordinator.pushed_messages) == 0

    async def test_idempotency_allows_new_version(self):
        """Auto-continue should inject for new version."""
        episode_manager = MockEpisodeManager()
        coordinator = MockTranscriptCoordinator()

        episode = Episode(
            episode_id="ep-1",
            task_id="task-1",
            session_id="session-1",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.TRANSCRIPT_VERSION: 5,
                MetadataKeys.AUTO_CONTINUE_ENABLED: True,
                MetadataKeys.LAST_AUTO_CONTINUE_VERSION: 3,  # Old version
            }
        )
        episode_manager.episodes["ep-1"] = episode

        manager = AutoContinueManager(episode_manager, coordinator)
        await manager.handle_auto_continue("ep-1", TranscriptState.WAITING_FOR_USER)

        # Should inject (new version)
        assert len(coordinator.pushed_messages) == 1

    async def test_missing_episode_skips_safely(self):
        """Auto-continue should skip if episode not found."""
        episode_manager = MockEpisodeManager()
        coordinator = MockTranscriptCoordinator()

        manager = AutoContinueManager(episode_manager, coordinator)
        await manager.handle_auto_continue("ep-nonexistent", TranscriptState.WAITING_FOR_USER)

        # Should not raise, no modifications
        assert len(coordinator.pushed_messages) == 0
