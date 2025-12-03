"""
Integration tests for InjectPromptExecutor with session manager injection.

Tests the full flow of session manager injection into executors via the
executor factory, enabling cross-episode transcript modification.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from saber.models.constants import MetadataKeys
from saber.server.base import Episode, EpisodeState
from saber.server.execution.executors import get_executor_class


class TestSessionManagerInjection:
    """Test that session manager can be injected into InjectPromptExecutor."""

    @pytest.fixture
    def blue_episode(self) -> Episode:
        """Create a blue team episode with initial transcript."""
        push_time = datetime.utcnow().isoformat()
        return Episode(
            episode_id="ep-blue-integration",
            task_id="blue-task",
            session_id="session-integration",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.CLIENT_TRANSCRIPT: [
                    {"role": "system", "content": "You are helpful..."},
                    {"role": "assistant", "content": "Hello! How can I help?"},
                ],
                MetadataKeys.TRANSCRIPT_LAST_PUSHED_AT: push_time,
                MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT: 0,
            },
        )

    @pytest.fixture
    def red_episode(self) -> Episode:
        """Create a red team episode."""
        return Episode(
            episode_id="ep-red-integration",
            task_id="red-task",
            session_id="session-integration",
            state=EpisodeState.ACTIVE,
            context={
                MetadataKeys.ORCHESTRATION_TARGET_EPISODES: ["ep-blue-integration"],
            },
        )

    @pytest.fixture
    def mock_session_manager(self, blue_episode: Episode, red_episode: Episode):
        """Create a mock session manager with episode lookup capability."""

        class MockConnectionManager:
            async def broadcast_to_episode(self, episode_id: str, message: dict):
                pass
            async def cleanup_episode(self, episode_id: str):
                pass

        class MockEpisodeManager:
            def __init__(self, blue_ep, red_ep):
                self.episodes = {
                    blue_ep.episode_id: blue_ep,
                    red_ep.episode_id: red_ep,
                }
                # Add transcript_coordinator
                from saber.server.episodes.transcript_coordinator import TranscriptCoordinator
                self.transcript_coordinator = TranscriptCoordinator(self, MockConnectionManager())

            def get_episode_by_id(self, episode_id: str):
                return self.episodes.get(episode_id)

        class MockSessionManager:
            def __init__(self, blue_ep, red_ep):
                self.episode_manager = MockEpisodeManager(blue_ep, red_ep)

        return MockSessionManager(blue_episode, red_episode)

    def test_executor_class_registered(self):
        """Test that inject_prompt executor is registered."""
        # Act
        executor_class = get_executor_class("inject_prompt")

        # Assert
        assert executor_class is not None
        assert executor_class.__name__ == "InjectPromptExecutor"

    def test_session_manager_can_be_set(self, mock_session_manager):
        """Test that session_manager can be set on executor instance."""
        # Arrange
        executor_class = get_executor_class("inject_prompt")
        executor = executor_class.create_with_config(
            sandbox_manager=None,
            session_manager=mock_session_manager
        )

        # Assert
        assert executor._session_manager is not None
        assert hasattr(executor._session_manager, "episode_manager")

    @pytest.mark.asyncio
    async def test_cross_episode_access_via_session_manager(
        self,
        blue_episode: Episode,
        red_episode: Episode,
        mock_session_manager
    ):
        """Test that red team can access blue team episode via session manager."""
        # Arrange
        executor_class = get_executor_class("inject_prompt")
        executor = executor_class.create_with_config(
            sandbox_manager=None,
            session_manager=mock_session_manager
        )

        parameters = {
            "message": "Cross-episode test injection",
        }
        context = {
            "session_id": "session-integration",
            "episode_id": "ep-red-integration",
        }

        # Act
        result = await executor.execute(parameters, context)

        # Assert
        assert result.success is True

        # Verify blue team episode was modified
        assert len(blue_episode.context[MetadataKeys.CLIENT_TRANSCRIPT]) == 3
        assert blue_episode.context[MetadataKeys.CLIENT_TRANSCRIPT][-1]["content"] == "Cross-episode test injection"
        assert blue_episode.context[MetadataKeys.TRANSCRIPT_MODIFICATION_COUNT] == 1
        assert MetadataKeys.TRANSCRIPT_LAST_MODIFIED_AT in blue_episode.context


class TestExecutorFactorySessionManagerInjection:
    """Test session manager injection through executor factory."""

    def test_inject_prompt_in_available_executors(self):
        """Test that inject_prompt is in the list of available executors."""
        from saber.server.execution.executors import get_available_executors

        # Act
        available = get_available_executors()

        # Assert
        assert "inject_prompt" in available

    def test_executor_metadata(self):
        """Test executor metadata is correctly set."""
        # Arrange
        executor_class = get_executor_class("inject_prompt")

        # Act
        metadata = executor_class._executor_metadata

        # Assert
        assert metadata["name"] == "inject_prompt"
        assert "inject" in metadata["description"].lower()
        assert "adversarial" in metadata["description"].lower() or "red team" in metadata["description"].lower()
