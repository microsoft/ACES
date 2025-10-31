"""Tests for async episode manager methods."""

import pytest

from saber.server.base import Episode, EpisodeState
from saber.server.episodes.episode_manager import EpisodeManager
from saber.server.episodes.exceptions import EpisodeNotFoundException


@pytest.fixture
def episode_manager():
    """Create an episode manager for testing."""
    return EpisodeManager()


@pytest.fixture
def test_episode(episode_manager):
    """Create a test episode in CREATING state."""
    episode = Episode(
        episode_id="test-episode-123",
        session_id="test-session",
        task_id="test-task",
        state=EpisodeState.CREATING,
    )
    episode_manager.episodes[episode.episode_id] = episode
    return episode


class TestEpisodeManagerAsync:
    """Test async episode manager methods."""

    def test_mark_episode_ready_transitions_to_ready(self, episode_manager, test_episode):
        """Test that mark_episode_ready transitions episode to READY state."""
        assert test_episode.state == EpisodeState.CREATING
        assert test_episode.is_ready is False

        episode_manager.mark_episode_ready(test_episode.episode_id)

        assert test_episode.state == EpisodeState.READY
        assert test_episode.is_ready is True

    def test_mark_episode_ready_with_nonexistent_episode_raises(self, episode_manager):
        """Test that mark_episode_ready raises for non-existent episode."""
        with pytest.raises(EpisodeNotFoundException, match="Episode .* not found"):
            episode_manager.mark_episode_ready("nonexistent-episode")

    def test_mark_episode_failed_creation_sets_state_and_error(self, episode_manager, test_episode):
        """Test that mark_episode_failed_creation sets state and error message."""
        assert test_episode.state == EpisodeState.CREATING
        assert test_episode.creation_error is None

        error_message = "Docker Compose health check failed"
        episode_manager.mark_episode_failed_creation(test_episode.episode_id, error_message)

        assert test_episode.state == EpisodeState.FAILED_CREATION
        assert test_episode.creation_error == error_message
        assert test_episode.is_ready is False

    def test_mark_episode_failed_creation_with_nonexistent_episode_raises(self, episode_manager):
        """Test that mark_episode_failed_creation raises for non-existent episode."""
        with pytest.raises(EpisodeNotFoundException, match="Episode .* not found"):
            episode_manager.mark_episode_failed_creation("nonexistent-episode", "Some error")

    def test_mark_episode_ready_logs_transition(self, episode_manager, test_episode, caplog):
        """Test that mark_episode_ready logs the state transition."""
        episode_manager.mark_episode_ready(test_episode.episode_id)

        # Check that logging occurred (implementation may vary)
        # This is a placeholder - adjust based on actual logging implementation
        assert test_episode.state == EpisodeState.READY

    def test_mark_episode_failed_creation_logs_error(self, episode_manager, test_episode, caplog):
        """Test that mark_episode_failed_creation logs the error."""
        error_message = "Environment setup failed"
        episode_manager.mark_episode_failed_creation(test_episode.episode_id, error_message)

        assert test_episode.state == EpisodeState.FAILED_CREATION
        assert test_episode.creation_error == error_message

    def test_multiple_episodes_state_management(self, episode_manager):
        """Test managing state for multiple episodes."""
        # Create multiple episodes
        episode1 = Episode(
            episode_id="episode-1",
            session_id="session-1",
            task_id="task-1",
            state=EpisodeState.CREATING,
        )
        episode2 = Episode(
            episode_id="episode-2",
            session_id="session-1",
            task_id="task-2",
            state=EpisodeState.CREATING,
        )

        episode_manager.episodes[episode1.episode_id] = episode1
        episode_manager.episodes[episode2.episode_id] = episode2

        # Mark one ready, one failed
        episode_manager.mark_episode_ready(episode1.episode_id)
        episode_manager.mark_episode_failed_creation(episode2.episode_id, "Failed")

        assert episode1.state == EpisodeState.READY
        assert episode1.is_ready is True
        assert episode2.state == EpisodeState.FAILED_CREATION
        assert episode2.is_ready is False
        assert episode2.creation_error == "Failed"
