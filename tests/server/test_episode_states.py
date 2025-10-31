"""Tests for episode state transitions and is_ready property."""

from saber.server.base import Episode, EpisodeState


class TestEpisodeStates:
    """Test episode state transitions and properties."""

    def test_is_ready_returns_true_for_ready_state(self):
        """Test that is_ready returns True for READY state."""
        episode = Episode(
            episode_id="test-episode-1",
            session_id="test-session",
            task_id="test-task",
            state=EpisodeState.READY,
        )
        assert episode.is_ready is True

    def test_is_ready_returns_true_for_active_state(self):
        """Test that is_ready returns True for ACTIVE state."""
        episode = Episode(
            episode_id="test-episode-2",
            session_id="test-session",
            task_id="test-task",
            state=EpisodeState.ACTIVE,
        )
        assert episode.is_ready is True

    def test_is_ready_returns_false_for_creating_state(self):
        """Test that is_ready returns False for CREATING state."""
        episode = Episode(
            episode_id="test-episode-3",
            session_id="test-session",
            task_id="test-task",
            state=EpisodeState.CREATING,
        )
        assert episode.is_ready is False

    def test_is_ready_returns_false_for_failed_creation_state(self):
        """Test that is_ready returns False for FAILED_CREATION state."""
        episode = Episode(
            episode_id="test-episode-4",
            session_id="test-session",
            task_id="test-task",
            state=EpisodeState.FAILED_CREATION,
        )
        assert episode.is_ready is False

    def test_is_ready_returns_false_for_failed_state(self):
        """Test that is_ready returns False for FAILED state."""
        episode = Episode(
            episode_id="test-episode-5",
            session_id="test-session",
            task_id="test-task",
            state=EpisodeState.FAILED,
        )
        assert episode.is_ready is False

    def test_is_ready_returns_false_for_completed_state(self):
        """Test that is_ready returns False for COMPLETED state."""
        episode = Episode(
            episode_id="test-episode-6",
            session_id="test-session",
            task_id="test-task",
            state=EpisodeState.COMPLETED,
        )
        assert episode.is_ready is False

    def test_episode_state_transitions_to_ready(self):
        """Test that episode state can be updated to READY."""
        episode = Episode(
            episode_id="test-episode-7",
            session_id="test-session",
            task_id="test-task",
            state=EpisodeState.CREATING,
        )
        assert episode.is_ready is False

        # Update state to READY
        episode.state = EpisodeState.READY
        assert episode.is_ready is True

    def test_episode_state_transitions_to_failed_creation(self):
        """Test that episode state can be updated to FAILED_CREATION."""
        episode = Episode(
            episode_id="test-episode-8",
            session_id="test-session",
            task_id="test-task",
            state=EpisodeState.CREATING,
        )
        assert episode.is_ready is False

        # Update state to FAILED_CREATION
        episode.state = EpisodeState.FAILED_CREATION
        episode.creation_error = "Health check failed"
        assert episode.is_ready is False
        assert episode.creation_error == "Health check failed"

    def test_episode_created_with_error_message(self):
        """Test that episode can be created with error message."""
        episode = Episode(
            episode_id="test-episode-9",
            session_id="test-session",
            task_id="test-task",
            state=EpisodeState.FAILED_CREATION,
            creation_error="Docker Compose failed to start",
        )
        assert episode.is_ready is False
        assert episode.creation_error == "Docker Compose failed to start"
