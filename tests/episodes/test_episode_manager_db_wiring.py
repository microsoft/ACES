"""Tests for EpisodeManager DB wiring.

Verifies that EpisodeManager correctly accepts and passes
an optional EpisodeEventRepository to TranscriptCoordinator.
"""

from unittest.mock import MagicMock

from saber.server.episodes.episode_manager import EpisodeManager


class TestEpisodeManagerDbWiring:
    """Test EpisodeManager ↔ TranscriptCoordinator DB wiring."""

    def test_default_no_event_repository(self) -> None:
        """EpisodeManager without event_repository → coordinator._repo is None."""
        manager = EpisodeManager()
        assert manager.transcript_coordinator._repo is None

    def test_with_event_repository(self) -> None:
        """EpisodeManager with event_repository → coordinator._repo is set."""
        mock_repo = MagicMock()
        manager = EpisodeManager(event_repository=mock_repo)
        assert manager.transcript_coordinator._repo is not None

    def test_event_repository_passed_to_coordinator(self) -> None:
        """The TranscriptRepository wraps the given event_repository."""
        mock_repo = MagicMock()
        manager = EpisodeManager(event_repository=mock_repo)
        # TranscriptRepository stores the event_repo as _event_repo
        assert manager.transcript_coordinator._repo._event_repo is mock_repo

    def test_backward_compat_no_args(self) -> None:
        """EpisodeManager() with no args still works (backward compat)."""
        manager = EpisodeManager()
        assert manager.episodes == {}
        assert manager.session_episodes == {}
        assert manager.transcript_coordinator is not None


class TestWireEventRepository:
    """Test EpisodeManager.wire_event_repository() public API."""

    def test_wire_event_repository_sets_repo(self) -> None:
        """wire_event_repository sets the coordinator's repository."""
        manager = EpisodeManager()
        assert manager.transcript_coordinator._repo is None

        mock_repo = MagicMock()
        manager.wire_event_repository(mock_repo)

        assert manager.transcript_coordinator._repo is not None
        assert manager.transcript_coordinator._repo._event_repo is mock_repo

    def test_wire_event_repository_overwrites_existing(self) -> None:
        """wire_event_repository can replace an existing repository."""
        mock_repo1 = MagicMock()
        manager = EpisodeManager(event_repository=mock_repo1)
        assert manager.transcript_coordinator._repo._event_repo is mock_repo1

        mock_repo2 = MagicMock()
        manager.wire_event_repository(mock_repo2)
        assert manager.transcript_coordinator._repo._event_repo is mock_repo2
