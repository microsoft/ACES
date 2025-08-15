"""Exceptions for the episode management system."""


class EpisodeManagerException(Exception):
    """Base exception for episode manager errors."""

    pass


class EpisodeNotFoundException(EpisodeManagerException):
    """Raised when an episode is not found for a session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"No active episode found for session: {session_id}")


class EpisodeStateException(EpisodeManagerException):
    """Raised when episode is in an invalid state for the requested operation."""

    def __init__(self, episode_id: str, current_state: str, expected_state: str):
        self.episode_id = episode_id
        self.current_state = current_state
        self.expected_state = expected_state
        super().__init__(f"Episode {episode_id} is in state '{current_state}', expected '{expected_state}'")
