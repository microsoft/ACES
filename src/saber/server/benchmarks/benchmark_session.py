"""BenchmarkSession model for tracking active benchmark sessions."""

from typing import Any, Dict, List, Optional

from ...models import BenchmarkInfo


class BenchmarkSession:
    """
    Tracks an active benchmark session.

    This is a wrapper around BenchmarkInfo that includes session-specific configuration.
    """

    def __init__(self, session_id: str, benchmark_info: BenchmarkInfo, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a benchmark session.

        Args:
            session_id: The session ID this benchmark is associated with
            benchmark_info: The benchmark information
            config: Optional benchmark configuration
        """
        self.session_id = session_id
        self.benchmark_info = benchmark_info
        self.config = config or {}
        self.active_episode_ids: List[str] = []  # Track episodes associated with this benchmark session

    def to_api_response(self, domain: str) -> Dict[str, Any]:
        """
        Convert to API response format.

        Args:
            domain: The domain name

        Returns:
            Dictionary containing benchmark session information in API format
        """
        response = self.benchmark_info.to_dict()
        response["benchmark_config"] = self.config
        response["active_episode_ids"] = self.active_episode_ids
        return response

    @property
    def domain(self) -> str:
        """Get the domain name."""
        return self.benchmark_info.domain

    @property
    def total_tasks(self) -> int:
        """Get the total number of tasks."""
        return self.benchmark_info.total_tasks

    @property
    def total_episodes(self) -> int:
        """Get the total number of episodes."""
        return self.benchmark_info.total_episodes

    def add_episode(self, episode_id: str) -> None:
        """Add an episode to this benchmark session."""
        if episode_id not in self.active_episode_ids:
            self.active_episode_ids.append(episode_id)

    def remove_episode(self, episode_id: str) -> None:
        """Remove an episode from this benchmark session."""
        if episode_id in self.active_episode_ids:
            self.active_episode_ids.remove(episode_id)

    @property
    def active_episode_count(self) -> int:
        """Get the number of active episodes."""
        return len(self.active_episode_ids)
