"""BenchmarkSession model for tracking active benchmark sessions."""

from typing import Any, Dict, Optional

from .benchmark_info import BenchmarkInfo


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
