"""Time source abstraction for testability.

This module provides a protocol-based abstraction for getting the current time,
enabling deterministic testing by injecting fake time sources.

Example usage in production:
    manager = ConnectionManager(time_source=UTCTimeSource())

Example usage in tests:
    fake_time = FakeTimeSource()
    manager = ConnectionManager(time_source=fake_time)

    # Advance time for testing
    fake_time.advance(60)  # Advance 60 seconds
"""

from datetime import datetime, timedelta, timezone
from typing import Protocol


class TimeSource(Protocol):
    """Protocol for getting current time (enables testing via dependency injection)."""

    def now(self) -> datetime:
        """Get current UTC time with timezone info.

        Returns:
            datetime: Current time in UTC with timezone info
        """
        ...


def utc_now() -> datetime:
    """Get current UTC time with timezone info.

    Convenience function for getting timezone-aware UTC datetime.
    Use this for default factories in Pydantic models and dataclasses.

    Returns:
        datetime: Current time in UTC with timezone info
    """
    return datetime.now(timezone.utc)


class UTCTimeSource:
    """Production time source using system clock.

    Returns UTC time with timezone info (timezone-aware).
    """

    def now(self) -> datetime:
        """Get current UTC time from system clock.

        Returns:
            datetime: Current time in UTC with timezone info
        """
        return utc_now()


class FakeTimeSource:
    """Fake time source for deterministic testing.

    Allows tests to control time progression and verify time-dependent behavior.

    Example:
        fake_time = FakeTimeSource()
        assert fake_time.now() == datetime(2025, 1, 1, tzinfo=timezone.utc)

        fake_time.advance(60)  # Advance 60 seconds
        assert fake_time.now() == datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc)

        fake_time.set(datetime(2025, 12, 25, 12, 0, tzinfo=timezone.utc))
        assert fake_time.now() == datetime(2025, 12, 25, 12, 0, tzinfo=timezone.utc)
    """

    def __init__(self, initial_time: datetime | None = None):
        """Initialize fake time source.

        Args:
            initial_time: Starting time (defaults to 2025-01-01 00:00:00 UTC)
        """
        if initial_time is None:
            initial_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        elif initial_time.tzinfo is None:
            # Ensure timezone-aware
            initial_time = initial_time.replace(tzinfo=timezone.utc)

        self._current = initial_time

    def now(self) -> datetime:
        """Get current fake time.

        Returns:
            datetime: Current fake time in UTC with timezone info
        """
        return self._current

    def advance(self, seconds: float) -> None:
        """Advance fake time by specified seconds.

        Args:
            seconds: Number of seconds to advance (can be negative to go back)
        """
        self._current += timedelta(seconds=seconds)

    def set(self, new_time: datetime) -> None:
        """Set fake time to specific value.

        Args:
            new_time: New time value (will be converted to UTC if timezone-naive)
        """
        if new_time.tzinfo is None:
            new_time = new_time.replace(tzinfo=timezone.utc)
        self._current = new_time


__all__ = ["TimeSource", "UTCTimeSource", "FakeTimeSource", "utc_now"]
