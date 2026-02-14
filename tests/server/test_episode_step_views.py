"""Tests for Episode.with_step_range() method for chunked evaluation."""

import pytest
from datetime import datetime, timezone
from saber.server.base import Episode, EpisodeState, Step, Action


def create_test_episode(num_steps: int = 10) -> Episode:
    """Create a test episode with specified number of steps."""
    episode = Episode(
        episode_id="test-episode-123",
        task_id="test-task",
        session_id="test-session",
        state=EpisodeState.COMPLETED,
        max_steps=num_steps,
    )

    # Add steps
    for i in range(num_steps):
        step = Step(
            step_number=i + 1,
            timestamp=datetime.now(timezone.utc),
            action=Action(
                tool_name="bash",
                parameters={"arguments": f"echo 'step {i + 1}'"},
            ),
            response={"exit_code": 0, "stdout": f"step {i + 1}\n", "stderr": ""},
        )
        episode.steps.append(step)

    return episode


def test_episode_with_step_range_basic():
    """Test Episode.with_step_range() creates proper view with correct steps."""
    episode = create_test_episode(10)

    # Get first 5 steps
    view = episode.with_step_range(0, 5)

    assert len(view.steps) == 5
    assert view.steps[0].step_number == 1
    assert view.steps[4].step_number == 5

    # Original episode should be unchanged
    assert len(episode.steps) == 10


def test_episode_with_step_range_middle_chunk():
    """Test getting a middle chunk of steps."""
    episode = create_test_episode(20)

    # Get steps 10-15
    view = episode.with_step_range(10, 15)

    assert len(view.steps) == 5
    assert view.steps[0].step_number == 11  # 0-indexed, so step 10 is step_number 11
    assert view.steps[4].step_number == 15


def test_episode_with_step_range_last_chunk():
    """Test getting the last chunk of steps."""
    episode = create_test_episode(25)

    # Get last 5 steps
    view = episode.with_step_range(20, 25)

    assert len(view.steps) == 5
    assert view.steps[0].step_number == 21
    assert view.steps[4].step_number == 25


def test_episode_view_shallow_copy():
    """Test that episode views share data with original (shallow copy)."""
    episode = create_test_episode(10)
    episode.metadata = {"test_key": "test_value"}

    view = episode.with_step_range(0, 5)

    # View should share the same metadata object
    assert view.metadata is episode.metadata
    assert view.episode_id == episode.episode_id
    assert view.task_id == episode.task_id
    assert view.session_id == episode.session_id

    # But steps should be different
    assert view.steps is not episode.steps
    assert len(view.steps) == 5
    assert len(episode.steps) == 10


def test_episode_view_preserves_metadata():
    """Test that all episode fields accessible from view."""
    episode = create_test_episode(10)
    episode.submission = "flag{test_flag}"
    episode.completion_reason = "max_steps_reached"
    episode.context = {"domain": "test"}

    view = episode.with_step_range(0, 5)

    # All metadata should be accessible
    assert view.submission == episode.submission
    assert view.completion_reason == episode.completion_reason
    assert view.context == episode.context
    assert view.state == episode.state
    assert view.max_steps == episode.max_steps


def test_episode_with_step_range_empty():
    """Test edge case: empty range."""
    episode = create_test_episode(10)

    view = episode.with_step_range(5, 5)

    assert len(view.steps) == 0


def test_episode_with_step_range_single_step():
    """Test edge case: single step."""
    episode = create_test_episode(10)

    view = episode.with_step_range(3, 4)

    assert len(view.steps) == 1
    assert view.steps[0].step_number == 4


def test_episode_with_step_range_full_range():
    """Test edge case: full range (same as original)."""
    episode = create_test_episode(10)

    view = episode.with_step_range(0, 10)

    assert len(view.steps) == 10
    # Should still be a copy, not the same list
    assert view.steps is not episode.steps
    # But individual steps should be the same objects (shallow copy)
    for i in range(10):
        assert view.steps[i] is episode.steps[i]


def test_episode_with_step_range_out_of_bounds():
    """Test that Python's slicing behavior applies (no error on out of bounds)."""
    episode = create_test_episode(10)

    # End index beyond length - should just return up to the end
    view = episode.with_step_range(5, 20)

    assert len(view.steps) == 5
    assert view.steps[0].step_number == 6
    assert view.steps[4].step_number == 10


def test_episode_with_step_range_chunking_simulation():
    """Test realistic chunking scenario for evaluation."""
    episode = create_test_episode(50)
    steps_per_message = 20

    chunks = []
    for chunk_start in range(0, len(episode.steps), steps_per_message):
        chunk_end = min(chunk_start + steps_per_message, len(episode.steps))
        chunk_view = episode.with_step_range(chunk_start, chunk_end)
        chunks.append(chunk_view)

    # Should have 3 chunks (0-19, 20-39, 40-49)
    assert len(chunks) == 3
    assert len(chunks[0].steps) == 20
    assert len(chunks[1].steps) == 20
    assert len(chunks[2].steps) == 10

    # Verify step numbers
    assert chunks[0].steps[0].step_number == 1
    assert chunks[0].steps[-1].step_number == 20
    assert chunks[1].steps[0].step_number == 21
    assert chunks[1].steps[-1].step_number == 40
    assert chunks[2].steps[0].step_number == 41
    assert chunks[2].steps[-1].step_number == 50
