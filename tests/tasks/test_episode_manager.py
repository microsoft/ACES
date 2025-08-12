"""
Unit tests for EpisodeManager.

Tests episode lifecycle management, RL interfaces, and state transitions.
Simplified after removing subtask progression tracking.
"""

import pytest
from datetime import datetime

from saber.server.tasks.episodes.episode_manager import EpisodeManager
from saber.server.tasks.episodes import Episode, Step, Action
from saber.server.tasks.base import EpisodeState
from saber.server.tasks.exceptions import EpisodeNotFoundException
from saber.server.execution.base import CommandResult


class TestEpisodeManager:
    """Test cases for EpisodeManager functionality."""

    def test_episode_manager_init(self):
        """Test EpisodeManager initialization."""
        manager = EpisodeManager()
        assert manager.active_episodes == {}

    def test_start_episode_minimal(self):
        """Test starting an episode with minimal parameters."""
        manager = EpisodeManager()

        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task"
        )

        assert isinstance(episode, Episode)
        assert episode.task_id == "test_task"
        assert episode.session_id == "test_session"
        assert episode.state == EpisodeState.ACTIVE
        assert episode.context == {}
        assert "test_session" in manager.active_episodes
        assert manager.active_episodes["test_session"] == episode

    def test_start_episode_with_context(self):
        """Test starting an episode with initial context."""
        manager = EpisodeManager()
        initial_context = {"sample_path": "/data/test.exe", "timeout": 300}

        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task",
            initial_context=initial_context
        )

        assert episode.context == initial_context

    def test_start_episode_replaces_existing(self):
        """Test that starting a new episode replaces any existing episode for the session."""
        manager = EpisodeManager()

        # Start first episode
        episode1 = manager.start_episode("test_session", "task1")
        episode1_id = episode1.episode_id

        # Start second episode (should replace first)
        episode2 = manager.start_episode("test_session", "task2")
        episode2_id = episode2.episode_id

        assert episode1_id != episode2_id
        assert manager.active_episodes["test_session"] == episode2
        assert episode2.task_id == "task2"

    def test_get_current_episode_exists(self):
        """Test getting an existing episode."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        retrieved = manager.get_current_episode("test_session")
        assert retrieved == episode

    def test_get_current_episode_not_exists(self):
        """Test getting a non-existent episode."""
        manager = EpisodeManager()

        result = manager.get_current_episode("nonexistent_session")
        assert result is None

    def test_step_basic(self):
        """Test basic step functionality."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        action = Action(tool_name="test_tool", parameters={"key": "value"})
        command_result = CommandResult(
            success=True,
            data={"output": "Test output", "file_type": "PE32"},
            error=None,
            execution_time=None
        )

        step = manager.step("test_session", action, command_result)

        assert isinstance(step, Step)
        assert step.action == action
        assert step.response["success"] is True
        assert step.response["data"]["output"] == "Test output"
        assert step.step_number == 0
        assert step.done is False

        # Verify step was added to episode
        assert len(episode.steps) == 1
        assert episode.steps[0] == step

    def test_step_session_not_found(self):
        """Test step with non-existent session."""
        manager = EpisodeManager()
        action = Action(tool_name="test_tool", parameters={})
        command_result = CommandResult(success=True, data={}, error=None, execution_time=None)

        with pytest.raises(EpisodeNotFoundException):
            manager.step("nonexistent_session", action, command_result)

    def test_step_increments_step_number(self):
        """Test that step numbers increment correctly."""
        manager = EpisodeManager()
        manager.start_episode("test_session", "test_task")

        action = Action(tool_name="test_tool", parameters={})
        command_result = CommandResult(success=True, data={}, error=None, execution_time=None)

        step1 = manager.step("test_session", action, command_result)
        step2 = manager.step("test_session", action, command_result)
        step3 = manager.step("test_session", action, command_result)

        assert step1.step_number == 0
        assert step2.step_number == 1
        assert step3.step_number == 2

    def test_end_episode_success(self):
        """Test successfully ending an episode."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        ended_episode = manager.end_episode("test_session", "task completed successfully")

        assert ended_episode == episode
        assert episode.state == EpisodeState.COMPLETED
        assert episode.completion_reason == "task completed successfully"
        assert episode.end_time is not None
        assert "test_session" not in manager.active_episodes

    def test_end_episode_not_found(self):
        """Test ending a non-existent episode."""
        manager = EpisodeManager()

        with pytest.raises(EpisodeNotFoundException):
            manager.end_episode("nonexistent_session", "reason")

    def test_reset_episode_existing(self):
        """Test resetting an existing episode."""
        manager = EpisodeManager()
        old_episode = manager.start_episode("test_session", "test_task")
        old_episode_id = old_episode.episode_id

        # Add some steps to the old episode
        action = Action(tool_name="test_tool", parameters={})
        command_result = CommandResult(success=True, data={}, error=None, execution_time=None)
        manager.step("test_session", action, command_result)

        # Reset the episode
        new_episode = manager.reset_episode("test_session", "test_task")

        assert new_episode.episode_id != old_episode_id
        assert new_episode.task_id == "test_task"
        assert new_episode.session_id == "test_session"
        assert new_episode.state == EpisodeState.ACTIVE
        assert len(new_episode.steps) == 0
        assert manager.active_episodes["test_session"] == new_episode

    def test_reset_episode_no_existing(self):
        """Test resetting when no episode exists."""
        manager = EpisodeManager()

        with pytest.raises(EpisodeNotFoundException):
            manager.reset_episode("test_session", "test_task")

    def test_get_episode_state_exists(self):
        """Test getting episode state when episode exists."""
        manager = EpisodeManager()
        manager.start_episode("test_session", "test_task")

        state = manager.get_episode_state("test_session")
        assert state == EpisodeState.ACTIVE

    def test_get_episode_state_not_exists(self):
        """Test getting episode state when episode doesn't exist."""
        manager = EpisodeManager()

        state = manager.get_episode_state("nonexistent_session")
        assert state is None

    def test_cleanup_session_with_episode(self):
        """Test cleaning up a session that has an active episode."""
        manager = EpisodeManager()
        manager.start_episode("test_session", "test_task")

        manager.cleanup_session("test_session")

        assert "test_session" not in manager.active_episodes

    def test_cleanup_session_no_episode(self):
        """Test cleaning up a session that has no active episode."""
        manager = EpisodeManager()

        # Should not raise an error
        manager.cleanup_session("nonexistent_session")

        assert manager.active_episodes == {}

    def test_create_step_basic(self):
        """Test basic step creation."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        action = Action(tool_name="test_tool", parameters={"key": "value"})
        response = CommandResult(success=True, data={"output": "test output"})

        step = manager.create_step(episode, action, response)

        assert isinstance(step, Step)
        assert step.action == action
        assert step.response["success"] is True
        assert step.response["data"]["output"] == "test output"
        assert step.step_number == 0
        assert step.done is False

    def test_update_episode_state_basic(self):
        """Test basic episode state update."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        action = Action(tool_name="test_tool", parameters={})
        step = Step(step_number=1, action=action, response={"success": True})

        manager.update_episode_state(episode, step)

        # Should not crash and episode should remain active
        assert episode.state == EpisodeState.ACTIVE
