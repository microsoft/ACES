"""
Unit tests for TaskManager - Episode Management.

Tests episode lifecycle, RL interfaces, step execution, and reset functionality.
"""

import pytest
from unittest.mock import Mock, patch

from saber.server.tasks.task_manager import TaskManager
from saber.server.tasks.episodes import Episode, Step, Action
from saber.server.tasks.episodes.episode_manager import EpisodeState
from saber.server.tasks.exceptions import (
    TaskNotFoundException,
    EpisodeNotFoundException
)
from saber.server.execution.base import CommandResult


class TestTaskManagerEpisodes:
    """Test cases for TaskManager episode management functionality."""

    def test_start_episode_success(self, temp_tasks_file):
        """Test successful episode creation."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        episode = manager.start_episode("test_session", "malware_family_analysis")

        assert isinstance(episode, Episode)
        assert episode.task_id == "malware_family_analysis"
        assert episode.session_id == "test_session"
        assert episode.state == EpisodeState.ACTIVE  # Episodes are activated when started
        assert len(episode.steps) == 0  # No initialization step in simplified framework

        # Verify episode is tracked by episode manager
        current = manager.episode_manager.get_current_episode("test_session")
        assert current == episode

    def test_start_episode_task_not_found(self, temp_tasks_file):
        """Test episode creation with nonexistent task."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        with pytest.raises(TaskNotFoundException):
            manager.start_episode("test_session", "nonexistent_task")

    def test_start_episode_initializes_task_context(self, temp_tasks_file):
        """Test that episode is initialized with task's initial context."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        episode = manager.start_episode("test_session", "malware_family_analysis")

        # Should copy initial context from task
        task = manager.get_task("malware_family_analysis")
        assert episode.context == task.initial_context

    def test_step_success(self, temp_tasks_file, sample_command_result):
        """Test successful step execution."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # Start episode
        episode = manager.start_episode("test_session", "malware_family_analysis")

        # Create action
        action = Action(
            tool_name="docker_cli_executor",
            parameters={"command": "file sample.exe"},
            command="file sample.exe"
        )

        # Execute step
        step = manager.step("test_session", action, sample_command_result)

        assert isinstance(step, Step)
        assert step.action == action
        assert step.response["success"] == sample_command_result.success
        assert step.step_number == 0  # First step starts at 0
        assert step.done is False

    def test_step_no_active_episode(self, temp_tasks_file, sample_command_result):
        """Test step execution when no active episode exists."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        action = Action(tool_name="test_tool", parameters={})

        with pytest.raises(EpisodeNotFoundException):
            manager.step("nonexistent_session", action, sample_command_result)

    def test_reset_success(self, temp_tasks_file):
        """Test successful episode reset."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # Start initial episode
        original_episode = manager.start_episode("test_session", "malware_family_analysis")
        original_id = original_episode.episode_id

        # Add some steps to the original episode
        action = Action(tool_name="test_tool", parameters={})
        step = Step(step_number=2, action=action, response={})
        original_episode.add_step(step)

        # Reset episode
        new_episode = manager.reset("test_session", "malware_family_analysis")

        assert isinstance(new_episode, Episode)
        assert new_episode.episode_id != original_id  # Should be different episode
        assert new_episode.task_id == "malware_family_analysis"
        assert new_episode.session_id == "test_session"
        assert new_episode.state == EpisodeState.ACTIVE
        assert len(new_episode.steps) == 0  # No initialization step in simplified framework

    def test_reset_task_not_found(self, temp_tasks_file):
        """Test reset with nonexistent task."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        with pytest.raises(TaskNotFoundException):
            manager.reset("test_session", "nonexistent_task")

    def test_reset_ends_current_episode_first(self, temp_tasks_file):
        """Test that reset ends current episode before starting new one."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # Start episode
        original_episode = manager.start_episode("test_session", "malware_family_analysis")

        # Mock episode manager methods
        with patch.object(manager.episode_manager, 'get_current_episode', return_value=original_episode), \
             patch.object(manager.episode_manager, 'end_episode') as mock_end, \
             patch.object(manager, 'start_episode', return_value=Mock()) as mock_start:

            manager.reset("test_session", "malware_family_analysis")

            mock_end.assert_called_once_with("test_session", "reset")
            mock_start.assert_called_once_with("test_session", "malware_family_analysis")

    def test_reset_no_current_episode_still_starts_new(self, temp_tasks_file):
        """Test that reset works even when no current episode exists."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # Mock no current episode
        with patch.object(manager.episode_manager, 'get_current_episode', return_value=None):
            new_episode = manager.reset("test_session", "malware_family_analysis")

            assert isinstance(new_episode, Episode)
            assert new_episode.task_id == "malware_family_analysis"

    def test_end_episode_success(self, temp_tasks_file):
        """Test successful episode ending."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        episode = manager.start_episode("test_session", "malware_family_analysis")

        ended_episode = manager.end_episode("test_session", "task completed")

        assert ended_episode == episode
        assert ended_episode.completion_reason == "task completed"
        assert ended_episode.end_time is not None

        # Episode should no longer be active
        current = manager.get_current_episode("test_session")
        assert current is None

    def test_end_episode_delegates_to_episode_manager(self, temp_tasks_file):
        """Test that end_episode delegates to episode manager."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        episode = manager.start_episode("test_session", "malware_family_analysis")

        with patch.object(manager.episode_manager, 'end_episode', return_value=episode) as mock_end:
            result = manager.end_episode("test_session", "test reason")

            assert result == episode
            mock_end.assert_called_once_with("test_session", "test reason")

    def test_reset_episode_success(self, temp_tasks_file):
        """Test successful episode reset using reset_episode method."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # Start episode
        original_episode = manager.start_episode("test_session", "malware_family_analysis")

        # Reset episode
        new_episode = manager.reset_episode("test_session")

        assert new_episode != original_episode
        assert new_episode.task_id == "malware_family_analysis"  # Same task
        assert new_episode.session_id == "test_session"

    def test_reset_episode_no_current_episode(self, temp_tasks_file):
        """Test reset_episode when no current episode exists."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        with pytest.raises(EpisodeNotFoundException):
            manager.reset_episode("nonexistent_session")

    def test_reset_episode_delegates_to_episode_manager(self, temp_tasks_file):
        """Test that reset_episode delegates to episode manager."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        episode = manager.start_episode("test_session", "malware_family_analysis")
        new_episode = Mock(spec=Episode)

        with patch.object(manager.episode_manager, 'get_current_episode', return_value=episode), \
             patch.object(manager.episode_manager, 'reset_episode', return_value=new_episode) as mock_reset:

            result = manager.reset_episode("test_session")

            assert result == new_episode
            mock_reset.assert_called_once_with("test_session", "malware_family_analysis")

    def test_rl_gym_style_interfaces(self, temp_tasks_file, sample_command_result):
        """Test RL gym-style step() and reset() methods work together."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # Test reset() - should return new episode
        episode1 = manager.reset("test_session", "malware_family_analysis")
        assert isinstance(episode1, Episode)
        assert episode1.state == EpisodeState.ACTIVE

        # Test step() - should return step with observation
        action = Action(tool_name="docker_cli_executor", parameters={"command": "file sample.exe"})
        step = manager.step("test_session", action, sample_command_result)
        assert isinstance(step, Step)

        # Test another reset() - should return new episode
        episode2 = manager.reset("test_session", "malware_family_analysis")
        assert episode2 != episode1
        assert episode2.state == EpisodeState.ACTIVE  # Episodes are activated when started

    def test_multiple_sessions_episode_isolation(self, temp_tasks_file, sample_command_result):
        """Test that episodes for different sessions are properly isolated."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # Start episodes for different sessions
        episode1 = manager.start_episode("session1", "malware_family_analysis")
        episode2 = manager.start_episode("session2", "malware_family_analysis")

        assert episode1 != episode2
        assert manager.get_current_episode("session1") == episode1
        assert manager.get_current_episode("session2") == episode2

        # Execute steps for different sessions
        action = Action(tool_name="test_tool", parameters={})

        step1 = manager.step("session1", action, sample_command_result)
        step2 = manager.step("session2", action, sample_command_result)

        assert step1 != step2

        # Refresh episode references to ensure we have the latest state
        episode1 = manager.get_current_episode("session1")
        episode2 = manager.get_current_episode("session2")

        assert len(episode1.steps) == 1  # Only action step
        assert len(episode2.steps) == 1  # Only action step

        # End one episode
        manager.end_episode("session1", "completed")

        assert manager.get_current_episode("session1") is None
        assert manager.get_current_episode("session2") == episode2

    def test_episode_context_inheritance(self, temp_tasks_file):
        """Test that episodes inherit context from task initial_context."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        task = manager.get_task("malware_family_analysis")
        expected_context = task.initial_context

        episode = manager.start_episode("test_session", "malware_family_analysis")

        assert episode.context == expected_context

    def test_logging_during_episode_operations(self, temp_tasks_file, sample_command_result, caplog):
        """Test that episode operations produce appropriate logging."""
        import logging

        manager = TaskManager("malware_classification", temp_tasks_file)

        with caplog.at_level(logging.INFO):
            # Start episode
            episode = manager.start_episode("test_session", "malware_family_analysis")

            # Execute step
            action = Action(tool_name="test_tool", parameters={})
            manager.step("test_session", action, sample_command_result)

            # Reset
            manager.reset("test_session", "malware_family_analysis")

            # End episode
            manager.end_episode("test_session", "completed")

        # Check for episode-related logs
        log_messages = [record.message for record in caplog.records]

        assert any("Starting new episode" in msg for msg in log_messages)
        assert any("RL reset" in msg for msg in log_messages)
        assert any("Ending episode" in msg for msg in log_messages)

    def test_error_handling_during_episode_operations(self, temp_tasks_file, sample_command_result):
        """Test error handling during episode operations."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # Test step with no episode
        action = Action(tool_name="test_tool", parameters={})
        with pytest.raises(EpisodeNotFoundException):
            manager.step("no_episode_session", action, sample_command_result)

        # Test end episode with no episode
        with pytest.raises(EpisodeNotFoundException):
            manager.end_episode("no_episode_session", "reason")

        # Test reset episode with no episode
        with pytest.raises(EpisodeNotFoundException):
            manager.reset_episode("no_episode_session")

    def test_episode_step_numbering(self, temp_tasks_file, sample_command_result):
        """Test that episode steps are numbered correctly."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        episode = manager.start_episode("test_session", "malware_family_analysis")

        # No initial steps in simplified framework
        assert len(episode.steps) == 0

        # Execute multiple steps
        action = Action(tool_name="test_tool", parameters={})

        step1 = manager.step("test_session", action, sample_command_result)
        assert step1.step_number == 0

        step2 = manager.step("test_session", action, sample_command_result)
        assert step2.step_number == 1

        step3 = manager.step("test_session", action, sample_command_result)
        assert step3.step_number == 2

        # Episode should have all steps
        assert len(episode.steps) == 3  # Steps 0, 1, 2

    def test_task_manager_episode_manager_integration(self, temp_tasks_file):
        """Test integration between TaskManager and EpisodeManager."""
        manager = TaskManager("malware_classification", temp_tasks_file)

        # TaskManager should have its own EpisodeManager instance
        assert manager.episode_manager is not None
        assert isinstance(manager.episode_manager, type(manager.episode_manager))

        # Episode operations should work through the episode manager
        episode = manager.start_episode("test_session", "malware_family_analysis")

        # Should be tracked in episode manager
        tracked_episode = manager.episode_manager.get_current_episode("test_session")
        assert tracked_episode == episode

        # End episode should remove from episode manager
        manager.end_episode("test_session", "completed")
        assert manager.episode_manager.get_current_episode("test_session") is None
