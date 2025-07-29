"""
Unit tests for EpisodeManager.

Tests episode lifecycle management, RL interfaces, and state transitions.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from saber.server.tasks.episodes.episode_manager import EpisodeManager, EpisodeState
from saber.server.tasks.episodes.episode import Episode, Step, Action
from saber.server.tasks.core.task import Task
from saber.server.tasks.core.subtask import SubTask
from saber.server.tasks.exceptions import EpisodeNotFoundException
from saber.server.tools.base import ToolResult


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
        assert episode.metadata["created_at"] is not None

    def test_start_episode_replaces_existing(self):
        """Test that starting a new episode replaces existing one for same session."""
        manager = EpisodeManager()

        # Start first episode
        episode1 = manager.start_episode(
            session_id="test_session",
            task_id="task1"
        )

        # Start second episode for same session
        episode2 = manager.start_episode(
            session_id="test_session",
            task_id="task2"
        )

        assert manager.active_episodes["test_session"] == episode2
        assert episode1 != episode2
        assert episode2.task_id == "task2"

    def test_get_current_episode_exists(self):
        """Test getting current episode when it exists."""
        manager = EpisodeManager()

        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task"
        )

        current = manager.get_current_episode("test_session")
        assert current == episode

    def test_get_current_episode_not_exists(self):
        """Test getting current episode when it doesn't exist."""
        manager = EpisodeManager()

        current = manager.get_current_episode("nonexistent_session")
        assert current is None

    def test_step_success(self, sample_tool_result):
        """Test successful episode step execution."""
        manager = EpisodeManager()

        # Start episode
        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task"
        )

        # Create action
        action = Action(
            tool_name="docker_cli_executor",
            parameters={"command": "file sample.exe"},
            command="file sample.exe"
        )

        # Execute step
        step = manager.step(
            session_id="test_session",
            action=action,
            tool_result=sample_tool_result,
            current_objective="Test objective"
        )

        assert isinstance(step, Step)
        assert step.action == action
        assert step.response["success"] == sample_tool_result.success
        assert step.response["data"] == sample_tool_result.data
        assert step.step_number == 0
        assert len(episode.steps) == 1

    def test_step_no_active_episode(self, sample_tool_result):
        """Test step execution when no active episode exists."""
        manager = EpisodeManager()

        action = Action(tool_name="test_tool", parameters={})

        with pytest.raises(EpisodeNotFoundException) as exc_info:
            manager.step(
                session_id="nonexistent_session",
                action=action,
                tool_result=sample_tool_result
            )

        assert "nonexistent_session" in str(exc_info.value)

    def test_step_command_extraction(self, sample_tool_result):
        """Test command extraction during step execution."""
        manager = EpisodeManager()

        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task"
        )

        # Test docker_cli_executor command extraction
        action = Action(
            tool_name="docker_cli_executor",
            parameters={"command": "strings sample.exe"}
        )

        step = manager.step(
            session_id="test_session",
            action=action,
            tool_result=sample_tool_result
        )

        assert action.command == "strings sample.exe"

    def test_step_other_tool_command_extraction(self, sample_tool_result):
        """Test command extraction for non-CLI tools."""
        manager = EpisodeManager()

        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task"
        )

        action = Action(
            tool_name="other_tool",
            parameters={"param": "value"}
        )

        step = manager.step(
            session_id="test_session",
            action=action,
            tool_result=sample_tool_result
        )

        assert action.command == "other_tool"

    def test_step_episode_completion(self, sample_tool_result):
        """Test that steps are created with done=False since EpisodeManager no longer handles completion."""
        manager = EpisodeManager()

        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task"
        )

        action = Action(tool_name="test_tool", parameters={})

        step = manager.step(
            session_id="test_session",
            action=action,
            tool_result=sample_tool_result
        )

        # EpisodeManager no longer determines completion, so done should always be False
        assert step.done is False

    def test_end_episode_success(self):
        """Test ending an episode successfully."""
        manager = EpisodeManager()

        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task"
        )

        # Add some steps to the episode
        action = Action(tool_name="test_tool", parameters={})
        step = Step(step_number=1, action=action, response={})
        episode.add_step(step)

        ended_episode = manager.end_episode("test_session", "task completed successfully")

        assert ended_episode == episode
        assert ended_episode.state == EpisodeState.COMPLETED
        assert ended_episode.completion_reason == "task completed successfully"
        assert ended_episode.end_time is not None
        assert "test_session" not in manager.active_episodes

    def test_end_episode_failure(self):
        """Test ending an episode with failure."""
        manager = EpisodeManager()

        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task"
        )

        ended_episode = manager.end_episode("test_session", "task failed due to error")

        assert ended_episode.state == EpisodeState.FAILED
        assert ended_episode.completion_reason == "task failed due to error"

    def test_end_episode_not_exists(self):
        """Test ending an episode that doesn't exist."""
        manager = EpisodeManager()

        with pytest.raises(EpisodeNotFoundException):
            manager.end_episode("nonexistent_session", "some reason")

    def test_reset_episode_success(self):
        """Test resetting an episode successfully."""
        manager = EpisodeManager()

        # Start original episode
        original_episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task",
            initial_context={"key": "value"}
        )

        # Add some steps
        action = Action(tool_name="test_tool", parameters={})
        step = Step(step_number=1, action=action, response={})
        original_episode.add_step(step)

        # Reset episode
        new_episode = manager.reset_episode("test_session", "test_task")

        assert new_episode != original_episode
        assert new_episode.task_id == "test_task"
        assert new_episode.session_id == "test_session"
        assert new_episode.context == {"key": "value"}  # Context should be copied
        assert len(new_episode.steps) == 0  # Steps should be fresh
        assert new_episode.state == EpisodeState.ACTIVE
        assert manager.active_episodes["test_session"] == new_episode

    def test_reset_episode_not_exists(self):
        """Test resetting an episode that doesn't exist."""
        manager = EpisodeManager()

        with pytest.raises(EpisodeNotFoundException):
            manager.reset_episode("nonexistent_session", "test_task")

    def test_get_episode_state_exists(self):
        """Test getting episode state when episode exists."""
        manager = EpisodeManager()

        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task"
        )

        state = manager.get_episode_state("test_session")
        assert state == EpisodeState.ACTIVE

    def test_get_episode_state_not_exists(self):
        """Test getting episode state when episode doesn't exist."""
        manager = EpisodeManager()

        state = manager.get_episode_state("nonexistent_session")
        assert state is None

    def test_create_step(self, sample_tool_result):
        """Test step creation from action and tool result."""
        manager = EpisodeManager()

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        action = Action(
            tool_name="docker_cli_executor",
            parameters={"command": "file sample.exe"}
        )

        step = manager.create_step(episode, action, sample_tool_result)

        assert step.step_number == 0  # First step
        assert step.action == action
        assert step.response["success"] == sample_tool_result.success
        assert step.response["data"] == sample_tool_result.data
        assert step.response["error"] == sample_tool_result.error
        assert step.response["execution_time"] == sample_tool_result.execution_time
        assert step.response["metadata"] == sample_tool_result.metadata
        assert step.context_snapshot == episode.context
        assert action.command == "file sample.exe"

    def test_create_step_with_existing_steps(self, sample_tool_result):
        """Test step creation when episode already has steps."""
        manager = EpisodeManager()

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Add existing step
        existing_action = Action(tool_name="existing", parameters={})
        existing_step = Step(step_number=0, action=existing_action, response={})
        episode.steps = [existing_step]

        action = Action(tool_name="new_tool", parameters={})
        step = manager.create_step(episode, action, sample_tool_result)

        assert step.step_number == 1  # Second step

    def test_update_episode_state(self, sample_tool_result):
        """Test updating episode state after a step."""
        manager = EpisodeManager()

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE,
            context={"original": "context"}
        )

        action = Action(tool_name="test_tool", parameters={})
        step = Step(
            step_number=1,
            action=action,
            response={},
            context_snapshot={"updated": "context", "new_key": "new_value"},
            done=False
        )

        manager.update_episode_state(episode, step)

        # Context should be updated
        assert episode.context["updated"] == "context"
        assert episode.context["new_key"] == "new_value"
        assert episode.state == EpisodeState.ACTIVE  # Should remain active

    def test_update_episode_state_done(self, sample_tool_result):
        """Test updating episode state when step indicates completion."""
        manager = EpisodeManager()

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        action = Action(tool_name="test_tool", parameters={})
        step = Step(
            step_number=1,
            action=action,
            response={},
            done=True
        )

        manager.update_episode_state(episode, step)

        assert episode.state == EpisodeState.COMPLETED

    def test_extract_command_from_action_docker_cli(self):
        """Test command extraction from DockerCLIExecutor action."""
        manager = EpisodeManager()

        action = Action(
            tool_name="docker_cli_executor",
            parameters={"command": "strings -a sample.exe"}
        )

        command = manager._extract_command_from_action(action)
        assert command == "strings -a sample.exe"

    def test_extract_command_from_action_empty_command(self):
        """Test command extraction when command parameter is empty."""
        manager = EpisodeManager()

        action = Action(
            tool_name="docker_cli_executor",
            parameters={"command": ""}
        )

        command = manager._extract_command_from_action(action)
        assert command is None

    def test_extract_command_from_action_no_command_param(self):
        """Test command extraction when no command parameter exists."""
        manager = EpisodeManager()

        action = Action(
            tool_name="docker_cli_executor",
            parameters={"other_param": "value"}
        )

        command = manager._extract_command_from_action(action)
        assert command is None

    def test_extract_command_from_action_other_tool(self):
        """Test command extraction from non-CLI tool."""
        manager = EpisodeManager()

        action = Action(
            tool_name="other_analysis_tool",
            parameters={"file_path": "/data/sample.exe"}
        )

        command = manager._extract_command_from_action(action)
        assert command == "other_analysis_tool"

    def test_get_episode_progress_info(self):
        """Test getting episode progress information."""
        manager = EpisodeManager()

        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 10, 5, 0)

        episode = Episode(
            episode_id="test_episode_id",
            task_id="test_task",
            session_id="test_session",
            start_time=start_time,
            end_time=end_time,
            state=EpisodeState.COMPLETED
        )

        # Add some steps to simulate progress
        action = Action(tool_name="test_tool", parameters={})
        step1 = Step(
            step_number=1,
            action=action,
            response={},
            current_subtask="subtask1",
            completed_subtasks={"subtask0"},
            in_progress_subtasks={"subtask1"},
            not_visited_subtasks={"subtask2", "subtask3"}
        )
        step2 = Step(
            step_number=2,
            action=action,
            response={},
            current_subtask="subtask2",
            completed_subtasks={"subtask0", "subtask1"},
            in_progress_subtasks={"subtask2"},
            not_visited_subtasks={"subtask3"}
        )
        episode.steps = [step1, step2]

        progress_info = manager._get_episode_progress_info(episode)

        assert progress_info["episode_id"] == "test_episode_id"
        assert progress_info["task_id"] == "test_task"
        assert progress_info["state"] == "completed"
        assert progress_info["total_steps"] == 2
        assert progress_info["current_subtask"] == "subtask2"
        assert set(progress_info["completed_subtasks"]) == {"subtask0", "subtask1"}
        assert set(progress_info["in_progress_subtasks"]) == {"subtask2"}
        assert set(progress_info["not_visited_subtasks"]) == {"subtask3"}
        assert progress_info["duration"] == 300.0  # 5 minutes
        assert progress_info["start_time"] == start_time.isoformat()
        assert progress_info["end_time"] == end_time.isoformat()

    def test_cleanup_session_with_active_episode(self):
        """Test cleaning up session with active episode."""
        manager = EpisodeManager()

        episode = manager.start_episode(
            session_id="test_session",
            task_id="test_task"
        )

        manager.cleanup_session("test_session")

        assert "test_session" not in manager.active_episodes

    def test_cleanup_session_no_active_episode(self):
        """Test cleaning up session with no active episode."""
        manager = EpisodeManager()

        # Should not raise any exception
        manager.cleanup_session("nonexistent_session")

        assert manager.active_episodes == {}

    def test_multiple_sessions_isolation(self, sample_tool_result):
        """Test that multiple sessions are properly isolated."""
        manager = EpisodeManager()

        # Start episodes for different sessions
        episode1 = manager.start_episode("session1", "task1")
        episode2 = manager.start_episode("session2", "task2")

        assert len(manager.active_episodes) == 2
        assert manager.get_current_episode("session1") == episode1
        assert manager.get_current_episode("session2") == episode2

        # End one episode
        manager.end_episode("session1", "completed")

        assert len(manager.active_episodes) == 1
        assert manager.get_current_episode("session1") is None
        assert manager.get_current_episode("session2") == episode2
