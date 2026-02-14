"""
Additional tests for EpisodeManager advanced functionality.

Tests episode dependency management, attachment features, error handling,
and edge cases that may not be covered in the main test file.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from saber.server.base import Action, CommandResult, EpisodeState
from saber.server.episodes.episode_manager import EpisodeManager


class TestEpisodeManagerAdvanced:
    """Advanced test cases for EpisodeManager edge cases and complex functionality."""

    def test_find_available_episode_for_dependency_success(self):
        """Test finding an available episode for dependency attachment."""
        manager = EpisodeManager()

        # Start a target episode that will be depended upon
        target_episode = manager.start_episode("test_session", "target_task")

        # Find available episode for dependency
        result = manager.find_available_episode_for_dependency(
            "test_session", "target_task", "dependent_task"
        )

        assert result == target_episode.episode_id

    def test_find_available_episode_for_dependency_none_available(self):
        """Test finding dependency when no episodes available."""
        manager = EpisodeManager()

        result = manager.find_available_episode_for_dependency(
            "test_session", "nonexistent_task", "dependent_task"
        )

        assert result is None

    def test_find_available_episode_for_dependency_circular(self):
        """Test preventing circular dependencies."""
        manager = EpisodeManager()

        with pytest.raises(ValueError, match="Circular dependency detected"):
            manager.find_available_episode_for_dependency(
                "test_session", "same_task", "same_task"
            )

    @pytest.mark.asyncio
    async def test_find_available_episode_with_retry_success(self):
        """Test retry mechanism finds episode when it becomes available."""
        manager = EpisodeManager()

        async def delayed_episode_creation():
            await asyncio.sleep(0.1)
            return manager.start_episode("test_session", "target_task")

        # Start the delayed creation
        create_task = asyncio.create_task(delayed_episode_creation())

        # Try to find with retry - should eventually succeed
        result = await manager.find_available_episode_for_dependency_with_retry(
            "test_session", "target_task", "dependent_task", max_wait_seconds=1.0
        )

        episode = await create_task
        assert result == episode.episode_id

    @pytest.mark.asyncio
    async def test_find_available_episode_with_retry_timeout(self):
        """Test retry mechanism times out when episode never becomes available."""
        manager = EpisodeManager()

        result = await manager.find_available_episode_for_dependency_with_retry(
            "test_session", "nonexistent_task", "dependent_task", max_wait_seconds=0.2
        )

        assert result is None

    def test_attach_episode_to_episode(self):
        """Test attaching one episode to another."""
        manager = EpisodeManager()

        # Create target and dependent episodes
        target_episode = manager.start_episode("test_session", "target_task")
        dependent_episode = manager.start_episode("test_session", "dependent_task")

        # Attach dependent to target
        manager.attach_episode_to_episode(dependent_episode.episode_id, target_episode.episode_id)

        # Verify attachment
        assert dependent_episode.attached_to_episode_id == target_episode.episode_id

    def test_should_terminate_episode_no_config(self):
        """Test termination check when episode has no configuration."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        should_terminate, reason = manager.should_terminate_episode(episode.episode_id)

        assert should_terminate is True
        assert "configuration_error" in reason

    @pytest.mark.asyncio
    async def test_should_terminate_episode_with_config(self):
        """Test termination check with proper configuration."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        # Configure episode
        mock_task = MagicMock()
        mock_task.episode_config = {"max_steps": 10}
        mock_task.initial_files = None  # No initial files to copy
        await manager.configure_for_task(episode.episode_id, mock_task)

        should_terminate, reason = manager.should_terminate_episode(episode.episode_id)

        assert should_terminate is False
        assert reason == ""

    def test_should_terminate_episode_not_found(self):
        """Test termination check for non-existent episode."""
        manager = EpisodeManager()

        should_terminate, reason = manager.should_terminate_episode("nonexistent")

        assert should_terminate is True
        assert reason == "episode_not_found"

    @pytest.mark.asyncio
    async def test_should_terminate_episode_completed(self):
        """Test termination check for already completed episode."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        # Complete the episode
        await manager.end_episode(episode.episode_id, "test_completion")

        should_terminate, reason = manager.should_terminate_episode(episode.episode_id)

        assert should_terminate is True
        assert reason == "test_completion"

    @pytest.mark.asyncio
    async def test_get_session_episodes_include_completed(self):
        """Test getting session episodes including completed ones."""
        manager = EpisodeManager()

        # Create and complete an episode
        episode1 = manager.start_episode("test_session", "task1")
        await manager.end_episode(episode1.episode_id, "completed")

        # Create an active episode
        episode2 = manager.start_episode("test_session", "task2")

        # Get all episodes (active + completed)
        all_episodes = manager.get_session_episodes("test_session", include_completed=True)
        episode_ids = [ep.episode_id for ep in all_episodes]

        assert len(all_episodes) == 2
        assert episode1.episode_id in episode_ids
        assert episode2.episode_id in episode_ids

    @pytest.mark.asyncio
    async def test_get_session_episodes_active_only(self):
        """Test getting only active session episodes."""
        manager = EpisodeManager()

        # Create and complete an episode
        episode1 = manager.start_episode("test_session", "task1")
        await manager.end_episode(episode1.episode_id, "completed")

        # Create an active episode
        episode2 = manager.start_episode("test_session", "task2")

        # Get only active episodes (default behavior)
        active_episodes = manager.get_session_episodes("test_session", include_completed=False)
        episode_ids = [ep.episode_id for ep in active_episodes]

        assert len(active_episodes) == 1
        assert episode2.episode_id in episode_ids
        assert episode1.episode_id not in episode_ids

    def test_extract_parameters_from_action(self):
        """Test parameter extraction from actions."""
        manager = EpisodeManager()

        # Test with docker_cli_executor command
        docker_action = Action(tool_name="docker_cli_executor", parameters={"command": "ls -la /home"})
        result = manager._extract_parameters_from_action(docker_action)
        assert result == "ls -la /home"

        # Test with non-docker action (returns tool name as fallback)
        other_action = Action(tool_name="other_tool", parameters={"key": "value"})
        result = manager._extract_parameters_from_action(other_action)
        assert result == "other_tool"

    def test_get_episode_progress_info(self):
        """Test getting episode progress information."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        # Add some steps
        action = Action(tool_name="test_tool", parameters={})
        response = CommandResult(exit_code=0, stdout="test", stderr="", execution_time=0.1)
        step = manager.create_step(episode, action, response)
        episode.add_step(step)

        progress_info = manager._get_episode_progress_info(episode)

        assert progress_info["episode_id"] == episode.episode_id
        assert progress_info["task_id"] == "test_task"
        assert progress_info["state"] == "active"
        assert progress_info["total_steps"] == 1
        assert "start_time" in progress_info
        assert "end_time" in progress_info

    def test_remove_episode_on_error(self):
        """Test removing episode when an error occurs."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        # Verify episode exists
        assert episode.episode_id in manager.episodes

        # Remove on error
        test_error = Exception("Test error occurred")
        manager.remove_episode_on_error(episode.episode_id, test_error)

        # Verify episode is moved to completed with failed state
        assert episode.episode_id not in manager.episodes
        assert episode.episode_id in manager.completed_episodes
        assert episode.state == EpisodeState.FAILED
        assert "error" in episode.completion_reason

    def test_remove_episode_on_error_not_found(self):
        """Test removing non-existent episode on error doesn't crash."""
        manager = EpisodeManager()

        # Should not raise an exception
        test_error = Exception("Test error")
        manager.remove_episode_on_error("nonexistent_episode", test_error)

        # Manager should remain in clean state
        assert len(manager.episodes) == 0
        assert len(manager.completed_episodes) == 0

    @pytest.mark.asyncio
    async def test_end_episode_with_result_submission(self):
        """Test ending episode with a result creates submission step."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        # End with result
        result_data = "flag{test_flag_found}"
        ended_episode = await manager.end_episode(episode.episode_id, "completed", result_data)

        # Check that a submission step was created
        assert len(ended_episode.steps) == 1
        final_step = ended_episode.steps[0]
        assert final_step.action.tool_name == "submission"
        assert final_step.action.parameters["result"] == result_data
        # Note: done is False for submission steps as they are created by create_step with default done=False

    def test_create_step_with_metadata(self):
        """Test step creation preserves command result metadata."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        action = Action(tool_name="test_tool", parameters={"param": "value"})
        response = CommandResult(
            exit_code=0,
            stdout="output text",
            stderr="error text",
            execution_time=1.5,
            metadata={"custom_key": "custom_value", "analysis": "complete"}
        )

        step = manager.create_step(episode, action, response)

        assert step.response["exit_code"] == 0
        assert step.response["stdout"] == "output text"
        assert step.response["stderr"] == "error text"
        assert step.response["execution_time"] == 1.5
        assert step.response["metadata"]["custom_key"] == "custom_value"
        assert step.response["metadata"]["analysis"] == "complete"

    @pytest.mark.asyncio
    async def test_configure_for_task_stores_config(self):
        """Test that configure_for_task properly stores episode configuration."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        # Create task with configuration
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.episode_config = {
            "max_steps": 15,
            "timeout_minutes": 30,
            "allowed_tools": ["shell", "python"]
        }
        mock_task.initial_files = None  # No initial files to copy

        await manager.configure_for_task(episode.episode_id, mock_task)

        # Verify configuration is stored
        assert episode.episode_id in manager.episode_configs
        stored_config = manager.episode_configs[episode.episode_id]
        assert stored_config["max_steps"] == 15
        # timeout_minutes is not stored as is - check for episode_timeout_minutes which is the actual key
        assert stored_config.get("episode_timeout_minutes") is not None
        assert stored_config.get("allowed_tools") == ["shell", "python"]

    @pytest.mark.asyncio
    async def test_step_with_command_failure(self):
        """Test step execution with command failure."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        # Configure episode
        mock_task = MagicMock()
        mock_task.episode_config = {"max_steps": 10}
        await manager.configure_for_task(episode.episode_id, mock_task)

        action = Action(tool_name="failing_tool", parameters={"will_fail": True})
        command_result = CommandResult(
            exit_code=1,  # Non-zero indicates failure
            stdout="",
            stderr="Command failed with error",
            execution_time=0.2,
            metadata={"error": "permission_denied"}
        )

        step_result = manager.step(episode.episode_id, action, command_result)

        # Verify failure is properly recorded
        step = step_result.step
        assert step.response["exit_code"] == 1
        assert step.response["stderr"] == "Command failed with error"
        assert step.response["metadata"]["error"] == "permission_denied"
        assert step.done is False  # Command failed, not done
