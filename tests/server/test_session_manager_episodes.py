"""
Unit tests for SessionManager episode management functionality.

Tests episode creation and task management integration.
Tool execution is tested separately for MCP API.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.base import Action, CommandResult, Episode, Step
from saber.server.session_manager import SessionManager


class TestSessionManagerEpisodes:
    """Test SessionManager episode management functionality."""

    @pytest.fixture
    def mock_episode(self):
        """Mock episode for testing."""
        episode = MagicMock(spec=Episode)
        episode.episode_id = "episode_123"
        episode.task_id = "task_456"
        episode.state = MagicMock()
        episode.state.value = "active"
        episode.steps = []
        return episode

    @pytest.fixture
    def mock_step(self):
        """Mock step result for testing."""
        from datetime import datetime

        step = MagicMock(spec=Step)
        step.step_number = 1
        step.done = False
        step.timestamp = datetime.utcnow()
        # Mock action attribute properly
        mock_action = MagicMock()
        mock_action.tool_name = "test_tool"
        mock_action.arguments = "test command"
        mock_action.parameters = {}
        step.action = mock_action
        step.response = {"output": "test output"}
        return step

    @pytest.fixture
    def mock_task(self):
        """Mock task for testing."""
        task = MagicMock()
        task.task_id = "task_456"
        task.title = "Test Task"
        task.description = "Test task description"

        # Mock the to_dict method to return expected dictionary
        task.to_dict.return_value = {
            "task_id": "task_456",
            "title": "Test Task",
            "description": "Test task description",
            "state": "active",
        }
        return task

    @pytest.fixture
    def session_manager_with_session(self):
        """Create SessionManager with mocked dependencies and a session."""
        mock_task_manager = MagicMock()
        mock_task_manager.get_task_prompt = MagicMock(return_value="test_prompt")

        mock_execution_manager = MagicMock()
        mock_execution_manager.step = AsyncMock()
        mock_execution_manager.configure_for_task = MagicMock()
        mock_policy_manager = MagicMock()
        mock_policy_manager.get_policy = AsyncMock()
        mock_policy_manager.set_episode_policy = MagicMock()  # Episode-first architecture
        mock_evaluation_manager = MagicMock()
        mock_evaluation_manager.log_session_start = AsyncMock()
        mock_evaluation_manager.log_episode_start = AsyncMock()
        mock_evaluation_manager.log_episode_end = AsyncMock()
        mock_evaluation_manager.log_action = AsyncMock()

        mock_episode_manager = MagicMock()
        mock_episode_manager.start_episode = MagicMock()
        mock_episode_manager.end_episode = MagicMock()
        mock_episode_manager.get_episode = MagicMock()
        mock_episode_manager.configure_for_task = AsyncMock()

        with (
            patch("saber.server.session_manager.BenchmarkManager", return_value=mock_task_manager),
            patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
            patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
            patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
            patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
        ):

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8002)
            return manager

    @pytest.mark.asyncio
    async def test_start_episode(self, session_manager_with_session, mock_episode):
        """Test starting an episode."""
        manager = session_manager_with_session

        # Create a session first
        session = await manager.create_session("test_client")
        session_id = session.session_id
        task_id = "task_456"

        # Mock task with proper initial_context
        mock_task = MagicMock()
        mock_task.initial_context = {"initial_data": "test"}
        mock_task.depends_on_task_id = None  # No dependencies
        manager.benchmark_manager.get_task.return_value = mock_task

        # Mock prompt generator to return expected prompts
        mock_prompts = {"instruction": "test_prompt", "assistant": "test_assistant", "submit": "test_submit"}
        manager.benchmark_manager.prompt_generator.render_agent_prompts_for_task.return_value = mock_prompts

        # Mock episode manager to return episode
        manager.episode_manager.start_episode.return_value = mock_episode

        # Start episode
        episode = await manager.start_episode(session_id, task_id)

        assert episode == mock_episode
        assert mock_episode.episode_id in session.active_episode_ids  # New multi-episode model

        # Verify task manager was called to get task
        manager.benchmark_manager.get_task.assert_called_once_with(task_id)

        # Verify episode manager was called
        manager.episode_manager.start_episode.assert_called_once_with(
            session_id=session_id, task_id=task_id, initial_context={"initial_data": "test"}, task=mock_task
        )

        # Verify evaluation manager was called
        manager.evaluation_manager.log_episode_start.assert_called_once_with(
            session_id, mock_episode.episode_id, task_id
        )

        # Verify PolicyManager set_episode_policy was called (episode-first architecture)
        manager.policy_manager.set_episode_policy.assert_called_once_with(mock_episode.episode_id, session_id, "test_prompt")

        # Verify EpisodeManager configure_for_task was called
        manager.episode_manager.configure_for_task.assert_called_once_with(mock_episode.episode_id, mock_task)

    @pytest.mark.asyncio
    async def test_start_episode_invalid_session(self, session_manager_with_session):
        """Test starting episode with invalid session."""
        manager = session_manager_with_session

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await manager.start_episode("invalid_session", "task_123")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_step_execution_success(self, session_manager_with_session, mock_episode, mock_step):
        """Test successful step execution."""
        manager = session_manager_with_session

        # Create session and start episode
        session = await manager.create_session("test_client")
        session_id = session.session_id
        # Add episode to session's active episodes (replacing current_episode_id)
        session.add_active_episode("episode_123")

        # Mock episode manager to return a valid episode
        mock_episode_obj = MagicMock()
        mock_episode_obj.episode_id = "episode_123"
        mock_episode_obj.add_step = MagicMock()
        manager.episode_manager.get_episode.return_value = mock_episode_obj

        # Mock execution and episode manager responses
        command_result = CommandResult(exit_code=0, stdout="test output", stderr="", execution_time=0.1)
        manager.execution_manager.step.return_value = command_result

        # Create StepResult mock that matches the new return type
        from saber.server.episodes.episode_manager import StepResult
        step_result = StepResult(
            step=mock_step,
            should_terminate=False,
            termination_reason=None
        )
        manager.episode_manager.step.return_value = step_result

        # Execute command
        action = Action(tool_name="bash", parameters={"arguments": "file test.txt", "param": "value"})
        response = await manager.execute_action(session_id, "episode_123", action)

        assert isinstance(response, CommandResult)
        assert response.success is True
        assert response.stdout == "test output"
        assert response.exit_code == 0

        # Verify execution manager was called with Action object
        call_args = manager.execution_manager.step.call_args
        assert call_args is not None
        action_arg = call_args[0][0]  # First positional argument
        assert isinstance(action_arg, Action)
        assert action_arg.parameters["arguments"] == "file test.txt"
        assert action_arg.parameters == {"arguments": "file test.txt", "param": "value"}
        assert action_arg.tool_name == "bash"

        # Verify episode manager was called with action
        call_args = manager.episode_manager.step.call_args
        assert call_args[0][0] == "episode_123"  # episode_id (new signature)
        assert isinstance(call_args[0][1], Action)  # action
        assert call_args[0][2] == command_result  # command_result

        # Verify evaluation manager was called
        manager.evaluation_manager.log_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_execution_no_active_episode(self, session_manager_with_session):
        """Test step execution without active episode."""
        manager = session_manager_with_session

        # Create session without episode
        session = await manager.create_session("test_client")
        session_id = session.session_id

        action = Action(tool_name="bash", parameters={"arguments": "file test.txt"})
        # New API requires explicit episode_id - test should fail fast with missing parameter
        with pytest.raises(TypeError):
            await manager.execute_action(session_id, action)

    @pytest.mark.asyncio
    async def test_step_execution_with_completion(self, session_manager_with_session, mock_step):
        """Test step execution that completes episode."""
        manager = session_manager_with_session

        # Create session and start episode
        session = await manager.create_session("test_client")
        session_id = session.session_id
        # Add episode to session's active episodes (replacing current_episode_id)
        session.add_active_episode("episode_123")

        # Mock episode manager to return a valid episode
        mock_episode_obj = MagicMock()
        mock_episode_obj.episode_id = "episode_123"
        mock_episode_obj.add_step = MagicMock()
        manager.episode_manager.get_episode.return_value = mock_episode_obj

        # Mock step as completed
        mock_step.done = True

        # Mock execution and episode manager responses
        command_result = CommandResult(exit_code=0, stdout="completed", stderr="", execution_time=0.1)
        manager.execution_manager.step.return_value = command_result

        # Create StepResult mock that indicates completion
        from saber.server.episodes.episode_manager import StepResult
        step_result = StepResult(
            step=mock_step,
            should_terminate=False,
            termination_reason=None
        )
        manager.episode_manager.step.return_value = step_result

        # Execute command
        action = Action(tool_name="bash", parameters={"arguments": "final command"})
        response = await manager.execute_action(session_id, "episode_123", action)

        assert response.success is True
        assert "episode_123" not in session.active_episode_ids  # Episode should be cleared

        # Verify episode was ended - should now use episode_id
        manager.episode_manager.end_episode.assert_called_once_with("episode_123", "completed")
        manager.evaluation_manager.log_episode_end.assert_called_once_with(session_id, "completed")

    @pytest.mark.asyncio
    async def test_step_execution_failure(self, session_manager_with_session):
        """Test step execution with failure."""
        manager = session_manager_with_session

        # Create session with episode
        session = await manager.create_session("test_client")
        session_id = session.session_id
        # Add episode to session's active episodes (replacing current_episode_id)
        session.add_active_episode("episode_123")

        # Mock execution manager to raise exception
        manager.execution_manager.step.side_effect = Exception("Execution failed")

        # Execute command
        action = Action(tool_name="bash", parameters={"arguments": "bad command"})
        response = await manager.execute_action(session_id, "episode_123", action)

        assert response.success is False
        assert response.error == "Execution failed"

    @pytest.mark.asyncio
    async def test_get_current_task(self, session_manager_with_session, mock_episode, mock_task):
        """Test getting current task information."""
        manager = session_manager_with_session

        # Create session with episode
        session = await manager.create_session("test_client")
        session_id = session.session_id
        # Add episode to session's active episodes (replacing current_episode_id)
        session.add_active_episode("episode_123")

        # Mock episode manager and task manager responses
        manager.episode_manager.get_episode_by_id.return_value = mock_episode
        manager.benchmark_manager.get_task.return_value = mock_task

        # Get current task - now requires episode_id
        task = await manager.get_current_task(session_id, "episode_123")
        task_info = task.to_dict()

        assert task_info["task_id"] == "task_456"
        assert task_info["title"] == "Test Task"
        assert task_info["description"] == "Test task description"
        # Note: episode_id would be added by the REST API endpoint, not the core method
        assert task_info["state"] == "active"

        # Verify episode manager and task manager were called
        manager.episode_manager.get_episode_by_id.assert_called_once_with("episode_123")
        manager.benchmark_manager.get_task.assert_called_once_with("task_456")

    @pytest.mark.asyncio
    async def test_get_current_task_no_episode(self, session_manager_with_session):
        """Test getting current task without active episode."""
        manager = session_manager_with_session

        # Create session without episode
        session = await manager.create_session("test_client")
        session_id = session.session_id

        from fastapi import HTTPException

        # New API requires explicit episode_id - test should fail fast with missing parameter
        with pytest.raises(TypeError):
            await manager.get_current_task(session_id)

    @pytest.mark.asyncio
    async def test_get_current_task_episode_not_found(self, session_manager_with_session):
        """Test getting current task when episode not found."""
        manager = session_manager_with_session

        # Create session with episode ID but no actual episode
        session = await manager.create_session("test_client")
        session_id = session.session_id
        # Add non-existent episode to session's active episodes (replacing current_episode_id)
        session.add_active_episode("episode_123")

        # Mock episode manager to return None
        manager.episode_manager.get_episode_by_id.return_value = None

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await manager.get_current_task(session_id, "episode_123")

        assert exc_info.value.status_code == 400
        assert "Episode episode_123 not found" in str(exc_info.value.detail)
