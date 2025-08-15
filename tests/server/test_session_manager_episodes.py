"""
Unit tests for SessionManager episode management functionality.

Tests episode creation and task management integration.
Tool execution is tested separately for MCP API.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saber.server.session_manager import SessionManager
from saber.server.base import Action, Step, CommandResult, Episode


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
        mock_action.command = "test command"
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
        return task

    @pytest.fixture
    def session_manager_with_session(self):
        """Create SessionManager with mocked dependencies and a session."""
        mock_task_manager = MagicMock()
        mock_execution_manager = MagicMock()
        mock_execution_manager.step = AsyncMock()
        mock_policy_manager = MagicMock()
        mock_policy_manager.get_policy = AsyncMock()
        mock_evaluation_manager = MagicMock()
        mock_evaluation_manager.log_session_start = AsyncMock()
        mock_evaluation_manager.log_episode_start = AsyncMock()
        mock_evaluation_manager.log_episode_end = AsyncMock()
        mock_evaluation_manager.log_action = AsyncMock()

        mock_episode_manager = MagicMock()
        mock_episode_manager.start_episode = MagicMock()
        mock_episode_manager.end_episode = MagicMock()
        mock_episode_manager.get_episode = MagicMock()

        with patch('saber.server.session_manager.TaskManager', return_value=mock_task_manager), \
             patch('saber.server.session_manager.ExecutionManager', return_value=mock_execution_manager), \
             patch('saber.server.session_manager.PolicyManager', return_value=mock_policy_manager), \
             patch('saber.server.session_manager.EvaluationManager', return_value=mock_evaluation_manager), \
             patch('saber.server.session_manager.EpisodeManager', return_value=mock_episode_manager):

            manager = SessionManager(
                domain_name="test_domain",
                config_dir="/tmp",
                host="127.0.0.1",
                port=8002
            )
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
        manager.task_manager.get_task.return_value = mock_task

        # Mock episode manager to return episode
        manager.episode_manager.start_episode.return_value = mock_episode

        # Start episode
        episode = await manager.start_episode(session_id, task_id)

        assert episode == mock_episode
        assert session.current_episode_id == mock_episode.episode_id

        # Verify task manager was called to get task
        manager.task_manager.get_task.assert_called_once_with(task_id)

        # Verify episode manager was called
        manager.episode_manager.start_episode.assert_called_once_with(
            session_id=session_id, task_id=task_id, initial_context={"initial_data": "test"}
        )

        # Verify evaluation manager was called
        manager.evaluation_manager.log_episode_start.assert_called_once_with(
            session_id, mock_episode.episode_id, task_id
        )

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
        session.current_episode_id = "episode_123"

        # Mock episode manager to return a valid episode
        mock_episode_obj = MagicMock()
        mock_episode_obj.episode_id = "episode_123"
        mock_episode_obj.add_step = MagicMock()
        manager.episode_manager.get_episode.return_value = mock_episode_obj

        # Mock execution and episode manager responses
        command_result = CommandResult(exit_code=0, stdout="test output", stderr="", execution_time=0.1)
        manager.execution_manager.step.return_value = command_result
        manager.episode_manager.step.return_value = mock_step

        # Execute command
        action = Action(tool_name="cli", command="file test.txt", parameters={"param": "value"})
        response = await manager.execute_command(session_id, action)

        assert isinstance(response, CommandResult)
        assert response.success is True
        assert response.stdout == "test output"
        assert response.exit_code == 0

        # Verify execution manager was called with Action object
        call_args = manager.execution_manager.step.call_args
        assert call_args is not None
        action_arg = call_args[0][0]  # First positional argument
        assert isinstance(action_arg, Action)
        assert action_arg.command == "file test.txt"
        assert action_arg.parameters == {"param": "value"}
        assert action_arg.tool_name == "cli"

        # Verify episode manager was called with action
        call_args = manager.episode_manager.step.call_args
        assert call_args[0][0] == session_id  # session_id
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

        action = Action(tool_name="cli", command="file test.txt", parameters={})
        result = await manager.execute_command(session_id, action)

        assert isinstance(result, CommandResult)
        assert not result.success
        assert "No active episode" in result.error

    @pytest.mark.asyncio
    async def test_step_execution_with_completion(self, session_manager_with_session, mock_step):
        """Test step execution that completes episode."""
        manager = session_manager_with_session

        # Create session and start episode
        session = await manager.create_session("test_client")
        session_id = session.session_id
        session.current_episode_id = "episode_123"

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
        manager.episode_manager.step.return_value = mock_step

        # Execute command
        action = Action(tool_name="cli", command="final command", parameters={})
        response = await manager.execute_command(session_id, action)

        assert response.success is True
        assert session.current_episode_id is None  # Episode should be cleared

        # Verify episode was ended
        manager.episode_manager.end_episode.assert_called_once_with(session_id, "completed")
        manager.evaluation_manager.log_episode_end.assert_called_once_with(session_id, "completed")

    @pytest.mark.asyncio
    async def test_step_execution_failure(self, session_manager_with_session):
        """Test step execution with failure."""
        manager = session_manager_with_session

        # Create session with episode
        session = await manager.create_session("test_client")
        session_id = session.session_id
        session.current_episode_id = "episode_123"

        # Mock execution manager to raise exception
        manager.execution_manager.step.side_effect = Exception("Execution failed")

        # Execute command
        action = Action(tool_name="cli", command="bad command", parameters={})
        response = await manager.execute_command(session_id, action)

        assert response.success is False
        assert response.error == "Execution failed"

    @pytest.mark.asyncio
    async def test_get_current_task(self, session_manager_with_session, mock_episode, mock_task):
        """Test getting current task information."""
        manager = session_manager_with_session

        # Create session with episode
        session = await manager.create_session("test_client")
        session_id = session.session_id
        session.current_episode_id = "episode_123"

        # Mock episode manager and task manager responses
        manager.episode_manager.get_current_episode.return_value = mock_episode
        manager.task_manager.get_task.return_value = mock_task

        # Get current task
        task_info = await manager.get_current_task(session_id)

        assert task_info["task_id"] == "task_456"
        assert task_info["title"] == "Test Task"
        assert task_info["description"] == "Test task description"
        assert task_info["episode_id"] == "episode_123"
        assert task_info["state"] == "active"

        # Verify episode manager and task manager were called
        manager.episode_manager.get_current_episode.assert_called_once_with(session_id)
        manager.task_manager.get_task.assert_called_once_with("task_456")

    @pytest.mark.asyncio
    async def test_get_current_task_no_episode(self, session_manager_with_session):
        """Test getting current task without active episode."""
        manager = session_manager_with_session

        # Create session without episode
        session = await manager.create_session("test_client")
        session_id = session.session_id

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await manager.get_current_task(session_id)

        assert exc_info.value.status_code == 400
        assert "No active episode" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_current_task_episode_not_found(self, session_manager_with_session):
        """Test getting current task when episode not found."""
        manager = session_manager_with_session

        # Create session with episode ID but no actual episode
        session = await manager.create_session("test_client")
        session_id = session.session_id
        session.current_episode_id = "episode_123"

        # Mock episode manager to return None
        manager.episode_manager.get_current_episode.return_value = None

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await manager.get_current_task(session_id)

        assert exc_info.value.status_code == 400
        assert "No active episode found" in str(exc_info.value.detail)
