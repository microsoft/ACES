"""
Unit tests for SessionManage        with patch('saber.server.session_manager.TaskManager', return_value=mock_task_manager), \
             patch('saber.server.session_manager.ExecutionManager', return_value=mock_execution_manager), \
             patch('saber.server.session_manager.PolicyManager', return_value=mock_policy_manager), \
             patch('saber.server.session_manager.EvaluationManager', return_value=mock_evaluation_manager):

            manager = SessionManager(
                domain_name="integration_test",
                config_dir="/tmp",
                host="127.0.0.1",
                port=8004
            )
            return manageron and error handling.

Tests component integration, error scenarios, and edge cases.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from saber.server.base import Action, CommandResult
from saber.server.session_manager import SessionManager


class TestSessionManagerIntegration:
    """Test SessionManager integration with other components."""

    @pytest.fixture
    def integration_session_manager(self):
        """Create SessionManager for integration testing."""
        mock_task_manager = MagicMock()
        mock_execution_manager = MagicMock()
        mock_execution_manager.step = AsyncMock()
        mock_policy_manager = MagicMock()
        mock_policy_manager.get_policy = AsyncMock()
        mock_evaluation_manager = MagicMock()
        mock_evaluation_manager.log_session_start = AsyncMock()
        mock_evaluation_manager.log_session_end = AsyncMock()
        mock_evaluation_manager.log_episode_start = AsyncMock()
        mock_evaluation_manager.log_episode_end = AsyncMock()
        mock_evaluation_manager.log_action = AsyncMock()

        mock_episode_manager = MagicMock()
        mock_episode_manager.start_episode = MagicMock()
        mock_episode_manager.end_episode = MagicMock()
        mock_episode_manager.get_episode = MagicMock()

        with (
            patch("saber.server.session_manager.TaskManager", return_value=mock_task_manager),
            patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
            patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
            patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
            patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
        ):

            manager = SessionManager(domain_name="integration_test", config_dir="/tmp", host="127.0.0.1", port=8004)
            return manager

    @pytest.mark.asyncio
    async def test_full_episode_workflow(self, integration_session_manager):
        """Test complete episode workflow from start to finish."""
        manager = integration_session_manager

        # Mock objects
        mock_episode = MagicMock()
        mock_episode.episode_id = "episode_123"
        mock_episode.task_id = "task_456"

        mock_step1 = MagicMock()
        mock_step1.step_number = 1
        mock_step1.done = False  # Explicitly ensure this step does not complete the episode
        mock_step1.current_subtask = "subtask_1"
        mock_step1.completed_subtasks = set()
        mock_step1.in_progress_subtasks = {"subtask_1"}
        mock_step1.not_visited_subtasks = {"subtask_2"}

        mock_step2 = MagicMock()
        mock_step2.step_number = 2
        mock_step2.done = True  # This step will complete the episode
        mock_step2.current_subtask = "subtask_2"
        mock_step2.completed_subtasks = {"subtask_1", "subtask_2"}
        mock_step2.in_progress_subtasks = set()
        mock_step2.not_visited_subtasks = set()

        # Configure mocks - use proper task/episode flow
        mock_task = MagicMock()
        mock_task.initial_context = {"test": "data"}
        manager.task_manager.get_task.return_value = mock_task
        manager.episode_manager.start_episode.return_value = mock_episode
        # Also mock get_current_episode for execute_action calls
        manager.episode_manager.get_current_episode.return_value = mock_episode
        # Mock step method to return the proper steps
        manager.episode_manager.step.side_effect = [mock_step1, mock_step2]

        command_result1 = CommandResult(exit_code=0, stdout="step1", stderr="", execution_time=0.1)
        command_result2 = CommandResult(exit_code=0, stdout="step2", stderr="", execution_time=0.1)
        manager.execution_manager.step.side_effect = [command_result1, command_result2]

        # 1. Create session
        session = await manager.create_session("test_client")
        session_id = session.session_id

        # 2. Start episode
        episode = await manager.start_episode(session_id, "task_456")
        assert session.current_episode_id == "episode_123"

        # 3. Execute first step
        action1 = Action(tool_name="cli", parameters={"arguments": "command1"})
        response1 = await manager.execute_action(session_id, action1)
        assert response1.success is True
        assert session.current_episode_id == "episode_123"  # Still active

        # 4. Execute final step (completes episode)
        action2 = Action(tool_name="cli", parameters={"arguments": "command2"})
        response2 = await manager.execute_action(session_id, action2)
        assert response2.success is True
        assert session.current_episode_id is None  # Episode completed

        # 5. Verify all components were called correctly
        manager.evaluation_manager.log_session_start.assert_called_once()
        manager.evaluation_manager.log_episode_start.assert_called_once()
        assert manager.evaluation_manager.log_action.call_count == 2
        manager.evaluation_manager.log_episode_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolation(self, integration_session_manager):
        """Test that multiple sessions are properly isolated."""
        manager = integration_session_manager

        # Create two sessions
        session1 = await manager.create_session("client1")
        session2 = await manager.create_session("client2")

        assert len(manager.active_sessions) == 2
        assert session1.session_id != session2.session_id
        assert session1.client_id == "client1"
        assert session2.client_id == "client2"

        # Update activity for session1
        session1.update_activity()
        original_session2_activity = session2.last_activity

        # Verify session2 activity wasn't affected
        assert session2.last_activity == original_session2_activity

        # Terminate session1
        await manager.terminate_session(session1.session_id)

        # Verify session2 is still active
        assert len(manager.active_sessions) == 1
        assert session2.session_id in manager.active_sessions
        assert session2.is_active is True

    @pytest.mark.asyncio
    async def test_component_initialization_order(self):
        """Test that components are initialized in the correct order."""
        with (
            patch("saber.server.session_manager.TaskManager") as mock_tm,
            patch("saber.server.session_manager.ExecutionManager") as mock_em,
            patch("saber.server.session_manager.PolicyManager") as mock_pm,
            patch("saber.server.session_manager.EvaluationManager") as mock_eval,
        ):

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")

            # Verify initialization order and parameters
            mock_tm.assert_called_once_with("test_domain", "/tmp")
            mock_em.assert_called_once_with("/tmp")
            mock_pm.assert_called_once_with("test_domain")
            mock_eval.assert_called_once()


class TestSessionManagerErrorHandling:
    """Test SessionManager error handling and edge cases."""

    @pytest.fixture
    def error_test_manager(self):
        """Create SessionManager for error testing."""
        mock_task_manager = MagicMock()
        mock_execution_manager = MagicMock()
        mock_execution_manager.step = AsyncMock()
        mock_policy_manager = MagicMock()
        mock_policy_manager.get_policy = AsyncMock()
        mock_evaluation_manager = MagicMock()
        mock_evaluation_manager.log_session_start = AsyncMock()
        mock_evaluation_manager.log_session_end = AsyncMock()
        mock_evaluation_manager.log_episode_start = AsyncMock()
        mock_evaluation_manager.log_episode_end = AsyncMock()
        mock_evaluation_manager.log_action = AsyncMock()

        mock_episode_manager = MagicMock()
        mock_episode_manager.start_episode = MagicMock()
        mock_episode_manager.end_episode = MagicMock()
        mock_episode_manager.get_episode = MagicMock()

        with (
            patch("saber.server.session_manager.TaskManager", return_value=mock_task_manager),
            patch("saber.server.session_manager.ExecutionManager", return_value=mock_execution_manager),
            patch("saber.server.session_manager.PolicyManager", return_value=mock_policy_manager),
            patch("saber.server.session_manager.EvaluationManager", return_value=mock_evaluation_manager),
            patch("saber.server.session_manager.EpisodeManager", return_value=mock_episode_manager),
        ):

            manager = SessionManager(domain_name="error_test", config_dir="/tmp")

            return manager

    @pytest.mark.asyncio
    async def test_session_termination_with_episode_error(self, error_test_manager):
        """Test session termination when episode ending fails."""
        manager = error_test_manager

        # Create session with episode
        session = await manager.create_session("test_client")
        session.current_episode_id = "episode_123"

        # Mock episode ending to raise exception
        manager.task_manager.end_episode.side_effect = Exception("Episode end failed")

        # Should not raise exception, but log warning
        await manager.terminate_session(session.session_id)

        # Session should still be terminated
        assert session.session_id not in manager.active_sessions

    @pytest.mark.asyncio
    async def test_step_execution_manager_failure(self, error_test_manager):
        """Test step execution when ExecutionManager fails."""
        manager = error_test_manager

        # Create session with episode
        session = await manager.create_session("test_client")
        session.current_episode_id = "episode_123"

        # Mock execution manager to fail
        manager.execution_manager.step.side_effect = Exception("Execution failed")

        # Execute command
        action = Action(tool_name="cli", parameters={"arguments": "failing_command"})
        response = await manager.execute_action(session.session_id, action)

        assert response.success is False
        assert response.error == "Execution failed"

    @pytest.mark.asyncio
    async def test_step_task_manager_failure(self, error_test_manager):
        """Test step execution when TaskManager fails."""
        manager = error_test_manager

        # Create session with episode
        session = await manager.create_session("test_client")
        session.current_episode_id = "episode_123"

        # Mock episode manager to return a valid episode
        mock_episode_obj = MagicMock()
        mock_episode_obj.episode_id = "episode_123"
        mock_episode_obj.add_step = MagicMock()
        manager.episode_manager.get_episode.return_value = mock_episode_obj

        # Mock execution to succeed but episode manager to fail
        command_result = CommandResult(exit_code=0, stdout="success", stderr="", execution_time=0.1)
        manager.execution_manager.step.return_value = command_result
        manager.episode_manager.step.side_effect = Exception("Episode manager failed")

        # Execute command
        action = Action(tool_name="cli", parameters={"arguments": "command"})
        response = await manager.execute_action(session.session_id, action)

        assert response.success is False
        assert "Episode manager failed" in response.error

    @pytest.mark.asyncio
    async def test_evaluation_manager_failure_resilience(self, error_test_manager):
        """Test that evaluation manager failures don't break functionality."""
        manager = error_test_manager

        # Mock evaluation manager to fail
        manager.evaluation_manager.log_session_start.side_effect = Exception("Eval failed")
        manager.evaluation_manager.log_action.side_effect = Exception("Eval failed")

        # Should still be able to create session and execute steps
        session = await manager.create_session("test_client")
        session.current_episode_id = "episode_123"

        # Mock episode manager to return a valid episode
        mock_episode_obj = MagicMock()
        mock_episode_obj.episode_id = "episode_123"
        mock_episode_obj.add_step = MagicMock()
        manager.episode_manager.get_episode.return_value = mock_episode_obj

        # Mock successful execution
        command_result = CommandResult(exit_code=0, stdout="success", stderr="", execution_time=0.1)
        manager.execution_manager.step.return_value = command_result

        mock_step = MagicMock()
        mock_step.step_number = 1
        mock_step.done = False
        manager.task_manager.step.return_value = mock_step

        # This should still work despite evaluation manager failures
        action = Action(tool_name="cli", parameters={"arguments": "command"})
        response = await manager.execute_action(session.session_id, action)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_policy_manager_failure(self, error_test_manager):
        """Test handling of policy manager failures."""
        manager = error_test_manager

        # Create session
        session = await manager.create_session("test_client")

        # Mock policy manager to fail
        manager.policy_manager.get_policy.side_effect = Exception("Policy failed")

        # Should propagate the exception
        with pytest.raises(Exception, match="Policy failed"):
            await manager.get_policy(session.session_id)

    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self, error_test_manager):
        """Test concurrent operations on different sessions."""
        manager = error_test_manager

        async def create_and_terminate_session(client_id):
            session = await manager.create_session(client_id)
            await asyncio.sleep(0.01)  # Small delay to simulate work
            await manager.terminate_session(session.session_id)
            return session.session_id

        # Run multiple concurrent operations
        tasks = [create_and_terminate_session(f"client_{i}") for i in range(5)]

        session_ids = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(session_ids) == 5
        assert len(set(session_ids)) == 5  # All unique
        assert len(manager.active_sessions) == 0  # All terminated

    def test_session_activity_update_thread_safety(self, error_test_manager):
        """Test that session activity updates are safe."""
        manager = error_test_manager

        # Create session directly (avoiding async)
        from saber.server.session_manager import ClientSession

        session = ClientSession(session_id="test_id", client_id="test_client")
        manager.active_sessions["test_id"] = session

        original_time = session.last_activity

        # Multiple rapid updates
        for _ in range(10):
            session.update_activity()

        assert session.last_activity >= original_time
