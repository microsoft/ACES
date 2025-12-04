"""
Unit tests for SessionManage        with patch('saber.server.session_manager.BenchmarkManager', return_value=mock_task_manager), \
             patch('saber.server.session_manager.ExecutionManager', return_value=mock_execution_manager), \
             patch('saber.server.session_manager.PolicyManager', return_value=mock_policy_manager), \
             patch('saber.server.session_manager.EvaluationManager', return_value=mock_evaluation_manager):

            manager = SessionManager(
                domain_name="integration_test",
                config_dir="        # Verify pentest_team_alpha has no active episodes but others are unaffected (real end_episode manages this)
        assert len(alpha_session.active_episode_ids) == 0
        # Note: episode_history is managed internally, we can't easily test that in this mock setup

        # Verify other sessions still have their episodes
        other_active_episodes = 0
                host="127.0.0.1",
                port=8004
            )
            return manageron and error handling.

Tests component integration, error scenarios, and edge cases.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from saber.server.base import Action, CommandResult, EpisodeState
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

        # Create a mock evaluation result for evaluate_episode
        from saber.server.evaluation.models import EvaluationResult
        mock_eval_result = EvaluationResult(
            episode_id="test_episode",
            task_id="test_task",
            strategy="static",
            raw_score=1.0,
            max_score=1.0,
            score=1.0,
            success=True,
            submission="test_submission",
            step_count=1
        )
        mock_evaluation_manager.evaluate_episode = AsyncMock(return_value=mock_eval_result)

        mock_episode_manager = MagicMock()
        mock_episode_manager.start_episode = MagicMock()
        mock_episode_manager.end_episode = AsyncMock()
        mock_episode_manager.get_episode = MagicMock()
        mock_episode_manager.configure_for_task = AsyncMock()

        with (
            patch("saber.server.session_manager.BenchmarkManager", return_value=mock_task_manager),
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
        mock_episode.state = EpisodeState.READY
        mock_episode.is_complete = False
        mock_episode.steps = []  # Start with empty steps

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
        mock_task.episode_config = {"max_steps": 20}  # Add episode_config to prevent early termination
        mock_task.dependency_template = None  # No dependencies
        manager.benchmark_manager.get_task.return_value = mock_task

        # Mock prompts
        manager.benchmark_manager.prompt_generator.render_agent_prompts_for_task.return_value = {
            "instruction": "test",
            "assistant": "test",
            "submit": "test"
        }

        # Mock execution manager async methods
        manager.execution_manager.configure_for_task_async = AsyncMock(return_value=("orchestrator", "compose_path"))
        manager.execution_manager.wait_for_episode_healthy = AsyncMock()
        manager.execution_manager.copy_initial_files_to_episode = AsyncMock()

        manager.episode_manager.start_episode.return_value = mock_episode
        # Also mock get_current_episode for execute_action calls
        manager.episode_manager.get_current_episode.return_value = mock_episode
        # Mock get_episode_by_id for execute_action validation
        manager.episode_manager.get_episode_by_id.return_value = mock_episode
        # Mock step method to return the proper StepResult objects
        from saber.server.episodes.episode_manager import StepResult

        step_result1 = StepResult(step=mock_step1, should_terminate=False, termination_reason=None)
        step_result2 = StepResult(step=mock_step2, should_terminate=False, termination_reason=None)
        manager.episode_manager.step.side_effect = [step_result1, step_result2]

        command_result1 = CommandResult(exit_code=0, stdout="step1", stderr="", execution_time=0.1)
        command_result2 = CommandResult(exit_code=0, stdout="step2", stderr="", execution_time=0.1)
        manager.execution_manager.step.side_effect = [command_result1, command_result2]

        # 1. Create session
        session = await manager.create_session("test_client")
        session_id = session.session_id

        # 2. Start episode (async - returns CREATING state)
        mock_episode.state = EpisodeState.CREATING
        episode = await manager.start_episode(session_id, "task_456")
        # Use the actual episode ID returned, not the mock one
        actual_episode_id = episode.episode_id

        # Update mock to return the actual episode when queried
        mock_episode.episode_id = actual_episode_id
        mock_episode.state = EpisodeState.READY  # Simulate finalization complete
        manager.episode_manager.get_episode_by_id.return_value = mock_episode

        # Manually move episode to active state (simulating finalization completion)
        session.move_to_active_episode(actual_episode_id)

        # 3. Execute first step (use actual episode ID)
        action1 = Action(tool_name="bash", parameters={"arguments": "command1"})
        response1 = await manager.execute_action(session_id, actual_episode_id, action1)
        assert response1.success is True
        assert actual_episode_id in session.active_episode_ids  # Still active

        # 4. Execute final step (completes episode)
        action2 = Action(tool_name="bash", parameters={"arguments": "command2"})
        response2 = await manager.execute_action(session_id, actual_episode_id, action2)
        assert response2.success is True
        assert actual_episode_id not in session.active_episode_ids  # Episode completed

        # 5. Verify all components were called correctly
        manager.evaluation_manager.log_session_start.assert_called_once()
        # Note: log_episode_start is called in background finalization task, not tested here
        assert manager.evaluation_manager.log_action.call_count == 2
        manager.evaluation_manager.log_episode_end.assert_called_once()

    # test_multiple_sessions_isolation removed - similar coverage in test_session_manager_core.py::test_create_multiple_sessions

    @pytest.mark.asyncio
    async def test_component_initialization_order(self):
        """Test that components are initialized in the correct order."""
        with (
            patch("saber.server.session_manager.BenchmarkManager") as mock_tm,
            patch("saber.server.session_manager.ExecutionManager") as mock_em,
            patch("saber.server.session_manager.PolicyManager") as mock_pm,
            patch("saber.server.session_manager.EvaluationManager") as mock_eval,
        ):
            # Set up ExecutionManager mock to have the required method
            mock_em.return_value.initialize_permanent_environment_manager = MagicMock()

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp")

            # Verify initialization order and parameters
            mock_tm.assert_called_once_with("test_domain", "/tmp")
            # ExecutionManager now takes only config_dir
            assert mock_em.call_count == 1
            call_args = mock_em.call_args[0]
            assert call_args[0] == "/tmp"  # config_dir
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

        # Create a mock evaluation result for evaluate_episode
        from saber.server.evaluation.models import EvaluationResult
        mock_eval_result = EvaluationResult(
            episode_id="test_episode",
            task_id="test_task",
            strategy="static",
            raw_score=1.0,
            max_score=1.0,
            score=1.0,
            success=True,
            submission="test_submission",
            step_count=1
        )
        mock_evaluation_manager.evaluate_episode = AsyncMock(return_value=mock_eval_result)

        mock_episode_manager = MagicMock()
        mock_episode_manager.start_episode = MagicMock()
        mock_episode_manager.end_episode = AsyncMock()
        mock_episode_manager.get_episode = MagicMock()
        mock_episode_manager.configure_for_task = AsyncMock()

        with (
            patch("saber.server.session_manager.BenchmarkManager", return_value=mock_task_manager),
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
        # Add episode to session's active episodes (replacing current_episode_id)
        session.add_active_episode("episode_123")

        # Mock episode ending to raise exception
        manager.benchmark_manager.end_episode.side_effect = Exception("Episode end failed")

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
        # Add episode to session's active episodes (replacing current_episode_id)
        session.add_active_episode("episode_123")

        # Mock episode manager to return a proper episode object
        from saber.server.base import EpisodeState
        mock_episode_obj = MagicMock()
        mock_episode_obj.episode_id = "episode_123"
        mock_episode_obj.state = EpisodeState.READY
        manager.episode_manager.get_episode_by_id = MagicMock(return_value=mock_episode_obj)

        # Mock execution manager to fail
        manager.execution_manager.step.side_effect = Exception("Execution failed")

        # Execute command
        action = Action(tool_name="bash", parameters={"arguments": "failing_command"})
        response = await manager.execute_action(session.session_id, "episode_123", action)

        assert response.success is False
        assert response.error == "Execution failed"

    @pytest.mark.asyncio
    async def test_step_task_manager_failure(self, error_test_manager):
        """Test step execution when BenchmarkManager fails."""
        manager = error_test_manager

        # Create session with episode
        session = await manager.create_session("test_client")
        # Add episode to session's active episodes (replacing current_episode_id)
        session.add_active_episode("episode_123")

        # Mock episode manager to return a valid episode
        mock_episode_obj = MagicMock()
        mock_episode_obj.episode_id = "episode_123"
        mock_episode_obj.state = EpisodeState.READY
        mock_episode_obj.add_step = MagicMock()
        manager.episode_manager.get_episode.return_value = mock_episode_obj
        manager.episode_manager.get_episode_by_id.return_value = mock_episode_obj

        # Mock execution to succeed but episode manager to fail
        command_result = CommandResult(exit_code=0, stdout="success", stderr="", execution_time=0.1)
        manager.execution_manager.step.return_value = command_result
        manager.episode_manager.step.side_effect = Exception("Episode manager failed")

        # Execute command
        action = Action(tool_name="bash", parameters={"arguments": "command"})
        response = await manager.execute_action(session.session_id, "episode_123", action)

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
        # Add episode to session's active episodes (replacing current_episode_id)
        session.add_active_episode("episode_123")

        # Mock episode manager to return a valid episode
        mock_episode_obj = MagicMock()
        mock_episode_obj.episode_id = "episode_123"
        mock_episode_obj.state = EpisodeState.READY
        mock_episode_obj.add_step = MagicMock()
        manager.episode_manager.get_episode.return_value = mock_episode_obj
        manager.episode_manager.get_episode_by_id.return_value = mock_episode_obj

        # Mock successful execution
        command_result = CommandResult(exit_code=0, stdout="success", stderr="", execution_time=0.1)
        manager.execution_manager.step.return_value = command_result

        mock_step = MagicMock()
        mock_step.step_number = 1
        mock_step.done = False
        manager.benchmark_manager.step.return_value = mock_step

        # This should still work despite evaluation manager failures
        action = Action(tool_name="bash", parameters={"arguments": "command"})
        response = await manager.execute_action(session.session_id, "episode_123", action)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_policy_manager_failure(self, error_test_manager):
        """Test handling of policy manager failures."""
        manager = error_test_manager

        # Create session and episode
        session = await manager.create_session("test_client")

        # Mock task for episode
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = {"timeout": 60}

        # Mock start_episode to return a valid episode ID and add it to session
        mock_episode_id = "test_episode_123"
        manager.start_episode = AsyncMock(return_value=mock_episode_id)

        # Start episode to trigger policy configuration
        episode_id = await manager.start_episode(session.session_id, mock_task)

        # Manually add episode to session for validation
        session.active_episode_ids.append(episode_id)

        # Mock policy manager to fail on retrieval
        manager.policy_manager.get_policy.side_effect = Exception("Policy failed")

        # Should propagate the exception
        with pytest.raises(Exception, match="Policy failed"):
            await manager.get_policy(session.session_id, episode_id)

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
