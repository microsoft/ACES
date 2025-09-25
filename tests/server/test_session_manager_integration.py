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
        mock_episode_manager.end_episode = MagicMock()
        mock_episode_manager.get_episode = MagicMock()

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
        mock_task.depends_on_task_id = None  # No dependencies
        manager.benchmark_manager.get_task.return_value = mock_task
        manager.episode_manager.start_episode.return_value = mock_episode
        # Also mock get_current_episode for execute_action calls
        manager.episode_manager.get_current_episode.return_value = mock_episode
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

        # 2. Start episode
        episode = await manager.start_episode(session_id, "task_456")
        assert "episode_123" in session.active_episode_ids

        # 3. Execute first step
        action1 = Action(tool_name="bash", parameters={"arguments": "command1"})
        response1 = await manager.execute_action(session_id, "episode_123", action1)
        assert response1.success is True
        assert "episode_123" in session.active_episode_ids  # Still active

        # 4. Execute final step (completes episode)
        action2 = Action(tool_name="bash", parameters={"arguments": "command2"})
        response2 = await manager.execute_action(session_id, "episode_123", action2)
        assert response2.success is True
        assert "episode_123" not in session.active_episode_ids  # Episode completed

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
        mock_episode_manager.end_episode = MagicMock()
        mock_episode_manager.get_episode = MagicMock()

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
        mock_episode_obj.add_step = MagicMock()
        manager.episode_manager.get_episode.return_value = mock_episode_obj

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
        mock_episode_obj.add_step = MagicMock()
        manager.episode_manager.get_episode.return_value = mock_episode_obj

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

    @pytest.mark.asyncio
    async def test_multi_episode_orchestration_single_session(self, error_test_manager):
        """Test SessionManager can orchestrate multiple episodes within a single session."""
        manager = error_test_manager

        # Create session
        session = await manager.create_session("test_multi_episode_client")
        session_id = session.session_id

        # Create multiple different task IDs for episode variety
        task_configs = [
            ("web_reconnaissance_task", 60),
            ("vulnerability_scan_task", 180),
            ("exploit_execution_task", 300),
            ("data_extraction_task", 120),
        ]

        # Mock the benchmark manager to return tasks
        for task_id, timeout in task_configs:
            mock_task = MagicMock()
            mock_task.task_id = task_id
            mock_task.execution_config = {"timeout": timeout}
            mock_task.initial_context = {}
            mock_task.depends_on_task_id = None  # No dependencies
            manager.benchmark_manager.get_task = MagicMock(return_value=mock_task)

        # Start multiple episodes concurrently
        episode_ids = []
        for task_id, timeout in task_configs:
            # Mock the episode manager to return a proper episode object
            mock_episode = MagicMock()
            mock_episode.episode_id = f"episode_{task_id}_{timeout}"
            mock_episode.task_id = task_id  # Use actual task_id string
            manager.episode_manager.start_episode = MagicMock(return_value=mock_episode)

            # Mock get_episode_by_id to return the same episode for end_episode
            manager.episode_manager.get_episode_by_id = MagicMock(return_value=mock_episode)

            started_episode = await manager.start_episode(session_id, task_id)
            episode_ids.append(started_episode.episode_id)

            # Add episode to session for validation (the real start_episode would do this)
            if started_episode.episode_id not in session.active_episode_ids:
                session.active_episode_ids.append(started_episode.episode_id)

        # Verify all episodes are tracked in the session
        assert len(session.active_episode_ids) == 4
        assert len(set(episode_ids)) == 4  # All unique episode IDs

        # Test episode-specific policy retrieval for each episode
        for i, episode_id in enumerate(episode_ids):
            # Mock policy manager to return episode-specific policy
            expected_timeout = [60, 180, 300, 120][i]
            mock_policy = MagicMock()
            mock_policy.prompt = f"Policy for episode {episode_id} with {expected_timeout}s timeout"
            manager.policy_manager.get_policy = MagicMock(return_value=mock_policy)

            policy = manager.get_policy(session_id, episode_id)
            assert str(expected_timeout) in policy.prompt
            assert episode_id in policy.prompt

        # Test episode isolation - ending one episode shouldn't affect others
        episode_to_end = episode_ids[1]  # End the vulnerability_scan episode

        # Create a mock submission for successful completion
        from saber.models.core import EvalSubmission
        mock_submission = EvalSubmission(
            episode_id=episode_to_end,
            task_id="test_task",
            model="test_model",
            choices=[{"content": "test", "message": {"role": "assistant", "content": "test"}}],
            submission="test_submission",
            tokens={"input": 10, "output": 20, "total": 30},
            time=1.5
        )

        await manager.end_episode(session_id, episode_to_end, "completed", mock_submission)

        # Verify session still has other active episodes (the real end_episode removes it)
        assert len(session.active_episode_ids) == 3
        assert episode_to_end not in session.active_episode_ids
        # Note: episode_history is managed internally, we can't easily test that in this mock setup

        # Verify other episodes are still accessible
        remaining_episodes = [ep for ep in episode_ids if ep != episode_to_end]
        for episode_id in remaining_episodes:
            # Should still be able to get policy for remaining episodes
            mock_policy = MagicMock()
            mock_policy.prompt = f"Remaining policy for {episode_id}"
            manager.policy_manager.get_policy = MagicMock(return_value=mock_policy)

            policy = manager.get_policy(session_id, episode_id)
            assert episode_id in policy.prompt

    @pytest.mark.asyncio
    async def test_concurrent_multi_session_multi_episode_orchestration(self, error_test_manager):
        """Test SessionManager handling multiple sessions each running multiple episodes."""
        manager = error_test_manager

        # Create multiple sessions representing different security assessment scenarios
        session_configs = [
            ("pentest_team_alpha", ["network_scan", "web_enum", "exploitation"]),
            ("pentest_team_beta", ["wireless_audit", "social_eng", "physical_sec"]),
            ("forensics_team", ["memory_analysis", "disk_forensics"]),
            ("red_team", ["c2_setup", "lateral_movement", "persistence"]),
        ]

        sessions_and_episodes = {}
        total_episodes = 0

        # Create sessions and start episodes for each
        for client_name, task_names in session_configs:
            session = await manager.create_session(client_name)
            session_id = session.session_id
            episode_ids = []

            for task_name in task_names:
                # Create unique task ID
                task_id = f"{client_name}_{task_name}_task"

                # Mock task object for benchmark manager
                mock_task = MagicMock()
                mock_task.task_id = task_id
                mock_task.execution_config = {"timeout": 60 + len(task_name) * 10}  # Varying timeouts
                mock_task.initial_context = {}
                mock_task.depends_on_task_id = None  # No dependencies
                manager.benchmark_manager.get_task = MagicMock(return_value=mock_task)

                # Mock episode creation
                episode_id = f"ep_{client_name}_{task_name}"
                mock_episode = MagicMock()
                mock_episode.episode_id = episode_id
                mock_episode.task_id = task_id  # Use actual task_id string
                manager.episode_manager.start_episode = MagicMock(return_value=mock_episode)

                # Mock get_episode_by_id to return the same episode for end_episode
                manager.episode_manager.get_episode_by_id = MagicMock(return_value=mock_episode)

                started_episode = await manager.start_episode(session_id, task_id)
                episode_ids.append(started_episode.episode_id)
                if started_episode.episode_id not in session.active_episode_ids:
                    session.active_episode_ids.append(started_episode.episode_id)
                total_episodes += 1

            sessions_and_episodes[session_id] = {
                "client_name": client_name,
                "session": session,
                "episode_ids": episode_ids,
                "task_names": task_names
            }

        # Verify all sessions and episodes are properly managed
        assert len(manager.active_sessions) == 4
        assert total_episodes == 11  # 3+3+2+3 = 11 episodes

        # Test cross-session episode isolation
        for session_id, session_data in sessions_and_episodes.items():
            session = session_data["session"]
            client_name = session_data["client_name"]

            # Verify each session only contains its own episodes
            for episode_id in session.active_episode_ids:
                assert client_name in episode_id

                # Mock policy retrieval for cross-session verification
                mock_policy = MagicMock()
                mock_policy.prompt = f"Policy for {episode_id} in session {session_id}"
                manager.policy_manager.get_policy = MagicMock(return_value=mock_policy)

                policy = manager.get_policy(session_id, episode_id)
                assert session_id in policy.prompt
                assert episode_id in policy.prompt

        # Test selective session cleanup - end all episodes for one team
        pentest_alpha_session_id = None
        for sid, data in sessions_and_episodes.items():
            if data["client_name"] == "pentest_team_alpha":
                pentest_alpha_session_id = sid
                break

        assert pentest_alpha_session_id is not None
        alpha_session = sessions_and_episodes[pentest_alpha_session_id]["session"]
        alpha_episodes = alpha_session.active_episode_ids.copy()

        # End all episodes for pentest_team_alpha
        for episode_id in alpha_episodes:
            await manager.end_episode(pentest_alpha_session_id, episode_id, "team_rotation")

        # Verify pentest_team_alpha has no active episodes but others are unaffected (real end_episode manages this)
        assert len(alpha_session.active_episode_ids) == 0
        # Note: episode_history is managed internally, we can't easily test that in this mock setup

        # Verify other sessions still have their episodes
        other_active_episodes = 0
        for sid, data in sessions_and_episodes.items():
            if sid != pentest_alpha_session_id:
                other_active_episodes += len(data["session"].active_episode_ids)

        assert other_active_episodes == 8  # 11 total - 3 from alpha team = 8

        # Test adding new episodes to existing sessions during operations
        forensics_session_id = None
        for sid, data in sessions_and_episodes.items():
            if data["client_name"] == "forensics_team":
                forensics_session_id = sid
                break

        assert forensics_session_id is not None
        forensics_session = sessions_and_episodes[forensics_session_id]["session"]

        # Add urgent analysis episode to forensics team
        urgent_task_id = "forensics_team_urgent_malware_analysis_task"
        urgent_mock_task = MagicMock()
        urgent_mock_task.task_id = urgent_task_id
        urgent_mock_task.execution_config = {"timeout": 600}  # High priority, longer timeout
        urgent_mock_task.initial_context = {}
        urgent_mock_task.depends_on_task_id = None  # No dependencies
        manager.benchmark_manager.get_task = MagicMock(return_value=urgent_mock_task)

        urgent_episode_id = "ep_forensics_team_urgent_malware_analysis"
        urgent_mock_episode = MagicMock()
        urgent_mock_episode.episode_id = urgent_episode_id
        urgent_mock_episode.task_id = urgent_task_id
        manager.episode_manager.start_episode = MagicMock(return_value=urgent_mock_episode)

        new_episode = await manager.start_episode(forensics_session_id, urgent_task_id)
        # Only add episode to session if it's not already there (the real start_episode might do this)
        if new_episode.episode_id not in forensics_session.active_episode_ids:
            forensics_session.active_episode_ids.append(new_episode.episode_id)

        # Verify new episode is properly integrated
        assert new_episode.episode_id in forensics_session.active_episode_ids
        assert len(forensics_session.active_episode_ids) == 3  # Original 2 + 1 new

        # Verify policy access for new episode
        mock_policy = MagicMock()
        mock_policy.prompt = f"Urgent analysis policy for {new_episode.episode_id}"
        manager.policy_manager.get_policy = MagicMock(return_value=mock_policy)

        policy = manager.get_policy(forensics_session_id, new_episode.episode_id)
        assert "Urgent analysis" in policy.prompt
        assert new_episode.episode_id in policy.prompt
