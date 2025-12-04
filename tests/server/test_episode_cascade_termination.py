"""
Test episode cascade termination functionality.

Tests that dependent episodes can end their parent episodes when they complete.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from saber.server.episodes.episode_manager import EpisodeManager
from saber.server.session_manager import SessionManager
from saber.server.base import Episode, EpisodeState
from saber.server.benchmarks.task import Task
from saber.models import EvalSubmission


class TestCascadeTermination:
    """Test cascade termination of attached episodes."""

    @pytest.fixture
    def fast_dependency_config(self):
        """Provide fast dependency configuration for test performance."""
        return {
            "wait_seconds": 1.0,
            "retry_interval": 0.1,
            "max_retry_interval": 0.2
        }

    @pytest.fixture
    def mock_benchmark_manager(self):
        """Create a mock benchmark manager."""
        manager = MagicMock()

        # Create mock tasks
        independent_task = Task(
            task_id="independent_task",
            domain="test_domain",
            title="Independent Task",
            description="A task with no dependencies",
            prompts={
                "instruction": "test_instruction.md",
                "assistant": "test_assistant.md",
                "submit": "test_submit.md"
            },
            initial_context={"test": "context"},
            episode_config={"max_steps": 10}
        )

        dependent_task = Task(
            task_id="dependent_task",
            domain="test_domain",
            title="Dependent Task",
            description="A task that depends on independent_task",
            prompts={
                "instruction": "test_instruction.md",
                "assistant": "test_assistant.md",
                "submit": "test_submit.md"
            },
            initial_context={"test": "context"},
            dependency_template="independent_task",
            role="dependent_role",
            episode_config={"max_steps": 10}
        )

        manager.get_task.side_effect = lambda task_id: {
            "independent_task": independent_task,
            "dependent_task": dependent_task,
        }.get(task_id)

        # Mock get_task_prompt method
        manager.get_task_prompt.return_value = "Test prompt for task"

        return manager

    @pytest.fixture
    def episode_manager(self):
        """Create episode manager instance."""
        return EpisodeManager()

    @pytest.fixture
    def session_manager(self, episode_manager, mock_benchmark_manager, fast_dependency_config):
        """Create session manager with mocked dependencies."""
        with patch('saber.server.session_manager.ExecutionManager'), \
             patch('saber.server.session_manager.PolicyManager'), \
             patch('saber.server.session_manager.EvaluationManager'), \
             patch('saber.server.session_manager.BenchmarkManager') as mock_bm_class:

            # Make the BenchmarkManager constructor return our mock
            mock_bm_class.return_value = mock_benchmark_manager

            # Configure fast dependency config for tests
            mock_benchmark_manager.get_dependency_config.return_value = fast_dependency_config

            manager = SessionManager(
                domain_name="test_domain",
                config_dir="/tmp/test"
            )
            manager.episode_manager = episode_manager
            manager.execution_manager = MagicMock()
            manager.execution_manager.configure_for_task = MagicMock()
            manager.execution_manager.configure_for_task_async = AsyncMock(return_value=("orchestrator", "compose_path"))
            manager.execution_manager.wait_for_episode_healthy = AsyncMock()
            manager.execution_manager.copy_initial_files_to_episode = AsyncMock()
            manager.execution_manager.cleanup_episode = MagicMock(return_value=True)
            manager.policy_manager = MagicMock()
            manager.policy_manager.set_episode_policy = MagicMock()
            manager.evaluation_manager = MagicMock()
            manager.evaluation_manager.configure_for_task = MagicMock()
            manager.evaluation_manager.log_episode_start = AsyncMock()
            manager.evaluation_manager.evaluate_episode = AsyncMock()

            # Mock prompt generator
            mock_benchmark_manager.prompt_generator.render_agent_prompts_for_task.return_value = {
                "instruction": "test",
                "assistant": "test",
                "submit": "test"
            }

            # Mock evaluation result
            mock_eval_result = MagicMock()
            mock_eval_result.success = True
            mock_eval_result.score = 100
            mock_eval_result.max_score = 100
            mock_eval_result.dict.return_value = {"score": 100, "max_score": 100, "success": True}
            manager.evaluation_manager.evaluate_episode.return_value = mock_eval_result

            # Mock _get_session method
            mock_session = MagicMock()
            mock_session.session_id = "test_session"
            mock_session.active_episode_ids = []

            # Mock session methods to track episodes properly
            def mock_add_active_episode(episode_id):
                if episode_id not in mock_session.active_episode_ids:
                    mock_session.active_episode_ids.append(episode_id)

            def mock_complete_episode(episode_id):
                if episode_id in mock_session.active_episode_ids:
                    mock_session.active_episode_ids.remove(episode_id)

            def mock_move_to_active_episode(episode_id):
                if episode_id not in mock_session.active_episode_ids:
                    mock_session.active_episode_ids.append(episode_id)

            mock_session.add_active_episode = mock_add_active_episode
            mock_session.complete_episode = mock_complete_episode
            mock_session.move_to_active_episode = mock_move_to_active_episode
            mock_session.update_activity = MagicMock()
            manager._get_session = MagicMock(return_value=mock_session)

            return manager

    async def test_cascade_termination_ends_parent_episode(self, session_manager):
        """Test that cascade termination ends the parent episode when dependent episode completes."""
        session_id = "test_session"

        # Start independent task first
        parent_episode = await session_manager.start_episode(session_id, "independent_task")

        # Move parent episode to ACTIVE state
        session_manager.episode_manager.mark_episode_ready(parent_episode.episode_id)
        session = session_manager._get_session(session_id)
        session.move_to_active_episode(parent_episode.episode_id)

        # Start dependent task (should attach to parent)
        dependent_episode = await session_manager.start_episode(session_id, "dependent_task")

        # Move dependent episode to ACTIVE state
        session_manager.episode_manager.mark_episode_ready(dependent_episode.episode_id)
        session.move_to_active_episode(dependent_episode.episode_id)

        # Verify attachment
        assert dependent_episode.attached_to_episode_id == parent_episode.episode_id

        # Create a submission for the dependent episode
        submission = EvalSubmission(
            episode_id=dependent_episode.episode_id,
            task_id="dependent_task",
            model="test_model",
            choices=[],
            submission="Test submission for dependent task",
            tokens={},
            time=1.0
        )

        # End dependent episode with cascade termination
        response = await session_manager.end_episode(
            session_id=session_id,
            episode_id=dependent_episode.episode_id,
            reason="completed",
            submission=submission,
            cascade_end_attached_episodes=True
        )

        # Verify dependent episode ended successfully
        assert response.episode_ended is True
        assert response.success is True

        # Verify parent episode was also ended (check episode manager state)
        parent_episode_after = session_manager.episode_manager.get_episode_by_id(parent_episode.episode_id)
        assert parent_episode_after.is_complete

    async def test_cascade_termination_skips_already_completed_parent(self, session_manager):
        """Test that cascade termination gracefully handles already completed parent episodes."""
        session_id = "test_session"

        # Start independent task first
        parent_episode = await session_manager.start_episode(session_id, "independent_task")

        # Move parent episode to ACTIVE state
        session_manager.episode_manager.mark_episode_ready(parent_episode.episode_id)
        session = session_manager._get_session(session_id)
        session.move_to_active_episode(parent_episode.episode_id)

        # Start dependent task (should attach to parent)
        dependent_episode = await session_manager.start_episode(session_id, "dependent_task")

        # Move dependent episode to ACTIVE state
        session_manager.episode_manager.mark_episode_ready(dependent_episode.episode_id)
        session.move_to_active_episode(dependent_episode.episode_id)

        # Manually complete the parent episode first
        parent_submission = EvalSubmission(
            episode_id=parent_episode.episode_id,
            task_id="independent_task",
            model="test_model",
            choices=[],
            submission="Parent completed first",
            tokens={},
            time=1.0
        )

        await session_manager.end_episode(
            session_id=session_id,
            episode_id=parent_episode.episode_id,
            reason="completed",
            submission=parent_submission,
            cascade_end_attached_episodes=False
        )

        # Now end dependent episode with cascade termination
        dependent_submission = EvalSubmission(
            episode_id=dependent_episode.episode_id,
            task_id="dependent_task",
            model="test_model",
            choices=[],
            submission="Dependent completed after parent",
            tokens={},
            time=1.0
        )

        # This should not fail even though parent is already complete
        response = await session_manager.end_episode(
            session_id=session_id,
            episode_id=dependent_episode.episode_id,
            reason="completed",
            submission=dependent_submission,
            cascade_end_attached_episodes=True
        )

        # Verify dependent episode ended successfully
        assert response.episode_ended is True
        assert response.success is True

    async def test_cascade_termination_without_attachment(self, session_manager):
        """Test that cascade termination is ignored for episodes without attachments."""
        session_id = "test_session"

        # Start independent task (no attachment)
        episode = await session_manager.start_episode(session_id, "independent_task")

        # Move episode to ACTIVE state
        session_manager.episode_manager.mark_episode_ready(episode.episode_id)
        session = session_manager._get_session(session_id)
        session.move_to_active_episode(episode.episode_id)

        # Verify no attachment
        assert episode.attached_to_episode_id is None

        # Create submission
        submission = EvalSubmission(
            episode_id=episode.episode_id,
            task_id="independent_task",
            model="test_model",
            choices=[],
            submission="Independent task submission",
            tokens={},
            time=1.0
        )

        # End episode with cascade termination flag (should be ignored)
        response = await session_manager.end_episode(
            session_id=session_id,
            episode_id=episode.episode_id,
            reason="completed",
            submission=submission,
            cascade_end_attached_episodes=True
        )

        # Verify episode ended successfully
        assert response.episode_ended is True
        assert response.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
