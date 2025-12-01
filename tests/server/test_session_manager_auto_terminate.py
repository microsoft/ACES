"""
Unit tests for SessionManager auto-termination on should_terminate flag.

Tests the execute_action code path where StepResult.should_terminate triggers
automatic episode termination.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saber.server.base import Action, CommandResult, Episode, EpisodeState
from saber.server.episodes.constants import EpisodeTerminationReason
from saber.server.episodes.episode_manager import StepResult
from saber.server.session_manager import SessionManager


class TestAutoTerminate:
    """Test auto-termination when should_terminate flag is set."""

    @pytest.fixture
    async def session_with_episode(self):
        """Create a session manager with an active episode."""
        with patch('saber.server.session_manager.BenchmarkManager'), \
             patch('saber.server.session_manager.ExecutionManager') as mock_exec, \
             patch('saber.server.session_manager.PolicyManager'), \
             patch('saber.server.session_manager.EvaluationManager') as mock_eval, \
             patch('saber.server.session_manager.EpisodeManager') as mock_ep:

            # Setup mocks
            mock_exec.return_value.step = AsyncMock()
            mock_eval.return_value.log_session_start = AsyncMock()
            mock_eval.return_value.log_action = AsyncMock()
            mock_eval.return_value.log_episode_end = AsyncMock()
            mock_ep.return_value.start_episode = MagicMock()
            mock_ep.return_value.end_episode = MagicMock()
            mock_ep.return_value.step = MagicMock()

            manager = SessionManager(domain_name="test_domain", config_dir="/tmp", host="127.0.0.1", port=8001)
            session = await manager.create_session("test-client")

            # Create a mock episode
            episode = MagicMock(spec=Episode)
            episode.episode_id = "episode_123"
            episode.task_id = "task_456"
            episode.state = EpisodeState.READY
            episode.steps = []
            episode.is_complete = False

            manager.episode_manager.get_episode_by_id.return_value = episode
            session.active_episode_ids.append("episode_123")

            yield manager, session, episode

    @pytest.mark.asyncio
    async def test_should_terminate_with_default_reason(self, session_with_episode):
        """Test auto-termination when should_terminate=True with default reason."""
        manager, session, episode = session_with_episode

        # Create StepResult with should_terminate=True and no custom reason
        from saber.server.base import Step
        mock_step = MagicMock(spec=Step)
        mock_step.done = False

        step_result = StepResult(
            step=mock_step,
            should_terminate=True,
            termination_reason=None  # Will use default TERMINATED
        )

        # Mock the execution manager to return a successful command result
        command_result = CommandResult(exit_code=0, stdout="test output", stderr="", execution_time=0.1)
        manager.execution_manager.step.return_value = command_result

        # Mock episode_manager.step to return our step_result
        manager.episode_manager.step.return_value = step_result

        # Execute an action
        action = Action(tool_name="bash", parameters={"arguments": "echo test"})
        result = await manager.execute_action(
            session_id=session.session_id,
            episode_id="episode_123",
            action=action
        )

        # Verify episode was ended with TERMINATED reason
        manager.episode_manager.end_episode.assert_called_once_with(
            "episode_123",
            EpisodeTerminationReason.TERMINATED
        )

        # Verify episode removed from active episodes
        assert "episode_123" not in session.active_episode_ids

        # Verify metadata added to result
        assert result.metadata["episode_terminated"] is True
        assert result.metadata["termination_reason"] == EpisodeTerminationReason.TERMINATED

    @pytest.mark.asyncio
    async def test_should_terminate_with_custom_reason(self, session_with_episode):
        """Test auto-termination when should_terminate=True with custom reason."""
        manager, session, episode = session_with_episode

        from saber.server.base import Step
        mock_step = MagicMock(spec=Step)
        mock_step.done = False

        # Use ERROR as custom termination reason
        step_result = StepResult(
            step=mock_step,
            should_terminate=True,
            termination_reason=EpisodeTerminationReason.ERROR
        )

        command_result = CommandResult(exit_code=0, stdout="error output", stderr="", execution_time=0.1)
        manager.execution_manager.step.return_value = command_result
        manager.episode_manager.step.return_value = step_result

        action = Action(tool_name="bash", parameters={"arguments": "echo error"})
        result = await manager.execute_action(
            session_id=session.session_id,
            episode_id="episode_123",
            action=action
        )

        # Verify episode was ended with ERROR reason
        manager.episode_manager.end_episode.assert_called_once_with(
            "episode_123",
            EpisodeTerminationReason.ERROR
        )

        assert result.metadata["termination_reason"] == EpisodeTerminationReason.ERROR

    @pytest.mark.asyncio
    async def test_should_terminate_logging_failure(self, session_with_episode):
        """Test auto-termination handles evaluation logging failures gracefully."""
        manager, session, episode = session_with_episode

        from saber.server.base import Step
        mock_step = MagicMock(spec=Step)
        mock_step.done = False

        step_result = StepResult(
            step=mock_step,
            should_terminate=True,
            termination_reason=EpisodeTerminationReason.TIMEOUT
        )

        command_result = CommandResult(exit_code=0, stdout="timeout", stderr="", execution_time=0.1)
        manager.execution_manager.step.return_value = command_result
        manager.episode_manager.step.return_value = step_result

        # Make log_episode_end raise an exception
        manager.evaluation_manager.log_episode_end.side_effect = Exception("Logging failed")

        action = Action(tool_name="bash", parameters={"arguments": "echo timeout"})
        # Should not raise despite logging failure
        result = await manager.execute_action(
            session_id=session.session_id,
            episode_id="episode_123",
            action=action
        )

        # Episode should still be ended
        manager.episode_manager.end_episode.assert_called_once()
        assert result.metadata["episode_terminated"] is True

    @pytest.mark.asyncio
    async def test_should_terminate_with_no_metadata_on_result(self, session_with_episode):
        """Test auto-termination when command_result has no metadata attribute."""
        manager, session, episode = session_with_episode

        from saber.server.base import Step
        mock_step = MagicMock(spec=Step)
        mock_step.done = False

        step_result = StepResult(
            step=mock_step,
            should_terminate=True,
            termination_reason=EpisodeTerminationReason.INTERRUPTED
        )

        # Create command result with metadata=None
        command_result = CommandResult(exit_code=0, stdout="interrupted", stderr="", execution_time=0.1)
        command_result.metadata = None

        manager.execution_manager.step.return_value = command_result
        manager.episode_manager.step.return_value = step_result

        action = Action(tool_name="bash", parameters={"arguments": "echo interrupted"})
        result = await manager.execute_action(
            session_id=session.session_id,
            episode_id="episode_123",
            action=action
        )

        # Metadata should be created and populated
        assert result.metadata is not None
        assert result.metadata["episode_terminated"] is True
        assert result.metadata["termination_reason"] == EpisodeTerminationReason.INTERRUPTED
