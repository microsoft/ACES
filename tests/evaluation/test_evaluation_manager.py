"""
Unit tests for EvaluationManager.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from saber.server.base import Episode, EpisodeState
from saber.server.benchmarks.task import Task
from saber.server.evaluation.constants import EVAL_STRATEGY_STATIC
from saber.server.evaluation.evaluation_manager import EvaluationManager
from saber.server.evaluation.exceptions import (
    EvaluationConfigError,
    IncompleteEpisodeError,
    InvalidEvaluationRequestError,
    InvalidEvaluationStrategyError,
    MissingSubmissionError,
)
from saber.server.evaluation.models import EvaluationResult


class TestEvaluationManager:
    """Test cases for EvaluationManager functionality."""

    @pytest.fixture
    def evaluation_manager(self):
        """Create EvaluationManager instance."""
        return EvaluationManager()

    @pytest.fixture
    def task_with_static_eval(self):
        """Create task with static evaluation config."""
        return Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            prompts={"instruction": "test_template.j2", "assistant": "test_template.j2", "submit": "test_template.j2", "continue": "test_continue.md"},
            submission_evaluation_config={
                "strategy": "static",
                "criteria": {"expected_answers": ["flag{correct}"]},
                "scoring": {"max_score": 1.0}
            }
        )

    @pytest.fixture
    def task_without_eval(self):
        """Create task without evaluation config."""
        return Task(
            task_id="no_eval_task",
            domain="test_domain",
            title="No Eval Task",
            description="Task without evaluation",
            prompts={"instruction": "test_template.j2", "assistant": "test_template.j2", "submit": "test_template.j2", "continue": "test_continue.md"}
        )

    @pytest.fixture
    def completed_episode(self):
        """Create completed episode with submission."""
        episode = Episode(
            episode_id="episode_123",
            task_id="test_task",
            session_id="session_123",
            state=EpisodeState.COMPLETED
        )
        episode.submission = "flag{correct}"
        return episode

    @pytest.fixture
    def incomplete_episode(self):
        """Create incomplete episode."""
        return Episode(
            episode_id="episode_456",
            task_id="test_task",
            session_id="session_123",
            state=EpisodeState.ACTIVE
        )

    def test_configure_for_task_success(self, evaluation_manager, task_with_static_eval):
        """Test successful task configuration."""
        evaluation_manager.configure_for_task(task_with_static_eval)

        assert "test_task" in evaluation_manager.evaluation_configs
        config = evaluation_manager.evaluation_configs["test_task"]
        assert config.strategy == "static"
        assert config.criteria["expected_answers"] == ["flag{correct}"]

    def test_configure_for_task_missing_config(self, evaluation_manager, task_without_eval):
        """Test configuration failure when task lacks submission_evaluation_config."""
        with pytest.raises(EvaluationConfigError, match="missing.*submission_evaluation_config"):
            evaluation_manager.configure_for_task(task_without_eval)

    def test_configure_for_task_invalid_strategy(self, evaluation_manager):
        """Test that custom strategies are now accepted.

        Strategy validation was relaxed - the server accepts any strategy string
        and validation happens at the client-side via the scorer registry.
        """
        task = Task(
            task_id="custom_task",
            domain="test_domain",
            title="Custom Task",
            description="Task with custom strategy",
            prompts={"instruction": "test_template.j2", "assistant": "test_template.j2", "submit": "test_template.j2", "continue": "test_continue.md"},
            submission_evaluation_config={
                "strategy": "custom_strategy",  # Custom strategies are now allowed
                "criteria": {},
                "scoring": {"max_score": 1.0}
            }
        )

        # Should not raise exception - custom strategies are validated at client-side
        evaluation_manager.configure_for_task(task)
        assert task.task_id in evaluation_manager.evaluation_configs

    def test_configure_for_task_invalid_static_config(self, evaluation_manager):
        """Test configuration failure with invalid static config."""
        task = Task(
            task_id="invalid_static_task",
            domain="test_domain",
            title="Invalid Static Task",
            description="Task with invalid static config",
            prompts={"instruction": "test_template.j2", "assistant": "test_template.j2", "submit": "test_template.j2", "continue": "test_continue.md"},
            submission_evaluation_config={
                "strategy": "static",
                "criteria": {},  # Missing expected_answers
                "scoring": {"max_score": 1.0}
            }
        )

        with pytest.raises(EvaluationConfigError, match="static strategy requires"):
            evaluation_manager.configure_for_task(task)

    def test_configure_for_task_invalid_max_score(self, evaluation_manager):
        """Test configuration failure with invalid max_score."""
        task = Task(
            task_id="invalid_score_task",
            domain="test_domain",
            title="Invalid Score Task",
            description="Task with invalid max_score",
            prompts={"instruction": "test_template.j2", "assistant": "test_template.j2", "submit": "test_template.j2", "continue": "test_continue.md"},
            submission_evaluation_config={
                "strategy": "static",
                "criteria": {"expected_answers": ["answer"]},
                "scoring": {"max_score": -1.0}  # Invalid negative score
            }
        )

        with pytest.raises(EvaluationConfigError, match="max_score must be a positive number"):
            evaluation_manager.configure_for_task(task)





    @pytest.mark.asyncio
    async def test_legacy_logging_methods(self, evaluation_manager):
        """Test that legacy logging methods work without errors."""
        # These should not raise exceptions
        await evaluation_manager.log_session_start("session_123", "client_456")
        await evaluation_manager.log_session_end("session_123")
        await evaluation_manager.log_episode_start("session_123", "episode_123", "task_123")
        await evaluation_manager.log_episode_end("session_123", "completed")

        # Mock action for log_action
        mock_action = MagicMock()
        mock_action.tool_name = "test_tool"
        await evaluation_manager.log_action("session_123", "episode_123", mock_action, {})

        # Get trajectory should return empty list
        trajectory = await evaluation_manager.get_trajectory("session_123")
        assert trajectory == []

    @pytest.mark.asyncio
    async def test_override_evaluation_result_success(self, evaluation_manager):
        """Test successful evaluation override."""
        # Mock the store
        mock_store = AsyncMock()
        evaluation_manager.store = mock_store

        # Create EpisodeEvaluationData
        from saber.server.evaluation.models import EpisodeEvaluationData
        evaluation_data = EpisodeEvaluationData(
            episode_id="test_episode",
            task_id="test_task",
            submission="flag{override}",
            executed_commands=["cat flag.txt"],
            completion_reason="success",
            step_count=1,
            model="gpt-4",
            choices=[{"message": {"content": "Found the flag"}}],
            tokens={"total": 100, "prompt": 20, "completion": 80},
            execution_time=30.5
        )

        # Call override method
        result = await evaluation_manager.override_evaluation_result(
            session_id="test_session",
            episode_id="test_episode",
            evaluation_data=evaluation_data,
            strategy="static",
            raw_score=1.0,
            max_score=1.0,
            score=1.0,
            success=True,
            details={"override": True}
        )

        # Verify result
        assert result.episode_id == "test_episode"
        assert result.task_id == "test_task"
        assert result.strategy == "static"
        assert result.raw_score == 1.0
        assert result.max_score == 1.0
        assert result.score == 1.0
        assert result.success is True
        assert result.submission == "flag{override}"
        assert result.executed_commands == ["cat flag.txt"]
        assert result.completion_reason == "success"
        assert result.step_count == 1
        assert result.model == "gpt-4"
        assert result.choices == [{"message": {"content": "Found the flag"}}]
        assert result.tokens == {"total": 100, "prompt": 20, "completion": 80}
        assert result.execution_time == 30.5
        assert result.details == {"override": True}

        # Verify store.save was called
        mock_store.save.assert_called_once()
        call_args = mock_store.save.call_args
        # Should be called with (result, session_id=session_id, golden_answer=None)
        saved_result = call_args[0][0]  # First positional argument
        session_id_arg = call_args[1]["session_id"]  # Keyword argument
        assert saved_result.episode_id == "test_episode"
        assert session_id_arg == "test_session"

    @pytest.mark.asyncio
    async def test_override_evaluation_result_episode_mismatch(self, evaluation_manager):
        """Test evaluation override with mismatched episode IDs."""
        from saber.server.evaluation.exceptions import InvalidEvaluationRequestError
        from saber.server.evaluation.models import EpisodeEvaluationData

        # Create EpisodeEvaluationData with different episode_id
        evaluation_data = EpisodeEvaluationData(
            episode_id="different_episode",
            task_id="test_task",
            submission="flag{override}",
            executed_commands=[],
            completion_reason="success",
            step_count=1,
            model="gpt-4",
            choices=[],
            tokens={},
            execution_time=0.0
        )

        # Should raise validation error for mismatched episode IDs
        with pytest.raises(InvalidEvaluationRequestError, match="Episode ID mismatch"):
            await evaluation_manager.override_evaluation_result(
                session_id="test_session",
                episode_id="test_episode",
                evaluation_data=evaluation_data,
                strategy="static",
                raw_score=1.0,
                max_score=1.0,
                score=1.0,
                success=True
            )
