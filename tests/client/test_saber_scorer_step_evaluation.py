"""
Unit tests for step evaluation integration in the SABER client scorer.

Tests the integration of step evaluation parsing with the inspect_ai
scorer implementation, focusing on the end-to-end evaluation workflow.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import asyncio

from inspect_ai.model import ChatMessage, ChatMessageUser
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState

from saber.client.inspect_ai.saber_scorer import saber_scorer
from saber.models.rest.evaluation import StepEvaluation


class TestSaberScorerStepEvaluation:
    """Test step evaluation integration in SaberScorer."""

    @pytest.fixture
    def sample_episode_data(self):
        """Create sample episode data for testing."""
        return {
            "episode_id": "ep_123",
            "steps": [
                {"step_number": 1, "command": "whoami", "output": "user"},
                {"step_number": 2, "command": "ls -la", "output": "total 8"},
                {"step_number": 3, "command": "grep -r suspicious /var/log", "output": "found suspicious activity"},
                {"step_number": 4, "command": "nslookup 198.43.121.209", "output": "malicious.domain.com"},
                {"step_number": 5, "command": "echo 198.43.121.209", "output": "198.43.121.209"},
            ],
            "task_id": "incident_investigation_1",
            "status": "completed"
        }

    @pytest.fixture
    def task_state_with_episode(self, sample_episode_data):
        """Create a TaskState with episode data."""
        state = Mock(spec=TaskState)
        state.metadata = {"episode_data": sample_episode_data}
        state.messages = [
            ChatMessageUser(content="Investigate the security incident")
        ]
        return state

    @pytest.fixture
    def scorer_config(self):
        """Create scorer configuration."""
        return {
            "judge_model": "gpt-4",
            "episode_endpoint": "http://localhost:8000/episodes",
            "evaluation_endpoint": "http://localhost:8000/evaluate",
            "concurrent_evaluations": 2
        }

    @pytest.fixture
    def saber_scorer(self, scorer_config):
        """Create SaberScorer instance."""
        return SaberScorer(**scorer_config)

    @pytest.mark.asyncio
    async def test_successful_step_evaluation(self, saber_scorer, task_state_with_episode):
        """Test successful step evaluation with task completion."""
        # Mock the evaluation endpoint response
        mock_evaluation_response = {
            "evaluation_result": {
                "grade": "1.0",
                "explanation": """
                Investigation completed successfully:

                STEP_EVALUATIONS:
                [1: initial_access] - Established user context
                [3: log_analysis] - Found suspicious activity in logs
                [4: threat_attribution] - Identified malicious domain
                [5: incident_investigation_1] - Successfully identified target IP: 198.43.121.209

                The agent completed all required steps.
                """,
                "metadata": {"judge_model": "gpt-4", "evaluation_time": "2025-01-15T10:30:00Z"}
            }
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock the HTTP response
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_evaluation_response)
            mock_response.raise_for_status = Mock()
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Execute scoring
            target = Target(target="198.43.121.209")
            score = await saber_scorer.score(task_state_with_episode, target)

            # Verify score results
            assert isinstance(score, Score)
            assert score.value == 1.0
            assert "Task completed at step 5" in score.explanation
            assert "initial_access, log_analysis, threat_attribution" in score.explanation

            # Verify metadata contains step evaluation details
            assert "step_evaluations" in score.metadata
            step_evals = score.metadata["step_evaluations"]
            assert len(step_evals) == 4

            # Check specific step evaluations
            assert step_evals[0]["step_number"] == 1
            assert step_evals[0]["objective_id"] == "initial_access"
            assert step_evals[0]["objective_type"] == "subtask"

            assert step_evals[3]["step_number"] == 5
            assert step_evals[3]["objective_id"] == "incident_investigation_1"
            assert step_evals[3]["objective_type"] == "task"

    @pytest.mark.asyncio
    async def test_partial_step_evaluation(self, saber_scorer, task_state_with_episode):
        """Test partial step evaluation with only subtasks completed."""
        # Mock the evaluation endpoint response
        mock_evaluation_response = {
            "evaluation_result": {
                "grade": "0.0",
                "explanation": """
                Investigation partially completed:

                STEP_EVALUATIONS:
                [1: initial_access] - Established user context
                [3: log_analysis] - Found suspicious activity in logs

                The agent completed some steps but did not reach the final objective.
                """,
                "metadata": {"judge_model": "gpt-4"}
            }
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock the HTTP response
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_evaluation_response)
            mock_response.raise_for_status = Mock()
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Execute scoring
            target = Target(target="198.43.121.209")
            score = await saber_scorer.score(task_state_with_episode, target)

            # Verify score results
            assert score.value == 0.0
            assert "Main task not completed" in score.explanation
            assert "initial_access, log_analysis" in score.explanation

            # Verify metadata
            step_evals = score.metadata["step_evaluations"]
            assert len(step_evals) == 2

            # All should be subtasks
            for step_eval in step_evals:
                assert step_eval["objective_type"] == "subtask"

    @pytest.mark.asyncio
    async def test_no_completions_evaluation(self, saber_scorer, task_state_with_episode):
        """Test evaluation with no completed objectives."""
        # Mock the evaluation endpoint response
        mock_evaluation_response = {
            "evaluation_result": {
                "grade": "0.0",
                "explanation": """
                Investigation failed:

                STEP_EVALUATIONS:
                [NO_COMPLETIONS]

                The agent did not complete any objectives.
                """,
                "metadata": {"judge_model": "gpt-4"}
            }
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock the HTTP response
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_evaluation_response)
            mock_response.raise_for_status = Mock()
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Execute scoring
            target = Target(target="198.43.121.209")
            score = await saber_scorer.score(task_state_with_episode, target)

            # Verify score results
            assert score.value == 0.0
            assert score.explanation == "Step evaluation: Main task not completed"

            # Verify metadata
            assert score.metadata["step_evaluations"] == []

    @pytest.mark.asyncio
    async def test_judge_parsing_error_handling(self, saber_scorer, task_state_with_episode):
        """Test handling of judge parsing errors."""
        # Mock the evaluation endpoint response with malformed step evaluations
        mock_evaluation_response = {
            "evaluation_result": {
                "grade": "0.5",
                "explanation": """
                This response is missing the STEP_EVALUATIONS section.
                The agent did some work but the format is incorrect.
                """,
                "metadata": {"judge_model": "gpt-4"}
            }
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock the HTTP response
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_evaluation_response)
            mock_response.raise_for_status = Mock()
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Execute scoring - should fail fast with RuntimeError
            target = Target(target="198.43.121.209")

            with pytest.raises(RuntimeError) as exc_info:
                await saber_scorer.score(task_state_with_episode, target)

            # Verify error message contains parsing details
            error_msg = str(exc_info.value)
            assert "STEP_EVALUATIONS section not found" in error_msg
            assert "Expected format" in error_msg

    @pytest.mark.asyncio
    async def test_evaluation_endpoint_error_handling(self, saber_scorer, task_state_with_episode):
        """Test handling of evaluation endpoint errors."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock HTTP error
            mock_response = AsyncMock()
            mock_response.raise_for_status.side_effect = Exception("HTTP 500: Internal Server Error")
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Execute scoring - should fail fast
            target = Target(target="test_target")

            with pytest.raises(Exception) as exc_info:
                await saber_scorer.score(task_state_with_episode, target)

            assert "HTTP 500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_episode_data_error(self, saber_scorer):
        """Test handling when episode data is missing from task state."""
        # Create task state without episode data
        state = Mock(spec=TaskState)
        state.metadata = {}  # No episode_data
        state.messages = [ChatMessageUser(content="Test task")]

        target = Target(target="test_target")

        with pytest.raises(RuntimeError) as exc_info:
            await saber_scorer.score(state, target)

        error_msg = str(exc_info.value)
        assert "Episode data not found" in error_msg

    @pytest.mark.asyncio
    async def test_step_evaluation_metadata_structure(self, saber_scorer, task_state_with_episode):
        """Test that step evaluation metadata has correct structure."""
        mock_evaluation_response = {
            "evaluation_result": {
                "grade": "1.0",
                "explanation": """
                STEP_EVALUATIONS:
                [2: checkpoint_alpha] - First checkpoint completed
                [5: checkpoint_beta] - Second checkpoint completed
                [8: main_task_id] - Main task completed successfully
                """,
                "metadata": {
                    "judge_model": "gpt-4",
                    "evaluation_time": "2025-01-15T10:30:00Z",
                    "additional_info": "test_data"
                }
            }
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_evaluation_response)
            mock_response.raise_for_status = Mock()
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Override task_id in episode data
            task_state_with_episode.metadata["episode_data"]["task_id"] = "main_task_id"

            target = Target(target="test_target")
            score = await saber_scorer.score(task_state_with_episode, target)

            # Verify score metadata structure
            metadata = score.metadata

            # Check step evaluations structure
            assert "step_evaluations" in metadata
            step_evals = metadata["step_evaluations"]
            assert isinstance(step_evals, list)
            assert len(step_evals) == 3

            # Check each step evaluation has required fields
            for step_eval in step_evals:
                assert "step_number" in step_eval
                assert "objective_id" in step_eval
                assert "objective_type" in step_eval
                assert "completed" in step_eval
                assert isinstance(step_eval["step_number"], int)
                assert isinstance(step_eval["objective_id"], str)
                assert isinstance(step_eval["objective_type"], str)
                assert isinstance(step_eval["completed"], bool)

            # Check step evaluation details
            assert step_evals[0]["step_number"] == 2
            assert step_evals[0]["objective_id"] == "checkpoint_alpha"
            assert step_evals[0]["objective_type"] == "subtask"

            assert step_evals[2]["step_number"] == 8
            assert step_evals[2]["objective_id"] == "main_task_id"
            assert step_evals[2]["objective_type"] == "task"

            # Check additional metadata preservation
            assert "task_completed_at_step" in metadata
            assert metadata["task_completed_at_step"] == 8
            assert "subtasks_completed" in metadata
            assert metadata["subtasks_completed"] == ["checkpoint_alpha", "checkpoint_beta"]

            # Check original judge metadata is preserved
            assert "judge_model" in metadata
            assert metadata["judge_model"] == "gpt-4"
            assert "evaluation_time" in metadata
            assert "additional_info" in metadata

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_handling(self, scorer_config):
        """Test that concurrent evaluations are handled correctly."""
        # Create scorer with higher concurrency
        scorer_config["concurrent_evaluations"] = 3
        scorer = SaberScorer(**scorer_config)

        # Create multiple task states
        task_states = []
        for i in range(3):
            state = Mock(spec=TaskState)
            state.metadata = {
                "episode_data": {
                    "episode_id": f"ep_{i}",
                    "steps": [{"step_number": 1, "command": "test", "output": "test"}],
                    "task_id": f"task_{i}",
                    "status": "completed"
                }
            }
            state.messages = [ChatMessageUser(content=f"Task {i}")]
            task_states.append(state)

        # Mock evaluation responses
        def create_mock_response(task_id):
            return {
                "evaluation_result": {
                    "grade": "1.0",
                    "explanation": f"""
                    STEP_EVALUATIONS:
                    [1: {task_id}] - Task {task_id} completed
                    """,
                    "metadata": {"judge_model": "gpt-4"}
                }
            }

        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock responses for each task
            responses = []
            for i in range(3):
                mock_response = AsyncMock()
                mock_response.json = AsyncMock(return_value=create_mock_response(f"task_{i}"))
                mock_response.raise_for_status = Mock()
                responses.append(mock_response)

            mock_post.return_value.__aenter__ = AsyncMock(side_effect=responses)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Execute concurrent scoring
            target = Target(target="test")
            tasks = [
                scorer.score(state, target)
                for state in task_states
            ]

            # Wait for all tasks to complete
            scores = await asyncio.gather(*tasks)

            # Verify all scores
            assert len(scores) == 3
            for i, score in enumerate(scores):
                assert score.value == 1.0
                assert f"task_{i}" in score.metadata["step_evaluations"][0]["objective_id"]
