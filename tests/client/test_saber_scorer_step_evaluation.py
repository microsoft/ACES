"""
                  mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Execute scoring
            target = Target(target="198.43.121.209")
            score = await saber_scorer_instance(task_state_with_episode, target)

            # Validate no completion scoreock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Execute scoring
            target = Target(target="198.43.121.209")
            score = await saber_scorer_instance(task_state_with_episode, target)

            # Validate partial completion scorests for step evaluation integration in the SABER client scorer.

Tests the integration of step evaluation parsing with the inspect_ai
sco            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            # Execute scoring
            target = Target(target="198.43.121.209")
            score = await saber_scorer_instance(task_state_with_episode, target)

            # Validate partial completion score
            assert score.value == 0.0ementation, focusing on the end-to-end evaluation workflow.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import asyncio

from inspect_ai.model import ChatMessage, ChatMessageUser
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.util import store

from saber.client.inspect_ai.saber_scorer import saber_scorer
from saber.models.rest.evaluation import EvaluationCriteriaResponse, StepEvaluation, TaskEvaluationContext, JudgeMessages
from saber.client.client_session import ClientSessionManager


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
    def mock_session_manager(self):
        """Create mock SABER session manager."""
        return Mock(spec=ClientSessionManager)

    @pytest.fixture
    def mock_episode(self):
        """Create mock episode for SABER context."""
        episode = Mock()
        episode.episode_id = "ep_123"
        return episode

    @pytest.fixture
    def saber_context(self, mock_session_manager, mock_episode):
        """Set up SABER context in inspect_ai store."""
        task_store = store()
        task_store.set("saber_session_manager", mock_session_manager)
        task_store.set("saber_session_id", "session_123")
        task_store.set("saber_current_episode", mock_episode)
        return {
            "session_manager": mock_session_manager,
            "session_id": "session_123",
            "episode": mock_episode
        }

    @pytest.fixture
    def saber_scorer_instance(self, scorer_config, saber_context):
        """Create saber_scorer instance with mocked context."""
        return saber_scorer(
            enable_override=scorer_config.get("enable_override", True),
            override_on_failure=scorer_config.get("override_on_failure", True),
            log_override_errors=scorer_config.get("log_override_errors", True)
        )

    @pytest.mark.asyncio
    async def test_successful_step_evaluation(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test successful step evaluation with task completion."""
        # Create mock evaluation criteria response
        mock_criteria = EvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            submission="Successfully identified target IP: 198.43.121.209",
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[
                    {"id": "initial_access", "description": "Establish user context"},
                    {"id": "log_analysis", "description": "Analyze logs for suspicious activity"},
                    {"id": "threat_attribution", "description": "Identify malicious domain"}
                ]
            ),
            evaluation_config={
                "strategy": "llm_judge",
                "scoring": {"max_score": 1.0}
            },
            judge_messages=JudgeMessages(
                system_message="You are an expert security analyst evaluating agent performance.",
                user_message="Evaluate the agent's investigation and provide step evaluations.",
                model="gpt-4"
            )
        )

        # Mock the session manager's get_evaluation_criteria method
        saber_context["session_manager"].get_evaluation_criteria.return_value = mock_criteria

        # Mock the LLM judge response
        mock_judge_response = """
        Investigation completed successfully:

        STEP_EVALUATIONS:
        [1: initial_access] - Established user context
        [3: log_analysis] - Found suspicious activity in logs
        [4: threat_attribution] - Identified malicious domain
        [5: incident_investigation_1] - Successfully identified target IP: 198.43.121.209

        The agent completed all required steps.
        """

        with patch('saber.client.inspect_ai.saber_scorer.get_model') as mock_get_model:
            # Mock the judge model and its response
            mock_model = AsyncMock()
            mock_response = Mock()
            mock_response.completion = mock_judge_response
            mock_model.generate.return_value = mock_response
            mock_get_model.return_value = mock_model

            # Execute scoring - saber_scorer_instance is the scoring function
            target = Target(target="198.43.121.209")
            score = await saber_scorer_instance(task_state_with_episode, target)

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
    async def test_partial_step_evaluation(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test partial step evaluation with only subtasks completed."""
        # Create mock evaluation criteria response for partial completion
        mock_criteria = EvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            submission="Found some suspicious activity in logs",
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[
                    {"id": "initial_access", "description": "Establish user context"},
                    {"id": "log_analysis", "description": "Analyze logs for suspicious activity"}
                ]
            ),
            evaluation_config={
                "strategy": "llm_judge",
                "scoring": {"max_score": 1.0}
            },
            judge_messages=JudgeMessages(
                system_message="You are an expert security analyst evaluating agent performance.",
                user_message="Evaluate the agent's investigation and provide step evaluations.",
                model="gpt-4"
            )
        )

        # Mock the session manager's get_evaluation_criteria method
        saber_context["session_manager"].get_evaluation_criteria.return_value = mock_criteria

        # Mock the LLM judge response with partial completion
        mock_judge_response = """
        Investigation partially completed:

        STEP_EVALUATIONS:
        [1: initial_access] - Established user context
        [3: log_analysis] - Found suspicious activity in logs

        The agent completed some steps but did not reach the final objective.
        """

        with patch('saber.client.inspect_ai.saber_scorer.get_model') as mock_get_model:
            # Mock the judge model and its response
            mock_model = AsyncMock()
            mock_response = Mock()
            mock_response.completion = mock_judge_response
            mock_model.generate.return_value = mock_response
            mock_get_model.return_value = mock_model

            # Execute scoring
            target = Target(target="198.43.121.209")
            score = await saber_scorer_instance(task_state_with_episode, target)

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
    async def test_no_completions_evaluation(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test evaluation with no completed objectives."""
        # Create mock evaluation criteria response for no completion
        mock_criteria = EvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            submission="Investigation started but no progress made",
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[]
            ),
            evaluation_config={
                "strategy": "llm_judge",
                "scoring": {"max_score": 1.0}
            },
            judge_messages=JudgeMessages(
                system_message="You are an expert security analyst evaluating agent performance.",
                user_message="Evaluate the agent's investigation and provide step evaluations.",
                model="gpt-4"
            )
        )

        # Mock the session manager's get_evaluation_criteria method
        saber_context["session_manager"].get_evaluation_criteria.return_value = mock_criteria

        # Mock the LLM judge response with no completions
        mock_judge_response = """
        Investigation failed:

        STEP_EVALUATIONS:
        [NO_COMPLETIONS]

        The agent did not complete any objectives.
        """

        with patch('saber.client.inspect_ai.saber_scorer.get_model') as mock_get_model:
            # Mock the judge model and its response
            mock_model = AsyncMock()
            mock_response = Mock()
            mock_response.completion = mock_judge_response
            mock_model.generate.return_value = mock_response
            mock_get_model.return_value = mock_model

            # Execute scoring
            target = Target(target="198.43.121.209")
            score = await saber_scorer_instance(task_state_with_episode, target)

            # Verify score results
            assert score.value == 0.0
            assert score.explanation == "Client-side Step evaluation: Main task not completed"

            # Verify metadata
            assert score.metadata["step_evaluations"] == []

    @pytest.mark.asyncio
    async def test_judge_parsing_error_handling(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test handling of judge parsing errors."""
        # Create mock evaluation criteria response
        mock_criteria = EvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            submission="Some work was done",
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[]
            ),
            evaluation_config={
                "strategy": "llm_judge",
                "scoring": {"max_score": 1.0}
            },
            judge_messages=JudgeMessages(
                system_message="You are an expert security analyst evaluating agent performance.",
                user_message="Evaluate the agent's investigation and provide step evaluations.",
                model="gpt-4"
            )
        )

        # Mock the session manager's get_evaluation_criteria method
        saber_context["session_manager"].get_evaluation_criteria.return_value = mock_criteria

        # Mock the LLM judge response with malformed step evaluations
        mock_judge_response = """
        This response is missing the STEP_EVALUATIONS section.
        The agent did some work but the format is incorrect.
        """

        with patch('saber.client.inspect_ai.saber_scorer.get_model') as mock_get_model:
            # Mock the judge model and its response
            mock_model = AsyncMock()
            mock_response = Mock()
            mock_response.completion = mock_judge_response
            mock_model.generate.return_value = mock_response
            mock_get_model.return_value = mock_model

            # Execute scoring - should fail fast with RuntimeError
            target = Target(target="198.43.121.209")

            with pytest.raises(RuntimeError) as exc_info:
                await saber_scorer_instance(task_state_with_episode, target)

            # Verify error message contains parsing details
            error_msg = str(exc_info.value)
            assert "STEP_EVALUATIONS section not found" in error_msg
            assert "Expected format" in error_msg

    @pytest.mark.asyncio
    async def test_evaluation_endpoint_error_handling(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test handling of evaluation endpoint errors."""
        # Mock the session manager to throw an exception when getting evaluation criteria
        saber_context["session_manager"].get_evaluation_criteria.side_effect = Exception("HTTP 500: Internal Server Error")

        # Execute scoring - should fail fast
        target = Target(target="test_target")

        with pytest.raises(Exception) as exc_info:
            await saber_scorer_instance(task_state_with_episode, target)

        assert "HTTP 500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_episode_data_error(self, scorer_config):
        """Test handling when episode data is missing from task state."""
        # Clear any existing store context first
        task_store = store()
        task_store.set("saber_session_manager", None)
        task_store.set("saber_session_id", None)
        task_store.set("saber_current_episode", None)

        # Create scorer instance without setting up SABER context
        scorer_instance = saber_scorer(
            enable_override=scorer_config.get("enable_override", True),
            override_on_failure=scorer_config.get("override_on_failure", True),
            log_override_errors=scorer_config.get("log_override_errors", True)
        )

        # Create task state without episode data
        state = Mock(spec=TaskState)
        state.metadata = {}  # No episode_data
        state.messages = [ChatMessageUser(content="Test task")]

        target = Target(target="test_target")

        with pytest.raises(RuntimeError) as exc_info:
            await scorer_instance(state, target)

        error_msg = str(exc_info.value)
        assert "SABER session manager not found in context" in error_msg

    @pytest.mark.asyncio
    async def test_step_evaluation_metadata_structure(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test that step evaluation metadata has correct structure."""
        # Create mock evaluation criteria response
        mock_criteria = EvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="main_task_id",
            submission="Task completed successfully",
            task_context=TaskEvaluationContext(
                task_id="main_task_id",
                title="Main Investigation Task",
                description="Main investigation task",
                domain="security",
                subtasks=[
                    {"id": "checkpoint_alpha", "description": "First checkpoint"},
                    {"id": "checkpoint_beta", "description": "Second checkpoint"}
                ]
            ),
            evaluation_config={
                "strategy": "llm_judge",
                "scoring": {"max_score": 1.0}
            },
            judge_messages=JudgeMessages(
                system_message="You are an expert security analyst evaluating agent performance.",
                user_message="Evaluate the agent's investigation and provide step evaluations.",
                model="gpt-4"
            )
        )

        # Mock the session manager's get_evaluation_criteria method
        saber_context["session_manager"].get_evaluation_criteria.return_value = mock_criteria

        mock_judge_response = """
        STEP_EVALUATIONS:
        [2: checkpoint_alpha] - First checkpoint completed
        [5: checkpoint_beta] - Second checkpoint completed
        [8: main_task_id] - Main task completed successfully
        """

        with patch('saber.client.inspect_ai.saber_scorer.get_model') as mock_get_model:
            mock_model = AsyncMock()
            mock_response = Mock()
            mock_response.completion = mock_judge_response
            mock_model.generate.return_value = mock_response
            mock_get_model.return_value = mock_model

            # Override task_id in episode data
            task_state_with_episode.metadata["episode_data"]["task_id"] = "main_task_id"

            target = Target(target="test_target")
            score = await saber_scorer_instance(task_state_with_episode, target)

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
            assert "strategy" in metadata
            assert metadata["strategy"] == "llm_judge_step_evaluation"

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_handling(self, scorer_config, saber_context):
        """Test that concurrent evaluations are handled correctly."""
        # Create scorer with higher concurrency
        scorer_config["concurrent_evaluations"] = 3
        scorer = saber_scorer(
            enable_override=scorer_config.get("enable_override", True),
            override_on_failure=scorer_config.get("override_on_failure", True),
            log_override_errors=scorer_config.get("log_override_errors", True)
        )

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

        # Mock evaluation criteria for each task
        def create_mock_criteria(task_id):
            return EvaluationCriteriaResponse(
                session_id="session_123",
                episode_id=f"ep_{task_id.split('_')[1]}",
                task_id=task_id,
                submission=f"Task {task_id} completed",
                task_context=TaskEvaluationContext(
                    task_id=task_id,
                    title=f"Task {task_id}",
                    description=f"Task {task_id}",
                    domain="security",
                    subtasks=[]
                ),
                evaluation_config={
                    "strategy": "llm_judge",
                    "scoring": {"max_score": 1.0}
                },
                judge_messages=JudgeMessages(
                    system_message="You are an expert security analyst evaluating agent performance.",
                    user_message=f"Evaluate task {task_id}.",
                    model="gpt-4"
                )
            )

        # Mock responses for each task
        def create_mock_response(task_id):
            return f"""
            STEP_EVALUATIONS:
            [1: {task_id}] - Task {task_id} completed
            """

        with patch('saber.client.inspect_ai.saber_scorer.get_model') as mock_get_model:
            # Mock the judge model for all evaluations
            mock_model = AsyncMock()
            mock_get_model.return_value = mock_model

            # Set up side effects for different evaluations
            mock_responses = []
            for i in range(3):
                mock_response = Mock()
                mock_response.completion = create_mock_response(f"task_{i}")
                mock_responses.append(mock_response)

            mock_model.generate.side_effect = mock_responses

            # Mock session manager responses
            criteria_responses = [create_mock_criteria(f"task_{i}") for i in range(3)]
            saber_context["session_manager"].get_evaluation_criteria.side_effect = criteria_responses

            # Execute concurrent scoring
            target = Target(target="test")
            tasks = [
                scorer(state, target)
                for state in task_states
            ]

            # Wait for all tasks to complete
            scores = await asyncio.gather(*tasks)

            # Verify all scores
            assert len(scores) == 3
            for i, score in enumerate(scores):
                assert score.value == 1.0
                assert f"task_{i}" in score.metadata["step_evaluations"][0]["objective_id"]
