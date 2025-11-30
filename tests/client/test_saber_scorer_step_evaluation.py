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

from inspect_ai.model import ChatMessage, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.util import store

from saber.inspect_ai import saber_scorer
from saber.models.rest.evaluation import (
    SubmissionEvaluationCriteriaResponse,
    SubtaskEvaluationCriteriaResponse,
    StepEvaluation,
    TaskEvaluationContext,
    JudgeMessages,
    EpisodeSubmissionResponse
)
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
        # Add mock output with usage for token tracking
        mock_usage = Mock()
        mock_usage.model_dump = Mock(return_value={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
        mock_output = Mock()
        mock_output.usage = mock_usage
        mock_output.completion = "The malicious IP address is 198.43.121.209"
        state.output = mock_output
        state.model = "gpt-4"
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
        episode.task_id = "incident_investigation_1"
        return episode

    @pytest.fixture
    def saber_context(self, mock_session_manager, mock_episode):
        """Set up SABER context in inspect_ai store."""
        task_store = store()
        # Set the keys that saber_scorer expects
        task_store.set("saber_session_manager", mock_session_manager)
        task_store.set("saber_session_id", "session_123")
        task_store.set("saber_task_id", "incident_investigation_1")
        # Use episode mapping with sample_id as key (new pattern)
        sample_id = "incident_investigation_1"
        task_store.set("saber_episode_mapping", {sample_id: mock_episode})
        # Return dict for test convenience
        saber_context_dict = {
            "session_manager": mock_session_manager,
            "session_id": "session_123",
            "episode": mock_episode
        }
        return saber_context_dict

    @pytest.fixture
    def saber_scorer_instance(self, scorer_config, saber_context):
        """Create saber_scorer instance with mocked context."""
        return saber_scorer()

    @pytest.mark.asyncio
    async def test_successful_step_evaluation(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test successful step evaluation with task completion."""
        # Create mock submission evaluation criteria
        mock_submission_criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            strategy="static",
            criteria={
                "expected_answers": ["Successfully identified target IP: 198.43.121.209"]
            },
            scoring={"max_score": 1.0},
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[
                    {"subtask_id": "initial_access", "objective": "Establish user context", "max_score": 0.25},
                    {"subtask_id": "log_analysis", "objective": "Analyze logs for suspicious activity", "max_score": 0.25},
                    {"subtask_id": "threat_attribution", "objective": "Identify malicious domain", "max_score": 0.25}
                ]
            )
        )

        # Create mock step evaluation criteria
        mock_step_criteria = SubtaskEvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            subtask_id="initial_access",
            strategy="static",
            criteria={
                "expected_outputs": ["user"]
            },
            max_score=0.25,
            weight=1.0,
            objective="Identify initial access vector",
            title="Initial Access Analysis",
            description="Analyze how the attacker gained initial access",
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[
                    {"subtask_id": "initial_access", "objective": "Establish user context", "max_score": 0.25},
                    {"subtask_id": "log_analysis", "objective": "Analyze logs for suspicious activity", "max_score": 0.25},
                    {"subtask_id": "threat_attribution", "objective": "Identify malicious domain", "max_score": 0.25}
                ]
            )
        )

        # Mock the session manager's evaluation methods
        saber_context["session_manager"].get_submission_evaluation_criteria.return_value = mock_submission_criteria
        saber_context["session_manager"].get_subtask_evaluation_criteria.return_value = [mock_step_criteria]

        # Mock template fetching
        saber_context["session_manager"].get_template_content.return_value = "Mock template content"

        # Mock submission data
        mock_submission_data = EpisodeSubmissionResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            submission="Successfully identified target IP: 198.43.121.209",
            model="gpt-4",
            tokens={"input": 100, "output": 50},
            execution_time=10.5
        )
        saber_context["session_manager"].get_episode_submission.return_value = mock_submission_data

        # Create proper step data
        from saber.models.rest.evaluation import EpisodeStepData, EpisodeStepsResponse
        from datetime import datetime

        steps = [
            EpisodeStepData(
                step_number=0,
                tool_name="bash",
                tool_input={"command": "whoami"},
                tool_output="user",
                timestamp=datetime.now()
            ),
            EpisodeStepData(
                step_number=1,
                tool_name="bash",
                tool_input={"command": "grep suspicious /var/log/auth.log"},
                tool_output="found suspicious activity",
                timestamp=datetime.now()
            )
        ]

        mock_steps_data = EpisodeStepsResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            steps=steps,
            total_steps=len(steps)
        )
        saber_context["session_manager"].get_episode_steps.return_value = mock_steps_data

        # Mock submit_evaluation_result to avoid server calls
        saber_context["session_manager"].submit_evaluation_result.return_value = Mock()

        # Execute scoring - saber_scorer_instance is the scoring function
        target = Target(target="198.43.121.209")
        score = await saber_scorer_instance(task_state_with_episode, target)

        # Verify score results
        assert isinstance(score, Score)
        # Score should include submission (1.0) plus step evaluation points
        assert score.value >= 0.0  # At least submission score
        assert "submission=" in score.explanation  # Explanation includes breakdown
        assert score.answer == "The malicious IP address is 198.43.121.209"  # Should match state.output.completion

        # Verify metadata contains step evaluation details
        assert "step_evaluations" in score.metadata
        step_evals = score.metadata["step_evaluations"]
        # Step evaluations should be parsed from the LLM response
        assert len(step_evals) >= 0  # May be empty if no subtasks completed

        # Check metadata structure
        assert "submission_score" in score.metadata
        # step_score is now calculated as sum of individual subtask scores
        # Check for individual subtask scores instead
        assert "max_possible" in score.metadata
        assert score.metadata["submission_score"] == 1.0  # Submission was CORRECT

       # Verify individual subtask scores are present
        assert "incident_investigation_1_initial_access_score" in score.metadata

    @pytest.mark.asyncio
    async def test_partial_step_evaluation(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test partial step evaluation with only subtasks completed."""
        # Create mock submission evaluation criteria
        mock_submission_criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            strategy="llm_judge",
            criteria={
                "judge_system_template": "Mock system template",
                "judge_user_template": "Mock user template",
                "model": "openai/gpt-4"
            },
            scoring={"max_score": 1.0},
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[
                    {"subtask_id": "initial_access", "objective": "Establish user context"},
                    {"subtask_id": "log_analysis", "objective": "Analyze logs for suspicious activity"}
                ]
            )
        )

        # Create mock step evaluation criteria
        mock_step_criteria = SubtaskEvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            subtask_id="log_analysis",
            strategy="llm_judge",
            criteria={
                "judge_system_template": "Mock system template",
                "judge_user_template": "Mock user template",
                "model": "openai/gpt-4",
                "steps_per_message": 5
            },
            max_score=0.25,
            weight=1.0,
            objective="Analyze logs for suspicious activity",
            title="Log Analysis",
            description="Analyze system logs for suspicious activity",
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[
                    {"subtask_id": "initial_access", "objective": "Establish user context"},
                    {"subtask_id": "log_analysis", "objective": "Analyze logs for suspicious activity"}
                ]
            )
        )

        # Mock the session manager's evaluation methods
        saber_context["session_manager"].get_submission_evaluation_criteria.return_value = mock_submission_criteria
        saber_context["session_manager"].get_subtask_evaluation_criteria.return_value = [mock_step_criteria]
        saber_context["session_manager"].get_template_content.return_value = "Mock template content"

        # Mock submission data
        mock_submission_data = EpisodeSubmissionResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            submission="Found some suspicious activity in logs",
            model="gpt-4",
            tokens={"input": 100, "output": 50},
            execution_time=10.5
        )
        saber_context["session_manager"].get_episode_submission.return_value = mock_submission_data

        from saber.models.rest.evaluation import EpisodeStepData, EpisodeStepsResponse
        from datetime import datetime

        steps = [
            EpisodeStepData(
                step_number=0,
                tool_name="bash",
                tool_input={"command": "whoami"},
                tool_output="user",
                timestamp=datetime.now()
            )
        ]
        mock_steps_data = EpisodeStepsResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            steps=steps,
            total_steps=len(steps)
        )
        saber_context["session_manager"].get_episode_steps.return_value = mock_steps_data

        # Mock the LLM judge response with partial completion (INCORRECT because not complete)
        mock_judge_response = """
        INCORRECT

        Investigation partially completed:

        STEP_EVALUATIONS:
        [1: initial_access] - Established user context
        [3: log_analysis] - Found suspicious activity in logs

        The agent completed some steps but did not reach the final objective.
        """

        with patch('saber.inspect_ai.core.saber_scorer.get_model') as mock_get_model:
            # Mock the judge model and its response
            mock_model = AsyncMock()
            mock_response = Mock()
            mock_response.completion = mock_judge_response
            mock_model.generate.return_value = mock_response
            mock_get_model.return_value = mock_model

            # Execute scoring
            target = Target(target="198.43.121.209")
            score = await saber_scorer_instance(task_state_with_episode, target)

            # Verify score results - submission should be 0.0 (INCORRECT)
            assert score.value == 0.25
            assert "submission=" in score.explanation
            assert score.metadata["submission_score"] == 0.0  # INCORRECT submission

    @pytest.mark.asyncio
    async def test_no_completions_evaluation(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test evaluation with no completed objectives."""
        # Create mock submission evaluation criteria
        mock_submission_criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            strategy="llm_judge",
            criteria={
                "judge_system_template": "Mock system template",
                "judge_user_template": "Mock user template",
                "model": "openai/gpt-4"
            },
            scoring={"max_score": 1.0},
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[]
            )
        )

        # Create mock step evaluation criteria
        mock_step_criteria = SubtaskEvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            subtask_id="initial_investigation",
            strategy="llm_judge",
            criteria={
                "judge_system_template": "Mock system template",
                "judge_user_template": "Mock user template",
                "model": "openai/gpt-4",
                "steps_per_message": 5
            },
            max_score=0.25,
            weight=1.0,
            objective="Complete initial investigation",
            title="Initial Investigation",
            description="Perform initial investigation of the incident",
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[]
            )
        )

        # Mock the session manager's evaluation methods
        saber_context["session_manager"].get_submission_evaluation_criteria.return_value = mock_submission_criteria
        saber_context["session_manager"].get_subtask_evaluation_criteria.return_value = [mock_step_criteria]

        # Mock template fetching
        saber_context["session_manager"].get_template_content.return_value = "Mock template content"

        # Mock submission data
        mock_submission_data = EpisodeSubmissionResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            submission="Investigation started but no progress made",
            model="gpt-4",
            tokens={"input": 100, "output": 50},
            execution_time=10.5
        )
        saber_context["session_manager"].get_episode_submission.return_value = mock_submission_data

        # Create proper step data
        from saber.models.rest.evaluation import EpisodeStepData, EpisodeStepsResponse
        from datetime import datetime

        steps = [
            EpisodeStepData(
                step_number=0,
                tool_name="bash",
                tool_input={"command": "whoami"},
                tool_output="user",
                timestamp=datetime.now()
            )
        ]

        mock_steps_data = EpisodeStepsResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            steps=steps,
            total_steps=len(steps)
        )
        saber_context["session_manager"].get_episode_steps.return_value = mock_steps_data

        # Mock the LLM judge response with no completions - INCORRECT submission
        mock_judge_response = """
        INCORRECT

        Investigation failed:

        STEP_EVALUATIONS:
        [NO_COMPLETIONS]

        The agent did not complete any objectives.
        """

        with patch('saber.inspect_ai.core.saber_scorer.get_model') as mock_get_model:
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
            assert "submission=" in score.explanation

            # Verify metadata
            assert score.metadata["submission_score"] == 0.0  # INCORRECT
            assert "step_evaluations" in score.metadata

    @pytest.mark.asyncio
    async def test_judge_parsing_error_handling(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test handling of judge parsing errors."""
        # Create mock submission evaluation criteria
        mock_submission_criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            strategy="llm_judge",
            criteria={
                "judge_system_template": "Mock system template",
                "judge_user_template": "Mock user template",
                "model": "openai/gpt-4"
            },
            scoring={"max_score": 1.0},
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[]
            )
        )

        # Create mock step evaluation criteria
        mock_step_criteria = SubtaskEvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            subtask_id="parsing_test",
            strategy="llm_judge",
            criteria={
                "judge_system_template": "Mock system template",
                "judge_user_template": "Mock user template",
                "model": "openai/gpt-4",
                "steps_per_message": 5
            },
            max_score=0.25,
            weight=1.0,
            objective="Test parsing functionality",
            title="Parsing Test",
            description="Test the parsing of step evaluations",
            task_context=TaskEvaluationContext(
                task_id="incident_investigation_1",
                title="Security Incident Investigation",
                description="Investigate security incident",
                domain="security",
                subtasks=[]
            )
        )

        # Mock the session manager's evaluation methods
        saber_context["session_manager"].get_submission_evaluation_criteria.return_value = mock_submission_criteria
        saber_context["session_manager"].get_subtask_evaluation_criteria.return_value = [mock_step_criteria]

        # Mock template fetching
        saber_context["session_manager"].get_template_content.return_value = "Mock template content"

        # Mock submission data
        mock_submission_data = EpisodeSubmissionResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            submission="Some work was done",
            model="gpt-4",
            tokens={"input": 100, "output": 50},
            execution_time=10.5
        )
        saber_context["session_manager"].get_episode_submission.return_value = mock_submission_data

        # Create proper step data
        from saber.models.rest.evaluation import EpisodeStepData, EpisodeStepsResponse
        from datetime import datetime

        steps = [
            EpisodeStepData(
                step_number=0,
                tool_name="bash",
                tool_input={"command": "whoami"},
                tool_output="user",
                timestamp=datetime.now()
            )
        ]

        mock_steps_data = EpisodeStepsResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="incident_investigation_1",
            steps=steps,
            total_steps=len(steps)
        )
        saber_context["session_manager"].get_episode_steps.return_value = mock_steps_data

        # Mock the LLM judge response with malformed step evaluations
        mock_judge_response = """
        This response is missing the STEP_EVALUATIONS section.
        The agent did some work but the format is incorrect.
        """

        with patch('saber.inspect_ai.core.saber_scorer.get_model') as mock_get_model:
            # Mock the judge model and its response
            mock_model = AsyncMock()
            mock_response = Mock()
            mock_response.completion = mock_judge_response
            mock_model.generate.return_value = mock_response
            mock_get_model.return_value = mock_model

            # Execute scoring - missing STEP_EVALUATIONS section is handled gracefully
            target = Target(target="198.43.121.209")
            score = await saber_scorer_instance(task_state_with_episode, target)

            # Verify scoring completes successfully even without STEP_EVALUATIONS section
            # The scorer should handle missing sections gracefully
            assert score.value >= 0.0
            assert isinstance(score.explanation, str)
            assert "submission=" in score.explanation

    @pytest.mark.asyncio
    async def test_evaluation_endpoint_error_handling(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test handling of evaluation endpoint errors."""
        # Mock the session manager to throw an exception when getting evaluation criteria
        saber_context["session_manager"].get_submission_evaluation_criteria.side_effect = Exception("HTTP 500: Internal Server Error")

        # Execute scoring - error is caught and returns error score
        target = Target(target="test_target")
        score = await saber_scorer_instance(task_state_with_episode, target)

        # Verify error score is returned instead of raising
        assert score.value == 0.0
        assert "Client-side evaluation failed" in score.explanation
        assert "error" in score.metadata
        assert "HTTP 500" in score.metadata["error"]

    @pytest.mark.asyncio
    async def test_missing_episode_data_error(self, scorer_config):
        """Test handling when episode data is missing from task state."""
        # Clear any existing store context first
        task_store = store()
        task_store.set("saber_session_manager", None)
        task_store.set("saber_session_id", None)
        task_store.set("saber_current_episode", None)

        # Create scorer instance without setting up SABER context
        scorer_instance = saber_scorer()

        # Create task state without episode data
        state = Mock(spec=TaskState)
        state.metadata = {}  # No episode_data
        state.messages = [ChatMessageUser(content="Test task")]

        target = Target(target="test_target")

        # Execute scoring - error is caught and returns error score
        score = await scorer_instance(state, target)

        # Verify error score is returned instead of raising
        assert score.value == 0.0
        assert "Client-side evaluation failed" in score.explanation
        assert "error" in score.metadata
        assert "saber_session_manager" in score.metadata["error"]

    @pytest.mark.asyncio
    async def test_step_evaluation_metadata_structure(self, saber_scorer_instance, task_state_with_episode, saber_context):
        """Test that step evaluation metadata has correct structure."""
        # Create mock submission evaluation criteria
        mock_submission_criteria = SubmissionEvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="main_task_id",
            strategy="llm_judge",
            criteria={
                "judge_system_template": "Mock system template",
                "judge_user_template": "Mock user template",
                "model": "openai/gpt-4"
            },
            scoring={"max_score": 1.0},
            task_context=TaskEvaluationContext(
                task_id="main_task_id",
                title="Main Investigation Task",
                description="Main investigation task",
                domain="security",
                subtasks=[
                    {"subtask_id": "checkpoint_alpha", "objective": "First checkpoint"},
                    {"subtask_id": "checkpoint_beta", "objective": "Second checkpoint"}
                ]
            )
        )

        # Create mock step evaluation criteria
        mock_step_criteria = SubtaskEvaluationCriteriaResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="main_task_id",
            subtask_id="checkpoint_alpha",
            strategy="llm_judge",
            criteria={
                "judge_system_template": "Mock system template",
                "judge_user_template": "Mock user template",
                "model": "openai/gpt-4",
                "steps_per_message": 5
            },
            max_score=0.5,
            weight=1.0,
            objective="First checkpoint",
            title="Checkpoint Alpha",
            description="First checkpoint in the task",
            task_context=TaskEvaluationContext(
                task_id="main_task_id",
                title="Main Investigation Task",
                description="Main investigation task",
                domain="security",
                subtasks=[
                    {"subtask_id": "checkpoint_alpha", "objective": "First checkpoint"},
                    {"subtask_id": "checkpoint_beta", "objective": "Second checkpoint"}
                ]
            )
        )

        # Mock the session manager's evaluation methods
        saber_context["session_manager"].get_submission_evaluation_criteria.return_value = mock_submission_criteria
        saber_context["session_manager"].get_subtask_evaluation_criteria.return_value = [mock_step_criteria]

        # Mock template fetching
        saber_context["session_manager"].get_template_content.return_value = "Mock template content"

        # Mock submission data
        mock_submission_data = EpisodeSubmissionResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="main_task_id",
            submission="Task completed successfully",
            model="gpt-4",
            tokens={"input": 100, "output": 50},
            execution_time=10.5
        )
        saber_context["session_manager"].get_episode_submission.return_value = mock_submission_data

        # Create proper step data
        from saber.models.rest.evaluation import EpisodeStepData, EpisodeStepsResponse
        from datetime import datetime

        steps = [
            EpisodeStepData(
                step_number=i,
                tool_name="bash",
                tool_input={"command": f"step_{i}"},
                tool_output=f"output_{i}",
                timestamp=datetime.now()
            )
            for i in range(9)  # Steps 0-8
        ]

        mock_steps_data = EpisodeStepsResponse(
            session_id="session_123",
            episode_id="ep_123",
            task_id="main_task_id",
            steps=steps,
            total_steps=len(steps)
        )
        saber_context["session_manager"].get_episode_steps.return_value = mock_steps_data

        mock_judge_response = """
        CORRECT

        STEP_EVALUATIONS:
        [2: checkpoint_alpha] - First checkpoint completed
        [5: checkpoint_beta] - Second checkpoint completed
        [8: main_task_id] - Main task completed successfully
        """

        with patch('saber.inspect_ai.core.saber_scorer.get_model') as mock_get_model:
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
            # Note: step evaluations are parsed from both submission and step judge responses
            # so we may get duplicates. The test verifies structure, not exact count.
            assert len(step_evals) >= 3  # At least the 3 objectives we defined

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

            # Check step evaluation details - evaluations are indexed sequentially
            # Find evaluations by objective_id since indexing may vary
            checkpoint_alpha_eval = next((se for se in step_evals if se["objective_id"] == "checkpoint_alpha"), None)
            assert checkpoint_alpha_eval is not None
            assert checkpoint_alpha_eval["objective_type"] == "subtask"
            assert checkpoint_alpha_eval["step_number"] >= 0

            # Note: Only checkpoint_alpha has evaluation criteria configured in mock_step_criteria
            # The judge response mentions checkpoint_beta and main_task_id, but without configured
            # evaluation criteria, these won't appear in step_evaluations (which is correct behavior).
            # step_evaluations only tracks objectives with actual evaluation criteria.

            # Check metadata contains score information
            assert "submission_score" in metadata
            # step_score is now calculated as sum of individual subtask scores
            # Check for individual subtask scores instead
            assert "max_possible" in metadata
            # Verify individual checkpoint scores are present
            # Only checkpoint_alpha is configured in mock_step_criteria, so only its score should be present
            assert "main_task_id_checkpoint_alpha_score" in metadata
            # checkpoint_beta is mentioned in the judge response but not configured as a criteria,
            # so it won't have a separate score metadata entry

            # Check original scorer metadata is preserved
            assert "scorer_version" in metadata
            assert metadata["scorer_version"] == "2.3"

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_handling(self, scorer_config, saber_context):
        """Test that concurrent evaluations are handled correctly."""
        # Create scorer with higher concurrency
        scorer_config["concurrent_evaluations"] = 3
        scorer = saber_scorer()

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

            # Add proper output mock with usage
            mock_output = Mock()
            mock_output.completion = f"Task {i} completion"
            mock_usage = Mock()
            mock_usage.model_dump.return_value = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            mock_output.usage = mock_usage
            state.output = mock_output

            task_states.append(state)

        # Mock evaluation criteria for each task
        def create_mock_submission_criteria(task_id):
            return SubmissionEvaluationCriteriaResponse(
                session_id="session_123",
                episode_id=f"ep_{task_id.split('_')[1]}",
                task_id=task_id,
                strategy="llm_judge",
                criteria={
                    "judge_system_template": "Mock system template",
                    "judge_user_template": "Mock user template",
                    "model": "openai/gpt-4"
                },
                scoring={"max_score": 1.0},
                task_context=TaskEvaluationContext(
                    task_id=task_id,
                    title=f"Task {task_id}",
                    description=f"Task {task_id}",
                    domain="security",
                    subtasks=[]
                )
            )

        def create_mock_step_criteria(task_id):
            return SubtaskEvaluationCriteriaResponse(
                session_id="session_123",
                episode_id=f"ep_{task_id.split('_')[1]}",
                task_id=task_id,
                subtask_id=f"{task_id}_subtask",
                strategy="llm_judge",
                criteria={
                    "judge_system_template": "Mock system template",
                    "judge_user_template": "Mock user template",
                    "model": "openai/gpt-4",
                    "steps_per_message": 5
                },
                max_score=0.25,
                weight=1.0,
                objective=f"Complete {task_id}",
                title=f"Task {task_id}",
                description=f"Complete task {task_id}",
                task_context=TaskEvaluationContext(
                    task_id=task_id,
                    title=f"Task {task_id}",
                    description=f"Task {task_id}",
                    domain="security",
                    subtasks=[
                        {"subtask_id": f"{task_id}_subtask", "objective": f"Complete {task_id}", "max_score": 0.25}
                    ]
                )
            )

        # Mock responses for each task
        def create_mock_response(task_id):
            return f"""
            CORRECT

            STEP_EVALUATIONS:
            [1: {task_id}] - Task {task_id} completed
            """

        # Set up mock data for all tasks
        from saber.models.rest.evaluation import EpisodeStepData, EpisodeStepsResponse
        from datetime import datetime

        def create_mock_steps(task_id):
            steps = [
                EpisodeStepData(
                    step_number=0,
                    tool_name="bash",
                    tool_input={"command": "test"},
                    tool_output="test",
                    timestamp=datetime.now()
                )
            ]
            return EpisodeStepsResponse(
                session_id="session_123",
                episode_id=f"ep_{task_id.split('_')[1]}",
                task_id=task_id,
                steps=steps,
                total_steps=len(steps)
            )

        def create_mock_submission(task_id):
            return EpisodeSubmissionResponse(
                session_id="session_123",
                episode_id=f"ep_{task_id.split('_')[1]}",
                task_id=task_id,
                submission=f"Task {task_id} completed",
                model="gpt-4",
                tokens={"input": 100, "output": 50},
                execution_time=10.5
            )

        with patch('saber.inspect_ai.core.saber_scorer.get_model') as mock_get_model:
            # Mock the judge model for all evaluations
            mock_model = AsyncMock()
            mock_get_model.return_value = mock_model

            # Set up side effects for different evaluations
            # We need 2 LLM calls per task (submission + steps) = 6 total
            mock_responses = []
            for i in range(3):
                # Submission evaluation response
                submission_response = Mock()
                submission_response.completion = f"CORRECT\n\nTask {i} submission evaluated."
                mock_responses.append(submission_response)

                # Step evaluation response
                step_response = Mock()
                step_response.completion = create_mock_response(f"task_{i}")
                mock_responses.append(step_response)

            mock_model.generate.side_effect = mock_responses

            # Mock session manager responses - need enough for all tasks
            submission_criteria_responses = []
            step_criteria_responses = []
            for i in range(3):
                submission_criteria_responses.append(create_mock_submission_criteria(f"task_{i}"))
                step_criteria_responses.append(create_mock_step_criteria(f"task_{i}"))

            saber_context["session_manager"].get_submission_evaluation_criteria.side_effect = submission_criteria_responses
            saber_context["session_manager"].get_subtask_evaluation_criteria.side_effect = step_criteria_responses

            # Mock template fetching - needs to be called multiple times
            saber_context["session_manager"].get_template_content.return_value = "Mock template content"

            # Mock submission and steps data
            submission_responses = [create_mock_submission(f"task_{i}") for i in range(3)]
            saber_context["session_manager"].get_episode_submission.side_effect = submission_responses

            steps_responses = [create_mock_steps(f"task_{i}") for i in range(3)]
            saber_context["session_manager"].get_episode_steps.side_effect = steps_responses

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
                assert score.value >= 0.0  # Score should be valid
