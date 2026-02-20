"""Regression tests for state.messages preservation during scoring.

The scoring functions previously mutated state.messages by calling state.messages.clear()
and then appending scorer messages. This destroyed the agent's conversation history that
gets written to the eval file. The fix was to use separate scorer_messages lists.

These tests verify the fix remains in place.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.solver import TaskState

from saber.inspect_ai.core.scoring.standard_submission import (
    score_submission_llm,
    score_submission_static,
)
from saber.inspect_ai.core.scoring.standard_subtask import score_subtask_llm
from saber.models.rest.evaluation import (
    EpisodeStepData,
    EpisodeStepsResponse,
    EpisodeSubmissionResponse,
    SubtaskEvaluationCriteriaResponse,
    SubmissionEvaluationCriteriaResponse,
    TaskEvaluationContext,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def state_with_messages() -> tuple[MagicMock, list[ChatMessageUser | ChatMessageAssistant]]:
    """Create a TaskState with pre-populated messages (simulating solver conversation).

    Returns:
        Tuple of (mock state, original messages list for comparison)
    """
    state = MagicMock(spec=TaskState)
    original_messages = [
        ChatMessageUser(content="What is the network topology?"),
        ChatMessageAssistant(content="Let me investigate using nmap..."),
        ChatMessageUser(content="Good, what did you find?"),
        ChatMessageAssistant(content="I found 3 hosts on the network."),
    ]
    state.messages = list(original_messages)  # Copy so we can compare
    state.output = MagicMock()
    state.output.completion = "test answer"
    return state, original_messages


@pytest.fixture
def task_context() -> TaskEvaluationContext:
    """Create a sample TaskEvaluationContext."""
    return TaskEvaluationContext(
        task_id="test-task",
        title="Test Task",
        description="Find the network topology",
        domain="test_domain",
        subtasks=[],
    )


@pytest.fixture
def submission_data() -> EpisodeSubmissionResponse:
    """Create sample submission data."""
    return EpisodeSubmissionResponse(
        session_id="session1",
        episode_id="ep1",
        task_id="test-task",
        submission="3 hosts found on the network",
        model="test-model",
        tokens={},
    )


@pytest.fixture
def llm_submission_criteria(task_context: TaskEvaluationContext) -> SubmissionEvaluationCriteriaResponse:
    """Create LLM judge submission criteria."""
    return SubmissionEvaluationCriteriaResponse(
        session_id="session1",
        episode_id="ep1",
        task_id="test-task",
        strategy="llm_judge",
        criteria={
            "judge_system_template": "You are a judge. Evaluate the answer.",
            "judge_user_template": "Question: {{ question }}\nAnswer: {{ submission }}\nExpected: {{ golden_answer }}",
            "model": "test-model",
            "golden_answer": "3 hosts",
        },
        scoring={"max_score": 1.0},
        task_context=task_context,
    )


@pytest.fixture
def static_submission_criteria(task_context: TaskEvaluationContext) -> SubmissionEvaluationCriteriaResponse:
    """Create static submission criteria."""
    return SubmissionEvaluationCriteriaResponse(
        session_id="session1",
        episode_id="ep1",
        task_id="test-task",
        strategy="static",
        criteria={"expected_answers": ["3 hosts"]},
        scoring={"max_score": 1.0},
        task_context=task_context,
    )


@pytest.fixture
def llm_subtask_criteria(task_context: TaskEvaluationContext) -> SubtaskEvaluationCriteriaResponse:
    """Create LLM judge subtask criteria."""
    return SubtaskEvaluationCriteriaResponse(
        session_id="session1",
        episode_id="ep1",
        task_id="test-task",
        subtask_id="subtask-1",
        strategy="llm_judge",
        criteria={
            "judge_system_template": "You are an evaluator.\nAssess whether the agent completed the subtask.",
            "judge_user_template": "Task: {{ question }}\nSubtask: {{ subtask.description }}\nSteps:\n{% for step in episode.steps %}Step {{ step.step_number }}: {{ step.tool_name }}\n{% endfor %}",
            "model": "test-model",
            "steps_per_message": 10,
        },
        max_score=1.0,
        weight=1.0,
        objective="Discover the network topology",
        title="Network Discovery",
        description="Use nmap to discover hosts on the network",
        task_context=task_context,
    )


@pytest.fixture
def steps_data() -> EpisodeStepsResponse:
    """Create sample steps data."""
    return EpisodeStepsResponse(
        session_id="session1",
        episode_id="ep1",
        task_id="test-task",
        steps=[
            EpisodeStepData(
                step_number=1,
                tool_name="execute_command",
                tool_input={"command": "nmap -sn 10.0.0.0/24"},
                tool_output="3 hosts up",
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                done=False,
                assistant_message="Let me scan the network.",
                reasoning="Need to discover hosts.",
            ),
        ],
        total_steps=1,
    )


# ============================================================================
# Regression Tests
# ============================================================================


class TestStateMessagesPreservation:
    """Regression tests ensuring scoring functions do NOT mutate state.messages.

    Previously, scoring functions called state.messages.clear() before appending
    scorer messages, which destroyed the agent's conversation history in the
    eval file. The fix uses separate scorer_messages lists.
    """

    @pytest.mark.asyncio
    async def test_score_submission_llm_preserves_state_messages(
        self,
        state_with_messages: tuple[MagicMock, list[ChatMessageUser | ChatMessageAssistant]],
        submission_data: EpisodeSubmissionResponse,
        llm_submission_criteria: SubmissionEvaluationCriteriaResponse,
    ) -> None:
        """Verify score_submission_llm does not mutate state.messages."""
        state, original_messages = state_with_messages

        mock_output = MagicMock(spec=ModelOutput)
        mock_output.completion = "CORRECT"
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        with patch(
            "saber.inspect_ai.core.scoring.standard_submission.resolve_template_content",
            new_callable=AsyncMock,
            side_effect=lambda _sm, val: val,
        ):
            with patch(
                "saber.inspect_ai.core.scoring.standard_submission.get_model",
                return_value=mock_model,
            ):
                await score_submission_llm(submission_data, llm_submission_criteria, MagicMock(), state)

        assert len(state.messages) == len(original_messages), (
            f"state.messages length changed from {len(original_messages)} to {len(state.messages)}"
        )
        assert state.messages == original_messages, "state.messages content was mutated by scorer"

    @pytest.mark.asyncio
    async def test_score_subtask_llm_preserves_state_messages(
        self,
        state_with_messages: tuple[MagicMock, list[ChatMessageUser | ChatMessageAssistant]],
        steps_data: EpisodeStepsResponse,
        submission_data: EpisodeSubmissionResponse,
        llm_subtask_criteria: SubtaskEvaluationCriteriaResponse,
        task_context: TaskEvaluationContext,
    ) -> None:
        """Verify score_subtask_llm does not mutate state.messages."""
        state, original_messages = state_with_messages

        # LLM returns JSON step evaluations
        llm_response = json.dumps({
            "step_evaluations": [
                {"step_number": 1, "completed": True},
            ]
        })
        mock_output = MagicMock(spec=ModelOutput)
        mock_output.completion = llm_response
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        with patch(
            "saber.inspect_ai.core.scoring.standard_subtask.get_model",
            return_value=mock_model,
        ):
            await score_subtask_llm(
                steps_data, llm_subtask_criteria, task_context, MagicMock(), state, submission_data
            )

        assert len(state.messages) == len(original_messages), (
            f"state.messages length changed from {len(original_messages)} to {len(state.messages)}"
        )
        assert state.messages == original_messages, "state.messages content was mutated by scorer"

    @pytest.mark.asyncio
    async def test_score_submission_static_preserves_state_messages(
        self,
        state_with_messages: tuple[MagicMock, list[ChatMessageUser | ChatMessageAssistant]],
        submission_data: EpisodeSubmissionResponse,
        static_submission_criteria: SubmissionEvaluationCriteriaResponse,
    ) -> None:
        """Verify score_submission_static does not mutate state.messages."""
        state, original_messages = state_with_messages

        await score_submission_static(submission_data, static_submission_criteria, None, state)

        assert len(state.messages) == len(original_messages), (
            f"state.messages length changed from {len(original_messages)} to {len(state.messages)}"
        )
        assert state.messages == original_messages, "state.messages content was mutated by scorer"
