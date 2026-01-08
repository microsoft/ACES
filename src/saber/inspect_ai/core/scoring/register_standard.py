"""Standard scoring methods registration.

This module registers all standard scoring methods with the scoring registry.
It runs automatically on import to make standard scorers available.
"""

from ....models.constants import StepEvaluationStrategy, SubmissionEvaluationStrategy
from .registry import ScorerMetadata, register_submission_scorer, register_subtask_scorer
from .standard_submission import score_submission_llm, score_submission_none, score_submission_static
from .standard_subtask import (
    score_subtask_llm,
    score_subtask_static,
    score_subtask_tool_call,
    score_subtask_tool_call_count,
)


def register_standard_scorers() -> None:
    """Register all standard scoring methods with the registry."""
    # Register submission scorers
    register_submission_scorer(SubmissionEvaluationStrategy.STATIC, score_submission_static)
    register_submission_scorer(SubmissionEvaluationStrategy.LLM_JUDGE, score_submission_llm)
    register_submission_scorer("none", score_submission_none)  # No-op scorer for tasks without submission evaluation

    # Register subtask scorers with metadata
    # All subtask scorers use consistent 6-parameter signature:
    # (steps_data, criteria, task_context, session_manager, state, submission_data)
    # uses_llm=True triggers state deepcopy to avoid race conditions
    register_subtask_scorer(
        StepEvaluationStrategy.STATIC,
        score_subtask_static,
        metadata=ScorerMetadata(uses_llm=False, requires_submission=True),
    )
    register_subtask_scorer(
        StepEvaluationStrategy.LLM_JUDGE,
        score_subtask_llm,
        metadata=ScorerMetadata(uses_llm=True, requires_submission=True),
    )
    register_subtask_scorer(
        StepEvaluationStrategy.TOOL_CALL,
        score_subtask_tool_call,
        metadata=ScorerMetadata(uses_llm=False, requires_submission=True),
    )
    register_subtask_scorer(
        StepEvaluationStrategy.TOOL_CALL_COUNT,
        score_subtask_tool_call_count,
        metadata=ScorerMetadata(uses_llm=False, requires_submission=True),
    )


# Auto-register on module import
register_standard_scorers()
