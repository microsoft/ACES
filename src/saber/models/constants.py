"""
SABER Models Constants - Shared constants and enumerations.

This module provides centralized constants that are used across multiple
components of the SABER system to ensure consistency.
"""

from enum import Enum


class SubmissionEvaluationStrategy(str, Enum):
    """
    Enumeration of supported evaluation strategies for main task submissions.

    These strategies are available for evaluating final task submissions:
    - STATIC: Pattern matching against expected answers
    - LLM_JUDGE: LLM-based evaluation using judge templates
    """

    STATIC = "static"
    """Static evaluation using pattern matching against expected answers"""

    LLM_JUDGE = "llm_judge"
    """LLM-based evaluation using judge templates and models"""

    def __str__(self) -> str:
        """Return the enum value as string for logging and serialization."""
        return self.value


class StepEvaluationStrategy(str, Enum):
    """
    Enumeration of supported evaluation strategies for subtask step evaluation.

    These strategies are available for evaluating individual steps/subtasks:
    - STATIC: Pattern matching against expected outputs in step results
    - LLM_JUDGE: LLM-based evaluation using judge templates
    - TOOL_CALL: Evaluation based on tool call analysis and execution patterns
    """

    STATIC = "static"
    """Static evaluation using pattern matching against expected step outputs"""

    LLM_JUDGE = "llm_judge"
    """LLM-based evaluation using judge templates and models"""

    TOOL_CALL = "tool_call"
    """Evaluation based on tool call analysis and execution (subtasks only)"""

    def __str__(self) -> str:
        """Return the enum value as string for logging and serialization."""
        return self.value


# Legacy unified enum for backwards compatibility
class EvaluationStrategy(str, Enum):
    """
    Legacy unified evaluation strategy enum.

    DEPRECATED: Use SubmissionEvaluationStrategy or StepEvaluationStrategy instead.
    Kept for backwards compatibility only.
    """

    STATIC = "static"
    LLM_JUDGE = "llm_judge"
    TOOL_CALL = "tool_call"

    def __str__(self) -> str:
        return self.value


# Backwards compatibility - provide the old constant names
EVAL_STRATEGY_STATIC = SubmissionEvaluationStrategy.STATIC
EVAL_STRATEGY_LLM_JUDGE = SubmissionEvaluationStrategy.LLM_JUDGE
EVAL_STRATEGY_TOOL_CALL = StepEvaluationStrategy.TOOL_CALL

# Valid strategies for different evaluation types
VALID_SUBMISSION_EVAL_STRATEGIES = [strategy.value for strategy in SubmissionEvaluationStrategy]
VALID_STEP_EVAL_STRATEGIES = [strategy.value for strategy in StepEvaluationStrategy]
VALID_EVAL_STRATEGIES = VALID_STEP_EVAL_STRATEGIES  # Legacy compatibility
