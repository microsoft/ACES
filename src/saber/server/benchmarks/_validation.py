"""Validation helpers for benchmark configuration loading.

Extracted from BenchmarkConfigLoader to improve testability and reduce file size.
"""

from typing import Any, Dict

from ...models.constants import (
    EVAL_STRATEGY_LLM_JUDGE,
    EVAL_STRATEGY_STATIC,
    EVAL_STRATEGY_TOOL_CALL,
    VALID_STEP_EVAL_STRATEGIES,
    VALID_SUBMISSION_EVAL_STRATEGIES,
)
from .exceptions import InvalidTaskDefinitionException

# Field name constants
FIELD_STRATEGY = "strategy"
FIELD_CRITERIA = "criteria"
FIELD_SCORING = "scoring"
FIELD_MAX_SCORE = "max_score"
FIELD_WEIGHT = "weight"
FIELD_MODEL = "model"
FIELD_EXPECTED_ANSWERS = "expected_answers"
FIELD_JUDGE_SYSTEM_TEMPLATE = "judge_system_template"
FIELD_JUDGE_USER_TEMPLATE = "judge_user_template"
FIELD_STEPS_PER_MESSAGE = "steps_per_message"
FIELD_SUBMISSION_EVALUATION_CONFIG = "submission_evaluation_config"
FIELD_STEP_EVALUATION_CONFIG = "step_evaluation_config"

DEFAULT_WEIGHT = 1.0
DEFAULT_MAX_SCORE = 1.0


def validate_template_path(template_path: str, field_name: str, task_id: str) -> None:
    """
    Validate that a template path is properly formatted.

    Template paths must:
    - End with .md extension
    - Not contain backslashes (use forward slashes only)
    - Not be empty or just whitespace
    - Not contain '..' path traversal

    Args:
        template_path: The template path to validate
        field_name: Name of the field being validated (for error messages)
        task_id: Task ID for error reporting

    Raises:
        InvalidTaskDefinitionException: If path is invalid
    """
    if not template_path or not template_path.strip():
        raise InvalidTaskDefinitionException(f"Task '{task_id}': {field_name} cannot be empty or whitespace")

    if "\\" in template_path:
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': {field_name} must use forward slashes, not backslashes: {template_path}"
        )

    if ".." in template_path:
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': {field_name} cannot contain '..' path traversal: {template_path}"
        )

    if not template_path.endswith(".md"):
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': {field_name} must end with .md extension: {template_path}"
        )


def validate_submission_evaluation_config(eval_config: Dict[str, Any], task_id: str) -> None:
    """
    Validate submission_evaluation_config.

    Args:
        eval_config: Submission evaluation configuration dictionary
        task_id: Task ID for error reporting

    Raises:
        InvalidTaskDefinitionException: If configuration is invalid
    """
    # Validate strategy
    strategy = eval_config.get(FIELD_STRATEGY)
    if strategy not in VALID_SUBMISSION_EVAL_STRATEGIES:
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': Invalid submission evaluation strategy. "
            f"Must be one of {VALID_SUBMISSION_EVAL_STRATEGIES}, got: {strategy}"
        )

    # Validate criteria section
    criteria = eval_config.get(FIELD_CRITERIA)
    if not isinstance(criteria, dict):
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': Missing or invalid {FIELD_CRITERIA} in {FIELD_SUBMISSION_EVALUATION_CONFIG}"
        )

    # Validate scoring section
    scoring = eval_config.get(FIELD_SCORING, {})
    if not isinstance(scoring, dict):
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': {FIELD_SCORING} must be a dictionary in {FIELD_SUBMISSION_EVALUATION_CONFIG}"
        )

    max_score = scoring.get(FIELD_MAX_SCORE, 1.0)
    if not isinstance(max_score, (int, float)) or max_score <= 0:
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': {FIELD_MAX_SCORE} must be a positive number, got: {max_score}"
        )

    # Strategy-specific validation
    if strategy == EVAL_STRATEGY_STATIC:
        expected_answers = criteria.get(FIELD_EXPECTED_ANSWERS)
        if not expected_answers or not isinstance(expected_answers, list):
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}': static strategy requires '{FIELD_EXPECTED_ANSWERS}' as a list"
            )

    elif strategy == EVAL_STRATEGY_LLM_JUDGE:
        model = criteria.get(FIELD_MODEL)
        if not model or not isinstance(model, str):
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}': llm_judge strategy requires '{FIELD_MODEL}' as a string"
            )

        # Template paths - must be explicitly defined
        judge_system_template = criteria.get(FIELD_JUDGE_SYSTEM_TEMPLATE)
        if not judge_system_template or not isinstance(judge_system_template, str):
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}': llm_judge requires '{FIELD_JUDGE_SYSTEM_TEMPLATE}' path"
            )
        validate_template_path(judge_system_template, FIELD_JUDGE_SYSTEM_TEMPLATE, task_id)

        judge_user_template = criteria.get(FIELD_JUDGE_USER_TEMPLATE)
        if not judge_user_template or not isinstance(judge_user_template, str):
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}': llm_judge requires '{FIELD_JUDGE_USER_TEMPLATE}' path"
            )
        validate_template_path(judge_user_template, FIELD_JUDGE_USER_TEMPLATE, task_id)


def validate_step_evaluation_config(eval_config: Dict[str, Any], task_id: str) -> None:
    """
    Validate step_evaluation_config.

    Args:
        eval_config: Step evaluation configuration dictionary
        task_id: Task ID for error reporting

    Raises:
        InvalidTaskDefinitionException: If configuration is invalid
    """
    # Validate scoring section
    scoring = eval_config.get(FIELD_SCORING, {})
    if not isinstance(scoring, dict):
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': scoring must be a dictionary in step_evaluation_config"
        )

    max_score = scoring.get(FIELD_MAX_SCORE, DEFAULT_MAX_SCORE)
    weight = scoring.get(FIELD_WEIGHT, DEFAULT_WEIGHT)

    if not isinstance(max_score, (int, float)) or max_score < 0:
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': max_score must be a non-negative number, got: {max_score}"
        )

    if not isinstance(weight, (int, float)) or weight < 0:
        raise InvalidTaskDefinitionException(f"Task '{task_id}': weight must be a non-negative number, got: {weight}")

    if weight > 1.0:
        raise InvalidTaskDefinitionException(f"Task '{task_id}': weight must not exceed 1.0, got: {weight}")

    # Validate strategy
    strategy = eval_config.get(FIELD_STRATEGY)
    if strategy not in VALID_STEP_EVAL_STRATEGIES:
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': Invalid step evaluation strategy. "
            f"Must be one of {VALID_STEP_EVAL_STRATEGIES}, got: {strategy}"
        )

    # Validate criteria section
    criteria = eval_config.get(FIELD_CRITERIA)
    if not isinstance(criteria, dict):
        raise InvalidTaskDefinitionException(
            f"Task '{task_id}': Missing or invalid {FIELD_CRITERIA} in {FIELD_STEP_EVALUATION_CONFIG}"
        )

    # Strategy-specific validation
    if strategy == EVAL_STRATEGY_STATIC:
        if "expected_outputs" not in criteria:
            raise InvalidTaskDefinitionException(
                "Subtask with static strategy must have 'expected_outputs' in criteria"
            )
    elif strategy == EVAL_STRATEGY_TOOL_CALL:
        if "expected_tools" not in criteria:
            raise InvalidTaskDefinitionException(
                "Subtask with tool_call strategy must have 'expected_tools' in criteria"
            )
    elif strategy == EVAL_STRATEGY_LLM_JUDGE:
        model = criteria.get(FIELD_MODEL)
        if not model or not isinstance(model, str):
            raise InvalidTaskDefinitionException(f"Task '{task_id}': step evaluation llm_judge requires 'model'")

        # Template paths
        judge_system_template = criteria.get(FIELD_JUDGE_SYSTEM_TEMPLATE)
        if not judge_system_template or not isinstance(judge_system_template, str):
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}': step evaluation requires 'judge_system_template' path"
            )
        validate_template_path(judge_system_template, "judge_system_template", task_id)

        judge_user_template = criteria.get(FIELD_JUDGE_USER_TEMPLATE)
        if not judge_user_template or not isinstance(judge_user_template, str):
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}': step evaluation requires 'judge_user_template' path"
            )
        validate_template_path(judge_user_template, "judge_user_template", task_id)

        # steps_per_message is optional
        steps_per_message = criteria.get(FIELD_STEPS_PER_MESSAGE, 10)
        if not isinstance(steps_per_message, int) or steps_per_message < 1:
            raise InvalidTaskDefinitionException(
                f"Task '{task_id}': steps_per_message must be a positive integer, got: {steps_per_message}"
            )
