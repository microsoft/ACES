"""
SABER Evaluation Utilities

Shared utilities for parsing and processing evaluation results.
Used by both client-side and server-side evaluation components.

Log Category: EVALUATION

Following SABER best practices:
- Fail fast on invalid formats
- No silent failures or defensive fallbacks
- Clean separation of concerns
"""

import re
from typing import List

from saber.logging_config import LogCategory, get_saber_logger

from .rest.evaluation import StepEvaluation

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


def parse_step_evaluations(judge_response: str, task_id: str) -> List[StepEvaluation]:
    """
    Parse step evaluation results from judge response.

    Expected format:
    STEP_EVALUATIONS:
    [step_number: objective_id] - explanation
    [step_number: objective_id] - explanation
    ...

    Args:
        judge_response: Full judge response text
        task_id: Main task ID to identify task completion

    Returns:
        List of StepEvaluation objects

    Raises:
        RuntimeError: If format is invalid (fail-fast principle)
    """
    # Find STEP_EVALUATIONS section
    step_section_pattern = r"STEP_EVALUATIONS:\s*(.*?)(?:\n\n|\Z)"
    section_match = re.search(step_section_pattern, judge_response, re.DOTALL | re.IGNORECASE)

    if not section_match:
        # Check for NO_COMPLETIONS case
        if "[NO_COMPLETIONS]" in judge_response:
            logger.info(
                "Judge response indicates no objectives were completed",
                extra={
                    "event": "step_evaluations_absent",
                    "judge_response_contains_no_completions": True,
                },
            )
            return []
        else:
            raise RuntimeError(
                "STEP_EVALUATIONS section not found in judge response. "
                "Expected format: 'STEP_EVALUATIONS:' followed by step entries or '[NO_COMPLETIONS]'"
            )

    section_content = section_match.group(1).strip()

    # Parse individual step entries - handle whitespace around step_number and objective_id
    step_pattern = r"\[\s*(\d+)\s*:\s*([^\]]+?)\s*\]"
    step_matches = re.findall(step_pattern, section_content)

    if not step_matches:
        # Empty section but properly formatted
        logger.info(
            "STEP_EVALUATIONS section found but contains no step entries",
            extra={"event": "step_evaluations_empty"},
        )
        return []

    step_evaluations = []

    for step_number_str, objective_id in step_matches:
        try:
            step_number = int(step_number_str)
            objective_id = objective_id.strip()

            # Validate step number (must be >= 0, steps are 0-indexed)
            if step_number < 0:
                logger.warning(
                    "Invalid step number encountered in evaluation entry",
                    extra={
                        "event": "step_evaluation_invalid_step_number",
                        "step_number": step_number,
                        "objective_id": objective_id,
                    },
                )
                continue

            # Determine objective type
            objective_type = "task" if objective_id == task_id else "subtask"

            step_eval = StepEvaluation(
                step_number=step_number,
                objective_id=objective_id,
                objective_type=objective_type,
                completed=True,
            )

            step_evaluations.append(step_eval)
            logger.debug(
                "Parsed step evaluation entry",
                extra={
                    "event": "step_evaluation_parsed",
                    "step_number": step_number,
                    "objective_id": objective_id,
                    "objective_type": objective_type,
                },
            )

        except ValueError as e:
            raise RuntimeError(f"Invalid step number in evaluation: '{step_number_str}' - {e}") from e

    logger.info(
        "Successfully parsed step evaluations from judge response",
        extra={
            "event": "step_evaluations_parsed",
            "count": len(step_evaluations),
        },
    )
    return step_evaluations


def calculate_step_evaluation_score(
    step_evaluations: List[StepEvaluation],
    task_id: str,
    max_score: float = 1.0,
    subtasks_with_scores: dict[str, float] | None = None,
) -> tuple[float, bool, int | None, List[str]]:
    """
    Calculate score and metadata from step evaluations using aggregation scoring.

    Args:
        step_evaluations: List of parsed step evaluations
        task_id: Main task ID to check for completion
        max_score: Maximum possible score for the main task (0 if task score shouldn't count)
        subtasks_with_scores: Dict mapping subtask_id to max_score (empty dict if no scored subtasks)

    Returns:
        Tuple of (score, is_correct, task_completed_at_step, subtasks_completed)

    Scoring Logic (Aggregation Mode):
        Score = (main task score if completed) + (sum of completed subtask scores)
        - Task not completed but max_score > 0: task score not added
        - Task completed with max_score = 0: no task score added (subtasks only)
        - Any subtask not in subtasks_with_scores or with score 0: contributes 0 to score
    """
    # Determine if main task was completed
    task_completed_at_step = None
    subtasks_completed_set = set()  # Use set to prevent duplicate subtask scoring
    subtasks_completed_order = []  # Track first occurrence order for consistent output

    for step_eval in step_evaluations:
        if step_eval.objective_type == "task" and step_eval.objective_id == task_id:
            if task_completed_at_step is None:  # Only use first occurrence
                task_completed_at_step = step_eval.step_number
        elif step_eval.objective_type == "subtask":
            # Only add if not already seen (maintains first-occurrence order)
            if step_eval.objective_id not in subtasks_completed_set:
                subtasks_completed_set.add(step_eval.objective_id)
                subtasks_completed_order.append(step_eval.objective_id)

    # Use ordered list for backwards compatibility with return type
    subtasks_completed = subtasks_completed_order

    # Determine if task was completed
    is_correct = task_completed_at_step is not None

    # Aggregation scoring: sum all completed objectives that have scores
    score_value = 0.0

    # Add task score if completed (and max_score > 0)
    if is_correct and max_score > 0:
        score_value += max_score
        logger.debug(
            "Added task score to total",
            extra={
                "event": "task_score_added",
                "task_id": task_id,
                "task_score": max_score,
                "running_total": score_value,
            },
        )

    # Add each completed subtask's score (if it has one)
    if subtasks_with_scores:
        for subtask_id in subtasks_completed:
            if subtask_id in subtasks_with_scores:
                subtask_score = subtasks_with_scores[subtask_id]
                if subtask_score > 0:
                    score_value += subtask_score
                    logger.debug(
                        "Added subtask score to total",
                        extra={
                            "event": "subtask_score_added",
                            "subtask_id": subtask_id,
                            "subtask_score": subtask_score,
                            "running_total": score_value,
                        },
                    )

    return score_value, is_correct, task_completed_at_step, subtasks_completed


def build_step_evaluation_explanation(
    is_correct: bool,
    task_completed_at_step: int | None,
    subtasks_completed: List[str],
    score_value: float | None = None,
    max_possible_score: float | None = None,
) -> str:
    """
    Build human-readable explanation for step evaluation results.

    Args:
        is_correct: Whether main task was completed
        task_completed_at_step: Step number where task was completed (if any)
        subtasks_completed: List of completed subtask IDs
        score_value: Optional actual score achieved (for aggregation mode)
        max_possible_score: Optional maximum possible score (for aggregation mode)

    Returns:
        Human-readable explanation string
    """
    if is_correct:
        explanation = f"Step evaluation: Task completed at step {task_completed_at_step}"
        if subtasks_completed:
            explanation += f", completed subtasks: {', '.join(subtasks_completed)}"
    else:
        explanation = "Step evaluation: Main task not completed"
        if subtasks_completed:
            explanation += f", but completed subtasks: {', '.join(subtasks_completed)}"

    # Add score information if provided (aggregation mode)
    if score_value is not None and max_possible_score is not None:
        explanation += f" (Score: {score_value}/{max_possible_score})"

    return explanation
