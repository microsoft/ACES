"""
SABER Evaluation Utilities

Shared utilities for parsing and processing evaluation results.
Used by both client-side and server-side evaluation components.

Following SABER best practices:
- Fail fast on invalid formats
- No silent failures or defensive fallbacks
- Clean separation of concerns
"""

import logging
import re
from typing import List

from .rest.evaluation import StepEvaluation

logger = logging.getLogger(__name__)


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
            logger.info("Judge response indicates no objectives were completed")
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
        logger.info("STEP_EVALUATIONS section found but contains no step entries")
        return []

    step_evaluations = []

    for step_number_str, objective_id in step_matches:
        try:
            step_number = int(step_number_str)
            objective_id = objective_id.strip()

            # Validate step number (must be >= 1)
            if step_number < 1:
                logger.warning(f"Invalid step number {step_number} in evaluation, skipping entry for '{objective_id}'")
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
            logger.debug(f"Parsed step evaluation: step {step_number} completed {objective_type} '{objective_id}'")

        except ValueError as e:
            raise RuntimeError(f"Invalid step number in evaluation: '{step_number_str}' - {e}") from e

    logger.info(f"Successfully parsed {len(step_evaluations)} step evaluations from judge response")
    return step_evaluations


def calculate_step_evaluation_score(
    step_evaluations: List[StepEvaluation], task_id: str, max_score: float = 1.0
) -> tuple[float, bool, int | None, List[str]]:
    """
    Calculate score and metadata from step evaluations.

    Args:
        step_evaluations: List of parsed step evaluations
        task_id: Main task ID to check for completion
        max_score: Maximum possible score

    Returns:
        Tuple of (score, is_correct, task_completed_at_step, subtasks_completed)
    """
    # Determine if main task was completed
    task_completed_at_step = None
    subtasks_completed = []

    for step_eval in step_evaluations:
        if step_eval.objective_type == "task" and step_eval.objective_id == task_id:
            if task_completed_at_step is None:  # Only use first occurrence
                task_completed_at_step = step_eval.step_number
        elif step_eval.objective_type == "subtask":
            subtasks_completed.append(step_eval.objective_id)

    # Calculate score: full score if main task completed, 0 otherwise
    is_correct = task_completed_at_step is not None
    score_value = max_score if is_correct else 0.0

    return score_value, is_correct, task_completed_at_step, subtasks_completed


def build_step_evaluation_explanation(
    is_correct: bool, task_completed_at_step: int | None, subtasks_completed: List[str]
) -> str:
    """
    Build human-readable explanation for step evaluation results.

    Args:
        is_correct: Whether main task was completed
        task_completed_at_step: Step number where task was completed (if any)
        subtasks_completed: List of completed subtask IDs

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

    return explanation
