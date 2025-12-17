"""Standard subtask scoring methods.

This module provides the standard scoring methods for subtasks:
- static: Pattern matching in step outputs
- llm_judge: LLM-based evaluation of step quality
- tool_call: Tool usage pattern matching
- tool_call_count: Count tool executions against threshold
"""

from typing import Any, List, Tuple

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.solver import TaskState
from jinja2 import Environment

from ....logging_config import LogCategory, get_saber_logger
from ....models.rest.evaluation import (
    EpisodeStepsResponse,
    EpisodeSubmissionResponse,
    StepEvaluation,
    SubtaskEvaluationCriteriaResponse,
)
from .template_utils import TemplateStringLoader

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


async def score_subtask_skip(
    steps_data: EpisodeStepsResponse,
    criteria: SubtaskEvaluationCriteriaResponse,
    task_context: Any,
    session_manager: Any,
    state: TaskState,
    submission_data: EpisodeSubmissionResponse,
) -> Tuple[float, List[StepEvaluation]]:
    """Handle subtasks without evaluation strategy (informational checkpoints).

    Args:
        steps_data: Episode steps data
        criteria: Step evaluation criteria
        task_context: Task context (unused)
        session_manager: Client session manager (unused)
        state: Task state (unused)
        submission_data: Submission data (unused)

    Returns:
        Tuple of (0.0, empty list) - no scoring for skipped subtasks
    """
    logger.info(
        f"Skipping subtask '{criteria.subtask_id}' - no evaluation strategy configured",
        extra={
            "subtask_id": criteria.subtask_id,
            "event": "subtask_skip_no_strategy",
        },
    )

    # Return 0 score and empty step evaluations for informational checkpoints
    step_evaluations = []
    for step in steps_data.steps:
        step_evaluations.append(
            StepEvaluation(
                step_number=step.step_number,
                objective_id=criteria.subtask_id,
                objective_type="subtask",
                completed=False,
            )
        )
    return 0.0, step_evaluations


class EpisodeContextForTemplate:
    """Helper class to provide episode-like interface for templates."""

    def __init__(self, steps: List[Any]) -> None:
        self.steps = steps

    def get_step_count(self) -> int:
        return len(self.steps)


class StepContextForTemplate:
    """Helper class to provide step-like interface for templates."""

    def __init__(self, step_data: dict) -> None:
        self.step_number = step_data["step_number"]
        self.done = step_data.get("done", False)
        # Create action object
        self.action = type(
            "Action",
            (),
            {
                "tool_name": step_data["tool_name"],
                "parameters": step_data["tool_input"],
                "assistant_message": step_data.get("assistant_message"),
                "reasoning": step_data.get("reasoning"),
            },
        )()
        # Response is tool output
        self.response = step_data["tool_output"]


async def score_subtask_static(
    steps_data: EpisodeStepsResponse,
    criteria: SubtaskEvaluationCriteriaResponse,
    task_context: Any,
    session_manager: Any,
    state: TaskState,
    submission_data: EpisodeSubmissionResponse,
) -> Tuple[float, List[StepEvaluation]]:
    """Score steps using static evaluation (pattern matching in outputs).

    Args:
        steps_data: Episode steps data
        criteria: Step evaluation criteria
        task_context: Task context (unused)
        session_manager: Client session manager (unused)
        state: Task state (unused)
        submission_data: Submission data (unused)

    Returns:
        Tuple of (subtask_score, list_of_step_evaluations)
    """
    logger.info(
        f"Scoring subtask '{criteria.subtask_id}' using STATIC strategy",
        extra={
            "subtask_id": criteria.subtask_id,
            "strategy": "STATIC",
            "max_score": criteria.max_score,
            "event": "scoring_subtask_start",
        },
    )
    expected_outputs = criteria.criteria.get("expected_outputs", [])
    max_score = criteria.max_score

    if not expected_outputs:
        logger.warning(
            "No expected_outputs configured for static subtask strategy", extra={"subtask_id": criteria.subtask_id}
        )
        return 0.0, []

    # Normalize to list
    if isinstance(expected_outputs, str):
        expected_outputs = [expected_outputs]

    all_graded_steps = []
    found_match = False

    expected_outputs_lower = [e.lower() for e in expected_outputs]
    for step in steps_data.steps:
        step_score = 0.0
        if step.tool_output:
            output_lower = step.tool_output.lower()
            if any(exp in output_lower for exp in expected_outputs_lower):
                step_score = max_score
                found_match = True
                logger.info(
                    "Static output match found at step %d",
                    step.step_number,
                    extra={
                        "step_number": step.step_number,
                        "expected": expected_outputs,
                        "subtask_id": criteria.subtask_id,
                        "event": "static_output_match",
                    },
                )
                break

        all_graded_steps.append(
            StepEvaluation(
                step_number=step.step_number,
                objective_id=criteria.subtask_id,
                objective_type="subtask",
                completed=step_score > 0,
            )
        )

    # Award subtask score if any step matched
    subtask_score = max_score if found_match else 0.0

    logger.info(
        f"Subtask '{criteria.subtask_id}' STATIC evaluation complete - Score: {subtask_score:.2f}/{criteria.max_score}",
        extra={
            "subtask_id": criteria.subtask_id,
            "score": subtask_score,
            "max_score": criteria.max_score,
            "found_match": found_match,
            "steps_evaluated": len(all_graded_steps),
            "event": "static_subtask_complete",
        },
    )

    return subtask_score, all_graded_steps


async def score_subtask_tool_call(
    steps_data: EpisodeStepsResponse,
    criteria: SubtaskEvaluationCriteriaResponse,
    task_context: Any,
    session_manager: Any,
    state: TaskState,
    submission_data: EpisodeSubmissionResponse,
) -> Tuple[float, List[StepEvaluation]]:
    """Score steps using tool call evaluation (matching tool names).

    Args:
        steps_data: Episode steps data
        criteria: Step evaluation criteria
        task_context: Task context (unused)
        session_manager: Client session manager (unused)
        state: Task state (unused)
        submission_data: Submission data (unused)

    Returns:
        Tuple of (subtask_score, list_of_step_evaluations)
    """
    logger.info(
        f"Scoring subtask '{criteria.subtask_id}' using TOOL_CALL strategy",
        extra={
            "subtask_id": criteria.subtask_id,
            "strategy": "TOOL_CALL",
            "max_score": criteria.max_score,
            "event": "scoring_subtask_start",
        },
    )
    expected_tools = criteria.criteria.get("expected_tools", [])
    max_score = criteria.max_score

    if not expected_tools:
        logger.warning(
            "No expected_tools configured for tool_call subtask strategy", extra={"subtask_id": criteria.subtask_id}
        )
        return 0.0, []

    # Normalize to list
    if isinstance(expected_tools, str):
        expected_tools = [expected_tools]

    all_graded_steps = []
    found_match = False

    for step in steps_data.steps:
        step_score = 0.0

        if step.tool_name and step.tool_name in expected_tools:
            step_score = max_score
            found_match = True
            logger.info(
                "Tool call match found at step %d with tool %s",
                step.step_number,
                step.tool_name,
                extra={
                    "step_number": step.step_number,
                    "tool_name": step.tool_name,
                    "subtask_id": criteria.subtask_id,
                    "event": "tool_call_match",
                },
            )
            break

        all_graded_steps.append(
            StepEvaluation(
                step_number=step.step_number,
                objective_id=criteria.subtask_id,
                objective_type="subtask",
                completed=step_score > 0,
            )
        )

    # Award subtask score if any step matched
    subtask_score = max_score if found_match else 0.0

    logger.info(
        f"Subtask '{criteria.subtask_id}' TOOL_CALL evaluation complete - "
        f"Score: {subtask_score:.2f}/{criteria.max_score}",
        extra={
            "subtask_id": criteria.subtask_id,
            "score": subtask_score,
            "max_score": criteria.max_score,
            "found_match": found_match,
            "steps_evaluated": len(all_graded_steps),
            "event": "tool_call_subtask_complete",
        },
    )

    return subtask_score, all_graded_steps


async def score_subtask_tool_call_count(
    steps_data: EpisodeStepsResponse,
    criteria: SubtaskEvaluationCriteriaResponse,
    task_context: Any,
    session_manager: Any,
    state: TaskState,
    submission_data: EpisodeSubmissionResponse,
) -> Tuple[float, List[StepEvaluation]]:
    """Score subtask based on tool execution count in steps.

    Binary scoring: pass if tool executed >= threshold times, fail otherwise.
    Useful for validating iterative development patterns.

    Args:
        steps_data: Episode steps data
        criteria: Subtask criteria with tool_name and min_executions
        task_context: Task context (unused)
        session_manager: Client session manager (unused)
        state: Task state (unused)
        submission_data: Submission data (unused)

    Returns:
        Tuple of (score, step_evaluations) - binary: full points or 0
    """
    logger.info(
        f"Scoring subtask '{criteria.subtask_id}' using TOOL_CALL_COUNT strategy",
        extra={
            "subtask_id": criteria.subtask_id,
            "strategy": "TOOL_CALL_COUNT",
            "max_score": criteria.max_score,
            "event": "scoring_subtask_start",
        },
    )

    tool_name = criteria.criteria.get("tool_name", "")
    min_executions = criteria.criteria.get("min_executions", 1)

    # Normalize tool_name to lowercase for comparison
    tool_name_lower = str(tool_name).lower() if tool_name else ""

    # Count tool executions
    execution_count = 0
    tool_steps = []

    for step in steps_data.steps:
        # Extract the actual tool from the tool_input parameters
        actual_tool = None
        if isinstance(step.tool_input, dict):
            actual_tool = step.tool_input.get("tool")

        # Check for tool name match (case-insensitive)
        if actual_tool and actual_tool.lower() == tool_name_lower:
            execution_count += 1
            tool_steps.append(step.step_number)

    # Binary scoring: pass if >= threshold
    achieved = execution_count >= int(min_executions)
    score = criteria.max_score if achieved else 0.0

    logger.info(
        f"Subtask '{criteria.subtask_id}' TOOL_CALL_COUNT evaluation complete - "
        f"Score: {score:.2f}/{criteria.max_score} ({execution_count}/{min_executions} executions)",
        extra={
            "subtask_id": criteria.subtask_id,
            "tool_name": tool_name,
            "execution_count": execution_count,
            "min_executions": min_executions,
            "achieved": achieved,
            "score": score,
            "max_score": criteria.max_score,
            "tool_steps": tool_steps,
            "event": "tool_call_count_complete",
        },
    )

    # Create step evaluations - mark tool execution steps as relevant
    step_evals = []
    for step in steps_data.steps:
        step_completed = achieved and step.step_number in tool_steps
        step_evals.append(
            StepEvaluation(
                step_number=step.step_number,
                objective_id=criteria.subtask_id,
                objective_type="subtask",
                completed=step_completed,
            )
        )

    return score, step_evals


async def score_subtask_llm(
    steps_data: EpisodeStepsResponse,
    criteria: SubtaskEvaluationCriteriaResponse,
    task_context: Any,
    session_manager: Any,
    state: TaskState,
    submission_data: EpisodeSubmissionResponse,
) -> Tuple[float, List[StepEvaluation]]:
    """Score steps using LLM evaluation.

    Args:
        steps_data: Episode steps data
        criteria: Step evaluation criteria
        task_context: Task context
        session_manager: Client session manager (unused)
        state: Task state for LLM calls
        submission_data: Submission data (unused)

    Returns:
        Tuple of (total_step_score, list_of_step_evaluations)
    """
    logger.info(
        f"Scoring subtask '{criteria.subtask_id}' using LLM_JUDGE strategy",
        extra={
            "subtask_id": criteria.subtask_id,
            "strategy": "LLM_JUDGE",
            "max_score": criteria.max_score,
            "event": "scoring_subtask_start",
        },
    )

    # Get template content from criteria
    system_template = criteria.criteria.get("judge_system_template")
    user_template = criteria.criteria.get("judge_user_template")
    model_name = criteria.criteria.get("model")
    steps_per_message = criteria.criteria.get("steps_per_message", 10)

    if not all([system_template, user_template, model_name]):
        raise RuntimeError("Missing required template content or model in step criteria")

    # Type narrowing after validation
    assert system_template is not None
    assert user_template is not None

    # Setup Jinja2
    env = Environment(loader=TemplateStringLoader({"system": system_template, "user": user_template}))

    # Chunk steps
    all_step_evaluations: List[StepEvaluation] = []
    step_chunks = [
        steps_data.steps[i : i + steps_per_message] for i in range(0, len(steps_data.steps), steps_per_message)
    ]

    logger.info(
        "Processing step chunks",
        extra={"total_steps": len(steps_data.steps), "chunks": len(step_chunks), "event": "step_chunking"},
    )

    for chunk_idx, chunk in enumerate(step_chunks):
        # Build context - create step objects that match template expectations
        step_objects = []
        for step in chunk:
            step_objects.append(
                StepContextForTemplate(
                    {
                        "step_number": step.step_number,
                        "tool_name": step.tool_name,
                        "tool_input": step.tool_input,
                        "tool_output": step.tool_output,
                        "done": False,
                        "assistant_message": step.assistant_message,
                        "reasoning": step.reasoning,
                    }
                )
            )

        episode = EpisodeContextForTemplate(step_objects)

        # Build context for subtask evaluation
        context = {
            "question": task_context.description,
            "episode": episode,
            "subtask": {
                "id": criteria.subtask_id,
                "description": criteria.description,
                "objective": criteria.objective,
                "title": criteria.title,
            },
            "model": criteria.criteria.get("model", ""),
            "domain": task_context.domain if hasattr(task_context, "domain") else None,
            "task_id": criteria.task_id,
            "task": {
                "task_id": task_context.task_id,
                "title": task_context.title,
                "description": task_context.description,
                "subtasks": task_context.subtasks,
            },
        }

        # Render templates
        system_message = env.get_template("system").render(context)
        user_message = env.get_template("user").render(context)

        # Execute LLM
        state.messages.clear()
        state.messages.append(ChatMessageSystem(content=system_message))
        state.messages.append(ChatMessageUser(content=user_message))

        model = get_model(model_name)
        response = await model.generate(state.messages)

        # Parse response for step evaluations (JSON format expected)
        import json

        try:
            result = json.loads(response.completion)
            step_evals_from_llm = result.get("step_evaluations", [])

            for step_eval in step_evals_from_llm:
                all_step_evaluations.append(
                    StepEvaluation(
                        step_number=step_eval["step_number"],
                        objective_id=criteria.subtask_id,
                        objective_type="subtask",
                        completed=step_eval.get("completed", False),
                    )
                )
        except json.JSONDecodeError:
            logger.warning(f"Could not parse LLM response as JSON: {response.completion}")

    # Calculate total score based on completed steps
    completed_steps = sum(1 for step in all_step_evaluations if step.completed)
    total_steps = len(all_step_evaluations)
    score = (completed_steps / total_steps * criteria.max_score) if total_steps > 0 else 0.0

    logger.info(
        f"Subtask '{criteria.subtask_id}' LLM_JUDGE evaluation complete - Score: {score:.2f}/{criteria.max_score}",
        extra={
            "subtask_id": criteria.subtask_id,
            "score": score,
            "max_score": criteria.max_score,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "event": "llm_subtask_complete",
        },
    )

    return score, all_step_evaluations
