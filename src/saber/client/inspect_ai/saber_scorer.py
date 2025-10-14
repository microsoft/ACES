"""
SABER Task Scorer for inspect_ai - CLIENT-SIDE EVALUATION (Breaking Change Migration)

Complete rewrite for client-side evaluation architecture.
ALL evaluation logic happens on the client:
- Fetch submission and steps from server
- Fetch evaluation criteria (template paths only)
- Fetch and render templates with Jinja2
- Execute LLM evaluation
- Calculate scores
- Submit results to server

NO server-side evaluation. NO backward compatibility.

Logging category: EVALUATION.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Scorer, Target, metric, scorer
from inspect_ai.scorer._metric import Metric, SampleScore, ValueToFloat, value_to_float
from inspect_ai.solver import TaskState
from inspect_ai.util import store
from jinja2 import BaseLoader, Environment, TemplateError

from ...logging_config import LogCategory, get_saber_logger
from ...models.evaluation_utils import parse_step_evaluations
from ...models.rest.evaluation import (
    EpisodeStepsResponse,
    EpisodeSubmissionResponse,
    EvaluationResultSubmission,
    StepEvaluation,
    StepEvaluationCriteriaResponse,
    SubmissionEvaluationCriteriaResponse,
)

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


# ============================================================================
# Template Context Helpers
# ============================================================================


class EpisodeContextForTemplate:
    """Helper class to provide episode-like interface for templates."""

    def __init__(self, steps: List[Any]) -> None:
        self.steps = steps

    def get_step_count(self) -> int:
        return len(self.steps)


class StepContextForTemplate:
    """Helper class to provide step-like interface for templates."""

    def __init__(self, step_data: Dict[str, Any]) -> None:
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


# ============================================================================
# Jinja2 Template Loader for Client-Side Rendering
# ============================================================================


class TemplateStringLoader(BaseLoader):
    """Load Jinja2 templates from string content."""

    def __init__(self, templates: Dict[str, str]):
        """
        Initialize template loader.

        Args:
            templates: Dict mapping template names to template content strings
        """
        self.templates = templates

    def get_source(self, environment: Any, template: str) -> Tuple[str, Optional[str], Callable[[], bool]]:
        """
        Get template source.

        Args:
            environment: Jinja2 environment
            template: Template name

        Returns:
            Tuple of (source, filename, uptodate_function)

        Raises:
            TemplateError: If template not found
        """
        if template in self.templates:
            return self.templates[template], None, lambda: True
        raise TemplateError(f"Template not found: {template}")


# ============================================================================
# Metric
# ============================================================================


@metric  # type: ignore[misc]
def saber_client_metric(to_float: ValueToFloat = value_to_float()) -> Metric:
    """
    Metric for SABER client-side evaluation score.

    Args:
        to_float: Function for mapping Value to float for computing metrics

    Returns:
        Client evaluation metric function
    """

    def metric_fn(scores: List[SampleScore]) -> float:
        total = 0.0
        for item in scores:
            total += to_float(item.score.value)
        return total / float(len(scores)) if scores else 0.0

    return metric_fn


# ============================================================================
# Main Scorer
# ============================================================================


@scorer(metrics=[saber_client_metric()])  # type: ignore[misc]
def saber_scorer() -> Scorer:
    """
    Client-side evaluation scorer (BREAKING CHANGE).

    Handles ALL evaluation logic on the client:
    1. Fetch submission and steps from server
    2. Fetch evaluation criteria (template paths)
    3. Fetch and render templates with Jinja2
    4. Execute LLM calls
    5. Calculate scores
    6. Submit results to server

    Returns:
        Scorer function that performs client-side evaluation
    """

    async def score(state: TaskState, target: Target) -> Score:
        """
        Perform complete client-side evaluation.

        Args:
            state: TaskState after execution
            target: Target criteria (unused)

        Returns:
            Score from client-side evaluation

        Raises:
            RuntimeError: If SABER context missing or evaluation fails
        """
        try:
            # Extract SABER context from inspect_ai store (same pattern as old working code)
            task_store = store()
            session_manager = task_store.get("saber_session_manager")
            session_id = task_store.get("saber_session_id")

            # Get episode_id from task store (set by SABER agent)
            current_episode = task_store.get("saber_current_episode")
            episode_id = current_episode.episode_id if current_episode else None

            # Validate SABER context (fail-fast)
            if not session_manager:
                raise RuntimeError("Missing saber_session_manager in store")

            if not session_id:
                raise RuntimeError("Missing saber_session_id in store")

            if not episode_id:
                raise RuntimeError("Missing episode_id from saber_current_episode in store")

            logger.info(
                "Starting client-side evaluation",
                extra={"session_id": session_id, "episode_id": episode_id, "event": "client_eval_start"},
            )

            # Step 1: Fetch submission data
            submission_data = await session_manager.get_episode_submission(session_id, episode_id)

            # Step 2: Fetch step history
            steps_data = await session_manager.get_episode_steps(session_id, episode_id)

            # Step 3: Fetch submission evaluation criteria
            submission_criteria = await session_manager.get_submission_evaluation_criteria(session_id, episode_id)

            # Step 4: Fetch step evaluation criteria (optional)
            step_criteria = await session_manager.get_step_evaluation_criteria(session_id, episode_id)

            # Step 5: Score submission
            submission_score = await _score_submission(submission_data, submission_criteria, session_manager, state)

            # Step 6: Score steps (if configured)
            step_score = 0.0
            step_evaluations: List[StepEvaluation] = []
            if step_criteria:
                step_score, step_evaluations = await _score_steps(
                    steps_data, step_criteria, submission_criteria.task_context, session_manager, state
                )

            # Calculate totals
            total_score = submission_score + step_score
            max_possible = submission_criteria.scoring.get("max_score", 1.0)
            if step_criteria:
                max_possible += sum(st.get("max_score", 0.0) for st in step_criteria.subtasks)

            # Step 7: Submit evaluation result
            evaluation_result = EvaluationResultSubmission(
                strategy="client_side_evaluation",
                raw_score=total_score,
                max_score=max_possible,
                score=total_score,
                success=submission_score > 0,
                details={
                    "task_id": submission_criteria.task_id,
                    "submission_score": submission_score,
                    "step_score": step_score,
                    "step_evaluations": [se.model_dump() for se in step_evaluations],
                    "client_scorer_version": "2.0",
                },
            )

            await session_manager.submit_evaluation_result(session_id, episode_id, evaluation_result)

            logger.info(
                "Client-side evaluation completed",
                extra={
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "total_score": total_score,
                    "submission_score": submission_score,
                    "step_score": step_score,
                    "event": "client_eval_complete",
                },
            )

            # Step 8: Return score for inspect_ai
            max_sub_score = submission_criteria.scoring.get("max_score", 1.0)
            return Score(
                value=total_score,
                answer=submission_data.submission,
                explanation=f"Client eval: submission={submission_score}/{max_sub_score}, steps={step_score}",
                metadata={
                    "submission_score": submission_score,
                    "step_score": step_score,
                    "max_possible": max_possible,
                    "step_evaluations": [se.model_dump() for se in step_evaluations],
                    "scorer_version": "2.0",
                },
            )

        except Exception as exc:
            logger.error(
                "Client-side evaluation failed",
                extra={"error": str(exc), "error_type": type(exc).__name__, "event": "client_eval_failure"},
                exc_info=True,
            )
            return Score(
                value=0.0,
                answer="",
                explanation=f"Client-side evaluation failed: {exc}",
                metadata={"error": str(exc), "scorer_version": "2.0"},
            )

    return score


# ============================================================================
# Submission Evaluation
# ============================================================================


async def _score_submission(
    submission_data: EpisodeSubmissionResponse,
    criteria: SubmissionEvaluationCriteriaResponse,
    session_manager: Any,
    state: TaskState,
) -> float:
    """
    Score submission using configured strategy.

    Args:
        submission_data: Episode submission data
        criteria: Submission evaluation criteria
        session_manager: Client session manager
        state: Task state

    Returns:
        Submission score (0.0 to max_score)
    """
    strategy = criteria.strategy
    if strategy == "static":
        return await _score_submission_static(submission_data, criteria)
    elif strategy == "llm_judge":
        return await _score_submission_llm(submission_data, criteria, session_manager, state)
    else:
        raise RuntimeError(f"Unknown submission strategy: {strategy}")


async def _score_submission_static(
    submission_data: EpisodeSubmissionResponse, criteria: SubmissionEvaluationCriteriaResponse
) -> float:
    """
    Static submission scoring (pattern matching).

    Args:
        submission_data: Episode submission data
        criteria: Submission evaluation criteria

    Returns:
        Score (0.0 or max_score)
    """
    expected_answers = criteria.criteria.get("expected_answers", [])
    max_score = criteria.scoring.get("max_score", 1.0)

    submission_lower = submission_data.submission.lower()
    for expected in expected_answers:
        if expected.lower() in submission_lower:
            logger.info(
                "Static submission match found",
                extra={"expected": expected, "score": max_score, "event": "static_submission_match"},
            )
            return max_score

    logger.info("Static submission no match", extra={"score": 0.0, "event": "static_submission_no_match"})
    return 0.0


async def _score_submission_llm(
    submission_data: EpisodeSubmissionResponse,
    criteria: SubmissionEvaluationCriteriaResponse,
    session_manager: Any,
    state: TaskState,
) -> float:
    """
    LLM submission scoring with client-side template rendering.

    Args:
        submission_data: Episode submission data
        criteria: Submission evaluation criteria
        session_manager: Client session manager
        state: Task state

    Returns:
        Score (0.0 or max_score based on LLM judgment)
    """
    # Get template content directly from criteria
    system_template = criteria.criteria.get("judge_system_template")
    user_template = criteria.criteria.get("judge_user_template")
    model_name = criteria.criteria.get("model")

    if not all([system_template, user_template, model_name]):
        raise RuntimeError("Missing required template content or model in criteria")

    # Type narrowing - we've verified these are not None above
    assert system_template is not None
    assert user_template is not None
    assert model_name is not None

    logger.debug(
        "Using templates from criteria",
        extra={"system_len": len(system_template), "user_len": len(user_template), "event": "using_criteria_templates"},
    )

    # Setup Jinja2
    env = Environment(loader=TemplateStringLoader({"system": system_template, "user": user_template}))

    # Build context
    context = {
        "question": criteria.task_context.description,
        "golden_answer": criteria.criteria.get("golden_answer", ""),
        "submission": submission_data.submission,
        "task_id": criteria.task_id,
        "domain": criteria.task_context.domain,
    }

    # Render templates
    system_message = env.get_template("system").render(context)
    user_message = env.get_template("user").render(context)

    logger.debug(
        "Rendered submission templates",
        extra={
            "system_len": len(system_message),
            "user_len": len(user_message),
            "event": "render_submission_templates",
        },
    )

    # Execute LLM
    state.messages.clear()
    state.messages.append(ChatMessageSystem(content=system_message))
    state.messages.append(ChatMessageUser(content=user_message))

    model = get_model(model_name)
    response = await model.generate(state.messages)
    state.output = response  # Update state with LLM response

    # Parse response (expect CORRECT/INCORRECT)
    judge_response = state.output.completion.upper()
    max_score = criteria.scoring.get("max_score", 1.0)

    # Check for INCORRECT first (since INCORRECT contains CORRECT as substring)
    if "INCORRECT" in judge_response:
        logger.info("LLM judge: INCORRECT", extra={"score": 0.0, "event": "llm_submission_incorrect"})
        return 0.0
    elif "CORRECT" in judge_response:
        logger.info("LLM judge: CORRECT", extra={"score": max_score, "event": "llm_submission_correct"})
        return max_score
    else:
        logger.info("LLM judge: UNCLEAR", extra={"score": 0.0, "event": "llm_submission_unclear"})
        return 0.0


# ============================================================================
# Step Evaluation
# ============================================================================


async def _score_steps(
    steps_data: EpisodeStepsResponse,
    criteria: StepEvaluationCriteriaResponse,
    task_context: Any,
    session_manager: Any,
    state: TaskState,
) -> Tuple[float, List[StepEvaluation]]:
    """
    Score steps with LLM evaluation and chunking.

    Args:
        steps_data: Episode steps data
        criteria: Step evaluation criteria
        task_context: Task context
        session_manager: Client session manager
        state: Task state

    Returns:
        Tuple of (total_step_score, list_of_step_evaluations)
    """
    # Get template content directly from criteria
    system_template = criteria.criteria.get("judge_system_template")
    user_template = criteria.criteria.get("judge_user_template")
    model_name = criteria.criteria.get("model")
    steps_per_message = criteria.criteria.get("steps_per_message", 10)

    if not all([system_template, user_template, model_name]):
        raise RuntimeError("Missing required template content or model in step criteria")

    # Type narrowing - we've verified these are not None above
    assert system_template is not None
    assert user_template is not None
    assert model_name is not None

    logger.debug(
        "Using step templates from criteria",
        extra={
            "system_len": len(system_template),
            "user_len": len(user_template),
            "event": "using_step_criteria_templates",
        },
    )

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
                        "done": False,  # We don't have this info from EpisodeStepData
                        # These may not be available in EpisodeStepData
                        "assistant_message": getattr(step, "assistant_message", None),
                        "reasoning": getattr(step, "reasoning", None),
                    }
                )
            )

        # Create episode-like object
        episode = EpisodeContextForTemplate(step_objects)

        # Build context matching template expectations
        context = {
            "question": task_context.description,
            "golden_answer": criteria.criteria.get("golden_answer", ""),  # Add golden_answer
            "episode": episode,  # Wrapped steps with get_step_count() method
            "task": {  # Wrap subtasks in task object
                "subtasks": [
                    type(
                        "Subtask",
                        (),
                        {
                            "subtask_id": st["subtask_id"],
                            "objective": st["objective"],
                        },
                    )()
                    for st in criteria.subtasks
                ]
            },
            "task_id": criteria.task_id,
        }

        # Render templates
        system_message = env.get_template("system").render(context)
        user_message = env.get_template("user").render(context)

        logger.debug(
            "Rendered step chunk templates",
            extra={
                "chunk": chunk_idx,
                "system_len": len(system_message),
                "user_len": len(user_message),
                "event": "render_step_chunk",
            },
        )

        # Execute LLM
        state.messages.clear()
        state.messages.append(ChatMessageSystem(content=system_message))
        state.messages.append(ChatMessageUser(content=user_message))

        model = get_model(model_name)
        response = await model.generate(state.messages)
        state.output = response  # Update state with LLM response

        # Parse step evaluations
        judge_response = state.output.completion
        chunk_evals = parse_step_evaluations(judge_response, criteria.task_id)
        all_step_evaluations.extend(chunk_evals)

        logger.debug(
            "Parsed step evaluations",
            extra={"chunk": chunk_idx, "evaluations": len(chunk_evals), "event": "parse_step_chunk"},
        )

    # Calculate score from subtasks
    subtasks_with_scores = {st["subtask_id"]: st.get("max_score", 0.0) for st in criteria.subtasks}

    total_score = 0.0
    for step_eval in all_step_evaluations:
        if step_eval.objective_id in subtasks_with_scores:
            total_score += subtasks_with_scores[step_eval.objective_id]

    logger.info(
        "Step evaluation complete",
        extra={"total_score": total_score, "evaluations": len(all_step_evaluations), "event": "step_eval_complete"},
    )

    return total_score, all_step_evaluations
