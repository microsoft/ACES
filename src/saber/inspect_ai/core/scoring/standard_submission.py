"""Standard submission scoring methods.

This module provides the standard scoring methods for task submissions:
- static: Pattern matching against expected answers
- llm_judge: LLM-based evaluation using judge templates
- none: No-op scorer that always returns 0 (for tasks without submission evaluation)
"""

from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.solver import TaskState
from jinja2 import Environment

from ....logging_config import LogCategory, get_saber_logger
from ....models.rest.evaluation import EpisodeSubmissionResponse, SubmissionEvaluationCriteriaResponse
from .template_utils import TemplateStringLoader, resolve_template_content

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


async def score_submission_none(
    submission_data: EpisodeSubmissionResponse,
    criteria: SubmissionEvaluationCriteriaResponse,
    session_manager: Any,
    state: TaskState,
) -> tuple[float, str]:
    """No-op submission scoring that always returns 0.

    Used for tasks without submission evaluation (e.g., continuous monitoring agents).

    Args:
        submission_data: Episode submission data (unused)
        criteria: Submission evaluation criteria (unused)
        session_manager: Client session manager (unused)
        state: Task state (unused)

    Returns:
        Tuple of (0.0, explanation)
    """
    logger.info(
        "No submission evaluation configured (none strategy)",
        extra={"event": "submission_none_strategy"},
    )
    return 0.0, "submission=0.0 (no submission evaluation configured)"


async def score_submission_static(
    submission_data: EpisodeSubmissionResponse,
    criteria: SubmissionEvaluationCriteriaResponse,
    session_manager: Any,
    state: TaskState,
) -> tuple[float, str]:
    """Static submission scoring using pattern matching.

    Args:
        submission_data: Episode submission data
        criteria: Submission evaluation criteria
        session_manager: Client session manager (unused)
        state: Task state (unused)

    Returns:
        Tuple of (score, explanation) where score is 0.0 or max_score
    """
    expected_answers = criteria.criteria.get("expected_answers", [])
    max_score = criteria.scoring.get("max_score", 1.0)

    # Normalize expected_answers to always be a list (handle string or list input)
    if isinstance(expected_answers, str):
        expected_answers = [expected_answers]

    submission_lower = submission_data.submission.lower()
    for expected in expected_answers:
        if expected.lower() in submission_lower:
            logger.info(
                "Static submission match found",
                extra={"expected": expected, "score": max_score, "event": "static_submission_match"},
            )
            explanation = f"submission={max_score} Expected: {expected_answers}"
            return max_score, explanation

    logger.info("Static submission no match", extra={"score": 0.0, "event": "static_submission_no_match"})
    explanation = f"submission=0.0 Expected: {expected_answers}"
    return 0.0, explanation


async def score_submission_llm(
    submission_data: EpisodeSubmissionResponse,
    criteria: SubmissionEvaluationCriteriaResponse,
    session_manager: Any,
    state: TaskState,
) -> tuple[float, str]:
    """LLM-based submission scoring using judge templates.

    Args:
        submission_data: Episode submission data
        criteria: Submission evaluation criteria
        session_manager: Client session manager (unused - templates in criteria)
        state: Task state

    Returns:
        Tuple of (score, explanation) from LLM judge
    """
    # Get template content from criteria (server pre-resolves paths for llm_judge,
    # but resolve_template_content handles it if they are still file paths)
    system_template_raw = criteria.criteria.get("judge_system_template")
    user_template_raw = criteria.criteria.get("judge_user_template")
    model_name = criteria.criteria.get("model")

    if not all([system_template_raw, user_template_raw, model_name]):
        raise RuntimeError(
            f"LLM submission evaluation requires judge_system_template, judge_user_template, "
            f"and model in criteria. Got keys: {list(criteria.criteria.keys())}"
        )

    # Type narrowing - we've verified these are not None above
    assert system_template_raw is not None
    assert user_template_raw is not None
    assert model_name is not None

    # Resolve templates (handles both pre-resolved content and file paths)
    system_template = await resolve_template_content(session_manager, system_template_raw)
    user_template = await resolve_template_content(session_manager, user_template_raw)

    logger.debug(
        "Using templates from criteria",
        extra={
            "system_len": len(system_template),
            "user_len": len(user_template),
            "model": model_name,
            "event": "using_criteria_templates",
        },
    )

    # Setup Jinja2 with inline templates
    env = Environment(loader=TemplateStringLoader({"system": system_template, "user": user_template}))

    # Build context
    golden_answer = criteria.criteria.get("golden_answer", "")
    context = {
        "question": criteria.task_context.description,
        "golden_answer": golden_answer,
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

    # Execute LLM with separate message list to preserve state.messages from solver
    # IMPORTANT: Do NOT modify state.messages - it contains the solver's conversation history
    # which gets written to the eval file as the sample's messages field
    scorer_messages = [
        ChatMessageSystem(content=system_message),
        ChatMessageUser(content=user_message),
    ]

    model = get_model(model_name)
    response = await model.generate(scorer_messages)
    # Note: Not updating state.output to preserve solver's output

    # Parse response (expect CORRECT/INCORRECT)
    judge_response = response.completion.upper()
    max_score = criteria.scoring.get("max_score", 1.0)

    # Check for INCORRECT first (since INCORRECT contains CORRECT as substring)
    if "INCORRECT" in judge_response:
        logger.info("LLM judge: INCORRECT", extra={"score": 0.0, "event": "llm_submission_incorrect"})
        explanation = f"submission=0.0 Expected: {golden_answer}" if golden_answer else "submission=0.0"
        return 0.0, explanation

    if "CORRECT" in judge_response:
        logger.info("LLM judge: CORRECT", extra={"score": max_score, "event": "llm_submission_correct"})
        explanation = f"submission={max_score}"
        return max_score, explanation

    # Fallback: couldn't parse response
    logger.warning(
        "Could not parse LLM judge response, defaulting to 0",
        extra={"response": judge_response[:200], "event": "llm_submission_parse_error"},
    )
    explanation = "submission=0.0 (unable to parse LLM response)"
    return 0.0, explanation
