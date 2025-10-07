"""
SABER Task Scorer for inspect_ai

Scorer implementation for SABER tasks within inspect_ai framework.
This provides both client-side and server-side scoring capabilities
while maintaining compatibility with SABER's server-side success criteria evaluation.

Logging category: EVALUATION.

Following SABER's philosophy:
- Fail fast when scoring prerequisites are not met
- Clean interface between inspect_ai scoring and SABER task evaluation
- Explicit handling of server-side vs client-side scoring
- No silent scoring failures that mask evaluation issues

Configuration Options:
- enable_override (bool): Submit client evaluation results to server override endpoint (default: True)
- override_on_failure (bool): Submit overrides even when client score is 0 (default: True)
- log_override_errors (bool): Log errors if override submission fails (default: True)

Usage:
    # Use default configuration (all overrides enabled)
    scorer = saber_scorer()

    # Disable override submission
    scorer = saber_scorer(enable_override=False)

    # Enable overrides but don't submit failures
    scorer = saber_scorer(enable_override=True, override_on_failure=False)
"""

from typing import Any, Dict, List

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Scorer, Target, metric, scorer
from inspect_ai.scorer._metric import Metric, SampleScore, ValueToFloat, value_to_float
from inspect_ai.solver import TaskState
from inspect_ai.util import store

from ...logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ...models.evaluation_utils import (
    build_step_evaluation_explanation,
    calculate_step_evaluation_score,
    parse_step_evaluations,
)
from ...models.rest.evaluation import EvaluationCriteriaResponse, EvaluationOverrideRequest

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


@metric  # type: ignore[misc]
def saber_client_score(to_float: ValueToFloat = value_to_float()) -> Metric:
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
            # Extract client score from metadata if available
            client_score = (
                item.score.metadata.get("client_score", item.score.value) if item.score.metadata else item.score.value
            )
            total += to_float(client_score)
        return total / float(len(scores))

    return metric_fn


@scorer(metrics=[saber_client_score()])  # type: ignore[misc]
def saber_scorer(
    enable_override: bool = True, override_on_failure: bool = True, log_override_errors: bool = True
) -> Scorer:
    """
    SABER client-side scorer using inspect_ai capabilities.

    Retrieves evaluation criteria AND submission from server, then performs
    local evaluation using inspect_ai's built-in scorers.

    This enables parallel client/server evaluation while leveraging
    inspect_ai's existing scoring infrastructure instead of reimplementing
    evaluation logic.

    Args:
        enable_override: Whether to submit evaluation overrides to server (default: True)
        override_on_failure: Whether to submit overrides even when score is 0 (default: True)
        log_override_errors: Whether to log override submission failures (default: True)

    Returns:
        Scorer function that performs client-side evaluation
    """

    async def score(state: TaskState, target: Target) -> Score:
        """
        Perform client-side evaluation using server's evaluation criteria.

        Args:
            state: TaskState after execution containing SABER context
            target: Target criteria (unused - server provides authoritative criteria)

        Returns:
            Score from client-side evaluation

        Raises:
            RuntimeError: If SABER context is missing or evaluation fails
        """
        try:
            # Get evaluation criteria + submission from server
            criteria = await _get_evaluation_criteria(state)

            # Use server's authoritative submission
            submission = criteria.submission

            if not submission:
                return Score(
                    value=0.0,
                    answer="",
                    explanation="Client-side evaluation: Empty submission from server",
                    metadata={
                        "scorer_type": "saber_client_side",
                        "error": "empty_submission",
                        "session_id": criteria.session_id,
                        "episode_id": criteria.episode_id,
                        "task_id": criteria.task_id,
                    },
                )

            # Delegate to appropriate evaluation strategy
            if criteria.evaluation_config.get("strategy") == "static":
                score_result = await _evaluate_static(submission, criteria)
            elif criteria.evaluation_config.get("strategy") == "llm_judge":
                score_result = await _evaluate_llm(submission, criteria, state)
            else:
                strategy = criteria.evaluation_config.get("strategy", "unknown")
                raise RuntimeError(f"Unsupported evaluation strategy: {strategy}")

            # Submit evaluation override to server (if enabled)
            await _submit_override_if_enabled(
                score_result, criteria, enable_override, override_on_failure, log_override_errors
            )

            return score_result

        except RuntimeError:
            # Re-raise RuntimeErrors (fail-fast principle)
            raise
        except Exception as exc:
            # Catch any other unexpected errors and fail fast
            logger.error(
                "Client-side scorer encountered unexpected error",
                extra={
                    "event": "client_scorer_unexpected_error",
                    "error": str(exc),
                },
            )
            raise RuntimeError(f"Client-side scorer error: {exc}") from exc

    return score


async def _get_evaluation_criteria(state: TaskState) -> EvaluationCriteriaResponse:
    """
    Retrieve evaluation criteria from SABER server.

    Args:
        state: TaskState with SABER context in inspect_ai store

    Returns:
        Complete evaluation criteria package

    Raises:
        RuntimeError: If SABER context is missing or criteria retrieval fails
    """
    # Extract SABER context from inspect_ai store
    task_store = store()
    session_manager = task_store.get("saber_session_manager")
    session_id = task_store.get("saber_session_id")

    # Get episode_id from task store (set by SABER agent)
    current_episode = task_store.get("saber_current_episode")
    episode_id = current_episode.episode_id if current_episode else None

    # Validate SABER context (fail-fast)
    if not session_manager:
        raise RuntimeError(
            "SABER session manager not found in context. " "This scorer requires SABER agent integration."
        )

    if not session_id:
        raise RuntimeError("SABER session ID not found in context. " "This scorer requires an active SABER session.")

    if not episode_id:
        raise RuntimeError("SABER episode ID not found in context. " "This scorer requires an active SABER episode.")

    log_operation_start(
        logger,
        "evaluation_criteria_fetch",
        session_id=session_id,
        episode_id=episode_id,
    )

    try:
        # Retrieve evaluation criteria from server (this needs to be implemented in session manager)
        criteria = await session_manager.get_evaluation_criteria(session_id, episode_id)
    except Exception as exc:
        log_operation_failure(
            logger,
            "evaluation_criteria_fetch",
            exc,
            session_id=session_id,
            episode_id=episode_id,
        )
        raise RuntimeError(f"Failed to retrieve evaluation criteria: {exc}") from exc

    log_operation_success(
        logger,
        "evaluation_criteria_fetch",
        session_id=session_id,
        episode_id=episode_id,
        strategy=criteria.evaluation_config.get("strategy", "unknown"),
    )
    logger.debug(
        "Evaluation criteria retrieved",
        extra={
            "event": "evaluation_criteria_received",
            "session_id": session_id,
            "episode_id": episode_id,
            "strategy": criteria.evaluation_config.get("strategy", "unknown"),
        },
    )
    return criteria  # type: ignore[no-any-return]


async def _evaluate_static(submission: str, criteria: EvaluationCriteriaResponse) -> Score:
    """
    Static evaluation using substring matching.

    Uses case-insensitive substring matching (contains) rather than exact
    string matching, which is more practical for security tasks where agents
    often provide longer responses containing expected flags/answers.

    Args:
        submission: Agent's submission to evaluate
        criteria: Complete evaluation criteria from server

    Returns:
        Score from static evaluation
    """
    expected_answers = criteria.evaluation_config.get("criteria", {}).get("expected_answers", [])
    max_score = criteria.evaluation_config.get("scoring", {}).get("max_score", 1.0)

    # Case-insensitive substring matching
    submission_lower = submission.lower()

    for expected in expected_answers:
        if expected.lower() in submission_lower:
            # Found a match - success!
            return Score(
                value=max_score,
                answer=submission,
                explanation=f"Client-side static evaluation: Found expected answer '{expected}' in submission",
                metadata={
                    "strategy": "static",
                    "scorer_type": "saber_client_side",
                    "expected_answers": expected_answers,
                    "matched_answer": expected,
                    "evaluation_method": "substring_match",
                    "client_score": max_score,
                    "is_correct": True,
                    "session_id": criteria.session_id,
                    "episode_id": criteria.episode_id,
                    "task_id": criteria.task_id,
                },
            )

    # No matches found - failure
    return Score(
        value=0.0,
        answer=submission,
        explanation=(
            f"Client-side static evaluation: Submission does not contain any expected answers: " f"{expected_answers}"
        ),
        metadata={
            "strategy": "static",
            "scorer_type": "saber_client_side",
            "expected_answers": expected_answers,
            "evaluation_method": "substring_match",
            "client_score": 0.0,
            "is_correct": False,
            "session_id": criteria.session_id,
            "episode_id": criteria.episode_id,
            "task_id": criteria.task_id,
        },
    )


async def _evaluate_llm(submission: str, criteria: EvaluationCriteriaResponse, state: TaskState) -> Score:
    """
    LLM evaluation using step-by-step analysis with STEP_EVALUATIONS format.

    Uses inspect_ai's generate() with pre-rendered judge prompts from the server.
    This follows inspect_ai's intended flow and ensures identical evaluation
    context between client and server.

    Following SABER best practices:
    - FAIL FAST: Clear error on missing judge messages or invalid response format
    - NO BACKWARDS COMPATIBILITY: Use modern step-evaluation patterns
    - Clean separation following AIR-Bench example patterns

    Args:
        submission: Agent's submission to evaluate
        criteria: Complete evaluation criteria from server
        state: TaskState for inspect_ai compatibility

    Returns:
        Score from step-by-step LLM evaluation
    """
    if not criteria.judge_messages:
        # Enhanced error message with actionable debugging information
        strategy = criteria.evaluation_config.get("strategy", "unknown")
        task_id = criteria.task_id
        episode_id = criteria.episode_id
        session_id = criteria.session_id

        error_details = [
            f"LLM evaluation strategy '{strategy}' requires judge_messages but none provided",
            f"Task: {task_id}, Episode: {episode_id}, Session: {session_id}",
            "This indicates server-side prompt template rendering failed.",
            "Check server logs for template rendering errors.",
            "Verify judge template files exist and are syntactically correct.",
        ]

        # Fail fast with comprehensive error context
        raise RuntimeError(" | ".join(error_details))

    judge_messages = criteria.judge_messages
    max_score = criteria.evaluation_config.get("scoring", {}).get("max_score", 1.0)

    # Following inspect_ai patterns (similar to AIR-Bench example):
    # 1. Clear the existing messages and start fresh for judge evaluation
    # 2. Add system message first (if provided)
    # 3. Add user message for judge evaluation
    # 4. Use inspect_ai's natural message flow

    # Clear existing messages for clean judge evaluation context
    state.messages.clear()

    # Add system message first (inspect_ai pattern)
    if judge_messages.system_message:
        state.messages.append(ChatMessageSystem(content=judge_messages.system_message))

    # Add user message for judge evaluation (following inspect_ai message patterns)
    state.messages.append(ChatMessageUser(content=judge_messages.user_message))

    try:
        # Get the judge model for evaluation
        judge_model = get_model(judge_messages.model)

        # Generate judge response using inspect_ai's proper model pattern
        response = await judge_model.generate(state.messages)

        # Update state with the judge response
        state.output = response

        if not state.output or not state.output.completion:
            raise RuntimeError("Judge model returned empty response")

        judge_response = state.output.completion

        # Parse step evaluations from judge response
        try:
            step_evaluations = parse_step_evaluations(judge_response, criteria.task_id)
        except RuntimeError as parse_error:
            raise RuntimeError(f"Step evaluation parsing failed: {parse_error}") from parse_error

        # Build subtask score mapping (always use aggregation mode)
        subtasks_with_scores = {}
        if criteria.task_context.subtasks:
            logger.info(f"Building subtasks_with_scores from {len(criteria.task_context.subtasks)} subtasks")
            for subtask in criteria.task_context.subtasks:
                subtask_id = subtask.get("subtask_id")
                subtask_max_score = subtask.get("max_score", 0.0)
                if subtask_id and subtask_max_score is not None and subtask_max_score > 0:
                    subtasks_with_scores[subtask_id] = float(subtask_max_score)
                    logger.debug(
                        "Subtask score registered",
                        extra={
                            "event": "subtask_score_registered",
                            "subtask_id": subtask_id,
                            "max_score": subtask_max_score,
                        },
                    )

            logger.info(f"Subtask scores: {len(subtasks_with_scores)} subtasks with scores defined")

        # Calculate score using aggregation mode
        score_value, is_correct, task_completed_at_step, subtasks_completed = calculate_step_evaluation_score(
            step_evaluations, criteria.task_id, max_score, subtasks_with_scores if subtasks_with_scores else None
        )
        logger.info(
            "Score calculated",
            extra={
                "event": "score_calculated",
                "score_value": score_value,
                "is_correct": is_correct,
                "task_completed_at_step": task_completed_at_step,
                "subtasks_completed": subtasks_completed,
            },
        )

        # Calculate max possible score (always aggregation)
        max_possible_score = max_score + sum(subtasks_with_scores.values()) if subtasks_with_scores else max_score

        # Build explanation using shared utility (always show max possible score)
        step_explanation = build_step_evaluation_explanation(
            is_correct, task_completed_at_step, subtasks_completed, score_value, max_possible_score
        )
        explanation = f"Client-side {step_explanation}"

        return Score(
            value=score_value,
            answer=submission,
            explanation=explanation,
            metadata={
                "strategy": "llm_judge_step_evaluation",
                "scorer_type": "saber_client_side",
                "model": judge_messages.model,
                "is_correct": is_correct,
                "client_score": score_value,
                "step_evaluations": [step_eval.model_dump() for step_eval in step_evaluations],
                "task_completed_at_step": task_completed_at_step,
                "subtasks_completed": subtasks_completed,
                "total_steps_evaluated": len(step_evaluations),
                "judge_response": judge_response,
                "session_id": criteria.session_id,
                "episode_id": criteria.episode_id,
                "task_id": criteria.task_id,
                "scoring_mode": "aggregation",
                "max_possible_score": max_possible_score,
                "subtasks_with_scores": subtasks_with_scores,
            },
        )

    except Exception as exc:
        logger.error(
            "LLM evaluation failed",
            extra={
                "event": "llm_evaluation_failed",
                "error": str(exc),
                "session_id": criteria.session_id,
                "episode_id": criteria.episode_id,
                "task_id": criteria.task_id,
                "strategy": criteria.evaluation_config.get("strategy", "unknown"),
                "judge_model": judge_messages.model if judge_messages else None,
            },
        )
        raise RuntimeError(f"LLM evaluation error: {exc}") from exc


def _build_override_request(
    score_result: Score, criteria: EvaluationCriteriaResponse, max_score: float
) -> EvaluationOverrideRequest:
    """
    Build an evaluation override request from client-side scoring results.

    Maps the inspect_ai Score object and evaluation context to the format
    expected by the server's override endpoint.

    Args:
        score_result: The Score object from client-side evaluation
        criteria: The evaluation criteria retrieved from server
        max_score: Maximum possible score for this evaluation

    Returns:
        EvaluationOverrideRequest object for server submission

    Raises:
        RuntimeError: If required data is missing from score or criteria
    """
    # Validate required inputs
    if not score_result:
        raise RuntimeError("Score result is required for override request")
    if not criteria:
        raise RuntimeError("Evaluation criteria is required for override request")
    if not criteria.episode_id:
        raise RuntimeError("Episode ID is required for override request")
    if not criteria.task_id:
        raise RuntimeError("Task ID is required for override request")

    # Extract metadata safely
    metadata = score_result.metadata or {}
    strategy = criteria.evaluation_config.get("strategy", "unknown")

    # Build evaluation_data section (EpisodeEvaluationData format)
    evaluation_data = {
        "episode_id": criteria.episode_id,
        "task_id": criteria.task_id,
        "submission": criteria.submission or "",
        "executed_commands": [],  # Client doesn't have command history
        "completion_reason": "client_evaluation",
        "step_count": 1,  # Client-side evaluation is single step
        "model": _extract_model_info(criteria, metadata),
        "choices": [{"message": {"content": score_result.explanation or ""}}],
        "tokens": metadata.get("tokens", {}),
        "execution_time": 0.0,  # Client doesn't track execution time
    }

    # Determine success status
    is_correct = metadata.get("is_correct")
    if is_correct is None:
        # Fallback: consider non-zero scores as success
        is_correct = score_result.value > 0

    # Build details section
    details = {
        "client_scorer_type": "saber_client_side",
        "original_explanation": score_result.explanation or "",
        "evaluation_method": metadata.get("evaluation_method", strategy),
        "client_metadata": metadata,
        "server_strategy": strategy,
        "override_timestamp": "client_submission",
    }

    # Create strongly typed override request
    override_request = EvaluationOverrideRequest(
        evaluation_data=evaluation_data,
        strategy=f"client_side_{strategy}",
        raw_score=score_result.value,
        max_score=max_score,
        score=score_result.value,
        success=is_correct,
        details=details,
    )

    logger.debug(
        "Override request constructed",
        extra={
            "event": "override_request_built",
            "episode_id": criteria.episode_id,
            "task_id": criteria.task_id,
            "score": score_result.value,
            "success": is_correct,
            "strategy": strategy,
        },
    )

    return override_request


def _extract_model_info(criteria: EvaluationCriteriaResponse, metadata: Dict[str, Any]) -> str:
    """
    Extract model information for override request.

    Args:
        criteria: Evaluation criteria containing judge messages
        metadata: Score metadata that might contain model info

    Returns:
        Model name/identifier for the override request
    """
    # Try judge messages first (for LLM evaluation)
    if criteria.judge_messages and criteria.judge_messages.model:
        return criteria.judge_messages.model

    # Try metadata
    if metadata.get("model"):
        return str(metadata["model"])

    # Fallback based on strategy
    strategy = criteria.evaluation_config.get("strategy", "unknown")
    if strategy == "llm_judge":
        return "unknown_llm_judge"
    elif strategy == "static":
        return "client_static_scorer"
    else:
        return "client_scorer"


async def _submit_override_if_enabled(
    score_result: Score,
    criteria: EvaluationCriteriaResponse,
    enable_override: bool,
    override_on_failure: bool,
    log_override_errors: bool,
) -> None:
    """
    Submit evaluation override to server if configuration permits.

    This function implements comprehensive error handling with graceful degradation:
    - Configuration validation (early returns for disabled features)
    - Input validation with detailed error messages
    - Network/service error handling with retry logic
    - Graceful degradation (never raises exceptions)

    Args:
        score_result: The Score result from client-side evaluation
        criteria: Evaluation criteria containing server context
        enable_override: Whether override submission is enabled
        override_on_failure: Whether to submit overrides even for failed scores
        log_override_errors: Whether to log detailed error information on override failures

    Raises:
        No exceptions - all errors are caught and optionally logged based on configuration
    """

    def _log_override_failure(message: str, extra: Dict[str, Any]) -> None:
        if log_override_errors:
            logger.error(message, extra=extra)
        else:
            logger.debug(message, extra=extra)

    if not score_result:
        _log_override_failure(
            "Override submission aborted: missing score result",
            extra={
                "event": "override_input_validation_failed",
                "reason": "missing_score_result",
            },
        )
        return

    if not criteria:
        _log_override_failure(
            "Override submission aborted: missing evaluation criteria",
            extra={
                "event": "override_input_validation_failed",
                "reason": "missing_criteria",
            },
        )
        return

    session_id = criteria.session_id
    episode_id = criteria.episode_id

    if not session_id:
        _log_override_failure(
            "Override submission aborted: missing session identifier",
            extra={
                "event": "override_input_validation_failed",
                "reason": "missing_session_id",
                "episode_id": episode_id,
            },
        )
        return

    if not episode_id:
        _log_override_failure(
            "Override submission aborted: missing episode identifier",
            extra={
                "event": "override_input_validation_failed",
                "reason": "missing_episode_id",
                "session_id": session_id,
            },
        )
        return

    if not enable_override:
        logger.debug(
            "Override submission disabled by configuration",
            extra={
                "event": "override_disabled",
                "session_id": session_id,
                "episode_id": episode_id,
            },
        )
        return

    is_failure = score_result.value == 0.0
    if is_failure and not override_on_failure:
        logger.debug(
            "Override submission skipped because score indicates failure",
            extra={
                "event": "override_skipped_for_failure",
                "session_id": session_id,
                "episode_id": episode_id,
                "score": score_result.value,
                "override_on_failure": override_on_failure,
            },
        )
        return

    strategy = criteria.evaluation_config.get("strategy", "unknown")
    max_score = criteria.evaluation_config.get("scoring", {}).get("max_score", 1.0)

    task_store = store()
    session_manager = task_store.get("saber_session_manager")

    if not session_manager:
        _log_override_failure(
            "Override submission failed: session manager unavailable",
            extra={
                "event": "override_session_manager_missing",
                "session_id": session_id,
                "episode_id": episode_id,
                "strategy": strategy,
                "override_on_failure": override_on_failure,
            },
        )
        return

    try:
        override_request = _build_override_request(score_result, criteria, max_score)
    except Exception as build_error:
        _log_override_failure(
            "Override submission failed while constructing request",
            extra={
                "event": "override_request_build_failed",
                "session_id": session_id,
                "episode_id": episode_id,
                "strategy": strategy,
                "error": str(build_error),
                "override_on_failure": override_on_failure,
            },
        )
        return

    log_operation_start(
        logger,
        "override_submission",
        session_id=session_id,
        episode_id=episode_id,
        strategy=strategy,
        score=score_result.value,
        submitted_on_failure=is_failure,
    )

    try:
        response = await session_manager.override_episode_evaluation(
            session_id=session_id,
            episode_id=episode_id,
            override_request=override_request,
        )
    except AttributeError as attr_error:
        log_operation_failure(
            logger,
            "override_submission",
            attr_error,
            session_id=session_id,
            episode_id=episode_id,
            strategy=strategy,
        )
        _log_override_failure(
            "Override submission failed: session manager missing override endpoint",
            extra={
                "event": "override_submission_method_missing",
                "session_id": session_id,
                "episode_id": episode_id,
                "strategy": strategy,
                "error": str(attr_error),
                "override_on_failure": override_on_failure,
            },
        )
        return
    except Exception as submission_error:
        log_operation_failure(
            logger,
            "override_submission",
            submission_error,
            session_id=session_id,
            episode_id=episode_id,
            strategy=strategy,
        )
        _log_override_failure(
            "Override submission failed during server call",
            extra={
                "event": "override_submission_failed",
                "session_id": session_id,
                "episode_id": episode_id,
                "strategy": strategy,
                "score": score_result.value,
                "max_score": max_score,
                "error": str(submission_error),
                "error_type": type(submission_error).__name__,
                "override_on_failure": override_on_failure,
            },
        )
        return

    log_operation_success(
        logger,
        "override_submission",
        session_id=session_id,
        episode_id=episode_id,
        strategy=strategy,
        score=score_result.value,
        submitted_on_failure=is_failure,
        response_type=type(response).__name__,
    )
    logger.info(
        "Override submission succeeded",
        extra={
            "event": "override_submission_success",
            "session_id": session_id,
            "episode_id": episode_id,
            "strategy": strategy,
            "score": score_result.value,
            "submitted_on_failure": is_failure,
            "response_type": type(response).__name__,
            "override_on_failure": override_on_failure,
        },
    )
