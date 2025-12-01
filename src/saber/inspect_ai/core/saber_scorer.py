"""
SABER Task Scorer for inspect_ai - CLIENT-SIDE EVALUATION (Breaking Change Migration)

Complete rewrite for client-side evaluation architecture.
ALL evaluation logic happens on the client:
- Extract and submit agent's answer to server (keeps episode alive)
- Fetch submission and steps from server
- Fetch evaluation criteria (template paths only)
- Fetch and render templates with Jinja2
- Execute LLM evaluation
- Calculate scores
- Submit results to server
- Episode remains alive until sample_cleanup ends it

NO server-side evaluation. NO backward compatibility.

Logging category: EVALUATION.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Scorer, Target, metric, scorer
from inspect_ai.scorer._metric import Metric, SampleScore, ValueToFloat, value_to_float
from inspect_ai.solver import TaskState
from inspect_ai.util import store
from jinja2 import BaseLoader, Environment, TemplateError

from ...logging_config import LogCategory, get_saber_logger
from ...models.constants import MetadataKeys, StepEvaluationStrategy, SubmissionEvaluationStrategy
from ...models.core import EvalSubmission
from ...models.rest.evaluation import (
    EpisodeStepsResponse,
    EpisodeSubmissionResponse,
    EvaluationResultSubmission,
    StepEvaluation,
    SubmissionEvaluationCriteriaResponse,
    SubtaskEvaluationCriteriaResponse,
)
from ..constants import SandboxTimeouts

logger = get_saber_logger(LogCategory.EVALUATION, __name__)


# ============================================================================
# Utility Functions
# ============================================================================


def clean_dict(payload: Any) -> Any:
    """Remove None/False values to prevent server-side type coercion issues."""
    if isinstance(payload, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in payload.items():
            if value is None or value is False:
                continue
            cleaned[key] = clean_dict(value)
        return cleaned
    if isinstance(payload, list):
        return [clean_dict(item) for item in payload if item is not None and item is not False]
    return payload


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
# Metrics
# ============================================================================


@metric  # type: ignore[misc]
def saber_score(to_float: ValueToFloat = value_to_float()) -> Metric:
    """
    Overall SABER evaluation score (submission + subtasks).

    Computes the average total score across all samples.

    Returns:
        Metric function that computes average total score
    """

    def metric_fn(scores: List[SampleScore]) -> float:
        total = 0.0
        for item in scores:
            total += to_float(item.score.value)
        return total / float(len(scores)) if scores else 0.0

    return metric_fn


@metric  # type: ignore[misc]
def submission_score() -> Metric:
    """
    Average submission score across all samples.

    This represents how well agents answered the main task question.

    Returns:
        Metric function that computes average submission score
    """

    def metric_fn(scores: List[SampleScore]) -> float:
        submission_scores = []
        for sample_score in scores:
            if sample_score.score.metadata:
                submission_score_value = sample_score.score.metadata.get(MetadataKeys.SUBMISSION_SCORE, 0.0)
                submission_scores.append(float(submission_score_value))

        return sum(submission_scores) / len(submission_scores) if submission_scores else 0.0

    return metric_fn


@metric  # type: ignore[misc]
def subtask_score() -> Metric:
    """
    Average subtask score across all samples.

    This represents how well agents completed intermediate checkpoints.
    Only reports when subtasks were actually scored (returns empty dict otherwise).

    Returns:
        Metric function that returns {"subtask_score": avg} or {}
    """

    def metric_fn(scores: List[SampleScore]) -> Dict[str, float]:
        subtask_scores = []
        for sample_score in scores:
            if sample_score.score.metadata:
                # Only include samples that have subtask_score in metadata
                # (i.e., tasks with step_evaluation_config configured)
                subtask_score_value = sample_score.score.metadata.get(MetadataKeys.SUBTASK_SCORE)
                if subtask_score_value is not None:
                    subtask_scores.append(float(subtask_score_value))

        # Return dict with score if we have any subtask scores, empty dict otherwise
        if subtask_scores:
            return {"subtask_score": sum(subtask_scores) / len(subtask_scores)}
        else:
            return {}

    return metric_fn


@metric  # type: ignore[misc]
def per_task_submission_scores() -> Metric:
    """
    Per-task submission score metrics.

    Creates a metric for each task showing the average submission score
    for samples of that task. Metric names: <task_id>_submission_score

    Returns:
        Metric function that computes per-task submission scores
    """

    def metric_fn(scores: List[SampleScore]) -> Dict[str, float]:
        # Track submission scores by task_id
        task_submission_scores: Dict[str, List[float]] = {}

        for sample_score in scores:
            if not sample_score.score.metadata:
                continue

            # Get task_id from sample metadata
            sample_metadata = sample_score.sample_metadata or {}
            task_id = sample_metadata.get(MetadataKeys.TASK_ID)
            if not task_id:
                continue

            # Get submission score
            submission_score_value = sample_score.score.metadata.get(MetadataKeys.SUBMISSION_SCORE, 0.0)

            # Track by task_id
            if task_id not in task_submission_scores:
                task_submission_scores[task_id] = []
            task_submission_scores[task_id].append(float(submission_score_value))

        # Calculate averages per task
        results = {}
        for task_id, score_list in task_submission_scores.items():
            avg_score = sum(score_list) / len(score_list) if score_list else 0.0
            sanitized_task = task_id.replace("-", "_").replace(" ", "_")
            metric_key = f"{sanitized_task}_submission_score"
            results[metric_key] = avg_score

        return results

    return metric_fn


@metric  # type: ignore[misc]
def per_task_subtask_scores() -> Metric:
    """
    Per-task subtask score metrics.

    Creates a metric for each task showing the average subtask score
    for samples of that task. Metric names: <task_id>_subtask_score
    Only includes tasks where subtasks were actually scored.

    Returns:
        Metric function that computes per-task subtask scores (empty dict if no subtask scoring)
    """

    def metric_fn(scores: List[SampleScore]) -> Dict[str, float]:
        # Track subtask scores by task_id
        task_subtask_scores: Dict[str, List[float]] = {}

        for sample_score in scores:
            if not sample_score.score.metadata:
                continue

            # Get task_id from sample metadata
            sample_metadata = sample_score.sample_metadata or {}
            task_id = sample_metadata.get(MetadataKeys.TASK_ID)
            if not task_id:
                continue

            # Only include samples that have subtask_score in metadata
            subtask_score_value = sample_score.score.metadata.get(MetadataKeys.SUBTASK_SCORE)
            if subtask_score_value is None:
                continue

            # Track by task_id
            if task_id not in task_subtask_scores:
                task_subtask_scores[task_id] = []
            task_subtask_scores[task_id].append(float(subtask_score_value))

        # Calculate averages per task (returns empty dict if no tasks have subtask scoring)
        results = {}
        for task_id, score_list in task_subtask_scores.items():
            avg_score = sum(score_list) / len(score_list) if score_list else 0.0
            sanitized_task = task_id.replace("-", "_").replace(" ", "_")
            metric_key = f"{sanitized_task}_subtask_score"
            results[metric_key] = avg_score

        return results

    return metric_fn


@metric  # type: ignore[misc]
def subtask_score_metrics() -> Metric:
    """
    Per-subtask average score metrics.

    Creates a metric for each task+subtask combination showing the average
    score earned. Metric names: <task_id>_<subtask_id>_score
    Only includes subtasks that were actually scored.

    Returns:
        Metric function that computes per-subtask average scores
    """

    def metric_fn(scores: List[SampleScore]) -> Dict[str, float]:
        # Track subtask scores by task_id + subtask_id
        subtask_scores: Dict[str, List[float]] = {}

        for sample_score in scores:
            if not sample_score.score.metadata:
                continue

            # Get task_id from sample metadata
            sample_metadata = sample_score.sample_metadata or {}
            task_id = sample_metadata.get(MetadataKeys.TASK_ID)
            if not task_id:
                continue

            # Get the subtask_scores dict from metadata (only exists when step_criteria configured)
            sample_subtask_scores = sample_score.score.metadata.get(MetadataKeys.SUBTASK_SCORES)
            if sample_subtask_scores is None:
                # Skip samples without subtask scoring
                continue

            # Add each subtask's score to our tracking with task_id prefix
            for subtask_id, score_value in sample_subtask_scores.items():
                # Create composite key: task_id:subtask_id
                composite_key = f"{task_id}:{subtask_id}"

                if composite_key not in subtask_scores:
                    subtask_scores[composite_key] = []
                subtask_scores[composite_key].append(float(score_value))

        # Calculate average scores
        results = {}
        for composite_key, score_list in subtask_scores.items():
            # Split composite key back into task_id and subtask_id
            if ":" in composite_key:
                task_id, subtask_id = composite_key.split(":", 1)
            else:
                # Fallback for malformed keys
                task_id = "unknown"
                subtask_id = composite_key

            avg_score = sum(score_list) / len(score_list) if score_list else 0.0

            # Create metric key: <task_id>_<subtask_id>_score
            sanitized_task = task_id.replace("-", "_").replace(" ", "_")
            sanitized_subtask = subtask_id.replace("-", "_").replace(" ", "_")
            metric_key = f"{sanitized_task}_{sanitized_subtask}_score"
            results[metric_key] = avg_score

        return results

    return metric_fn


# ============================================================================
# Main Scorer
# ============================================================================


@scorer(  # type: ignore[misc]
    metrics=[
        saber_score(),  # Total score (submission + subtasks)
        submission_score(),  # Average submission score across all samples
        subtask_score(),  # Average subtask score (only shown when subtasks are scored)
        per_task_submission_scores(),  # Per-task submission scores
        per_task_subtask_scores(),  # Per-task subtask scores (only shown when subtasks are scored)
        subtask_score_metrics(),  # Per-subtask scores (<task_id>_<subtask_id>_score) - only when subtasks are scored
    ]
)
def saber_scorer() -> Scorer:
    """
    Client-side evaluation scorer (BREAKING CHANGE).

    Handles ALL evaluation logic on the client:
    1. Submit agent's answer to server (keeps episode alive during scoring)
    2. Fetch submission and steps from server
    3. Fetch evaluation criteria (template paths)
    4. Fetch and render templates with Jinja2
    5. Execute LLM calls
    6. Calculate scores
    7. Submit results to server
    8. Episode remains alive - sample_cleanup will end it

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
            # Log scorer invocation for debugging
            logger.info(
                "Scorer invoked",
                extra={
                    "event": "scorer_invoked",
                    "state_is_none": state is None,
                    "state_messages_is_none": state.messages is None if state else "state_is_none",
                    "state_metadata": state.metadata if state else None,
                },
            )

            # Extract SABER context from inspect_ai store (same pattern as old working code)
            task_store = store()
            session_manager = task_store.get("saber_session_manager")
            session_id = task_store.get("saber_session_id")
            task_id = task_store.get("saber_task_id", "unknown")

            # FIX: Get episode_id from sample_id-keyed mapping to prevent cross-contamination
            # CRITICAL: Multiple attempts (attempt_1, attempt_2, attempt_3) share the same task_id
            # but run concurrently. Must use sample_id (which includes attempt suffix) as key.
            sample_id = state.metadata.get(MetadataKeys.SAMPLE_ID, task_id)  # Fallback to task_id if not set
            episode_mapping = task_store.get("saber_episode_mapping", {})
            current_episode = episode_mapping.get(sample_id)
            episode_id = current_episode.episode_id if current_episode else None

            # Log context state for debugging
            logger.info(
                "SABER context state",
                extra={
                    "event": "saber_context_check",
                    "session_manager_present": session_manager is not None,
                    "session_id_present": session_id is not None,
                    "current_episode_present": current_episode is not None,
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "sample_id": sample_id,
                },
            )

            # Validate SABER context (fail-fast)
            if not session_manager:
                logger.error(
                    "Scorer called after session_manager cleanup",
                    extra={"event": "scorer_missing_session_manager"},
                )
                raise RuntimeError("Missing saber_session_manager in store - likely called after cleanup")

            if not session_id:
                logger.error(
                    "Scorer called without session_id",
                    extra={"event": "scorer_missing_session_id"},
                )
                raise RuntimeError("Missing saber_session_id in store")

            if not episode_id:
                logger.error(
                    "Scorer called without episode_id",
                    extra={"event": "scorer_missing_episode_id", "current_episode": current_episode},
                )
                raise RuntimeError("Missing episode_id from saber_current_episode in store")

            logger.info(
                "Starting client-side evaluation",
                extra={"session_id": session_id, "episode_id": episode_id, "event": "client_eval_start"},
            )

            # Step 1: Submit the agent's answer to the server (keeps episode alive during scoring)
            # Extract submission from state.output.completion (the agent's final answer)
            agent_answer = state.output.completion if state.output else ""

            # Get task_id from current_episode for the EvalSubmission
            task_id = current_episode.task_id if hasattr(current_episode, "task_id") else "unknown"

            # Extract model name as string (state.model is a ModelName object)
            model_name = str(state.model) if hasattr(state, "model") and state.model else "unknown"

            # Extract token usage from state.output if available
            tokens = {}
            if state.output and hasattr(state.output, "usage") and state.output.usage:
                # Get token usage and ensure all values are integers (convert None to 0)
                usage_dict = state.output.usage.model_dump()
                tokens = {k: v if v is not None else 0 for k, v in usage_dict.items()}

            # Create EvalSubmission from agent's completion
            eval_submission = EvalSubmission(
                episode_id=episode_id,
                task_id=task_id,
                model=model_name,
                choices=[],  # Not available in state
                submission=agent_answer,
                tokens=tokens,
                time=0.0,  # Could calculate if needed
            )

            # Post submission to server (episode stays alive)
            await session_manager.post_episode_submission(session_id, episode_id, eval_submission)

            logger.info(
                "Agent submission posted to server (episode remains active)",
                extra={
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "submission_length": len(agent_answer),
                    "event": "scorer_submission_posted",
                },
            )

            # Step 2: Fetch submission data (should now contain what we just posted)
            submission_data = await session_manager.get_episode_submission(session_id, episode_id)

            logger.debug(
                "Fetched submission data",
                extra={
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "submission": submission_data.submission,
                    "submission_length": (len(submission_data.submission) if submission_data.submission else 0),
                    "event": "submission_data_fetched",
                },
            )

            # Step 3: Fetch step history
            steps_data = await session_manager.get_episode_steps(session_id, episode_id)

            # Step 4: Fetch submission evaluation criteria
            submission_criteria = await session_manager.get_submission_evaluation_criteria(session_id, episode_id)

            # Step 4: Fetch list of all subtask evaluation criteria,
            # to be applied to the list of steps in score_steps (optional)
            subtasks_criteria_list = await session_manager.get_subtask_evaluation_criteria(session_id, episode_id)

            # Step 6: Score submission
            submission_score, submission_explanation = await _score_submission(
                submission_data, submission_criteria, session_manager, state
            )

            # Step 6: Score steps (if configured)
            total_subtask_score = 0.0
            step_evaluations: List[List[StepEvaluation]] = []
            subtask_scores_weighted: Dict[str, float] = {}  # Track weighted subtask scores for total
            subtask_scores_unweighted: Dict[str, float] = {}  # Track unweighted scores for individual metrics
            has_scorable_subtasks = False  # Track if any subtask has a valid evaluation strategy

            if subtasks_criteria_list is not None and subtasks_criteria_list != []:
                # Check if any subtask has a valid strategy (not empty/None)
                has_scorable_subtasks = any(
                    criteria.strategy and criteria.strategy != "" for criteria in subtasks_criteria_list
                )

                total_subtask_score, subtask_scores_raw_results, step_evaluations = await _score_all_subtasks(
                    steps_data, subtasks_criteria_list, submission_criteria.task_context, session_manager, state
                )

                # Map individual scores back to subtask IDs
                # Store both weighted (for totals) and unweighted (for individual metrics)
                for i, criteria in enumerate(subtasks_criteria_list):
                    if i < len(subtask_scores_raw_results):
                        # Unweighted score (raw from strategy)
                        unweighted_score = subtask_scores_raw_results[i]
                        subtask_scores_unweighted[criteria.subtask_id] = unweighted_score

                        # Weighted score (for total calculation)
                        weighted_score = unweighted_score * criteria.weight
                        subtask_scores_weighted[criteria.subtask_id] = weighted_score
                    else:
                        subtask_scores_unweighted[criteria.subtask_id] = 0.0
                        subtask_scores_weighted[criteria.subtask_id] = 0.0

                # Recalculate total with weights applied
                total_subtask_score = sum(subtask_scores_weighted.values())

            # Calculate totals
            total_score = submission_score + total_subtask_score
            max_possible = submission_criteria.scoring.get("max_score", 1.0)
            if has_scorable_subtasks:
                # Calculate max possible with weights applied - only for subtasks with strategies
                weighted_max_possible = sum(
                    criteria.max_score * criteria.weight
                    for criteria in subtasks_criteria_list
                    if criteria.strategy and criteria.strategy != ""
                )
                max_possible += weighted_max_possible

            # Step 8: Submit evaluation result
            evaluation_result = EvaluationResultSubmission(
                strategy="client_side_evaluation",
                raw_score=total_score,
                max_score=max_possible,
                score=total_score,
                success=submission_score > 0,
                details={
                    "task_id": submission_criteria.task_id,
                    "submission_score": submission_score,
                    "total_subtask_score": total_subtask_score,
                    "individual_subtasks_score_weighted": subtask_scores_weighted,
                    "individual_subtasks_score_unweighted": subtask_scores_unweighted,
                    "step_evaluations_by_subtask": {
                        subtasks_criteria_list[i].subtask_id: {
                            "strategy": subtasks_criteria_list[i].strategy,
                            "max_score": subtasks_criteria_list[i].max_score,
                            "weight": subtasks_criteria_list[i].weight,
                            "step_evaluations": [se.model_dump() for se in step_evals],
                        }
                        for i, step_evals in enumerate(step_evaluations)
                        if i < len(subtasks_criteria_list)
                    },
                    "client_scorer_version": "2.3",
                },
            )

            await session_manager.submit_evaluation_result(session_id, episode_id, evaluation_result)

            if has_scorable_subtasks:
                eval_complete_msg = (
                    f"Client-side evaluation completed: score={total_score:.2f} "
                    f"(submission={submission_score:.2f}, weighted_steps={total_subtask_score:.2f})"
                )
            else:
                eval_complete_msg = (
                    f"Client-side evaluation completed: score={total_score:.2f} " f"(submission={submission_score:.2f})"
                )
            logger.info(
                "Client-side evaluation completed",
                extra={
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "total_score": total_score,
                    "submission_score": submission_score,
                    "subtask_score": total_subtask_score,
                    "scoring_method": "sum",
                    "event": "client_eval_complete",
                },
            )

            # Extract task_id for metadata
            task_id = submission_criteria.task_id

            # NOTE: Submission already posted to server in Step 1
            # Episode remains alive through scoring and will be ended by sample_cleanup

            # Step 9: Return score for inspect_ai
            max_sub_score = submission_criteria.scoring.get("max_score", 1.0)

            # Calculate weighted max possible for explanation
            weighted_max_possible = 0.0
            if subtasks_criteria_list:
                weighted_max_possible = sum(criteria.max_score * criteria.weight for criteria in subtasks_criteria_list)

            # Ensure task_id is in state.metadata so it appears in sample_metadata
            if state.metadata is None:
                state.metadata = {}
            state.metadata[MetadataKeys.TASK_ID] = task_id

            # Build metadata - only include subtask data if subtasks_criteria_list exists
            metadata = {
                MetadataKeys.SUBMISSION_SCORE: submission_score,
                "max_possible": max_possible,
                MetadataKeys.TASK_ID: task_id,
                "scorer_version": "2.3",
                "scoring_method": "sum",
            }

            # Only add subtask-related metadata when subtasks are actually being scored
            if has_scorable_subtasks:
                metadata[MetadataKeys.SUBTASK_SCORE] = total_subtask_score
                metadata[MetadataKeys.WEIGHTED_SUBTASK_SCORE] = total_subtask_score
                metadata[MetadataKeys.STEP_EVALUATIONS] = [se.model_dump() for se in sum(step_evaluations, [])]
                metadata[MetadataKeys.SUBTASK_SCORES] = subtask_scores_weighted  # Keep nested for programmatic access

                # Add individual subtask scores as top-level metadata fields
                # Format: <task_id>_<subtask_id>_score
                # Use UNWEIGHTED scores so individual checkpoints show 0.0 or 1.0 (or max_score)
                for subtask_id, subtask_score_value in subtask_scores_unweighted.items():
                    # Sanitize IDs for metric names
                    sanitized_task = task_id.replace("-", "_").replace(" ", "_")
                    sanitized_subtask = subtask_id.replace("-", "_").replace(" ", "_")
                    metric_key = f"{sanitized_task}_{sanitized_subtask}_score"
                    metadata[metric_key] = subtask_score_value

            # Build explanation based on whether subtasks were actually scored
            if has_scorable_subtasks:
                explanation = (
                    f"{submission_explanation}, "
                    f"weighted_subtasks={total_subtask_score:.2f}, "
                    f"sum(submission, subtasks)={total_score}"
                )
            else:
                explanation = f"{submission_explanation}"

            # === ORCHESTRATION COORDINATION ===
            # Wait for all orchestrated samples to complete scoring before returning Score
            orchestration_id = state.metadata.get(MetadataKeys.ORCHESTRATION_ID)
            role = state.metadata.get(MetadataKeys.SUB_TASK_ROLE)

            if orchestration_id and role:
                from .orchestration_coordinator import OrchestrationCoordinator

                coordinator = OrchestrationCoordinator()

                logger.info(
                    f"Orchestrated sample {role} starting score coordination",
                    extra={
                        "orchestration_id": orchestration_id,
                        "role": role,
                        "score": total_score,
                        "event": "scorer_coordination_start",
                    },
                )

                try:
                    # Wait for all siblings to finish scoring
                    # This blocks until all samples in orchestration reach this point
                    await coordinator.wait_for_all_scored(
                        orchestration_id=orchestration_id,
                        role=role,
                        score=total_score,
                        timeout=SandboxTimeouts.ORCHESTRATION_SCORE_SYNC_SECONDS,
                    )

                    logger.info(
                        f"Score coordination complete for {role}",
                        extra={
                            "orchestration_id": orchestration_id,
                            "role": role,
                            "event": "scorer_coordination_complete",
                        },
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"Score coordination timeout for {role} - proceeding anyway",
                        extra={
                            "orchestration_id": orchestration_id,
                            "role": role,
                            "event": "scorer_coordination_timeout",
                        },
                    )
                    # Proceed with partial results rather than failing

            return Score(
                value=total_score,
                answer=agent_answer,  # Use the agent answer we extracted earlier
                explanation=explanation,
                metadata=metadata,
            )

        except Exception as exc:
            error_msg = f"Client-side evaluation failed: {type(exc).__name__}: {exc}"
            logger.error(
                "Client-side evaluation failed",
                extra={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "event": "client_eval_failure",
                    "state_is_none": state is None if "state" in locals() else "not_in_scope",
                    "state_messages_is_none": (
                        state.messages is None if state and "state" in locals() else "cannot_check"
                    ),
                },
                exc_info=True,
            )
            return Score(
                value=0.0,
                answer="",
                explanation=f"Client-side evaluation failed: {exc}",
                metadata={"error": str(exc), "scorer_version": "2.3"},
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
) -> tuple[float, str]:
    """
    Score submission using configured strategy.

    Args:
        submission_data: Episode submission data
        criteria: Submission evaluation criteria
        session_manager: Client session manager
        state: Task state

    Returns:
        Tuple of (submission_score, explanation) where score is 0.0 to max_score
    """
    strategy = criteria.strategy
    if strategy == SubmissionEvaluationStrategy.STATIC:
        return await _score_submission_static(submission_data, criteria)
    elif strategy == SubmissionEvaluationStrategy.LLM_JUDGE:
        return await _score_submission_llm(submission_data, criteria, session_manager, state)
    else:
        raise RuntimeError(f"Unknown submission strategy: {strategy}")


async def _score_submission_static(
    submission_data: EpisodeSubmissionResponse, criteria: SubmissionEvaluationCriteriaResponse
) -> tuple[float, str]:
    """
    Static submission scoring (pattern matching).

    Args:
        submission_data: Episode submission data
        criteria: Submission evaluation criteria

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


async def _score_submission_llm(
    submission_data: EpisodeSubmissionResponse,
    criteria: SubmissionEvaluationCriteriaResponse,
    session_manager: Any,
    state: TaskState,
) -> tuple[float, str]:
    """
    LLM submission scoring with client-side template rendering.

    Args:
        submission_data: Episode submission data
        criteria: Submission evaluation criteria
        session_manager: Client session manager
        state: Task state

    Returns:
        Tuple of (score, explanation) where score is 0.0 or max_score based on LLM judgment
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
        explanation = f"submission=0.0 Expected: {golden_answer}" if golden_answer else "submission=0.0"
        return 0.0, explanation
    elif "CORRECT" in judge_response:
        logger.info("LLM judge: CORRECT", extra={"score": max_score, "event": "llm_submission_correct"})
        explanation = (
            f"submission={max_score} Expected: {golden_answer}" if golden_answer else f"submission={max_score}"
        )
        return max_score, explanation
    else:
        logger.info("LLM judge: UNCLEAR", extra={"score": 0.0, "event": "llm_submission_unclear"})
        explanation = (
            f"submission=0.0 (unclear) Expected: {golden_answer}" if golden_answer else "submission=0.0 (unclear)"
        )
        return 0.0, explanation


# ============================================================================
# Step Evaluation
# ============================================================================


async def _score_all_subtasks(
    steps_data: EpisodeStepsResponse,
    list_of_all_subtask_criteria: List[SubtaskEvaluationCriteriaResponse],
    task_context: Any,
    session_manager: Any,
    state: TaskState,
) -> Tuple[float, List[float], List[List[StepEvaluation]]]:
    """
    Score all subtasks using their configured evaluation strategies.

    Args:
        steps_data: Episode steps data
        list_of_all_subtask_criteria: List of subtask evaluation criteria
        task_context: Task context
        session_manager: Client session manager
        state: Task state

    Returns:
        Tuple of (total_subtask_score, individual_scores, all_step_evaluations)
    """
    all_scores = []
    total_subtask_score = 0.0
    all_step_evaluations: List[List[StepEvaluation]] = []

    logger.info(
        "Starting subtask evaluation",
        extra={
            "subtask_count": len(list_of_all_subtask_criteria),
            "subtasks": [
                {"id": c.subtask_id, "strategy": c.strategy, "max_score": c.max_score, "weight": c.weight}
                for c in list_of_all_subtask_criteria
            ],
            "event": "subtask_eval_start",
        },
    )

    tasks = []

    for criteria in list_of_all_subtask_criteria:
        # Skip subtasks without a configured strategy (informational checkpoints only)
        if not criteria.strategy or criteria.strategy == "":
            logger.info(
                "Skipping subtask without evaluation strategy (informational checkpoint only)",
                extra={
                    "subtask_id": criteria.subtask_id,
                    "strategy": criteria.strategy,
                    "event": "subtask_skip_no_strategy",
                },
            )
            # Add a placeholder result with 0 score for this subtask
            tasks.append(_score_subtask_skip(steps_data, criteria))
            continue

        logger.info(
            "Evaluating subtask with strategy",
            extra={
                "subtask_id": criteria.subtask_id,
                "strategy": criteria.strategy,
                "max_score": criteria.max_score,
                "weight": criteria.weight,
                "event": "subtask_eval_strategy",
            },
        )
        if criteria.strategy == StepEvaluationStrategy.STATIC:
            tasks.append(_score_subtask_static(steps_data, criteria))
        elif criteria.strategy == StepEvaluationStrategy.TOOL_CALL:
            tasks.append(_score_subtask_tool_call(steps_data, criteria))
        elif criteria.strategy == StepEvaluationStrategy.LLM_JUDGE:
            tasks.append(
                _score_subtask_llm(
                    steps_data, criteria, task_context, session_manager, state, list_of_all_subtask_criteria
                )
            )
        else:
            raise RuntimeError(f"Unknown subtask strategy: {criteria.strategy}")

    results = await asyncio.gather(*tasks)
    for i, (subtask_score, step_evals) in enumerate(results):
        criteria = list_of_all_subtask_criteria[i]
        total_subtask_score += subtask_score
        all_scores.append(subtask_score)
        all_step_evaluations.append(step_evals)

        logger.debug(
            "Subtask scored",
            extra={
                "subtask_id": criteria.subtask_id,
                "strategy": criteria.strategy,
                "score": subtask_score,
                "max_score": criteria.max_score,
                "event": "subtask_scored",
            },
        )

    logger.info(
        "All subtasks scored",
        extra={
            "total_score": total_subtask_score,
            "subtask_count": len(list_of_all_subtask_criteria),
            "step_evaluations": len(all_step_evaluations[0]) if all_step_evaluations else 0,
            "event": "all_subtasks_scored",
        },
    )

    return total_subtask_score, all_scores, all_step_evaluations


async def _score_subtask_skip(
    steps_data: EpisodeStepsResponse,
    criteria: SubtaskEvaluationCriteriaResponse,
) -> Tuple[float, List[StepEvaluation]]:
    """
    Skip scoring for subtasks without step_evaluation_config (informational checkpoints only).

    Args:
        steps_data: Episode steps data
        criteria: Subtask criteria (without strategy)

    Returns:
        Tuple of (0.0, empty_list) - no score for informational subtasks
    """
    logger.info(
        "Subtask skipped - no evaluation strategy configured (informational only)",
        extra={
            "subtask_id": criteria.subtask_id,
            "event": "subtask_skip_complete",
        },
    )
    return 0.0, []


async def _score_subtask_static(
    steps_data: EpisodeStepsResponse,
    criteria: SubtaskEvaluationCriteriaResponse,
) -> Tuple[float, List[StepEvaluation]]:
    """
    Score steps using static evaluation (pattern matching in outputs).

    Args:
        steps_data: Episode steps data
        criteria: Step evaluation criteria

    Returns:
        Tuple of (subtask_score, list_of_step_evaluations)
        where subtask_score is max_score if any step matches criteria, 0.0 otherwise
    """
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
                    "Static output match found with step tool output: %s at step %d",
                    step.tool_output,
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
        "Static subtask evaluation complete",
        extra={
            "subtask_id": criteria.subtask_id,
            "subtask_score": subtask_score,
            "found_match": found_match,
            "steps_evaluated": len(all_graded_steps),
            "event": "static_subtask_complete",
        },
    )

    return subtask_score, all_graded_steps


async def _score_subtask_tool_call(
    steps_data: EpisodeStepsResponse,
    criteria: SubtaskEvaluationCriteriaResponse,
) -> Tuple[float, List[StepEvaluation]]:
    """
    Score steps using tool call evaluation (matching tool names).

    Args:
        steps_data: Episode steps data
        criteria: Step evaluation criteria

    Returns:
        Tuple of (subtask_score, list_of_step_evaluations)
        where subtask_score is max_score if any step uses expected tools, 0.0 otherwise
    """
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
        "Tool call subtask evaluation complete",
        extra={
            "subtask_id": criteria.subtask_id,
            "subtask_score": subtask_score,
            "found_match": found_match,
            "steps_evaluated": len(all_graded_steps),
            "event": "tool_call_subtask_complete",
        },
    )

    return subtask_score, all_graded_steps


async def _score_subtask_llm(
    steps_data: EpisodeStepsResponse,
    criteria: SubtaskEvaluationCriteriaResponse,
    task_context: Any,
    session_manager: Any,
    state: TaskState,
    all_subtask_criteria: Optional[List[SubtaskEvaluationCriteriaResponse]] = None,
) -> Tuple[float, List[StepEvaluation]]:
    """
    Score steps using LLM evaluation.

    Args:
        steps_data: Episode steps data
        criteria: Step evaluation criteria
        task_context: Task context
        session_manager: Client session manager
        state: Task state
        all_subtask_criteria: Optional list of all subtask criteria for context

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

        # Build simplified context for subtask evaluation
        # Use information directly from criteria instead of searching task_context
        # Build subtasks list from all_subtask_criteria if available
        subtasks_for_template = []
        if all_subtask_criteria:
            for subtask in all_subtask_criteria:
                subtasks_for_template.append(
                    {
                        "subtask_id": subtask.subtask_id,
                        "title": subtask.title,
                        "description": subtask.description,
                        "objective": subtask.objective,
                    }
                )
        elif hasattr(task_context, "subtasks"):
            subtasks_for_template = task_context.subtasks

        context = {
            "question": task_context.description,
            "episode": episode,  # Steps to evaluate
            # current subtask being evaluated
            "subtask": {
                "id": criteria.subtask_id,
                "description": criteria.description,
                "objective": criteria.objective,
                "title": criteria.title,
            },
            "model": criteria.criteria.get("model", ""),
            "domain": task_context.domain if hasattr(task_context, "domain") else None,
            "task_id": criteria.task_id,
            # Add task object with subtasks for template compatibility
            "task": {
                "task_id": task_context.task_id,
                "title": task_context.title,
                "description": task_context.description,
                "domain": task_context.domain if hasattr(task_context, "domain") else None,
                "subtasks": subtasks_for_template,
            },
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

        # Parse LLM response for simple completion judgment
        judge_response = state.output.completion.upper()

        logger.debug(
            "LLM response for chunk",
            extra={"chunk": chunk_idx, "response_preview": judge_response[:100], "event": "llm_chunk_response"},
        )

    # After processing all chunks, determine if subtask was completed
    # Combine all LLM responses to make final judgment
    # For now, use the last chunk's response (could be enhanced to consider all chunks)
    completed = False
    if "NO_COMPLETIONS" in judge_response:
        completed = False
    else:
        completed = True

    # Create step evaluations for all steps
    # Mark all steps with the same completion status based on LLM judgment
    for step in steps_data.steps:
        all_step_evaluations.append(
            StepEvaluation(
                step_number=step.step_number,
                objective_id=criteria.subtask_id,
                objective_type="subtask",
                completed=completed,
            )
        )

    # Calculate final score
    total_score = criteria.max_score if completed else 0.0

    logger.info(
        "Subtask LLM evaluation completed.\nContext: %s\nSystem prompt: %s\nUser prompt: %s\nJudge response: %s",
        context,
        system_message,
        user_message,
        judge_response,
        extra={
            "subtask_id": criteria.subtask_id,
            "total_score": total_score,
            "max_score": criteria.max_score,
            "completed": completed,
            "chunks_processed": len(step_chunks),
            "step_evaluations_created": len(all_step_evaluations),
            "event": "subtask_llm_eval_complete",
            "context": context,
            "system_prompt": system_message,
            "user_prompt": user_message,
            "judge_response": judge_response,
        },
    )

    return total_score, all_step_evaluations
