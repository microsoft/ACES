"""
SABER Dataset Conversion for inspect_ai

Converts SABER tasks from the server into inspect_ai Sample format for
eval_async integration. This maintains SABER's task structure while
providing clean inspect_ai dataset integration.

Logging category: HARNESS.

Key Principles:
- Sample input = task description (NOT agent prompt - that's set when creating the agent)
- Episode attempts = multiple samples (one per attempt for fresh starts)
- Subtask count is NOT shared with agents (internal SABER orchestration)
- Full SABER context preserved in metadata for proper evaluation

Following SABER's philosophy:
- Fail fast when SABER server is unavailable or returns invalid data
- Clean conversion between SABER task format and inspect_ai Sample format
- Explicit error handling for network and data format issues
- No silent data loss during conversion
"""

from typing import List

from inspect_ai.dataset import Sample

from ...logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ...models import TaskInfo

logger = get_saber_logger(LogCategory.HARNESS, __name__)


async def create_saber_dataset(tasks_data: List[TaskInfo]) -> List[Sample]:
    """
    Convert SABER task data to inspect_ai dataset samples.

    Each SABER task becomes one or more inspect_ai Samples based on episode_attempts:
    - input: Task description (NOT the agent prompt - that's set when creating the agent)
    - target: Success criteria for evaluation
    - metadata: Full SABER task definition for solver access
    - Multiple samples: One sample per episode attempt for fresh starts

    Args:
        tasks_data: List of TaskInfo objects from session manager

    Returns:
        List of inspect_ai Sample objects (multiple per task if episode_attempts > 1)

    Raises:
        ValueError: If task data is invalid
        RuntimeError: If task conversion fails
    """

    if not tasks_data:
        error = ValueError("tasks_data cannot be empty")
        log_operation_failure(logger, "inspect_dataset_conversion", error)
        raise error

    log_operation_start(
        logger,
        "inspect_dataset_conversion",
        task_count=len(tasks_data),
    )

    # Convert each SABER task to one or more inspect_ai Samples
    samples = []
    conversion_errors = []

    for task_info in tasks_data:
        try:
            # Create multiple samples for episode attempts (fresh starts)
            for attempt in range(1, task_info.episode_attempts + 1):
                sample = _convert_saber_task_to_sample(task_info, attempt)
                samples.append(sample)
                logger.debug(
                    "Task attempt converted",
                    extra={
                        "event": "inspect_task_attempt_converted",
                        "task_id": task_info.task_id,
                        "attempt": attempt,
                    },
                )

        except Exception as exc:
            conversion_errors.append(f"Task {task_info.task_id}: {exc}")
            logger.error(
                "Task conversion failed",
                extra={
                    "event": "inspect_task_conversion_failed",
                    "task_id": task_info.task_id,
                    "error": str(exc),
                },
            )

    # Fail fast if any conversions failed
    if conversion_errors:
        preview_failures = conversion_errors[:3]
        additional_failures = max(len(conversion_errors) - 3, 0)
        failure_message = (
            "Failed to convert {count} SABER tasks: {preview}".format(
                count=len(conversion_errors),
                preview=preview_failures,
            )
            if additional_failures == 0
            else "Failed to convert {count} SABER tasks: {preview} (and {extra} more)".format(
                count=len(conversion_errors),
                preview=preview_failures,
                extra=additional_failures,
            )
        )
        conversion_error = RuntimeError(failure_message)
        log_operation_failure(
            logger,
            "inspect_dataset_conversion",
            conversion_error,
            event="inspect_dataset_conversion_failed",
            failed_task_count=len(conversion_errors),
            sample_failures=preview_failures,
            additional_failures=additional_failures,
        )
        raise conversion_error

    log_operation_success(
        logger,
        "inspect_dataset_conversion",
        task_count=len(tasks_data),
        sample_count=len(samples),
    )
    return samples


def _convert_saber_task_to_sample(task_info: TaskInfo, attempt: int = 1) -> Sample:
    """
    Convert a single SABER task to an inspect_ai Sample.

    Args:
        task_info: SABER TaskInfo object
        attempt: Which episode attempt this sample represents

    Returns:
        inspect_ai Sample object

    Raises:
        ValueError: If task data is invalid
    """

    if not task_info:
        raise ValueError("task_info cannot be None")

    if attempt < 1:
        raise ValueError("attempt must be >= 1")

    # Create inspect_ai input from SABER task (this is NOT the agent prompt)
    task_input = _create_task_input(task_info)

    # Create inspect_ai target from SABER success criteria
    task_target = _create_task_target(task_info)

    # Package full SABER task in metadata for solver access
    task_metadata = {
        "saber_task": task_info.model_dump(),
        "task_id": task_info.task_id,
        # NEW: Three distinct prompts (replaces initial_prompt)
        "instruction_prompt": task_info.instruction_prompt,
        "assistant_prompt": task_info.assistant_prompt,
        "submit_prompt": task_info.submit_prompt,
        "attempt": attempt,
        "total_attempts": task_info.episode_attempts,
        "tool_call_limit": task_info.max_steps,
    }

    logger.debug(
        "Task metadata prepared",
        extra={
            "event": "inspect_task_metadata_prepared",
            "task_id": task_info.task_id,
            "attempt": attempt,
            "total_attempts": task_info.episode_attempts,
        },
    )

    # Create unique sample ID for each attempt
    # Use double underscore to avoid collision with task IDs containing single underscores
    sample_id = f"{task_info.task_id}__attempt_{attempt}"
    task_metadata["sample_id"] = sample_id

    # Create inspect_ai Sample
    sample = Sample(id=sample_id, input=task_input, target=task_target, metadata=task_metadata)

    return sample


def _create_task_input(task_info: TaskInfo) -> str:
    """
    Create inspect_ai input string from SABER TaskInfo object.

    Returns a concise task summary for the initial user message.
    The full instruction prompt is passed to the agent via metadata
    and becomes part of the system message through AgentPrompt.instructions.

    This prevents duplication - the instruction prompt should NOT appear
    as both a user message AND a system message.

    Args:
        task_info: SABER TaskInfo object

    Returns:
        Concise task summary (title + description)
    """
    # Return a simple task summary for the user message
    # The full instruction_prompt is already in metadata and will be used
    # in AgentPrompt.instructions (system message), so we don't duplicate it here
    return f"Title: {task_info.title}\nTask: {task_info.description}"


def _create_task_target(task_info: TaskInfo) -> str:
    """
    Create inspect_ai target string from SABER TaskInfo.

    This provides a simple success criteria for evaluation.

    Args:
        task_info: SABER TaskInfo object

    Returns:
        Target string for evaluation
    """

    # For now, use a simple default target
    # The actual success criteria will be handled by SABER's evaluation system
    return f"Successfully complete the task: {task_info.title or task_info.description}"
