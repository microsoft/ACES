"""
SABER Dataset Conversion for inspect_ai

Converts SABER tasks from the server into inspect_ai Sample format for
eval_async integration. This maintains SABER's task structure while
providing clean inspect_ai dataset integration.

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

import logging
from typing import List

from inspect_ai.dataset import Sample

from ...models import TaskInfo

logger = logging.getLogger(__name__)


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
        raise ValueError("tasks_data cannot be empty")

    logger.info(f"Converting {len(tasks_data)} SABER tasks to inspect_ai samples")

    # Convert each SABER task to one or more inspect_ai Samples
    samples = []
    conversion_errors = []

    for task_info in tasks_data:
        try:
            # Create multiple samples for episode attempts (fresh starts)
            for attempt in range(1, task_info.episode_attempts + 1):
                sample = _convert_saber_task_to_sample(task_info, attempt)
                samples.append(sample)
                logger.debug(f"Converted task {task_info.task_id} attempt {attempt}")

        except Exception as e:
            conversion_errors.append(f"Task {task_info.task_id}: {e}")
            logger.warning(f"Failed to convert task {task_info.task_id}: {e}")

    # Fail fast if any conversions failed
    if conversion_errors:
        error_summary = f"Failed to convert {len(conversion_errors)} tasks: {conversion_errors[:3]}"
        if len(conversion_errors) > 3:
            error_summary += f" (and {len(conversion_errors) - 3} more)"
        raise RuntimeError(error_summary)

    logger.info(f"Successfully converted {len(samples)} SABER task attempts to inspect_ai samples")
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

    logger.info(f"Task metadata for task {task_info.task_id}: {task_metadata}")

    # Create unique sample ID for each attempt
    sample_id = f"{task_info.task_id}_attempt_{attempt}"

    # Create inspect_ai Sample
    sample = Sample(id=sample_id, input=task_input, target=task_target, metadata=task_metadata)

    return sample


def _create_task_input(task_info: TaskInfo) -> str:
    """
    Create inspect_ai input string from SABER TaskInfo object.

    This is the task description that will be provided to the agent as context
    (NOT the agent prompt - that's set when creating the agent).

    Args:
        task_info: SABER TaskInfo object

    Returns:
        Formatted task input string for the agent to understand the task
    """

    input_parts = [
        f"Task: {task_info.description}",
    ]

    # Add title if different from description
    if task_info.title and task_info.title != task_info.description:
        input_parts.insert(0, f"Title: {task_info.title}")

    return "\n".join(input_parts)


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
