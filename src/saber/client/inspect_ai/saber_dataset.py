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

from typing import List, Union

from inspect_ai.dataset import MemoryDataset, Sample

from ...logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ...models import (
    BenchmarkTask,
    MetadataKeys,
    OrchestratedTask,
    OrchestrationStrategy,
    SingleEpisodeTask,
    SubTaskDefinition,
    TaskExecutionMode,
)

logger = get_saber_logger(LogCategory.HARNESS, __name__)


class SABERDataset(MemoryDataset):
    """SABER-specific dataset that preserves orchestration boundaries when slicing.

    This dataset subclass ensures that when Inspect AI applies --limit or other
    slicing operations, complete orchestrations are preserved. Orchestrated tasks
    create multiple samples that must execute together as a unit - breaking them
    apart would cause coordination failures.

    Key behavior:
    - When slicing would cut an orchestration, automatically expands to include
      all samples from that orchestration
    - Only affects samples with 'orchestration_id' in metadata
    - Non-orchestrated samples are sliced normally
    """

    def __getitem__(self, index: Union[int, slice]) -> Union[Sample, "MemoryDataset"]:
        """Override slicing to preserve orchestration boundaries."""
        if isinstance(index, int):
            # Single sample access - use parent implementation
            return super().__getitem__(index)

        # Slice operation - check for orchestration boundaries
        expanded_slice = self._expand_slice_for_orchestrations(index)

        # Log if we expanded the slice
        if expanded_slice != index:
            original_count = len(range(*index.indices(len(self.samples))))
            expanded_count = len(range(*expanded_slice.indices(len(self.samples))))
            logger.info(
                f"Expanded slice to preserve orchestrations: {original_count} → {expanded_count} samples",
                extra={
                    "original_slice": f"{index.start}:{index.stop}:{index.step}",
                    "expanded_slice": f"{expanded_slice.start}:{expanded_slice.stop}:{expanded_slice.step}",
                    "original_count": original_count,
                    "expanded_count": expanded_count,
                },
            )

        # Return MemoryDataset (not SABERDataset) to avoid deep recursion issues
        # The expanded slice already includes complete orchestrations
        return MemoryDataset(
            samples=self.samples[expanded_slice],
            name=self.name,
            location=self.location,
            shuffled=self.shuffled,
        )

    def _expand_slice_for_orchestrations(self, s: slice) -> slice:
        """Expand slice to include complete orchestrations.

        Algorithm:
        1. Identify orchestration_ids in the requested slice range
        2. Scan forward from slice.stop to find remaining samples from those orchestrations
        3. Return expanded slice that includes all samples

        Args:
            s: Original slice from __getitem__

        Returns:
            Expanded slice that includes complete orchestrations
        """
        start, stop, step = s.indices(len(self.samples))

        # Only handle simple forward slices
        if step != 1:
            logger.warning(
                f"Complex slice with step={step} cannot be expanded for orchestrations",
                extra={"slice": f"{start}:{stop}:{step}"},
            )
            return s

        # Find all orchestration_ids in the slice range
        orchestrations_in_slice = set()
        for i in range(start, stop):
            sample_metadata = self.samples[i].metadata
            if sample_metadata:
                orch_id = sample_metadata.get(MetadataKeys.ORCHESTRATION_ID)
                if orch_id:
                    orchestrations_in_slice.add(orch_id)

        # If no orchestrations found, return original slice
        if not orchestrations_in_slice:
            return s

        # Scan forward to find all samples from these orchestrations
        new_stop = stop
        for i in range(stop, len(self.samples)):
            sample_metadata = self.samples[i].metadata
            orch_id = sample_metadata.get(MetadataKeys.ORCHESTRATION_ID) if sample_metadata else None
            if orch_id in orchestrations_in_slice:
                # This sample belongs to an orchestration we're including
                new_stop = i + 1
            elif orch_id:
                # Different orchestration - we can stop here
                break
            # else: non-orchestrated sample, keep scanning

        return slice(start, new_stop, step)


async def create_saber_dataset(tasks_data: List[BenchmarkTask]) -> SABERDataset:
    """
    Convert SABER task data to a SABERDataset.

    Each SABER task becomes one or more inspect_ai Samples based on episode_attempts:
    - input: Task description (NOT agent prompt - that's set when creating the agent)
    - target: Success criteria for evaluation
    - metadata: Full SABER task definition for solver access
    - Multiple samples: One sample per episode attempt for fresh starts

    Supports both legacy TaskInfo and new polymorphic BenchmarkTask types:
    - SingleEpisodeTask: Traditional one-episode-per-task behavior
    - OrchestratedTask: Multi-episode coordinated execution

    The returned SABERDataset automatically preserves orchestration boundaries when
    slicing operations (like --limit) are applied. This ensures orchestrated tasks
    are never broken apart mid-execution.

    Args:
        tasks_data: List of TaskInfo or BenchmarkTask objects from session manager

    Returns:
        SABERDataset containing samples with orchestration-aware slicing

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

    for task_data in tasks_data:
        try:
            # Determine episode attempts based on task type
            if isinstance(task_data, BenchmarkTask):
                episode_attempts = task_data.episode_attempts
                task_id = task_data.benchmark_task_id

            # Check if this is an orchestrated task (multi-sample approach)
            if isinstance(task_data, OrchestratedTask):
                # Create one sample per sub-task per attempt
                for attempt in range(1, episode_attempts + 1):
                    for sub_task in task_data.sub_tasks:
                        sample = _convert_sub_task_to_sample(task_data, sub_task, attempt)
                        samples.append(sample)
                        logger.debug(
                            "Orchestrated sub-task converted",
                            extra={
                                "event": "inspect_orchestrated_sub_task_converted",
                                "orchestration_id": task_id,
                                "sub_task_role": sub_task.role,
                                "attempt": attempt,
                            },
                        )
            else:
                # Single episode or legacy - one sample per attempt
                for attempt in range(1, episode_attempts + 1):
                    sample = _convert_task_to_sample(task_data, attempt)
                    samples.append(sample)
                    logger.debug(
                        "Task attempt converted",
                        extra={
                            "event": "inspect_task_attempt_converted",
                            "task_id": task_id,
                            "attempt": attempt,
                        },
                    )

        except Exception as exc:
            # Extract task ID for error message
            if isinstance(task_data, BenchmarkTask):
                error_task_id = task_data.benchmark_task_id

            conversion_errors.append(f"Task {error_task_id}: {exc}")
            logger.error(
                "Task conversion failed",
                extra={
                    "event": "inspect_task_conversion_failed",
                    "task_id": error_task_id,
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

    # Sort samples to keep orchestrations together
    # This ensures that when --limit is applied, complete orchestrations are more likely to be included
    # Orchestrated samples are sorted by (orchestration_id, order) to keep them sequential
    # Non-orchestrated samples maintain their original order
    samples.sort(
        key=lambda s: (
            s.metadata.get(MetadataKeys.ORCHESTRATION_ID, s.id) if s.metadata else s.id,  # Group by orchestration
            s.metadata.get(MetadataKeys.ORDER, 0) if s.metadata else 0,  # Then by execution order within orchestration
            s.id,  # Finally by sample ID for stability
        )
    )

    # Log detailed sample information for debugging orchestrated tasks
    logger.info(
        f"Dataset conversion complete: {len(samples)} samples created",
        extra={
            "event": "dataset_conversion_complete",
            "sample_count": len(samples),
            "sample_ids": [s.id for s in samples],
        },
    )

    # Return SABERDataset with orchestration-aware slicing
    return SABERDataset(
        samples=samples,
        name="saber_dataset",
        location=None,
        shuffled=False,
    )


def _convert_task_to_sample(task_data: BenchmarkTask, attempt: int = 1) -> Sample:
    """
    Convert a SABER BenchmarkTask to an inspect_ai Sample.

    Handles polymorphic BenchmarkTask types (SingleEpisodeTask, OrchestratedTask).

    NOTE: For OrchestratedTask, this is a fallback that converts the entire orchestration
    into a single sample. Normal flow uses _convert_sub_task_to_sample to create one sample
    per sub-task.

    Args:
        task_data: SABER BenchmarkTask object
        attempt: Which episode attempt this sample represents

    Returns:
        inspect_ai Sample object

    Raises:
        ValueError: If task data is invalid
    """
    if isinstance(task_data, SingleEpisodeTask):
        return _convert_benchmark_task_to_sample(task_data, attempt)
    elif isinstance(task_data, OrchestratedTask):
        # For orchestrated tasks, create a single sample representing the entire orchestration
        # This is used when the orchestration itself is the unit of work (rare case)
        # Normal flow creates one sample per sub-task via _convert_sub_task_to_sample
        sample_id = f"{task_data.benchmark_task_id}__attempt_{attempt}"

        # Merge all sub-task descriptions with role labels
        input_parts = [f"Title: {task_data.benchmark_task_id}"]
        for sub_task in task_data.sub_tasks:
            input_parts.append(f"[{sub_task.role}] {sub_task.title}: {sub_task.description}")
        task_input = "\n".join(input_parts)

        task_target = "Successfully complete orchestrated task"

        # Use first sub-task's prompts as defaults
        first_sub_task = task_data.sub_tasks[0] if task_data.sub_tasks else None

        task_metadata = {
            MetadataKeys.BENCHMARK_TASK: task_data.model_dump(),
            MetadataKeys.EXECUTION_MODE: TaskExecutionMode.ORCHESTRATED.value,
            MetadataKeys.ORCHESTRATION_ID: task_data.benchmark_task_id,
            MetadataKeys.ORCHESTRATION_STRATEGY: (
                task_data.orchestration_strategy.value
                if isinstance(task_data.orchestration_strategy, OrchestrationStrategy)
                else task_data.orchestration_strategy
            ),
            MetadataKeys.INSTRUCTION_PROMPT: first_sub_task.instruction_prompt if first_sub_task else "",
            "assistant_prompt": first_sub_task.assistant_prompt if first_sub_task else "",
            "submit_prompt": first_sub_task.submit_prompt if first_sub_task else "",
            MetadataKeys.ATTEMPT: attempt,
            MetadataKeys.TOTAL_ATTEMPTS: task_data.episode_attempts,
            MetadataKeys.TOOL_CALL_LIMIT: first_sub_task.max_steps if first_sub_task else 30,
            MetadataKeys.SAMPLE_ID: sample_id,
        }

        return Sample(id=sample_id, input=task_input, target=task_target, metadata=task_metadata)
    else:
        raise ValueError(f"Unknown BenchmarkTask type: {type(task_data)}")


def _convert_benchmark_task_to_sample(benchmark_task: BenchmarkTask, attempt: int = 1) -> Sample:
    """
    Convert a BenchmarkTask to an inspect_ai Sample.

    NOTE: This only handles SingleEpisodeTask. OrchestratedTask samples are created
    by _convert_sub_task_to_sample (one sample per sub-task).

    Args:
        benchmark_task: SingleEpisodeTask instance
        attempt: Which episode attempt this sample represents

    Returns:
        inspect_ai Sample object

    Raises:
        ValueError: If task data is invalid or if called with OrchestratedTask
    """
    if not benchmark_task:
        raise ValueError("benchmark_task cannot be None")

    if attempt < 1:
        raise ValueError("attempt must be >= 1")

    # This function should only be called for SingleEpisodeTask
    # OrchestratedTask samples are created by _convert_sub_task_to_sample
    if isinstance(benchmark_task, OrchestratedTask):
        raise ValueError("OrchestratedTask should be converted using _convert_sub_task_to_sample, not this function")

    if not isinstance(benchmark_task, SingleEpisodeTask):
        raise ValueError(f"Unknown BenchmarkTask type: {type(benchmark_task)}")

    # Single episode - use task description
    task_input = f"Title: {benchmark_task.title}\nTask: {benchmark_task.description}"
    task_target = f"Successfully complete the task: {benchmark_task.title or benchmark_task.description}"

    task_metadata = {
        MetadataKeys.BENCHMARK_TASK: benchmark_task.model_dump(),
        MetadataKeys.EXECUTION_MODE: TaskExecutionMode.SINGLE.value,
        MetadataKeys.TASK_ID: benchmark_task.task_id,
        MetadataKeys.INSTRUCTION_PROMPT: benchmark_task.instruction_prompt,
        "assistant_prompt": benchmark_task.assistant_prompt,
        "submit_prompt": benchmark_task.submit_prompt,
        MetadataKeys.ATTEMPT: attempt,
        MetadataKeys.TOTAL_ATTEMPTS: benchmark_task.episode_attempts,
        MetadataKeys.TOOL_CALL_LIMIT: benchmark_task.max_steps,
    }

    sample_id = f"{benchmark_task.task_id}__attempt_{attempt}"

    task_metadata[MetadataKeys.SAMPLE_ID] = sample_id

    logger.debug(
        "Benchmark task metadata prepared",
        extra={
            "event": "inspect_benchmark_task_metadata_prepared",
            "benchmark_task_id": benchmark_task.benchmark_task_id,
            "execution_mode": task_metadata[MetadataKeys.EXECUTION_MODE],
            "attempt": attempt,
        },
    )

    # Create inspect_ai Sample
    sample = Sample(id=sample_id, input=task_input, target=task_target, metadata=task_metadata)

    return sample


def _convert_sub_task_to_sample(
    orchestrated_task: OrchestratedTask,
    sub_task: SubTaskDefinition,  # SubTaskDefinition from orchestrated_task.sub_tasks
    attempt: int = 1,
) -> Sample:
    """
    Convert a sub-task from an OrchestratedTask to an inspect_ai Sample.

    This creates one sample per sub-task, allowing each to execute independently
    while coordinating through the OrchestrationCoordinator.

    Args:
        orchestrated_task: Parent OrchestratedTask containing this sub-task
        sub_task: SubTaskDefinition for this specific sub-task
        attempt: Which episode attempt this sample represents

    Returns:
        inspect_ai Sample object configured for orchestrated execution
    """
    # Create sample ID: task_id_attempt_N (e.g., saber_dual_blue_team_attempt_1)
    sample_id = f"{sub_task.task_id}_attempt_{attempt}"

    # Input is the sub-task description with role prefix
    task_input = f"[{sub_task.role}] {sub_task.title}: {sub_task.description}"

    # Target is success for this sub-task
    task_target = f"Successfully complete {sub_task.role} task: {sub_task.title}"

    # Metadata includes orchestration coordination info
    task_metadata = {
        MetadataKeys.BENCHMARK_TASK: orchestrated_task.model_dump(),
        MetadataKeys.EXECUTION_MODE: "orchestrated_sub_task",  # NEW mode for multi-sample orchestration
        MetadataKeys.ORCHESTRATION_ID: orchestrated_task.benchmark_task_id,
        MetadataKeys.SUB_TASK_ROLE: sub_task.role,
        MetadataKeys.TASK_ID: sub_task.task_id,
        MetadataKeys.DEPENDS_ON_ROLE: sub_task.depends_on_role,
        MetadataKeys.ORDER: sub_task.order,
        # Use this sub-task's prompts
        MetadataKeys.INSTRUCTION_PROMPT: sub_task.instruction_prompt,
        "assistant_prompt": sub_task.assistant_prompt,
        "submit_prompt": sub_task.submit_prompt,
        MetadataKeys.ATTEMPT: attempt,
        MetadataKeys.TOTAL_ATTEMPTS: orchestrated_task.episode_attempts,
        MetadataKeys.TOOL_CALL_LIMIT: sub_task.max_steps,
        MetadataKeys.SAMPLE_ID: sample_id,
    }

    # Add blocking_config if present in sub-task
    if sub_task.blocking_config:
        task_metadata["blocking_config"] = sub_task.blocking_config

    logger.debug(
        "Orchestrated sub-task sample created",
        extra={
            "event": "inspect_orchestrated_sub_task_sample_created",
            "orchestration_id": orchestrated_task.benchmark_task_id,
            "sub_task_role": sub_task.role,
            "task_id": sub_task.task_id,
            "depends_on_role": sub_task.depends_on_role,
            "sample_id": sample_id,
        },
    )

    return Sample(id=sample_id, input=task_input, target=task_target, metadata=task_metadata)
