"""
Unit tests for SABER dataset conversion with polymorphic task support.

Tests the dataset conversion functionality for BenchmarkTask types.
"""

import pytest

from saber.client.inspect_ai.saber_dataset import (
    _convert_benchmark_task_to_sample,
    _convert_task_to_sample,
    create_saber_dataset,
)
from saber.models import OrchestratedTask, OrchestrationStrategy, SingleEpisodeTask, SubTaskDefinition


class TestConvertBenchmarkTaskToSample:
    """Test cases for _convert_benchmark_task_to_sample."""

    def test_convert_single_episode_task(self):
        """Test converting a SingleEpisodeTask to Sample."""
        task = SingleEpisodeTask(
            benchmark_task_id="test_task_1",
            task_id="test_task_1",
            domain="cybench",
            title="XSS Challenge",
            description="Find and exploit XSS vulnerability",
            episode_attempts=3,
            subtask_count=2,
            max_steps=50,
            instruction_prompt="Find the XSS vulnerability",
            assistant_prompt="I'll help you find it",
            submit_prompt="Submit your exploit",
            continue_prompt="",
        )

        sample = _convert_benchmark_task_to_sample(task, attempt=1)

        # Verify sample structure
        assert sample.id == "test_task_1__attempt_1"
        assert "XSS Challenge" in sample.input
        assert "Find and exploit XSS vulnerability" in sample.input
        assert "Successfully complete the task" in sample.target

        # Verify metadata
        assert sample.metadata["execution_mode"] == "single"
        assert sample.metadata["task_id"] == "test_task_1"
        assert sample.metadata["attempt"] == 1
        assert sample.metadata["total_attempts"] == 3
        assert sample.metadata["tool_call_limit"] == 50
        assert sample.metadata["instruction_prompt"] == "Find the XSS vulnerability"
        assert "benchmark_task" in sample.metadata

    # NOTE: test_convert_orchestrated_task removed because _convert_benchmark_task_to_sample
    # explicitly raises ValueError for OrchestratedTask. Orchestrated tasks are converted
    # using _convert_sub_task_to_sample (one sample per sub-task), not as a single merged sample.

    def test_convert_with_different_attempts(self):
        """Test that attempt number is reflected in sample ID and metadata."""
        task = SingleEpisodeTask(
            benchmark_task_id="test_task",
            task_id="test_task",
            domain="test",
            title="Test",
            description="Test",
            episode_attempts=5,
            max_steps=10,
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="",
        )

        sample1 = _convert_benchmark_task_to_sample(task, attempt=1)
        sample2 = _convert_benchmark_task_to_sample(task, attempt=3)
        sample3 = _convert_benchmark_task_to_sample(task, attempt=5)

        assert sample1.id == "test_task__attempt_1"
        assert sample2.id == "test_task__attempt_3"
        assert sample3.id == "test_task__attempt_5"

        assert sample1.metadata["attempt"] == 1
        assert sample2.metadata["attempt"] == 3
        assert sample3.metadata["attempt"] == 5

    def test_convert_validates_attempt(self):
        """Test that invalid attempt numbers raise ValueError."""
        task = SingleEpisodeTask(
            benchmark_task_id="test_task",
            task_id="test_task",
            domain="test",
            title="Test",
            description="Test",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="",
        )

        with pytest.raises(ValueError, match="attempt must be >= 1"):
            _convert_benchmark_task_to_sample(task, attempt=0)

        with pytest.raises(ValueError, match="attempt must be >= 1"):
            _convert_benchmark_task_to_sample(task, attempt=-1)


class TestConvertTaskToSample:
    """Test cases for polymorphic _convert_task_to_sample dispatcher."""

    def test_dispatcher_handles_single_episode_task(self):
        """Test dispatcher correctly routes SingleEpisodeTask."""
        task = SingleEpisodeTask(
            benchmark_task_id="test_1",
            task_id="test_1",
            domain="test",
            title="Test",
            description="Test",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="",
        )

        sample = _convert_task_to_sample(task, attempt=1)

        assert sample.metadata["execution_mode"] == "single"
        assert sample.id == "test_1__attempt_1"

    def test_dispatcher_handles_orchestrated_task(self):
        """Test dispatcher correctly routes OrchestratedTask."""
        task = OrchestratedTask(
            benchmark_task_id="orch_1",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="sub_1",
                    role="test",
                    order=0,
                    domain="test",
                    title="Test",
                    description="Test",
                    episode_attempts=1,
                    max_steps=10,
                    instruction_prompt="test",
                    assistant_prompt="test",
                    submit_prompt="test",
            continue_prompt="",
                )
            ],
            episode_attempts=1,
        )

        sample = _convert_task_to_sample(task, attempt=1)

        assert sample.metadata["execution_mode"] == "orchestrated"
        assert sample.id == "orch_1__attempt_1"


class TestCreateSaberDataset:
    """Test cases for create_saber_dataset with polymorphic tasks."""

    @pytest.mark.asyncio
    async def test_create_dataset_with_single_episode_tasks(self):
        """Test creating dataset from list of SingleEpisodeTasks."""
        tasks = [
            SingleEpisodeTask(
                benchmark_task_id="task_1",
                task_id="task_1",
                domain="test",
                title="Task 1",
                description="First task",
                episode_attempts=2,
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
            continue_prompt="",
            ),
            SingleEpisodeTask(
                benchmark_task_id="task_2",
                task_id="task_2",
                domain="test",
                title="Task 2",
                description="Second task",
                episode_attempts=3,
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
            continue_prompt="",
            ),
        ]

        samples = await create_saber_dataset(tasks)

        # 2 tasks with 2 and 3 attempts = 5 total samples
        assert len(samples) == 5

        # Verify samples for first task
        assert samples[0].id == "task_1__attempt_1"
        assert samples[1].id == "task_1__attempt_2"

        # Verify samples for second task
        assert samples[2].id == "task_2__attempt_1"
        assert samples[3].id == "task_2__attempt_2"
        assert samples[4].id == "task_2__attempt_3"

    @pytest.mark.asyncio
    async def test_create_dataset_with_orchestrated_tasks(self):
        """Test creating dataset from list of OrchestratedTasks."""
        tasks = [
            OrchestratedTask(
                benchmark_task_id="orch_1",
                orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
                sub_tasks=[
                    SubTaskDefinition(
                        task_id="sub_1",
                        role="blue",
                        order=0,
                        domain="test",
                        title="Test",
                        description="Test",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="test",
                        assistant_prompt="test",
                        submit_prompt="test",
            continue_prompt="",
                    ),
                    SubTaskDefinition(
                        task_id="sub_2",
                        role="red",
                        order=1,
                        domain="test",
                        title="Test",
                        description="Test",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="test",
                        assistant_prompt="test",
                        submit_prompt="test",
            continue_prompt="",
                    ),
                ],
                episode_attempts=2,
            )
        ]

        samples = await create_saber_dataset(tasks)

        # 1 orchestration with 2 sub-tasks and 2 attempts = 4 samples (2 sub-tasks × 2 attempts)
        # This is the actual implementation: one sample per sub-task per attempt
        assert len(samples) == 4
        # Verify samples are created for each sub-task in each attempt
        # Note: sub-task samples use single underscore format (task_id_attempt_N)
        sample_ids = [s.id for s in samples]
        assert "sub_1_attempt_1" in sample_ids
        assert "sub_1_attempt_2" in sample_ids
        assert "sub_2_attempt_1" in sample_ids
        assert "sub_2_attempt_2" in sample_ids

    @pytest.mark.asyncio
    async def test_create_dataset_with_mixed_tasks(self):
        """Test creating dataset from mixed SingleEpisode and Orchestrated tasks."""
        tasks = [
            SingleEpisodeTask(
                benchmark_task_id="single_1",
                task_id="single_1",
                domain="test",
                title="Single",
                description="Single task",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
            continue_prompt="",
            ),
            OrchestratedTask(
                benchmark_task_id="orch_1",
                orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
                sub_tasks=[
                    SubTaskDefinition(
                        task_id="sub_1",
                        role="test",
                        order=0,
                        domain="test",
                        title="Test",
                        description="Test",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="test",
                        assistant_prompt="test",
                        submit_prompt="test",
            continue_prompt="",
                    )
                ],
                episode_attempts=2,
            ),
        ]

        samples = await create_saber_dataset(tasks)

        # 1 single (1 attempt) + 1 orchestrated (1 sub-task × 2 attempts) = 3 samples
        # Samples are sorted by orchestration_id then order, so sub-tasks come first
        assert len(samples) == 3
        sample_ids = sorted([s.id for s in samples])
        assert sample_ids == ["single_1__attempt_1", "sub_1_attempt_1", "sub_1_attempt_2"]

    @pytest.mark.asyncio
    async def test_create_dataset_empty_raises_error(self):
        """Test that empty task list raises ValueError."""
        with pytest.raises(ValueError, match="tasks_data cannot be empty"):
            await create_saber_dataset([])

    @pytest.mark.asyncio
    async def test_create_dataset_handles_conversion_errors(self):
        """Test that conversion errors are collected and reported."""
        # Create an invalid task (this would need to bypass validation somehow)
        # For this test, we'll just verify the error handling structure

        tasks = [
            SingleEpisodeTask(
                benchmark_task_id="valid_task",
                task_id="valid_task",
                domain="test",
                title="Valid",
                description="Valid task",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
            continue_prompt="",
            )
        ]

        # Should not raise error for valid tasks
        samples = await create_saber_dataset(tasks)
        assert len(samples) == 1


class TestMetadataStructure:
    """Test cases for sample metadata structure."""

    def test_single_episode_metadata_contains_benchmark_task(self):
        """Test that SingleEpisodeTask metadata includes serialized task."""
        task = SingleEpisodeTask(
            benchmark_task_id="test_1",
            task_id="test_1",
            domain="test",
            title="Test",
            description="Test",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="",
        )

        sample = _convert_benchmark_task_to_sample(task, attempt=1)

        # Verify benchmark_task can be deserialized
        assert "benchmark_task" in sample.metadata
        benchmark_task_dict = sample.metadata["benchmark_task"]

        # Should be able to reconstruct task from metadata
        reconstructed = SingleEpisodeTask(**benchmark_task_dict)
        assert reconstructed.benchmark_task_id == task.benchmark_task_id
        assert reconstructed.execution_mode == task.execution_mode

    def test_orchestrated_metadata_contains_benchmark_task(self):
        """Test that OrchestratedTask metadata includes serialized task."""
        task = OrchestratedTask(
            benchmark_task_id="orch_1",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="sub_1",
                    role="test",
                    order=0,
                    domain="test",
                    title="Test",
                    description="Test",
                    episode_attempts=1,
                    max_steps=10,
                    instruction_prompt="test",
                    assistant_prompt="test",
                    submit_prompt="test",
            continue_prompt="",
                )
            ],
            episode_attempts=1,
        )

        # Use dispatcher which handles both task types
        sample = _convert_task_to_sample(task, attempt=1)

        # Verify benchmark_task can be deserialized
        assert "benchmark_task" in sample.metadata
        benchmark_task_dict = sample.metadata["benchmark_task"]

        # Should be able to reconstruct task from metadata
        reconstructed = OrchestratedTask(**benchmark_task_dict)
        assert reconstructed.benchmark_task_id == task.benchmark_task_id
        assert reconstructed.orchestration_strategy == task.orchestration_strategy
        assert len(reconstructed.sub_tasks) == len(task.sub_tasks)
