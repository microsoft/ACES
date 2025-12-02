"""Unit tests for SABERDataset and dataset conversion functions.

Tests focus on increasing coverage for:
- SABERDataset slicing with orchestration preservation
- create_saber_dataset conversion
- _convert_task_to_sample
- _convert_benchmark_task_to_sample
- _convert_sub_task_to_sample
- Error handling and validation
"""

import pytest
from unittest.mock import Mock, patch

from inspect_ai.dataset import Sample

from saber.inspect_ai.core.saber_dataset import (
    SABERDataset,
    create_saber_dataset,
    _convert_task_to_sample,
    _convert_benchmark_task_to_sample,
    _convert_sub_task_to_sample,
)
from saber.models import (
    BenchmarkTask,
    SingleEpisodeTask,
    OrchestratedTask,
    SubTaskDefinition,
    MetadataKeys,
    TaskExecutionMode,
    OrchestrationStrategy,
)


class TestSABERDataset:
    """Test SABERDataset slicing behavior."""

    def test_getitem_single_index(self):
        """Test accessing a single sample by index."""
        samples = [
            Sample(id="s1", input="task 1", target="target 1"),
            Sample(id="s2", input="task 2", target="target 2"),
        ]
        dataset = SABERDataset(samples=samples)

        result = dataset[0]
        assert isinstance(result, Sample)
        assert result.id == "s1"

    def test_getitem_slice_no_orchestrations(self):
        """Test slicing without orchestrations returns normal slice."""
        samples = [
            Sample(id=f"s{i}", input=f"task {i}", target=f"target {i}")
            for i in range(10)
        ]
        dataset = SABERDataset(samples=samples)

        result = dataset[2:5]
        # Returns MemoryDataset to avoid recursion issues
        assert len(result.samples) == 3
        assert result.samples[0].id == "s2"
        assert result.samples[2].id == "s4"

    def test_getitem_slice_with_orchestrations_no_expansion(self):
        """Test slicing that naturally includes complete orchestrations."""
        samples = [
            Sample(id="s1", input="task 1", target="target 1", metadata={MetadataKeys.ORCHESTRATION_ID: "orch1"}),
            Sample(id="s2", input="task 2", target="target 2", metadata={MetadataKeys.ORCHESTRATION_ID: "orch1"}),
            Sample(id="s3", input="task 3", target="target 3", metadata={MetadataKeys.ORCHESTRATION_ID: "orch2"}),
            Sample(id="s4", input="task 4", target="target 4", metadata={MetadataKeys.ORCHESTRATION_ID: "orch2"}),
        ]
        dataset = SABERDataset(samples=samples)

        # Slice that includes both samples from orch1
        result = dataset[0:2]
        assert len(result.samples) == 2
        assert result.samples[0].id == "s1"
        assert result.samples[1].id == "s2"

    def test_getitem_slice_expansion_for_orchestration(self):
        """Test that slicing expands to include complete orchestrations."""
        samples = [
            Sample(id="s1", input="task 1", target="target 1", metadata={MetadataKeys.ORCHESTRATION_ID: "orch1"}),
            Sample(id="s2", input="task 2", target="target 2", metadata={MetadataKeys.ORCHESTRATION_ID: "orch1"}),
            Sample(id="s3", input="task 3", target="target 3", metadata={MetadataKeys.ORCHESTRATION_ID: "orch1"}),
            Sample(id="s4", input="task 4", target="target 4"),  # Non-orchestrated
        ]
        dataset = SABERDataset(samples=samples)

        # Slice [0:2] would cut orch1, should expand to [0:3]
        result = dataset[0:2]
        assert len(result.samples) == 3
        assert result.samples[0].id == "s1"
        assert result.samples[2].id == "s3"

    def test_getitem_slice_stops_at_different_orchestration(self):
        """Test that expansion stops when hitting a different orchestration."""
        samples = [
            Sample(id="s1", input="task 1", target="target 1", metadata={MetadataKeys.ORCHESTRATION_ID: "orch1"}),
            Sample(id="s2", input="task 2", target="target 2", metadata={MetadataKeys.ORCHESTRATION_ID: "orch1"}),
            Sample(id="s3", input="task 3", target="target 3", metadata={MetadataKeys.ORCHESTRATION_ID: "orch2"}),
            Sample(id="s4", input="task 4", target="target 4", metadata={MetadataKeys.ORCHESTRATION_ID: "orch2"}),
        ]
        dataset = SABERDataset(samples=samples)

        # Slice [0:1] should expand to include orch1 but stop before orch2
        result = dataset[0:1]
        assert len(result.samples) == 2
        assert result.samples[0].id == "s1"
        assert result.samples[1].id == "s2"

    def test_getitem_slice_complex_step_warning(self):
        """Test that complex slices (step != 1) log warning but don't expand."""
        samples = [
            Sample(id=f"s{i}", input=f"task {i}", target=f"target {i}",
                   metadata={MetadataKeys.ORCHESTRATION_ID: "orch1"})
            for i in range(10)
        ]
        dataset = SABERDataset(samples=samples)

        # Slice with step=2 should not expand
        result = dataset[0:6:2]
        assert len(result.samples) == 3
        assert result.samples[0].id == "s0"
        assert result.samples[1].id == "s2"
        assert result.samples[2].id == "s4"

    def test_getitem_slice_mixed_orchestrated_and_non_orchestrated(self):
        """Test slicing with mix of orchestrated and non-orchestrated samples."""
        samples = [
            Sample(id="s1", input="task 1", target="target 1"),  # Non-orchestrated
            Sample(id="s2", input="task 2", target="target 2", metadata={MetadataKeys.ORCHESTRATION_ID: "orch1"}),
            Sample(id="s3", input="task 3", target="target 3", metadata={MetadataKeys.ORCHESTRATION_ID: "orch1"}),
            Sample(id="s4", input="task 4", target="target 4"),  # Non-orchestrated
        ]
        dataset = SABERDataset(samples=samples)

        # Slice [1:2] should expand to include all of orch1 [1:3]
        result = dataset[1:2]
        assert len(result.samples) == 2
        assert result.samples[0].id == "s2"
        assert result.samples[1].id == "s3"


class TestCreateSABERDataset:
    """Test create_saber_dataset conversion function."""

    @pytest.mark.asyncio
    async def test_empty_tasks_raises_error(self):
        """Test that empty task list raises ValueError."""
        with pytest.raises(ValueError, match="tasks_data cannot be empty"):
            await create_saber_dataset([])

    @pytest.mark.asyncio
    async def test_single_episode_task_single_attempt(self):
        """Test converting SingleEpisodeTask with one attempt."""
        task = SingleEpisodeTask(
            benchmark_task_id="task1",
            task_id="task1",
            domain="test_domain",
            title="Test Task",
            description="Test description",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="Do the task",
            assistant_prompt="I'll help",
            submit_prompt="Submit",
        )

        dataset = await create_saber_dataset([task])

        assert len(dataset.samples) == 1
        sample = dataset.samples[0]
        assert sample.id == "task1__attempt_1"
        assert "Test Task" in sample.input
        assert "Test description" in sample.input
        assert sample.metadata[MetadataKeys.EXECUTION_MODE] == TaskExecutionMode.SINGLE.value
        assert sample.metadata[MetadataKeys.ATTEMPT] == 1
        assert sample.metadata[MetadataKeys.TOTAL_ATTEMPTS] == 1

    @pytest.mark.asyncio
    async def test_single_episode_task_multiple_attempts(self):
        """Test converting SingleEpisodeTask with multiple attempts."""
        task = SingleEpisodeTask(
            benchmark_task_id="task2",
            task_id="task2",
            domain="test_domain",
            title="Multi Attempt Task",
            description="Test with retries",
            episode_attempts=3,
            max_steps=20,
            instruction_prompt="Do the task",
            assistant_prompt="I'll help",
            submit_prompt="Submit",
        )

        dataset = await create_saber_dataset([task])

        assert len(dataset.samples) == 3
        for i, sample in enumerate(dataset.samples, start=1):
            assert sample.id == f"task2__attempt_{i}"
            assert sample.metadata[MetadataKeys.ATTEMPT] == i
            assert sample.metadata[MetadataKeys.TOTAL_ATTEMPTS] == 3

    @pytest.mark.asyncio
    async def test_orchestrated_task_creates_samples_per_subtask(self):
        """Test converting OrchestratedTask creates one sample per sub-task per attempt."""
        sub_tasks = [
            SubTaskDefinition(
                role="blue",
                task_id="blue_task",
                domain="test_domain",
                title="Blue Team",
                description="Blue team description",
                order=1,
                depends_on_role=None,
                episode_attempts=2,
                max_steps=15,
                instruction_prompt="Blue instructions",
                assistant_prompt="Blue assistant",
                submit_prompt="Blue submit",
            ),
            SubTaskDefinition(
                role="red",
                task_id="red_task",
                domain="test_domain",
                title="Red Team",
                description="Red team description",
                order=2,
                depends_on_role="blue",
                episode_attempts=2,
                max_steps=15,
                instruction_prompt="Red instructions",
                assistant_prompt="Red assistant",
                submit_prompt="Red submit",
            ),
        ]

        task = OrchestratedTask(
            benchmark_task_id="orch1",
            episode_attempts=2,
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=sub_tasks,
        )

        dataset = await create_saber_dataset([task])

        # 2 sub-tasks * 2 attempts = 4 samples
        assert len(dataset.samples) == 4

        # Check blue team samples
        blue_samples = [s for s in dataset.samples if "blue" in s.id]
        assert len(blue_samples) == 2
        assert blue_samples[0].id == "blue_task_attempt_1"
        assert blue_samples[1].id == "blue_task_attempt_2"
        assert blue_samples[0].metadata[MetadataKeys.SUB_TASK_ROLE] == "blue"
        assert blue_samples[0].metadata[MetadataKeys.EXECUTION_MODE] == "orchestrated_sub_task"

        # Check red team samples
        red_samples = [s for s in dataset.samples if "red" in s.id]
        assert len(red_samples) == 2
        assert red_samples[0].metadata[MetadataKeys.SUB_TASK_ROLE] == "red"
        assert red_samples[0].metadata[MetadataKeys.DEPENDS_ON_ROLE] == "blue"

    @pytest.mark.asyncio
    async def test_mixed_tasks_conversion(self):
        """Test converting mix of SingleEpisodeTask and OrchestratedTask."""
        single_task = SingleEpisodeTask(
            benchmark_task_id="single1",
            task_id="single1",
            domain="test_domain",
            title="Single",
            description="Single task",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="Do it",
            assistant_prompt="OK",
            submit_prompt="Done",
        )

        sub_tasks = [
            SubTaskDefinition(
                role="alpha",
                task_id="alpha_task",
                domain="test_domain",
                title="Alpha",
                description="Alpha desc",
                order=1,
                depends_on_role=None,
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="Alpha",
                assistant_prompt="Alpha",
                submit_prompt="Alpha",
            ),
        ]

        orch_task = OrchestratedTask(
            benchmark_task_id="orch1",
            episode_attempts=1,
            orchestration_strategy=OrchestrationStrategy.PARALLEL,
            sub_tasks=sub_tasks,
        )

        dataset = await create_saber_dataset([single_task, orch_task])

        # 1 single task + 1 orchestrated sub-task = 2 samples
        assert len(dataset.samples) == 2

    @pytest.mark.asyncio
    async def test_conversion_error_handling(self):
        """Test that conversion errors are collected and raised."""
        # Create a mock task that will fail conversion
        bad_task = Mock(spec=BenchmarkTask)
        bad_task.benchmark_task_id = "bad_task"
        bad_task.episode_attempts = 1
        # Make it raise when accessed
        type(bad_task).episode_attempts = property(lambda self: exec('raise ValueError("Bad task")'))

        with pytest.raises(RuntimeError, match="Failed to convert.*SABER tasks"):
            await create_saber_dataset([bad_task])

    @pytest.mark.asyncio
    async def test_dataset_sorting_by_orchestration(self):
        """Test that samples are sorted to keep orchestrations together."""
        sub_tasks_1 = [
            SubTaskDefinition(
                role="role1",
                task_id="task1_role1",
                domain="test_domain",
                title="Task 1 Role 1",
                description="Desc",
                order=2,
                depends_on_role=None,
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
            SubTaskDefinition(
                role="role2",
                task_id="task1_role2",
                domain="test_domain",
                title="Task 1 Role 2",
                description="Desc",
                order=1,
                depends_on_role=None,
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
        ]

        orch_task = OrchestratedTask(
            benchmark_task_id="orch_sort",
            episode_attempts=1,
            orchestration_strategy=OrchestrationStrategy.PARALLEL,
            sub_tasks=sub_tasks_1,
        )

        dataset = await create_saber_dataset([orch_task])

        # Should be sorted by order within orchestration
        assert dataset.samples[0].metadata[MetadataKeys.ORDER] == 1
        assert dataset.samples[1].metadata[MetadataKeys.ORDER] == 2


class TestConvertTaskToSample:
    """Test _convert_task_to_sample function."""

    def test_convert_single_episode_task(self):
        """Test converting SingleEpisodeTask."""
        task = SingleEpisodeTask(
            benchmark_task_id="task1",
            task_id="task1",
            domain="test_domain",
            title="Title",
            description="Description",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="Instruction",
            assistant_prompt="Assistant",
            submit_prompt="Submit",
        )

        sample = _convert_task_to_sample(task, attempt=1)

        assert sample.id == "task1__attempt_1"
        assert "Title" in sample.input
        assert "Description" in sample.input
        assert sample.metadata[MetadataKeys.EXECUTION_MODE] == TaskExecutionMode.SINGLE.value

    def test_convert_orchestrated_task_fallback(self):
        """Test converting OrchestratedTask as single sample (fallback)."""
        sub_tasks = [
            SubTaskDefinition(
                role="role1",
                task_id="task1",
                domain="test_domain",
                title="Sub Task 1",
                description="Sub desc 1",
                order=1,
                depends_on_role=None,
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="Inst",
                assistant_prompt="Asst",
                submit_prompt="Sub",
            ),
        ]

        task = OrchestratedTask(
            benchmark_task_id="orch1",
            episode_attempts=1,
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=sub_tasks,
        )

        sample = _convert_task_to_sample(task, attempt=1)

        assert sample.id == "orch1__attempt_1"
        assert "[role1]" in sample.input
        assert "Sub Task 1" in sample.input
        assert sample.metadata[MetadataKeys.EXECUTION_MODE] == TaskExecutionMode.ORCHESTRATED.value
        assert sample.metadata[MetadataKeys.ORCHESTRATION_ID] == "orch1"

    def test_convert_unknown_task_type_raises_error(self):
        """Test that unknown task type raises ValueError."""
        bad_task = Mock()
        bad_task.__class__.__name__ = "UnknownTaskType"

        with pytest.raises(ValueError, match="Unknown BenchmarkTask type"):
            _convert_task_to_sample(bad_task, attempt=1)


class TestConvertBenchmarkTaskToSample:
    """Test _convert_benchmark_task_to_sample function."""

    def test_convert_valid_single_episode_task(self):
        """Test converting valid SingleEpisodeTask."""
        task = SingleEpisodeTask(
            benchmark_task_id="task1",
            task_id="task1",
            domain="test_domain",
            title="Test",
            description="Test desc",
            episode_attempts=2,
            max_steps=15,
            instruction_prompt="Do it",
            assistant_prompt="OK",
            submit_prompt="Submit",
        )

        sample = _convert_benchmark_task_to_sample(task, attempt=2)

        assert sample.id == "task1__attempt_2"
        assert sample.metadata[MetadataKeys.ATTEMPT] == 2
        assert sample.metadata[MetadataKeys.TOTAL_ATTEMPTS] == 2
        assert sample.metadata[MetadataKeys.TOOL_CALL_LIMIT] == 15

    def test_convert_none_task_raises_error(self):
        """Test that None task raises ValueError."""
        with pytest.raises(ValueError, match="benchmark_task cannot be None"):
            _convert_benchmark_task_to_sample(None, attempt=1)

    def test_convert_invalid_attempt_raises_error(self):
        """Test that attempt < 1 raises ValueError."""
        task = SingleEpisodeTask(
            benchmark_task_id="task1",
            task_id="task1",
            domain="test_domain",
            title="Test",
            description="Test",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="",
            assistant_prompt="",
            submit_prompt="",
        )

        with pytest.raises(ValueError, match="attempt must be >= 1"):
            _convert_benchmark_task_to_sample(task, attempt=0)

    def test_convert_orchestrated_task_raises_error(self):
        """Test that OrchestratedTask raises ValueError (should use different converter)."""
        sub_tasks = [
            SubTaskDefinition(
                role="role1",
                task_id="task1",
                domain="test_domain",
                title="Sub",
                description="Sub",
                order=1,
                depends_on_role=None,
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="",
                assistant_prompt="",
                submit_prompt="",
            ),
        ]

        task = OrchestratedTask(
            benchmark_task_id="orch1",
            episode_attempts=1,
            orchestration_strategy=OrchestrationStrategy.PARALLEL,
            sub_tasks=sub_tasks,
        )

        with pytest.raises(ValueError, match="OrchestratedTask should be converted using _convert_sub_task_to_sample"):
            _convert_benchmark_task_to_sample(task, attempt=1)

    def test_convert_unknown_benchmark_task_type(self):
        """Test that unknown BenchmarkTask type raises ValueError."""
        bad_task = Mock(spec=BenchmarkTask)
        bad_task.__class__ = type('UnknownTask', (BenchmarkTask,), {})

        with pytest.raises(ValueError, match="Unknown BenchmarkTask type"):
            _convert_benchmark_task_to_sample(bad_task, attempt=1)


class TestConvertSubTaskToSample:
    """Test _convert_sub_task_to_sample function."""

    def test_convert_sub_task_basic(self):
        """Test converting a basic sub-task."""
        sub_task = SubTaskDefinition(
            role="defender",
            task_id="defender_task",
            domain="test_domain",
            title="Defend System",
            description="Defend against attacks",
            order=1,
            depends_on_role=None,
            episode_attempts=3,
            max_steps=20,
            instruction_prompt="Defend the system",
            assistant_prompt="I will defend",
            submit_prompt="Submit defense",
        )

        orch_task = OrchestratedTask(
            benchmark_task_id="security_test",
            episode_attempts=3,
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[sub_task],
        )

        sample = _convert_sub_task_to_sample(orch_task, sub_task, attempt=2)

        assert sample.id == "defender_task_attempt_2"
        assert "[defender]" in sample.input
        assert "Defend System" in sample.input
        assert sample.metadata[MetadataKeys.EXECUTION_MODE] == "orchestrated_sub_task"
        assert sample.metadata[MetadataKeys.ORCHESTRATION_ID] == "security_test"
        assert sample.metadata[MetadataKeys.SUB_TASK_ROLE] == "defender"
        assert sample.metadata[MetadataKeys.TASK_ID] == "defender_task"
        assert sample.metadata[MetadataKeys.DEPENDS_ON_ROLE] is None
        assert sample.metadata[MetadataKeys.ORDER] == 1
        assert sample.metadata[MetadataKeys.ATTEMPT] == 2
        assert sample.metadata[MetadataKeys.TOTAL_ATTEMPTS] == 3
        assert sample.metadata[MetadataKeys.TOOL_CALL_LIMIT] == 20

    def test_convert_sub_task_with_dependency(self):
        """Test converting a sub-task with role dependency."""
        sub_task = SubTaskDefinition(
            role="attacker",
            task_id="attacker_task",
            domain="test_domain",
            title="Attack System",
            description="Attack the defended system",
            order=2,
            depends_on_role="defender",
            episode_attempts=1,
            max_steps=25,
            instruction_prompt="Attack",
            assistant_prompt="I will attack",
            submit_prompt="Submit attack",
        )

        orch_task = OrchestratedTask(
            benchmark_task_id="security_test",
            episode_attempts=1,
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[sub_task],
        )

        sample = _convert_sub_task_to_sample(orch_task, sub_task, attempt=1)

        assert sample.metadata[MetadataKeys.SUB_TASK_ROLE] == "attacker"
        assert sample.metadata[MetadataKeys.DEPENDS_ON_ROLE] == "defender"
        assert sample.metadata[MetadataKeys.ORDER] == 2

    def test_convert_sub_task_preserves_orchestration_metadata(self):
        """Test that sub-task conversion includes full orchestration metadata."""
        sub_task = SubTaskDefinition(
            role="role1",
            task_id="task1",
            domain="test_domain",
            title="Title",
            description="Desc",
            order=1,
            depends_on_role=None,
            episode_attempts=5,
            max_steps=10,
            instruction_prompt="Inst",
            assistant_prompt="Asst",
            submit_prompt="Sub",
        )

        orch_task = OrchestratedTask(
            benchmark_task_id="orch_preserve",
            episode_attempts=5,
            orchestration_strategy=OrchestrationStrategy.PARALLEL,
            sub_tasks=[sub_task],
        )

        sample = _convert_sub_task_to_sample(orch_task, sub_task, attempt=3)

        # Verify full orchestration task is in metadata
        assert MetadataKeys.BENCHMARK_TASK in sample.metadata
        benchmark_data = sample.metadata[MetadataKeys.BENCHMARK_TASK]
        assert benchmark_data["benchmark_task_id"] == "orch_preserve"
        assert benchmark_data["episode_attempts"] == 5

    def test_convert_sub_task_with_blocking_config(self):
        """Test that sub-task with blocking_config includes it in metadata."""
        blocking_config = {
            "enabled": True,
            "poll_interval": 2.0,
            "max_iterations": 50,
            "timeout": 100.0,
            "skip_first_iteration": True,
        }

        sub_task = SubTaskDefinition(
            role="blue",
            task_id="blue_task",
            domain="test_domain",
            title="Blue Team Task",
            description="Blue team with blocking",
            order=1,
            depends_on_role=None,
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="Defend",
            assistant_prompt="I will defend",
            submit_prompt="Submit",
            blocking_config=blocking_config,
        )

        orch_task = OrchestratedTask(
            benchmark_task_id="red_blue_test",
            episode_attempts=1,
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[sub_task],
        )

        sample = _convert_sub_task_to_sample(orch_task, sub_task, attempt=1)

        # Verify blocking_config is in metadata
        assert "blocking_config" in sample.metadata
        assert sample.metadata["blocking_config"]["enabled"] is True
        assert sample.metadata["blocking_config"]["poll_interval"] == 2.0
        assert sample.metadata["blocking_config"]["max_iterations"] == 50
        assert sample.metadata["blocking_config"]["timeout"] == 100.0
        assert sample.metadata["blocking_config"]["skip_first_iteration"] is True

    def test_convert_sub_task_without_blocking_config(self):
        """Test that sub-task without blocking_config doesn't include it in metadata."""
        sub_task = SubTaskDefinition(
            role="red",
            task_id="red_task",
            domain="test_domain",
            title="Red Team Task",
            description="Red team without blocking",
            order=0,
            depends_on_role=None,
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="Attack",
            assistant_prompt="I will attack",
            submit_prompt="Submit",
            blocking_config=None,
        )

        orch_task = OrchestratedTask(
            benchmark_task_id="red_blue_test",
            episode_attempts=1,
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[sub_task],
        )

        sample = _convert_sub_task_to_sample(orch_task, sub_task, attempt=1)

        # Verify blocking_config is NOT in metadata
        assert "blocking_config" not in sample.metadata

    def test_convert_orchestrated_task_with_mixed_blocking_configs(self):
        """Test orchestrated task where only some sub-tasks have blocking_config."""
        red_task = SubTaskDefinition(
            role="red",
            task_id="red_task",
            domain="test_domain",
            title="Red Team",
            description="Red team",
            order=0,
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="Attack",
            assistant_prompt="I will attack",
            submit_prompt="Submit",
            blocking_config=None,  # No blocking
        )

        blue_task = SubTaskDefinition(
            role="blue",
            task_id="blue_task",
            domain="test_domain",
            title="Blue Team",
            description="Blue team",
            order=1,
            depends_on_role="red",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="Defend",
            assistant_prompt="I will defend",
            submit_prompt="Submit",
            blocking_config={"enabled": True, "skip_first_iteration": True},  # With blocking
        )

        orch_task = OrchestratedTask(
            benchmark_task_id="red_blue",
            episode_attempts=1,
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[red_task, blue_task],
        )

        # Test red team sample (no blocking)
        red_sample = _convert_sub_task_to_sample(orch_task, red_task, attempt=1)
        assert "blocking_config" not in red_sample.metadata

        # Test blue team sample (with blocking)
        blue_sample = _convert_sub_task_to_sample(orch_task, blue_task, attempt=1)
        assert "blocking_config" in blue_sample.metadata
        assert blue_sample.metadata["blocking_config"]["enabled"] is True
