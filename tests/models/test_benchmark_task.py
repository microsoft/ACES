"""
Unit tests for BenchmarkTask polymorphic models.

Tests the polymorphic task abstraction that enables orchestrated task execution,
including SingleEpisodeTask and OrchestratedTask types.
"""

import pytest

from saber.models import (
    BenchmarkTask,
    OrchestratedTask,
    OrchestrationStrategy,
    SingleEpisodeTask,
    SubTaskDefinition,
    TaskExecutionMode,
)


class TestSingleEpisodeTask:
    """Test cases for SingleEpisodeTask functionality."""

    def test_single_episode_task_creation(self):
        """Test creating a single episode task."""
        task = SingleEpisodeTask(
            benchmark_task_id="test_task_1",
            task_id="test_task_1",
            domain="cybench",
            title="Test Task",
            description="A test task",
            episode_attempts=3,
            subtask_count=2,
            max_steps=50,
            instruction_prompt="Do the task",
            assistant_prompt="I'll help",
            submit_prompt="Submit your answer",
            continue_prompt="",
        )

        assert task.execution_mode == TaskExecutionMode.SINGLE
        assert task.benchmark_task_id == "test_task_1"
        assert task.task_id == "test_task_1"
        assert task.episode_attempts == 3
        assert task.get_total_episodes() == 3

    def test_single_episode_get_task_ids(self):
        """Test get_task_ids returns single ID."""
        task = SingleEpisodeTask(
            benchmark_task_id="task_1",
            task_id="task_1",
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

        assert task.get_task_ids() == ["task_1"]

    def test_single_episode_matches_filter_exact(self):
        """Test exact match filtering."""
        task = SingleEpisodeTask(
            benchmark_task_id="xss_0_flag_capture",
            task_id="xss_0_flag_capture",
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

        assert task.matches_filter("xss_0_flag_capture") is True
        assert task.matches_filter("xss_1_flag_capture") is False

    def test_single_episode_matches_filter_glob(self):
        """Test glob pattern filtering."""
        task = SingleEpisodeTask(
            benchmark_task_id="xss_0_flag_capture",
            task_id="xss_0_flag_capture",
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

        assert task.matches_filter("xss_*") is True
        assert task.matches_filter("*_flag_capture") is True
        assert task.matches_filter("sql_*") is False


class TestOrchestratedTask:
    """Test cases for OrchestratedTask functionality."""

    @pytest.fixture
    def sample_orchestrated_task(self):
        """Create a sample orchestrated task for testing."""
        blue_subtask = SubTaskDefinition(
            task_id="blue_task_1",
            role="blue",
            order=0,
            depends_on_role=None,
            domain="saber_dual",
            title="Blue Team Task",
            description="Defend the system",
            episode_attempts=1,
            max_steps=30,
            instruction_prompt="Defend",
            assistant_prompt="Helping defend",
            submit_prompt="Submit defense",
            continue_prompt="",
        )

        red_subtask = SubTaskDefinition(
            task_id="red_task_1",
            role="red",
            order=1,
            depends_on_role="blue",
            domain="saber_dual",
            title="Red Team Task",
            description="Attack the system",
            episode_attempts=1,
            max_steps=30,
            instruction_prompt="Attack",
            assistant_prompt="Helping attack",
            submit_prompt="Submit attack",
            continue_prompt="",
        )

        return OrchestratedTask(
            benchmark_task_id="dual_task_pair_1",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[blue_subtask, red_subtask],
            episode_attempts=2,
        )

    def test_orchestrated_task_creation(self, sample_orchestrated_task):
        """Test creating an orchestrated task."""
        task = sample_orchestrated_task

        assert task.execution_mode == TaskExecutionMode.ORCHESTRATED
        assert task.benchmark_task_id == "dual_task_pair_1"
        assert task.orchestration_strategy == "sequential_paired"
        assert len(task.sub_tasks) == 2
        assert task.episode_attempts == 2

    def test_orchestrated_get_task_ids(self, sample_orchestrated_task):
        """Test get_task_ids returns all sub-task IDs."""
        task = sample_orchestrated_task

        task_ids = task.get_task_ids()
        assert len(task_ids) == 2
        assert "blue_task_1" in task_ids
        assert "red_task_1" in task_ids

    def test_orchestrated_get_total_episodes(self, sample_orchestrated_task):
        """Test get_total_episodes returns attempts (not attempts * sub_tasks)."""
        task = sample_orchestrated_task

        # Each attempt creates all sub-episodes, so total is just episode_attempts
        assert task.get_total_episodes() == 2

    def test_orchestrated_matches_filter_orchestration_id(self, sample_orchestrated_task):
        """Test filtering by orchestration ID."""
        task = sample_orchestrated_task

        assert task.matches_filter("dual_task_pair_1") is True
        assert task.matches_filter("dual_task_pair_2") is False

    def test_orchestrated_matches_filter_subtask_id(self, sample_orchestrated_task):
        """Test filtering by any sub-task ID includes entire orchestration."""
        task = sample_orchestrated_task

        # Matching any sub-task includes the whole orchestration
        assert task.matches_filter("blue_task_1") is True
        assert task.matches_filter("red_task_1") is True
        assert task.matches_filter("other_task") is False

    def test_orchestrated_matches_filter_glob_orchestration(self, sample_orchestrated_task):
        """Test glob pattern matching on orchestration ID."""
        task = sample_orchestrated_task

        assert task.matches_filter("dual_*") is True
        assert task.matches_filter("*_pair_1") is True
        assert task.matches_filter("single_*") is False

    def test_orchestrated_matches_filter_glob_subtask(self, sample_orchestrated_task):
        """Test glob pattern matching on sub-task IDs."""
        task = sample_orchestrated_task

        assert task.matches_filter("blue_*") is True
        assert task.matches_filter("red_*") is True
        assert task.matches_filter("*_task_1") is True
        assert task.matches_filter("xss_*") is False

    def test_subtask_ordering(self):
        """Test that sub-tasks maintain their order."""
        subtasks = [
            SubTaskDefinition(
                task_id=f"task_{i}",
                role=f"role_{i}",
                order=i,
                domain="test",
                title=f"Task {i}",
                description=f"Description {i}",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
            continue_prompt="",
            )
            for i in range(5)
        ]

        task = OrchestratedTask(
            benchmark_task_id="ordered_task",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=subtasks,
            episode_attempts=1,
        )

        # Verify ordering is preserved
        for i, subtask in enumerate(task.sub_tasks):
            assert subtask.order == i
            assert subtask.task_id == f"task_{i}"


class TestSubTaskDefinition:
    """Test cases for SubTaskDefinition."""

    def test_subtask_creation_minimal(self):
        """Test creating a sub-task with minimal fields."""
        subtask = SubTaskDefinition(
            task_id="test_subtask",
            role="worker",
            order=0,
            domain="test",
            title="Test Subtask",
            description="A test subtask",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="Do it",
            assistant_prompt="Helping",
            submit_prompt="Submit",
            continue_prompt="",
        )

        assert subtask.task_id == "test_subtask"
        assert subtask.role == "worker"
        assert subtask.order == 0
        assert subtask.depends_on_role is None

    def test_subtask_with_dependency(self):
        """Test creating a sub-task with dependency."""
        subtask = SubTaskDefinition(
            task_id="dependent_task",
            role="dependent",
            order=1,
            depends_on_role="root",
            domain="test",
            title="Dependent Task",
            description="Depends on root",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="",
        )

        assert subtask.depends_on_role == "root"
        assert subtask.order == 1


class TestPolymorphicBehavior:
    """Test polymorphic behavior across task types."""

    def test_single_and_orchestrated_share_interface(self):
        """Test that both task types implement the BenchmarkTask interface."""
        single = SingleEpisodeTask(
            benchmark_task_id="single_1",
            task_id="single_1",
            domain="test",
            title="Single",
            description="Test",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="",
        )

        orchestrated = OrchestratedTask(
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

        # Both should implement the same interface
        assert hasattr(single, "get_task_ids")
        assert hasattr(single, "get_total_episodes")
        assert hasattr(single, "matches_filter")

        assert hasattr(orchestrated, "get_task_ids")
        assert hasattr(orchestrated, "get_total_episodes")
        assert hasattr(orchestrated, "matches_filter")

        # Both should be instances of BenchmarkTask
        assert isinstance(single, BenchmarkTask)
        assert isinstance(orchestrated, BenchmarkTask)

    def test_serialization_roundtrip(self):
        """Test that tasks can be serialized and deserialized."""
        original = SingleEpisodeTask(
            benchmark_task_id="test_1",
            task_id="test_1",
            domain="test",
            title="Test",
            description="Test task",
            episode_attempts=3,
            max_steps=50,
            instruction_prompt="test",
            assistant_prompt="test",
            submit_prompt="test",
            continue_prompt="",
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back
        restored = SingleEpisodeTask(**data)

        assert restored.benchmark_task_id == original.benchmark_task_id
        assert restored.task_id == original.task_id
        assert restored.episode_attempts == original.episode_attempts
        assert restored.execution_mode == original.execution_mode

    def test_orchestrated_serialization_roundtrip(self):
        """Test orchestrated task serialization roundtrip."""
        original = OrchestratedTask(
            benchmark_task_id="orch_test",
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
                    depends_on_role="blue",
                    domain="test",
                    title="Test 2",
                    description="Test 2",
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

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back
        restored = OrchestratedTask(**data)

        assert restored.benchmark_task_id == original.benchmark_task_id
        assert len(restored.sub_tasks) == len(original.sub_tasks)
        assert restored.sub_tasks[0].task_id == original.sub_tasks[0].task_id
        assert restored.sub_tasks[1].depends_on_role == original.sub_tasks[1].depends_on_role


class TestSubTaskDefinitionTranscriptConfig:
    """Test SubTaskDefinition with transcript_config field."""

    def test_subtask_without_transcript_config(self):
        """Test creating SubTaskDefinition without transcript_config."""
        subtask = SubTaskDefinition(
            task_id="test_task",
            role="blue",
            order=0,
            domain="test",
            title="Test Task",
            description="Test description",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="test instruction",
            assistant_prompt="test assistant",
            submit_prompt="test submit",
            continue_prompt="",
        )

        assert subtask.transcript_config is None

    def test_subtask_with_transcript_config(self):
        """Test creating SubTaskDefinition with transcript_config."""
        transcript_config = {
            "websocket": {
                "connection_timeout": 10.0,
                "pull": {
                    "enabled": True,
                    "blocking": True,
                    "event_timeout": 100.0,
                },
                "push": {
                    "enabled": True,
                }
            }
        }

        subtask = SubTaskDefinition(
            task_id="test_task",
            role="blue",
            order=0,
            domain="test",
            title="Test Task",
            description="Test description",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="test instruction",
            assistant_prompt="test assistant",
            submit_prompt="test submit",
            continue_prompt="",
            transcript_config=transcript_config,
        )

        assert subtask.transcript_config is not None
        assert subtask.transcript_config["websocket"]["connection_timeout"] == 10.0
        assert subtask.transcript_config["websocket"]["pull"]["enabled"] is True
        assert subtask.transcript_config["websocket"]["pull"]["blocking"] is True
        assert subtask.transcript_config["websocket"]["pull"]["event_timeout"] == 100.0
        assert subtask.transcript_config["websocket"]["push"]["enabled"] is True

    def test_subtask_transcript_config_serialization(self):
        """Test SubTaskDefinition with transcript_config serializes correctly."""
        transcript_config = {
            "websocket": {
                "pull": {
                    "enabled": True,
                    "event_timeout": 300.0,
                }
            }
        }

        original = SubTaskDefinition(
            task_id="test_task",
            role="blue",
            order=0,
            domain="test",
            title="Test Task",
            description="Test description",
            episode_attempts=1,
            max_steps=10,
            instruction_prompt="test instruction",
            assistant_prompt="test assistant",
            submit_prompt="test submit",
            continue_prompt="",
            transcript_config=transcript_config,
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back
        restored = SubTaskDefinition(**data)

        assert restored.transcript_config is not None
        assert restored.transcript_config["websocket"]["pull"]["enabled"] is True
        assert restored.transcript_config["websocket"]["pull"]["event_timeout"] == 300.0

    def test_orchestrated_task_with_transcript_config_in_subtasks(self):
        """Test OrchestratedTask with transcript_config in sub-tasks."""
        blue_transcript_config = {
            "websocket": {
                "pull": {
                    "enabled": True,
                    "blocking": True,
                }
            }
        }

        orchestrated = OrchestratedTask(
            benchmark_task_id="test_orch",
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
            sub_tasks=[
                SubTaskDefinition(
                    task_id="red_task",
                    role="red",
                    order=0,
                    domain="test",
                    title="Red Task",
                    description="Red team task",
                    episode_attempts=1,
                    max_steps=10,
                    instruction_prompt="red instruction",
                    assistant_prompt="red assistant",
                    submit_prompt="red submit",
            continue_prompt="",
                    transcript_config=None,  # Red team doesn't use transcript sync
                ),
                SubTaskDefinition(
                    task_id="blue_task",
                    role="blue",
                    order=1,
                    depends_on_role="red",
                    domain="test",
                    title="Blue Task",
                    description="Blue team task",
                    episode_attempts=1,
                    max_steps=10,
                    instruction_prompt="blue instruction",
                    assistant_prompt="blue assistant",
                    submit_prompt="blue submit",
            continue_prompt="",
                    transcript_config=blue_transcript_config,  # Blue team uses transcript sync
                ),
            ],
            episode_attempts=1,
        )

        assert orchestrated.sub_tasks[0].transcript_config is None
        assert orchestrated.sub_tasks[1].transcript_config is not None
        assert orchestrated.sub_tasks[1].transcript_config["websocket"]["pull"]["enabled"] is True
        assert orchestrated.sub_tasks[1].transcript_config["websocket"]["pull"]["blocking"] is True
