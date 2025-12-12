"""
Unit tests for polymorphic task filtering.

Tests the _apply_task_filter function with BenchmarkTask types.
"""

import pytest
from inspect_ai._util.error import PrerequisiteError

from saber.inspect_ai.core.task_filter import apply_task_filter as _apply_task_filter
from saber.models import OrchestratedTask, OrchestrationStrategy, SingleEpisodeTask, SubTaskDefinition


class TestTaskFilteringWithSingleEpisodeTasks:
    """Test filtering with SingleEpisodeTask objects."""

    @pytest.fixture
    def sample_single_tasks(self):
        """Create sample SingleEpisodeTask objects for filtering tests."""
        return [
            SingleEpisodeTask(
                benchmark_task_id="xss_0_flag_capture",
                task_id="xss_0_flag_capture",
                domain="cybench",
                title="XSS Challenge 0",
                description="Test",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
            continue_prompt="",
            ),
            SingleEpisodeTask(
                benchmark_task_id="xss_1_flag_capture",
                task_id="xss_1_flag_capture",
                domain="cybench",
                title="XSS Challenge 1",
                description="Test",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
            continue_prompt="",
            ),
            SingleEpisodeTask(
                benchmark_task_id="sql_injection_0",
                task_id="sql_injection_0",
                domain="cybench",
                title="SQL Injection 0",
                description="Test",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
            continue_prompt="",
            ),
        ]

    def test_exact_match_filter(self, sample_single_tasks):
        """Test exact match filtering."""
        filtered = _apply_task_filter(
            tasks=sample_single_tasks,
            task_filter="xss_0_flag_capture",
            domain_slug="cybench",
        )

        assert len(filtered) == 1
        assert filtered[0].benchmark_task_id == "xss_0_flag_capture"

    def test_glob_pattern_filter(self, sample_single_tasks):
        """Test glob pattern filtering."""
        filtered = _apply_task_filter(
            tasks=sample_single_tasks,
            task_filter="xss_*",
            domain_slug="cybench",
        )

        assert len(filtered) == 2
        assert all(task.benchmark_task_id.startswith("xss_") for task in filtered)

    def test_wildcard_suffix_filter(self, sample_single_tasks):
        """Test wildcard suffix pattern."""
        filtered = _apply_task_filter(
            tasks=sample_single_tasks,
            task_filter="*_flag_capture",
            domain_slug="cybench",
        )

        assert len(filtered) == 2
        assert all(task.benchmark_task_id.endswith("_flag_capture") for task in filtered)

    def test_multiple_filters_or_logic(self, sample_single_tasks):
        """Test multiple filters with OR logic."""
        filtered = _apply_task_filter(
            tasks=sample_single_tasks,
            task_filter="xss_0_flag_capture,sql_*",
            domain_slug="cybench",
        )

        assert len(filtered) == 2
        task_ids = {t.benchmark_task_id for t in filtered}
        assert "xss_0_flag_capture" in task_ids
        assert "sql_injection_0" in task_ids

    def test_list_filter_input(self, sample_single_tasks):
        """Test that list input is supported (Inspect AI may parse comma-separated as list)."""
        filtered = _apply_task_filter(
            tasks=sample_single_tasks,
            task_filter=["xss_*", "sql_*"],
            domain_slug="cybench",
        )

        assert len(filtered) == 3  # All tasks match

    def test_deduplication_with_multiple_patterns(self, sample_single_tasks):
        """Test that tasks are deduplicated when multiple patterns match."""
        # Both patterns match xss_0_flag_capture
        filtered = _apply_task_filter(
            tasks=sample_single_tasks,
            task_filter="xss_*,*_flag_capture",
            domain_slug="cybench",
        )

        # Should have 2 tasks (xss_0, xss_1), deduplicated
        assert len(filtered) == 2

    def test_no_match_raises_error(self, sample_single_tasks):
        """Test that no matches raises PrerequisiteError."""
        with pytest.raises(PrerequisiteError, match="No tasks matched filter"):
            _apply_task_filter(
                tasks=sample_single_tasks,
                task_filter="nonexistent_task",
                domain_slug="cybench",
            )

    def test_deterministic_ordering(self, sample_single_tasks):
        """Test that filtered results are consistently ordered."""
        filtered1 = _apply_task_filter(
            tasks=sample_single_tasks,
            task_filter="*",
            domain_slug="cybench",
        )

        filtered2 = _apply_task_filter(
            tasks=sample_single_tasks,
            task_filter="*",
            domain_slug="cybench",
        )

        # Should have same order
        assert [t.benchmark_task_id for t in filtered1] == [t.benchmark_task_id for t in filtered2]


class TestTaskFilteringWithOrchestratedTasks:
    """Test filtering with OrchestratedTask objects."""

    @pytest.fixture
    def sample_orchestrated_tasks(self):
        """Create sample OrchestratedTask objects."""
        return [
            OrchestratedTask(
                benchmark_task_id="dual_blue_red_1",
                orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
                sub_tasks=[
                    SubTaskDefinition(
                        task_id="blue_defend_1",
                        role="blue",
                        order=0,
                        domain="saber_dual",
                        title="Blue Defend 1",
                        description="Test",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="test",
                        assistant_prompt="test",
                        submit_prompt="test",
            continue_prompt="",
                    ),
                    SubTaskDefinition(
                        task_id="red_attack_1",
                        role="red",
                        order=1,
                        domain="saber_dual",
                        title="Red Attack 1",
                        description="Test",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="test",
                        assistant_prompt="test",
                        submit_prompt="test",
            continue_prompt="",
                    ),
                ],
                episode_attempts=1,
            ),
            OrchestratedTask(
                benchmark_task_id="dual_blue_red_2",
                orchestration_strategy=OrchestrationStrategy.SEQUENTIAL_PAIRED,
                sub_tasks=[
                    SubTaskDefinition(
                        task_id="blue_defend_2",
                        role="blue",
                        order=0,
                        domain="saber_dual",
                        title="Blue Defend 2",
                        description="Test",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="test",
                        assistant_prompt="test",
                        submit_prompt="test",
            continue_prompt="",
                    ),
                    SubTaskDefinition(
                        task_id="red_attack_2",
                        role="red",
                        order=1,
                        domain="saber_dual",
                        title="Red Attack 2",
                        description="Test",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="test",
                        assistant_prompt="test",
                        submit_prompt="test",
            continue_prompt="",
                    ),
                ],
                episode_attempts=1,
            ),
        ]

    def test_filter_by_orchestration_id(self, sample_orchestrated_tasks):
        """Test filtering by orchestration ID."""
        filtered = _apply_task_filter(
            tasks=sample_orchestrated_tasks,
            task_filter="dual_blue_red_1",
            domain_slug="saber_dual",
        )

        assert len(filtered) == 1
        assert filtered[0].benchmark_task_id == "dual_blue_red_1"

    def test_filter_by_subtask_includes_whole_orchestration(self, sample_orchestrated_tasks):
        """Test that matching a sub-task includes the entire orchestration."""
        # Filter by blue team sub-task
        filtered = _apply_task_filter(
            tasks=sample_orchestrated_tasks,
            task_filter="blue_defend_1",
            domain_slug="saber_dual",
        )

        # Should get the entire orchestration, not just the sub-task
        assert len(filtered) == 1
        assert filtered[0].benchmark_task_id == "dual_blue_red_1"
        assert len(filtered[0].sub_tasks) == 2  # Both blue and red

    def test_glob_pattern_on_subtask(self, sample_orchestrated_tasks):
        """Test glob pattern matching on sub-task IDs."""
        # Match all blue team tasks
        filtered = _apply_task_filter(
            tasks=sample_orchestrated_tasks,
            task_filter="blue_*",
            domain_slug="saber_dual",
        )

        # Should get both orchestrations (each has a blue sub-task)
        assert len(filtered) == 2

    def test_glob_pattern_on_orchestration_id(self, sample_orchestrated_tasks):
        """Test glob pattern on orchestration ID."""
        filtered = _apply_task_filter(
            tasks=sample_orchestrated_tasks,
            task_filter="dual_*",
            domain_slug="saber_dual",
        )

        assert len(filtered) == 2

    def test_multiple_subtasks_matched_only_one_orchestration(self, sample_orchestrated_tasks):
        """Test that matching multiple sub-tasks from same orchestration returns it once."""
        # Match both sub-tasks from first orchestration
        filtered = _apply_task_filter(
            tasks=sample_orchestrated_tasks,
            task_filter="blue_defend_1,red_attack_1",
            domain_slug="saber_dual",
        )

        # Should only get one orchestration (deduplicated)
        assert len(filtered) == 1
        assert filtered[0].benchmark_task_id == "dual_blue_red_1"


class TestTaskFilteringMixedTypes:
    """Test filtering with mixed SingleEpisode and Orchestrated tasks."""

    @pytest.fixture
    def mixed_tasks(self):
        """Create a mix of SingleEpisode and Orchestrated tasks."""
        return [
            SingleEpisodeTask(
                benchmark_task_id="xss_0",
                task_id="xss_0",
                domain="test",
                title="XSS 0",
                description="Test",
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
                        task_id="blue_1",
                        role="blue",
                        order=0,
                        domain="test",
                        title="Blue 1",
                        description="Test",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="test",
                        assistant_prompt="test",
                        submit_prompt="test",
            continue_prompt="",
                    ),
                    SubTaskDefinition(
                        task_id="red_1",
                        role="red",
                        order=1,
                        domain="test",
                        title="Red 1",
                        description="Test",
                        episode_attempts=1,
                        max_steps=10,
                        instruction_prompt="test",
                        assistant_prompt="test",
                        submit_prompt="test",
            continue_prompt="",
                    ),
                ],
                episode_attempts=1,
            ),
            SingleEpisodeTask(
                benchmark_task_id="sql_0",
                task_id="sql_0",
                domain="test",
                title="SQL 0",
                description="Test",
                episode_attempts=1,
                max_steps=10,
                instruction_prompt="test",
                assistant_prompt="test",
                submit_prompt="test",
            continue_prompt="",
            ),
        ]

    def test_filter_mixed_types_single_pattern(self, mixed_tasks):
        """Test filtering mixed types with single pattern."""
        # Should match only single episode task
        filtered = _apply_task_filter(
            tasks=mixed_tasks,
            task_filter="xss_*",
            domain_slug="test",
        )

        assert len(filtered) == 1
        assert isinstance(filtered[0], SingleEpisodeTask)

    def test_filter_mixed_types_multiple_patterns(self, mixed_tasks):
        """Test filtering mixed types with multiple patterns."""
        filtered = _apply_task_filter(
            tasks=mixed_tasks,
            task_filter="xss_*,blue_*",
            domain_slug="test",
        )

        # Should get xss_0 (SingleEpisode) and dual_1 (Orchestrated, matched by blue_1)
        assert len(filtered) == 2

        # Check types
        task_types = {type(t).__name__ for t in filtered}
        assert "SingleEpisodeTask" in task_types
        assert "OrchestratedTask" in task_types


class TestErrorHandling:
    """Test error handling in task filtering."""

    def test_empty_pattern_skipped(self):
        """Test that empty patterns are skipped."""
        tasks = [
            SingleEpisodeTask(
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
        ]

        # Empty pattern in comma-separated list should be skipped
        filtered = _apply_task_filter(
            tasks=tasks,
            task_filter="test_1,",  # Trailing comma creates empty pattern
            domain_slug="test",
        )

        assert len(filtered) == 1

    def test_error_message_includes_available_tasks(self):
        """Test that error message lists available tasks."""
        tasks = [
            SingleEpisodeTask(
                benchmark_task_id="available_1",
                task_id="available_1",
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
        ]

        with pytest.raises(PrerequisiteError) as exc_info:
            _apply_task_filter(
                tasks=tasks,
                task_filter="nonexistent",
                domain_slug="test",
            )

        error_message = str(exc_info.value)
        assert "available_1" in error_message
        assert "Available tasks" in error_message
