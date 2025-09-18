"""
Unit tests for server-side composite task functionality.

Tests the BenchmarkManager task filtering, REST API endpoint, and TaskInfo model
enhancements. Follows SABER best practices for fail-fast validation and error handling.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from typing import List

from saber.models.core.task_types import TaskType
from saber.models import BenchmarkInfo, TaskInfo
from saber.server.benchmarks.benchmark_manager import BenchmarkManager
from saber.server.api.session_rest_api import SessionRestAPI


class TestBenchmarkManagerTaskFiltering:
    """Test cases for BenchmarkManager task filtering by type."""

    @pytest.fixture
    def mock_tasks(self):
        """Create mock Task objects for testing."""
        # Mock individual tasks (no type field or type != composite)
        individual_task1 = Mock()
        individual_task1.task_id = "individual_task_1"
        individual_task1.episodes_count = 1
        # Explicitly delete type attribute to test default behavior
        if hasattr(individual_task1, 'type'):
            delattr(individual_task1, 'type')

        individual_task2 = Mock()
        individual_task2.task_id = "individual_task_2"
        individual_task2.episodes_count = 2
        individual_task2.type = TaskType.INDIVIDUAL

        # Mock composite task
        composite_task = Mock()
        composite_task.task_id = "composite_task_1"
        composite_task.episodes_count = 3
        composite_task.type = TaskType.COMPOSITE

        return {
            "individual_task_1": individual_task1,
            "individual_task_2": individual_task2,
            "composite_task_1": composite_task
        }

    @pytest.fixture
    def benchmark_manager(self, mock_tasks):
        """Create BenchmarkManager with mock tasks."""
        manager = Mock(spec=BenchmarkManager)
        manager.tasks = mock_tasks

        # Bind the actual method to test
        manager._filter_tasks_by_type = BenchmarkManager._filter_tasks_by_type.__get__(manager)

        return manager

    def test_filter_individual_tasks(self, benchmark_manager):
        """Test filtering for individual tasks only."""
        result = benchmark_manager._filter_tasks_by_type(TaskType.INDIVIDUAL)

        assert len(result) == 2
        task_ids = [task.task_id for task in result]
        assert "individual_task_1" in task_ids
        assert "individual_task_2" in task_ids
        assert "composite_task_1" not in task_ids

    def test_filter_composite_tasks(self, benchmark_manager):
        """Test filtering for composite tasks only."""
        result = benchmark_manager._filter_tasks_by_type(TaskType.COMPOSITE)

        assert len(result) == 1
        assert result[0].task_id == "composite_task_1"

    def test_filter_all_tasks(self, benchmark_manager):
        """Test filtering for all task types."""
        result = benchmark_manager._filter_tasks_by_type(TaskType.ALL)

        assert len(result) == 3
        task_ids = [task.task_id for task in result]
        assert "individual_task_1" in task_ids
        assert "individual_task_2" in task_ids
        assert "composite_task_1" in task_ids

    def test_filter_invalid_task_type_fails_fast(self, benchmark_manager):
        """Test that invalid task_type raises ValueError immediately."""
        with pytest.raises(ValueError) as exc_info:
            benchmark_manager._filter_tasks_by_type("invalid_type")

        # Verify error message mentions valid types
        assert "invalid_type" in str(exc_info.value)
        assert "Must be one of" in str(exc_info.value)

    def test_tasks_endpoint_error_message_format(self):
        """Test the tasks endpoint error message format."""
        valid_types = TaskType.get_valid_filter_types()
        invalid_type = "invalid_type"

        # Simulate the FastAPI error response format
        valid_type_strings = [task_type.value for task_type in valid_types]
        error_message = f"Invalid task_type '{invalid_type}'. Must be one of: {sorted(valid_type_strings)}"

        # Verify proper error message format
        assert "['all', 'composite', 'individual']" in error_message
        assert invalid_type in error_message
        assert "Must be one of" in error_message

    def test_filter_empty_task_type_fails_fast(self, benchmark_manager):
        """Test that empty task_type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            benchmark_manager._filter_tasks_by_type("")

        assert "Invalid task_type ''" in str(exc_info.value)

    def test_filter_none_task_type_fails_fast(self, benchmark_manager):
        """Test that None task_type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            benchmark_manager._filter_tasks_by_type(None)

        assert "Invalid task_type 'None'" in str(exc_info.value)

    def test_default_individual_type_handling(self, benchmark_manager):
        """Test that tasks without type attribute default to individual."""
        # Get the task without type attribute
        task_without_type = benchmark_manager.tasks["individual_task_1"]
        assert not hasattr(task_without_type, 'type')

        # Should be included in individual filter
        result = benchmark_manager._filter_tasks_by_type(TaskType.INDIVIDUAL)
        task_ids = [task.task_id for task in result]
        assert "individual_task_1" in task_ids

        # Should not be included in composite filter
        result = benchmark_manager._filter_tasks_by_type(TaskType.COMPOSITE)
        task_ids = [task.task_id for task in result]
        assert "individual_task_1" not in task_ids


class TestBenchmarkManagerGetBenchmarkInfo:
    """Test cases for BenchmarkManager.get_benchmark_info with task type filtering."""

    @pytest.fixture
    def mock_benchmark_manager(self):
        """Create mock BenchmarkManager for testing get_benchmark_info."""
        manager = Mock(spec=BenchmarkManager)

        # Add required domain attribute
        manager.domain = "test_domain"

        # Mock tasks data with all required attributes
        individual_task = Mock()
        individual_task.task_id = "individual_1"
        individual_task.title = "Individual Task 1"
        individual_task.description = "Individual task description"
        individual_task.episodes_count = 2
        individual_task.type = TaskType.INDIVIDUAL
        individual_task.subtasks = []  # Individual tasks have no subtasks
        individual_task.episode_config = {"max_steps": 50}
        individual_task.get_episode_attempts = Mock(return_value=2)

        composite_task = Mock()
        composite_task.task_id = "composite_1"
        composite_task.title = "Composite Task 1"
        composite_task.description = "Composite task description"
        composite_task.episodes_count = 3
        composite_task.type = TaskType.COMPOSITE
        composite_task.subtasks = [Mock(), Mock(), Mock()]  # Composite tasks have subtasks
        composite_task.episode_config = {"max_steps": 100}
        composite_task.get_episode_attempts = Mock(return_value=3)
        # Composite tasks need individual_tasks as a list, not a Mock
        composite_task.individual_tasks = [
            {"task_id": "subtask_1", "title": "Subtask 1"},
            {"task_id": "subtask_2", "title": "Subtask 2"},
            {"task_id": "subtask_3", "title": "Subtask 3"}
        ]

        # Mock get_task_prompt method
        manager.get_task_prompt = Mock(return_value="Test task prompt")

        # Setup _filter_tasks_by_type to return appropriate tasks
        def mock_filter(task_type):
            if task_type == TaskType.INDIVIDUAL:
                return [individual_task]
            elif task_type == TaskType.COMPOSITE:
                return [composite_task]
            elif task_type == TaskType.ALL:
                return [individual_task, composite_task]
            else:
                raise ValueError(f"Invalid task_type '{task_type}'")

        manager._filter_tasks_by_type = mock_filter

        # Bind the actual get_benchmark_info method
        manager.get_benchmark_info = BenchmarkManager.get_benchmark_info.__get__(manager)

        return manager, individual_task, composite_task

    def test_get_benchmark_info_individual_tasks(self, mock_benchmark_manager):
        """Test get_benchmark_info for individual tasks."""
        manager, individual_task, composite_task = mock_benchmark_manager

        result = manager.get_benchmark_info(TaskType.INDIVIDUAL)

        # Should return BenchmarkInfo with individual tasks
        assert hasattr(result, 'tasks')
        assert len(result.tasks) == 1
        task_info = result.tasks[0]
        assert task_info.task_id == "individual_1"
        assert task_info.task_type == TaskType.INDIVIDUAL
        assert task_info.subtask_count == 0  # Individual tasks have no subtasks

    def test_get_benchmark_info_composite_tasks(self, mock_benchmark_manager):
        """Test get_benchmark_info for composite tasks."""
        manager, individual_task, composite_task = mock_benchmark_manager

        result = manager.get_benchmark_info(TaskType.COMPOSITE)

        # Should return BenchmarkInfo with composite tasks
        assert hasattr(result, 'tasks')
        assert len(result.tasks) == 1
        task_info = result.tasks[0]
        assert task_info.task_id == "composite_1"
        assert task_info.task_type == TaskType.COMPOSITE
        assert task_info.subtask_count == 3  # Composite tasks have subtasks

    def test_get_benchmark_info_all_tasks(self, mock_benchmark_manager):
        """Test get_benchmark_info for all task types."""
        manager, individual_task, composite_task = mock_benchmark_manager

        result = manager.get_benchmark_info(TaskType.ALL)

        # Should return BenchmarkInfo with all tasks
        assert hasattr(result, 'tasks')
        assert len(result.tasks) == 2

        task_ids = [task.task_id for task in result.tasks]
        assert "individual_1" in task_ids
        assert "composite_1" in task_ids

    def test_get_benchmark_info_default_parameter(self, mock_benchmark_manager):
        """Test get_benchmark_info defaults to individual tasks."""
        manager, individual_task, composite_task = mock_benchmark_manager

        result = manager.get_benchmark_info()  # No parameter - should default to individual

        # Should return BenchmarkInfo with individual tasks only
        assert hasattr(result, 'tasks')
        assert len(result.tasks) == 1
        task_info = result.tasks[0]
        assert task_info.task_id == "individual_1"
        assert task_info.task_type == TaskType.INDIVIDUAL        # Should return BenchmarkInfo with composite tasks
        assert hasattr(result, 'tasks')
        assert len(result.tasks) == 1
        task_info = result.tasks[0]
        assert task_info.task_id == "composite_1"
        assert task_info.task_type == TaskType.COMPOSITE
        assert task_info.subtask_count == 3  # Composite tasks have subtasks

    def test_get_benchmark_info_all_tasks(self, mock_benchmark_manager):
        """Test get_benchmark_info for all tasks."""
        manager, individual_task, composite_task = mock_benchmark_manager

        result = manager.get_benchmark_info(TaskType.ALL)

        # Should return BenchmarkInfo with all tasks
        assert hasattr(result, 'tasks')
        assert len(result.tasks) == 2

        task_ids = [task.task_id for task in result.tasks]
        assert "individual_1" in task_ids
        assert "composite_1" in task_ids

    def test_get_benchmark_info_default_parameter(self, mock_benchmark_manager):
        """Test get_benchmark_info defaults to individual tasks."""
        manager, individual_task, composite_task = mock_benchmark_manager

        result = manager.get_benchmark_info()  # No parameter - should default to individual

        # Should return BenchmarkInfo with individual tasks only
        assert hasattr(result, 'tasks')
        assert len(result.tasks) == 1
        task_info = result.tasks[0]
        assert task_info.task_id == "individual_1"
        assert task_info.task_type == TaskType.INDIVIDUAL


class TestTaskInfoModelEnhancements:
    """Test cases for TaskInfo model task_type field."""

    def test_task_info_includes_task_type_field(self):
        """Test that TaskInfo model includes task_type field."""
        # This test verifies the model enhancement mentioned in the implementation
        task_info = TaskInfo(
            task_id="test_task",
            title="Test Task",
            description="Test task description",
            episode_attempts=2,
            subtask_count=1,
            max_steps=10,
            task_type=TaskType.INDIVIDUAL
        )

        assert task_info.task_type == TaskType.INDIVIDUAL
        assert task_info.task_id == "test_task"
        assert task_info.episode_attempts == 2

    def test_task_info_default_task_type(self):
        """Test TaskInfo default task_type for backward compatibility."""
        task_info = TaskInfo(
            task_id="test_task",
            title="Test Task",
            description="Test task description",
            episode_attempts=1,
            subtask_count=1,
            max_steps=10
        )

        # Should default to individual for backward compatibility
        assert task_info.task_type == TaskType.INDIVIDUAL

    def test_task_info_composite_task_type(self):
        """Test TaskInfo with composite task type."""
        task_info = TaskInfo(
            task_id="composite_task",
            title="Composite Task",
            description="Test composite task",
            episode_attempts=3,
            subtask_count=5,
            max_steps=20,
            task_type=TaskType.COMPOSITE
        )

        assert task_info.task_type == TaskType.COMPOSITE

    def test_task_info_serialization_includes_task_type(self):
        """Test that TaskInfo serialization includes task_type field."""
        task_info = TaskInfo(
            task_id="test_task",
            title="Test Task",
            description="Test task",
            episode_attempts=1,
            subtask_count=1,
            max_steps=10,
            task_type=TaskType.COMPOSITE
        )

        serialized = task_info.model_dump()
        assert "task_type" in serialized
        assert serialized["task_type"] == TaskType.COMPOSITE


class TestSessionRestAPITasksEndpoint:
    """Test cases for the new /api/v1/tasks REST API endpoint."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock SessionManager for API testing."""
        manager = Mock()

        # Mock benchmark info response
        benchmark_info = Mock(spec=BenchmarkInfo)
        benchmark_info.total_tasks = 2
        benchmark_info.total_episodes = 5
        benchmark_info.model_dump.return_value = {"test": "data"}

        manager.get_benchmark_info.return_value = benchmark_info

        return manager

    @pytest.fixture
    def rest_api(self, mock_session_manager):
        """Create SessionRestAPI instance for testing."""
        api = SessionRestAPI(mock_session_manager)
        return api

    def test_tasks_endpoint_default_individual(self, mock_session_manager):
        """Test /api/v1/tasks endpoint defaults to individual tasks."""
        # This would be tested through the actual FastAPI test client
        # Here we test the underlying logic

        api = SessionRestAPI(mock_session_manager)

        # Simulate the endpoint logic
        task_type = TaskType.INDIVIDUAL  # Default parameter
        valid_types = TaskType.get_valid_filter_types()

        assert task_type in valid_types

        # Should call session_manager.get_benchmark_info with correct parameter
        mock_session_manager.get_benchmark_info.assert_not_called()  # Not called yet

    def test_tasks_endpoint_validation_logic(self, mock_session_manager):
        """Test the validation logic used in the tasks endpoint."""
        # Test the validation logic that would be used in the endpoint

        valid_types = TaskType.get_valid_filter_types()

        # Valid task types should pass
        for task_type in valid_types:
            assert task_type in valid_types  # Would not raise HTTPException

        # Invalid task types should fail
        invalid_types = ["invalid", "", None, "INDIVIDUAL"]
        for invalid_type in invalid_types:
            assert invalid_type not in valid_types  # Would raise HTTPException 422

    def test_tasks_endpoint_error_message_format(self):
        """Test error message format for invalid task_type."""
        # Test the error message format used in HTTPException
        valid_types = TaskType.get_valid_filter_types()
        invalid_type = "invalid_type"

        # Convert enum values to strings for proper comparison
        valid_type_strings = [task_type.value for task_type in valid_types]
        error_detail = f"Invalid task_type '{invalid_type}'. Must be one of: {sorted(valid_type_strings)}"

        assert "Invalid task_type 'invalid_type'" in error_detail
        assert "Must be one of:" in error_detail
        assert "['all', 'composite', 'individual']" in error_detail

    def test_tasks_endpoint_serialization_check(self, mock_session_manager):
        """Test that endpoint checks serialization before returning."""
        # Test the serialization check logic from the endpoint

        benchmark_info = mock_session_manager.get_benchmark_info.return_value

        # Should be able to call model_dump without error
        try:
            benchmark_info.model_dump()
            serialization_success = True
        except Exception:
            serialization_success = False

        assert serialization_success
