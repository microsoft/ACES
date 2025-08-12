"""
Unit tests for Task class.

Tests basic task creation and informational retrieval.
Simplified after removing progression logic.
"""

import pytest

from saber.server.tasks.core.task import Task
from saber.server.tasks.core.subtask import SubTask


class TestTask:
    """Test cases for Task functionality."""

    def test_task_creation_minimal(self):
        """Test creating a basic task with minimal parameters."""
        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[]
        )

        assert task.task_id == "test_task"
        assert task.domain == "test_domain"
        assert task.title == "Test Task"
        assert task.description == "A test task"
        assert task.subtasks == []
        assert task.initial_context == {}

    def test_task_creation_with_subtasks(self):
        """Test creating a task with subtasks."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="First Subtask",
            description="First test subtask",
            objective="Complete first step"
        )
        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="Second Subtask",
            description="Second test subtask",
            objective="Complete second step"
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        assert len(task.subtasks) == 2
        assert task.subtasks[0].subtask_id == "subtask1"
        assert task.subtasks[1].subtask_id == "subtask2"

    def test_task_creation_with_context(self):
        """Test creating a task with additional context."""
        context = {
            "domain": "malware_analysis",
            "timeout": 300,
            "sample_path": "/data/test.exe"
        }

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[],
            initial_context=context
        )

        assert task.initial_context == context
        assert task.initial_context["domain"] == "malware_analysis"
        assert task.initial_context["timeout"] == 300

    def test_task_get_subtask_by_id(self):
        """Test retrieving a subtask by ID."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="First Subtask",
            description="First test subtask",
            objective="Complete first step"
        )
        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="Second Subtask",
            description="Second test subtask",
            objective="Complete second step"
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        # Test finding existing subtask
        found_subtask = task.get_subtask_by_id("subtask1")
        assert found_subtask is not None
        assert found_subtask.subtask_id == "subtask1"
        assert found_subtask.title == "First Subtask"

        # Test finding non-existent subtask
        not_found = task.get_subtask_by_id("nonexistent")
        assert not_found is None
