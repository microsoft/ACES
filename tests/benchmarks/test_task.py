"""
Unit tests for Task class.

Tests meaningful task functionality - pruned basic data structure tests.
"""

import pytest

from saber.server.benchmarks.subtask import SubTask
from saber.server.benchmarks.task import Task


class TestTask:
    """Test cases for Task functionality."""

    def test_task_get_subtask_by_id(self):
        """Test retrieving a subtask by ID."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="First Subtask",
            description="First test subtask",
            objective="Complete first step",
        )
        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="Second Subtask",
            description="Second test subtask",
            objective="Complete second step",
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            prompts={"instruction": "test_task_prompt.md", "assistant": "test_task_prompt.md", "submit": "test_task_prompt.md"},
            subtasks=[subtask1, subtask2],
        )

        # Test finding existing subtask
        found_subtask = task.get_subtask_by_id("subtask1")
        assert found_subtask is not None
        assert found_subtask.subtask_id == "subtask1"
        assert found_subtask.title == "First Subtask"

        # Test finding non-existent subtask
        not_found = task.get_subtask_by_id("nonexistent")
        assert not_found is None
