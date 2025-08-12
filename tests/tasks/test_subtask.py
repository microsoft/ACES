"""
Unit tests for SubTask class.

Tests basic subtask creation and informational functionality.
Simplified after removing entry/exit conditions and completion logic.
"""

import pytest

from saber.server.tasks.core.subtask import SubTask


class TestSubTask:
    """Test cases for SubTask functionality."""

    def test_subtask_creation_minimal(self):
        """Test creating a basic subtask with minimal parameters."""
        subtask = SubTask(
            subtask_id="test_subtask",
            task_id="test_task",
            title="Test SubTask",
            description="A test subtask",
            objective="Complete test objective"
        )

        assert subtask.subtask_id == "test_subtask"
        assert subtask.task_id == "test_task"
        assert subtask.title == "Test SubTask"
        assert subtask.description == "A test subtask"
        assert subtask.objective == "Complete test objective"

    def test_subtask_required_fields(self):
        """Test that required fields are enforced."""
        # Test missing subtask_id
        with pytest.raises(ValueError):
            SubTask(task_id="test", title="Test", description="Test", objective="Test")

        # Test missing title
        with pytest.raises(ValueError):
            SubTask(subtask_id="test", task_id="test", description="Test", objective="Test")

        # Test missing description
        with pytest.raises(ValueError):
            SubTask(subtask_id="test", task_id="test", title="Test", objective="Test")

        # Test missing objective
        with pytest.raises(ValueError):
            SubTask(subtask_id="test", task_id="test", title="Test", description="Test")

    def test_subtask_str_representation(self):
        """Test string representation of subtask."""
        subtask = SubTask(
            subtask_id="test_subtask",
            task_id="test_task",
            title="Test SubTask",
            description="A test subtask",
            objective="Test objective"
        )

        str_repr = str(subtask)
        assert "test_subtask" in str_repr
        assert "Test SubTask" in str_repr

    def test_subtask_equality(self):
        """Test subtask equality comparison."""
        subtask1 = SubTask(
            subtask_id="test_subtask",
            task_id="test_task",
            title="Test SubTask",
            description="A test subtask",
            objective="Test objective"
        )

        subtask2 = SubTask(
            subtask_id="test_subtask",
            task_id="test_task",
            title="Test SubTask",
            description="A test subtask",
            objective="Test objective"
        )

        subtask3 = SubTask(
            subtask_id="different_subtask",
            task_id="test_task",
            title="Different SubTask",
            description="A different subtask",
            objective="Different objective"
        )

        # Same content should be equal
        assert subtask1 == subtask2

        # Different content should not be equal
        assert subtask1 != subtask3

    def test_subtask_dict_conversion(self):
        """Test converting subtask to dictionary."""
        subtask = SubTask(
            subtask_id="test_subtask",
            task_id="test_task",
            title="Test SubTask",
            description="A test subtask",
            objective="Test objective"
        )

        subtask_dict = subtask.model_dump()
        assert subtask_dict["subtask_id"] == "test_subtask"
        assert subtask_dict["task_id"] == "test_task"
        assert subtask_dict["title"] == "Test SubTask"
        assert subtask_dict["description"] == "A test subtask"
        assert subtask_dict["objective"] == "Test objective"

    def test_is_dependent_on(self):
        """Test dependency checking (should always return False)."""
        subtask = SubTask(
            subtask_id="test_subtask",
            task_id="test_task",
            title="Test SubTask",
            description="A test subtask",
            objective="Test objective"
        )

        # Should always return False since dependencies are removed
        assert not subtask.is_dependent_on("any_subtask")
        assert not subtask.is_dependent_on("test_subtask")
