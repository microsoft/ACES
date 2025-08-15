"""
Unit tests for SubTask class.

Tests meaningful validation functionality - pruned basic data structure tests.
"""

import pytest

from saber.server.tasks.subtask import SubTask


class TestSubTask:
    """Test cases for SubTask functionality."""

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
