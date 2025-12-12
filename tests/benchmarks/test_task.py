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
            prompts={"instruction": "test_task_prompt.md", "assistant": "test_task_prompt.md", "submit": "test_task_prompt.md", "continue": "test_continue.md"},
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

    def test_task_with_role(self):
        """Test task loads with role field."""
        task = Task(
            task_id="blue_task",
            domain="test_domain",
            title="Blue Team Task",
            description="Test blue team task",
            prompts={
                "instruction": "blue_instruction.md",
                "assistant": "blue_assistant.md",
                "submit": "blue_submit.md",
                "continue": "test_continue.md",
            },
            role="blue",
        )
        assert task.role == "blue"

    def test_task_without_role(self):
        """Test task can be created without role for single episode tasks."""
        task = Task(
            task_id="single_task",
            domain="test_domain",
            title="Single Task",
            description="Test single task",
            prompts={
                "instruction": "instruction.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
        )
        assert task.role is None

    def test_dependent_task_requires_role(self):
        """Test dependent task fails without role."""
        with pytest.raises(ValueError, match="'role' is required when 'dependency_template' is set"):
            Task(
                task_id="red_task",
                domain="test_domain",
                title="Red Team Task",
                description="Test red team task",
                prompts={
                    "instruction": "red_instruction.md",
                    "assistant": "red_assistant.md",
                    "submit": "red_submit.md",
                    "continue": "test_continue.md",
                },
                dependency_template="blue_task",
                role=None,  # Should fail - dependent task needs role
            )

    def test_dependent_task_with_role(self):
        """Test dependent task successfully created with role."""
        task = Task(
            task_id="red_task",
            domain="test_domain",
            title="Red Team Task",
            description="Test red team task",
            prompts={
                "instruction": "red_instruction.md",
                "assistant": "red_assistant.md",
                "submit": "red_submit.md",
                "continue": "test_continue.md",
            },
            dependency_template="blue_task",
            role="red",
        )
        assert task.role == "red"
        assert task.dependency_template == "blue_task"

    def test_task_role_with_custom_semantic_names(self):
        """Test tasks can use custom semantic role names."""
        # Test defender role
        defender = Task(
            task_id="defender_task",
            domain="test_domain",
            title="Defender Task",
            description="Test defender",
            prompts={
                "instruction": "defender.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            role="defender",
        )
        assert defender.role == "defender"

        # Test attacker role
        attacker = Task(
            task_id="attacker_task",
            domain="test_domain",
            title="Attacker Task",
            description="Test attacker",
            prompts={
                "instruction": "attacker.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            dependency_template="defender_task",
            role="attacker",
        )
        assert attacker.role == "attacker"
