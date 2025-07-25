"""Pytest test suite for the task management system."""

import pytest
from pathlib import Path

from saber.server.tasks import (
    TaskManager,
    SessionState,
    TaskNotFoundException,
    SubTaskNotFoundException,
    SessionNotFoundException
)


@pytest.fixture
def task_manager():
    """Create a TaskManager instance for testing."""
    yaml_path = Path(__file__).parent.parent / "data" / "malware_classification" / "tasks.yaml"
    return TaskManager("malware_classification", str(yaml_path))


def test_task_manager_initialization(task_manager):
    """Test TaskManager initialization and task loading."""
    assert task_manager.domain == "malware_classification"
    assert len(task_manager.tasks) == 2
    assert "malware_family_analysis" in task_manager.tasks
    assert "incident_response_triage" in task_manager.tasks


def test_get_task(task_manager):
    """Test getting a specific task."""
    task = task_manager.get_task("malware_family_analysis")
    assert task.task_id == "malware_family_analysis"
    assert task.domain == "malware_classification"
    assert len(task.subtasks) == 4

    # Test non-existent task
    with pytest.raises(TaskNotFoundException):
        task_manager.get_task("non_existent_task")


def test_get_subtask(task_manager):
    """Test getting a specific subtask."""
    subtask = task_manager.get_subtask("malware_family_analysis", "static_analysis")
    assert subtask.subtask_id == "static_analysis"
    assert subtask.task_id == "malware_family_analysis"

    # Test non-existent subtask
    with pytest.raises(SubTaskNotFoundException):
        task_manager.get_subtask("malware_family_analysis", "non_existent_subtask")


def test_session_creation_and_management(task_manager):
    """Test session creation and basic management."""
    session = task_manager.create_task_session("test_client", "malware_family_analysis")

    assert session.task_id == "malware_family_analysis"
    assert session.client_id == "test_client"
    assert session.state == SessionState.CREATED
    assert "sample_path" in session.context

    # Test getting session
    retrieved_session = task_manager.get_session(session.session_id)
    assert retrieved_session.session_id == session.session_id

    # Test non-existent session
    with pytest.raises(SessionNotFoundException):
        task_manager.get_session("non_existent_session")


def test_subtask_advancement(task_manager):
    """Test subtask advancement workflow."""
    session = task_manager.create_task_session("test_client", "malware_family_analysis")

    # Advance to first subtask
    first_subtask = task_manager.advance_subtask(session.session_id)
    assert first_subtask.subtask_id == "static_analysis"
    assert session.state == SessionState.ACTIVE

    # Complete subtask and advance to next
    session.complete_subtask("static_analysis")
    next_subtask = task_manager.advance_subtask(session.session_id)
    assert next_subtask.subtask_id == "dynamic_analysis"
    assert "static_analysis" in session.completed_subtasks


def test_dependency_validation(task_manager):
    """Test subtask dependency validation."""
    task = task_manager.get_task("malware_family_analysis")

    # Get dynamic_analysis subtask which depends on static_analysis
    dynamic_subtask = task.get_subtask_by_id("dynamic_analysis")
    assert dynamic_subtask.depends_on == ["static_analysis"]

    # Test dependency validation
    assert not dynamic_subtask.validate_dependencies(set())
    assert dynamic_subtask.validate_dependencies({"static_analysis"})


def test_task_completion(task_manager):
    """Test task completion workflow."""
    session = task_manager.create_task_session("test_client", "malware_family_analysis")
    task = task_manager.get_task("malware_family_analysis")

    # Complete all subtasks
    for subtask in task.subtasks:
        session.complete_subtask(subtask.subtask_id)

    # Check if task is complete
    assert task.is_complete(session.completed_subtasks)
    assert task.get_completion_percentage(session.completed_subtasks) == 1.0

    # Complete the task
    result = task_manager.complete_task(session.session_id)
    assert result.success
    assert result.task_id == "malware_family_analysis"
    assert len(result.completed_subtasks) == 4


def test_list_tasks(task_manager):
    """Test listing available tasks."""
    tasks = task_manager.list_tasks()
    assert len(tasks) == 2

    task_ids = [task["task_id"] for task in tasks]
    assert "malware_family_analysis" in task_ids
    assert "incident_response_triage" in task_ids
