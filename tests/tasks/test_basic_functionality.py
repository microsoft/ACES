"""Simple test to verify the task manager implementation works."""

import os
import sys
from pathlib import Path

from saber.server.tasks import TaskManager, SessionState


def test_task_manager_basic_functionality():
    """Test basic TaskManager functionality with the malware classification tasks."""

    # Get path to test YAML file
    yaml_path = Path(__file__).parent.parent / "data" / "malware_classification" / "tasks.yaml"

    # Initialize TaskManager
    manager = TaskManager("malware_classification", str(yaml_path))

    # Test listing tasks
    tasks = manager.list_tasks()
    print(f"Found {len(tasks)} tasks:")
    for task in tasks:
        print(f"  - {task['task_id']}: {task['title']}")

    # Test getting a specific task
    task = manager.get_task("malware_family_analysis")
    print(f"\nTask: {task.title}")
    print(f"Subtasks: {len(task.subtasks)}")

    # Test creating a session
    session = manager.create_task_session("test_client", "malware_family_analysis")
    print(f"\nCreated session: {session.session_id}")
    print(f"Session state: {session.state}")
    print(f"Initial context: {session.context}")

    # Test advancing to first subtask
    first_subtask = manager.advance_subtask(session.session_id)
    print(f"\nFirst subtask: {first_subtask.subtask_id if first_subtask else 'None'}")
    if first_subtask:
        print(f"  Title: {first_subtask.title}")
        print(f"  Objective: {first_subtask.objective}")
        print(f"  Required tools: {first_subtask.required_tools}")

    # Test completing a subtask and advancing
    if first_subtask:
        session.complete_subtask(first_subtask.subtask_id)
        print(f"\nCompleted subtask: {first_subtask.subtask_id}")

        next_subtask = manager.advance_subtask(session.session_id)
        print(f"Next subtask: {next_subtask.subtask_id if next_subtask else 'None'}")


if __name__ == "__main__":
    test_task_manager_basic_functionality()
