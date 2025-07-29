"""
Unit tests for Task.

Tests task creation, subtask management, episode progression, and DAG logic.
"""

import pytest
from unittest.mock import Mock, patch

from saber.server.tasks.core.task import Task
from saber.server.tasks.core.subtask import SubTask
from saber.server.tasks.episodes import Episode, Step, Action
from saber.server.tasks.episodes.episode_manager import EpisodeState


class TestTask:
    """Test cases for Task functionality."""

    def test_task_creation_minimal(self):
        """Test creating a task with minimal required fields."""
        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task for unit testing"
        )

        assert task.task_id == "test_task"
        assert task.domain == "test_domain"
        assert task.title == "Test Task"
        assert task.description == "A test task for unit testing"
        assert task.subtasks == []
        assert task.initial_context == {}
        assert task._subtask_map == {}

    def test_task_creation_with_all_fields(self):
        """Test creating a task with all fields specified."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="SubTask 1",
            description="First subtask",
            objective="Complete first objective"
        )

        initial_context = {"sample_path": "/data/test.exe", "timeout": 300}

        task = Task(
            task_id="test_task",
            domain="malware_classification",
            title="Test Task",
            description="A comprehensive test task",
            subtasks=[subtask1],
            initial_context=initial_context
        )

        assert len(task.subtasks) == 1
        assert task.initial_context == initial_context
        assert "subtask1" in task._subtask_map
        assert task._subtask_map["subtask1"] == subtask1

    def test_add_subtask(self):
        """Test adding a subtask to a task."""
        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task"
        )

        subtask = SubTask(
            subtask_id="new_subtask",
            task_id="different_task",  # Will be overridden
            title="New SubTask",
            description="A new subtask",
            objective="Complete new objective"
        )

        task.add_subtask(subtask)

        assert len(task.subtasks) == 1
        assert task.subtasks[0] == subtask
        assert subtask.task_id == "test_task"  # Should be updated
        assert "new_subtask" in task._subtask_map

    def test_get_subtask_by_id_success(self):
        """Test successful subtask retrieval by ID."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="SubTask 1",
            description="First subtask",
            objective="Complete first objective"
        )

        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="SubTask 2",
            description="Second subtask",
            objective="Complete second objective"
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        assert task.get_subtask_by_id("subtask1") == subtask1
        assert task.get_subtask_by_id("subtask2") == subtask2

    def test_get_subtask_by_id_not_found(self):
        """Test subtask retrieval when ID doesn't exist."""
        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task"
        )

        assert task.get_subtask_by_id("nonexistent") is None

    def test_initialize_episode_no_dependencies(self):
        """Test episode initialization with subtasks that have no dependencies."""
        subtask1 = SubTask(
            subtask_id="entry1",
            task_id="test_task",
            title="Entry Point 1",
            description="First entry point",
            objective="Start here",
            depends_on=[]
        )

        subtask2 = SubTask(
            subtask_id="entry2",
            task_id="test_task",
            title="Entry Point 2",
            description="Second entry point",
            objective="Also start here",
            depends_on=[]
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        task.initialize_episode(episode)

        assert len(episode.steps) == 1
        initial_step = episode.steps[0]
        assert initial_step.step_number == 0
        assert initial_step.action.tool_name == "system_init"
        assert initial_step.in_progress_subtasks == {"entry1", "entry2"}
        assert initial_step.not_visited_subtasks == set()
        assert initial_step.completed_subtasks == set()

    def test_initialize_episode_with_dependencies(self):
        """Test episode initialization with subtask dependencies."""
        subtask1 = SubTask(
            subtask_id="init",
            task_id="test_task",
            title="Initialization",
            description="Initialize task",
            objective="Setup",
            depends_on=[]
        )

        subtask2 = SubTask(
            subtask_id="dependent",
            task_id="test_task",
            title="Dependent Task",
            description="Depends on initialization",
            objective="Complete after init",
            depends_on=["init"]
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        task.initialize_episode(episode)

        initial_step = episode.steps[0]
        assert initial_step.in_progress_subtasks == {"init"}
        assert initial_step.not_visited_subtasks == {"dependent"}

    def test_initialize_episode_existing_steps(self):
        """Test episode initialization when episode already has steps."""
        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[]
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Add existing step
        existing_action = Action(tool_name="existing", parameters={})
        existing_step = Step(step_number=1, action=existing_action, response={})
        episode.steps = [existing_step]

        task.initialize_episode(episode)

        # Should not add another initialization step
        assert len(episode.steps) == 1
        assert episode.steps[0] == existing_step

    def test_check_episode_progression_completion(self):
        """Test episode progression when subtasks complete."""
        # Create subtask with completion conditions
        subtask = SubTask(
            subtask_id="completable",
            task_id="test_task",
            title="Completable Task",
            description="Can be completed",
            objective="Execute commands",
            completion_conditions=["test_command"],
            depends_on=[]
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask]
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Initialize episode
        task.initialize_episode(episode)

        # The subtask should be in progress since it has no dependencies
        assert "completable" in episode.in_progress_subtasks

        # Create step with command that completes the subtask
        action = Action(
            tool_name="docker_cli_executor",
            parameters={"command": "test_command"},
            command="test_command"
        )

        # Get the current subtask state from the latest step
        latest_step = episode.steps[-1] if episode.steps else None
        current_subtask = latest_step.current_subtask if latest_step else None
        completed = latest_step.completed_subtasks.copy() if latest_step else set()
        in_progress = latest_step.in_progress_subtasks.copy() if latest_step else set()
        not_visited = latest_step.not_visited_subtasks.copy() if latest_step else set()

        step = Step(
            step_number=len(episode.steps) + 1,
            action=action,
            response={"success": True},
            current_subtask=current_subtask,
            completed_subtasks=completed,
            in_progress_subtasks=in_progress,
            not_visited_subtasks=not_visited
        )

        # Add step to episode before checking progression
        episode.steps.append(step)

        # Mock the SubTask.check_exit_conditions method to return True
        with patch.object(SubTask, 'check_exit_conditions', return_value=True):
            task.check_episode_progression(episode, step)

        assert "completable" in step.completed_subtasks

    def test_check_episode_progression_new_available(self):
        """Test episode progression when new subtasks become available."""
        subtask1 = SubTask(
            subtask_id="prerequisite",
            task_id="test_task",
            title="Prerequisite",
            description="Must complete first",
            objective="Complete prerequisite",
            depends_on=[]
        )

        subtask2 = SubTask(
            subtask_id="dependent",
            task_id="test_task",
            title="Dependent",
            description="Depends on prerequisite",
            objective="Complete after prerequisite",
            depends_on=["prerequisite"]
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Initialize episode
        task.initialize_episode(episode)

        # Create step with proper subtask state initialization
        action = Action(tool_name="test", parameters={})
        latest_step = episode.steps[-1] if episode.steps else None
        current_subtask = latest_step.current_subtask if latest_step else None
        completed = latest_step.completed_subtasks.copy() if latest_step else set()
        in_progress = latest_step.in_progress_subtasks.copy() if latest_step else set()
        not_visited = latest_step.not_visited_subtasks.copy() if latest_step else set()

        step = Step(
            step_number=len(episode.steps) + 1,
            action=action,
            response={},
            current_subtask=current_subtask,
            completed_subtasks=completed,
            in_progress_subtasks=in_progress,
            not_visited_subtasks=not_visited
        )
        episode.steps.append(step)

        # Mock the SubTask.check_entry_conditions method to return True for the dependent subtask
        def mock_check_entry_conditions(self, episode):
            return self.subtask_id == "dependent"

        with patch.object(SubTask, 'check_entry_conditions', mock_check_entry_conditions):
            task.check_episode_progression(episode, step)

        # dependent should now be in progress and removed from not_visited
        assert "dependent" in step.in_progress_subtasks
        assert "dependent" not in step.not_visited_subtasks

    def test_update_step_subtask_state_first_step(self):
        """Test updating step state when it's the first step."""
        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task"
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Create initial step to establish episode state
        initial_step = Step(
            step_number=0,
            action=Action(tool_name="init", parameters={}),
            response={},
            completed_subtasks={"completed1"},
            in_progress_subtasks={"in_progress1"},
            not_visited_subtasks={"not_visited1"},
            current_subtask="current1"
        )
        episode.steps.append(initial_step)

        action = Action(tool_name="test", parameters={})
        step = Step(step_number=1, action=action, response={})

        task._update_step_subtask_state(episode, step)

        # Should copy from episode (which gets values from latest step)
        assert step.completed_subtasks == {"completed1"}
        assert step.in_progress_subtasks == {"in_progress1"}
        assert step.not_visited_subtasks == {"not_visited1"}
        assert step.current_subtask == "current1"

    def test_update_step_subtask_state_subsequent_step(self):
        """Test updating step state when previous steps exist."""
        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task"
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Add previous step
        previous_action = Action(tool_name="previous", parameters={})
        previous_step = Step(
            step_number=1,
            action=previous_action,
            response={},
            completed_subtasks={"completed1"},
            in_progress_subtasks={"in_progress1"},
            not_visited_subtasks={"not_visited1"},
            current_subtask="current1"
        )
        episode.steps = [previous_step]

        action = Action(tool_name="test", parameters={})
        step = Step(step_number=2, action=action, response={})

        task._update_step_subtask_state(episode, step)

        # Should copy from previous step
        assert step.completed_subtasks == {"completed1"}
        assert step.in_progress_subtasks == {"in_progress1"}
        assert step.not_visited_subtasks == {"not_visited1"}
        assert step.current_subtask == "current1"

    def test_check_completed_subtasks(self):
        """Test checking which subtasks have completed."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="SubTask 1",
            description="First subtask",
            objective="Complete first"
        )

        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="SubTask 2",
            description="Second subtask",
            objective="Complete second"
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Set up episode state by creating a step
        initial_step = Step(
            step_number=0,
            action=Action(tool_name="init", parameters={}),
            response={},
            completed_subtasks={"already_completed"},
            in_progress_subtasks={"subtask1", "subtask2"}
        )
        episode.steps.append(initial_step)

        # Mock exit conditions for different subtasks
        def mock_check_exit_conditions(self, episode):
            if self.subtask_id == "subtask1":
                return True
            elif self.subtask_id == "subtask2":
                return False
            return False

        with patch.object(SubTask, 'check_exit_conditions', mock_check_exit_conditions):
            completed = task._check_completed_subtasks(episode)

        assert completed == {"already_completed", "subtask1"}

    def test_check_available_subtasks(self):
        """Test checking which subtasks are now available."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="SubTask 1",
            description="First subtask",
            objective="Complete first"
        )

        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="SubTask 2",
            description="Second subtask",
            objective="Complete second"
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Set up episode state with initial step
        from saber.server.tasks.base import Step, Action
        initial_step = Step(
            step_number=1,
            action=Action(tool_name="init", parameters={"content": "Episode started"}),
            response={"status": "started"},
            not_visited_subtasks={"subtask1", "subtask2"},
            current_subtask=None
        )
        episode.steps = [initial_step]

        # Mock entry conditions
        original_check = SubTask.check_entry_conditions

        def mock_check_entry_conditions(self, episode):
            if self.subtask_id == "subtask1":
                return True
            elif self.subtask_id == "subtask2":
                return False
            return False

        with patch.object(SubTask, 'check_entry_conditions', mock_check_entry_conditions):
            available = task._check_available_subtasks(episode)

        assert available == {"subtask1"}

    def test_get_current_subtask_success(self):
        """Test getting current subtask when one is active."""
        subtask = SubTask(
            subtask_id="current_subtask",
            task_id="test_task",
            title="Current SubTask",
            description="Currently active",
            objective="Active objective"
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask]
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Set up episode state with step containing current subtask
        from saber.server.tasks.base import Step, Action
        step = Step(
            step_number=1,
            action=Action(tool_name="start", parameters={"content": "Started subtask"}),
            response={"status": "in_progress"},
            current_subtask="current_subtask",
            in_progress_subtasks={"current_subtask"}
        )
        episode.steps = [step]

        current = task.get_current_subtask(episode)
        assert current == subtask

    def test_get_current_subtask_none(self):
        """Test getting current subtask when none is active."""
        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task"
        )

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Set up episode state with no current subtask (empty steps or non-active steps)
        episode.steps = []

        current = task.get_current_subtask(episode)
        assert current is None

    def test_is_complete_all_completed(self):
        """Test task completion when all subtasks are completed."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="SubTask 1",
            description="First subtask",
            objective="Complete first"
        )

        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="SubTask 2",
            description="Second subtask",
            objective="Complete second"
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        completed_subtasks = {"subtask1", "subtask2"}
        assert task.is_complete(completed_subtasks) is True

    def test_is_complete_partial_completion(self):
        """Test task completion when only some subtasks are completed."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="SubTask 1",
            description="First subtask",
            objective="Complete first"
        )

        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="SubTask 2",
            description="Second subtask",
            objective="Complete second"
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        completed_subtasks = {"subtask1"}
        assert task.is_complete(completed_subtasks) is False

    def test_is_complete_no_subtasks(self):
        """Test task completion when task has no subtasks."""
        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[]
        )

        completed_subtasks = set()
        assert task.is_complete(completed_subtasks) is True

    def test_validate_dependencies_valid(self):
        """Test dependency validation with valid dependencies."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="SubTask 1",
            description="First subtask",
            objective="Complete first",
            depends_on=[]
        )

        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="SubTask 2",
            description="Second subtask",
            objective="Complete second",
            depends_on=["subtask1"]
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        errors = task.validate_dependencies()
        assert errors == []

    def test_validate_dependencies_invalid(self):
        """Test dependency validation with invalid dependencies."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="SubTask 1",
            description="First subtask",
            objective="Complete first",
            depends_on=["nonexistent"]
        )

        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="SubTask 2",
            description="Second subtask",
            objective="Complete second",
            depends_on=["subtask1", "another_nonexistent"]
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        errors = task.validate_dependencies()
        assert len(errors) == 2
        assert any("nonexistent" in error for error in errors)
        assert any("another_nonexistent" in error for error in errors)

    def test_get_all_subtask_ids(self):
        """Test getting all subtask IDs."""
        subtask1 = SubTask(
            subtask_id="subtask1",
            task_id="test_task",
            title="SubTask 1",
            description="First subtask",
            objective="Complete first"
        )

        subtask2 = SubTask(
            subtask_id="subtask2",
            task_id="test_task",
            title="SubTask 2",
            description="Second subtask",
            objective="Complete second"
        )

        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[subtask1, subtask2]
        )

        all_ids = task.get_all_subtask_ids()
        assert all_ids == {"subtask1", "subtask2"}

    def test_get_all_subtask_ids_empty(self):
        """Test getting all subtask IDs when no subtasks exist."""
        task = Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="A test task",
            subtasks=[]
        )

        all_ids = task.get_all_subtask_ids()
        assert all_ids == set()
