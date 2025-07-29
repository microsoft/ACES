"""
Unit tests for SubTask.

Tests subtask creation, entry/exit conditions, command matching, and dependency logic.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from saber.server.tasks.core.subtask import SubTask
from saber.server.tasks.episodes.episode import Episode, Step, Action
from saber.server.tasks.episodes.episode_manager import EpisodeState


class TestSubTask:
    """Test cases for SubTask functionality."""

    def test_subtask_creation_with_all_fields(self, sample_subtask_data):
        """Test creating a subtask with all fields specified."""
        subtask = SubTask(**sample_subtask_data)

        assert subtask.subtask_id == "test_subtask"
        assert subtask.task_id == "test_task"
        assert subtask.title == "Test SubTask"
        assert subtask.description == "A test subtask for unit testing"
        assert subtask.objective == "Complete the test objectives"
        assert subtask.completion_conditions == ["test_command", "another_command"]
        assert subtask.depends_on == ["prerequisite_subtask"]

    def test_subtask_creation_minimal_fields(self):
        """Test creating a subtask with only required fields."""
        subtask = SubTask(
            subtask_id="minimal_subtask",
            task_id="minimal_task",
            title="Minimal SubTask",
            description="A minimal subtask",
            objective="Complete minimal objectives"
        )

        assert subtask.subtask_id == "minimal_subtask"
        assert subtask.task_id == "minimal_task"
        assert subtask.completion_conditions == []
        assert subtask.depends_on == []

    def test_check_entry_conditions_no_dependencies(self):
        """Test entry conditions when subtask has no dependencies."""
        subtask = SubTask(
            subtask_id="no_deps",
            task_id="test_task",
            title="No Dependencies",
            description="Subtask with no dependencies",
            objective="Complete without prerequisites",
            depends_on=[]
        )

        # Create mock episode
        episode = Mock()
        episode.completed_subtasks = set()

        # Should be able to start immediately
        assert subtask.check_entry_conditions(episode) is True

    def test_check_entry_conditions_with_satisfied_dependencies(self):
        """Test entry conditions when all dependencies are satisfied."""
        subtask = SubTask(
            subtask_id="with_deps",
            task_id="test_task",
            title="With Dependencies",
            description="Subtask with dependencies",
            objective="Complete after prerequisites",
            depends_on=["dep1", "dep2"]
        )

        # Create mock episode with completed dependencies
        episode = Mock()
        episode.completed_subtasks = {"dep1", "dep2", "other_subtask"}

        assert subtask.check_entry_conditions(episode) is True

    def test_check_entry_conditions_with_unsatisfied_dependencies(self):
        """Test entry conditions when dependencies are not satisfied."""
        subtask = SubTask(
            subtask_id="with_deps",
            task_id="test_task",
            title="With Dependencies",
            description="Subtask with dependencies",
            objective="Complete after prerequisites",
            depends_on=["dep1", "dep2"]
        )

        # Create mock episode with only partial dependencies completed
        episode = Mock()
        episode.completed_subtasks = {"dep1"}  # Missing dep2

        assert subtask.check_entry_conditions(episode) is False

    def test_check_exit_conditions_no_completion_conditions(self):
        """Test exit conditions when no completion conditions are specified."""
        subtask = SubTask(
            subtask_id="no_conditions",
            task_id="test_task",
            title="No Conditions",
            description="Subtask with no completion conditions",
            objective="Always complete",
            completion_conditions=[]
        )

        # Create mock episode
        episode = Mock()
        episode.steps = []

        # Should always be complete when no conditions specified
        assert subtask.check_exit_conditions(episode) is True

    def test_check_exit_conditions_with_satisfied_commands(self):
        """Test exit conditions when all required commands were executed."""
        subtask = SubTask(
            subtask_id="with_conditions",
            task_id="test_task",
            title="With Conditions",
            description="Subtask with completion conditions",
            objective="Complete after commands",
            completion_conditions=["file sample.exe", "strings sample.exe"]
        )

        # Create episode with executed commands
        steps = [
            Mock(action=Mock(command="file sample.exe")),
            Mock(action=Mock(command="strings sample.exe")),
            Mock(action=Mock(command="other_command"))
        ]
        episode = Mock()
        episode.steps = steps

        assert subtask.check_exit_conditions(episode) is True

    def test_check_exit_conditions_with_missing_commands(self):
        """Test exit conditions when some required commands were not executed."""
        subtask = SubTask(
            subtask_id="with_conditions",
            task_id="test_task",
            title="With Conditions",
            description="Subtask with completion conditions",
            objective="Complete after commands",
            completion_conditions=["file sample.exe", "strings sample.exe", "hexdump sample.exe"]
        )

        # Create episode with only some executed commands
        steps = [
            Mock(action=Mock(command="file sample.exe")),
            Mock(action=Mock(command="other_command"))
        ]
        episode = Mock()
        episode.steps = steps

        assert subtask.check_exit_conditions(episode) is False

    def test_command_was_executed_direct_match(self):
        """Test direct command matching."""
        subtask = SubTask(
            subtask_id="test",
            task_id="test_task",
            title="Test",
            description="Test",
            objective="Test"
        )

        executed_commands = ["file sample.exe", "strings sample.exe"]

        assert subtask._command_was_executed("file sample.exe", executed_commands) is True
        assert subtask._command_was_executed("strings sample.exe", executed_commands) is True
        assert subtask._command_was_executed("hexdump sample.exe", executed_commands) is False

    def test_command_was_executed_template_variables(self):
        """Test command matching with template variables."""
        subtask = SubTask(
            subtask_id="test",
            task_id="test_task",
            title="Test",
            description="Test",
            objective="Test"
        )

        executed_commands = ["file /data/sample.exe", "strings /path/to/malware.bin"]

        # Template variables should match
        assert subtask._command_was_executed("file ${sample_path}", executed_commands) is True
        assert subtask._command_was_executed("strings ${sample_path}", executed_commands) is True
        assert subtask._command_was_executed("hexdump ${sample_path}", executed_commands) is False

    def test_command_was_executed_partial_matching(self):
        """Test partial command matching (base command)."""
        subtask = SubTask(
            subtask_id="test",
            task_id="test_task",
            title="Test",
            description="Test",
            objective="Test"
        )

        executed_commands = ["file -v /data/sample.exe", "strings -a malware.bin"]

        # Base command should match even with different arguments
        assert subtask._command_was_executed("file sample.exe", executed_commands) is True
        assert subtask._command_was_executed("strings malware", executed_commands) is True
        assert subtask._command_was_executed("hexdump sample", executed_commands) is False

    def test_command_was_executed_command_in_middle(self):
        """Test command matching when base command appears in middle of executed command."""
        subtask = SubTask(
            subtask_id="test",
            task_id="test_task",
            title="Test",
            description="Test",
            objective="Test"
        )

        executed_commands = ["docker run --rm -v /data:/data file /data/sample.exe"]

        # Should find 'file' even when it's not the first command
        assert subtask._command_was_executed("file sample.exe", executed_commands) is True

    def test_command_was_executed_empty_commands(self):
        """Test command matching with empty command lists."""
        subtask = SubTask(
            subtask_id="test",
            task_id="test_task",
            title="Test",
            description="Test",
            objective="Test"
        )

        # Empty executed commands
        assert subtask._command_was_executed("file sample.exe", []) is False

        # Executed commands with None/empty values
        executed_commands = [None, "", "file sample.exe"]
        assert subtask._command_was_executed("file sample.exe", executed_commands) is True

    def test_command_was_executed_complex_template(self):
        """Test command matching with complex template variables."""
        subtask = SubTask(
            subtask_id="test",
            task_id="test_task",
            title="Test",
            description="Test",
            objective="Test"
        )

        executed_commands = ["python3 -c 'import analysis; analysis.run(\"/data/sample.exe\")'"]

        # Template variable in middle of command
        assert subtask._command_was_executed("python3 ${analysis_script}", executed_commands) is True

    def test_is_dependent_on_true(self):
        """Test dependency checking when subtask depends on another."""
        subtask = SubTask(
            subtask_id="dependent",
            task_id="test_task",
            title="Dependent SubTask",
            description="Depends on other subtasks",
            objective="Complete after dependencies",
            depends_on=["prerequisite1", "prerequisite2"]
        )

        assert subtask.is_dependent_on("prerequisite1") is True
        assert subtask.is_dependent_on("prerequisite2") is True

    def test_is_dependent_on_false(self):
        """Test dependency checking when subtask doesn't depend on another."""
        subtask = SubTask(
            subtask_id="independent",
            task_id="test_task",
            title="Independent SubTask",
            description="No dependencies",
            objective="Complete independently",
            depends_on=["prerequisite1"]
        )

        assert subtask.is_dependent_on("prerequisite2") is False
        assert subtask.is_dependent_on("nonexistent") is False

    def test_is_dependent_on_no_dependencies(self):
        """Test dependency checking when subtask has no dependencies."""
        subtask = SubTask(
            subtask_id="no_deps",
            task_id="test_task",
            title="No Dependencies",
            description="Independent subtask",
            objective="Complete independently",
            depends_on=[]
        )

        assert subtask.is_dependent_on("any_subtask") is False

    def test_exit_conditions_with_real_episode_structure(self):
        """Test exit conditions using real Episode structure."""
        subtask = SubTask(
            subtask_id="real_test",
            task_id="test_task",
            title="Real Test",
            description="Test with real episode",
            objective="Complete with real structure",
            completion_conditions=["file sample.exe", "strings sample.exe"]
        )

        # Create real episode with real steps
        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        # Add steps with actions containing commands
        action1 = Action(tool_name="docker_cli_executor", parameters={"command": "file sample.exe"})
        action1.command = "file sample.exe"
        step1 = Step(step_number=1, action=action1, response={"success": True})

        action2 = Action(tool_name="docker_cli_executor", parameters={"command": "strings sample.exe"})
        action2.command = "strings sample.exe"
        step2 = Step(step_number=2, action=action2, response={"success": True})

        episode.steps = [step1, step2]

        assert subtask.check_exit_conditions(episode) is True

    def test_exit_conditions_handles_missing_command_attribute(self):
        """Test exit conditions handling when action doesn't have command attribute."""
        subtask = SubTask(
            subtask_id="robust_test",
            task_id="test_task",
            title="Robust Test",
            description="Test robustness",
            objective="Handle missing attributes",
            completion_conditions=["file sample.exe"]
        )

        # Create episode with steps that don't have command attribute
        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            state=EpisodeState.ACTIVE
        )

        action_without_command = Action(tool_name="other_tool", parameters={})
        # Don't set command attribute
        step = Step(step_number=1, action=action_without_command, response={"success": True})

        episode.steps = [step]

        # Should not crash and should return False (condition not met)
        assert subtask.check_exit_conditions(episode) is False

    def test_pydantic_validation(self):
        """Test Pydantic validation of SubTask fields."""
        # Test that required fields are enforced
        with pytest.raises(ValueError):
            SubTask()  # Missing required fields

        # Test that valid subtask can be created
        subtask = SubTask(
            subtask_id="valid",
            task_id="valid_task",
            title="Valid SubTask",
            description="Valid description",
            objective="Valid objective"
        )

        assert subtask.subtask_id == "valid"
        assert isinstance(subtask.completion_conditions, list)
        assert isinstance(subtask.depends_on, list)

    def test_subtask_serialization(self, sample_subtask_data):
        """Test that SubTask can be serialized/deserialized."""
        subtask = SubTask(**sample_subtask_data)

        # Test dict conversion
        subtask_dict = subtask.dict()
        assert subtask_dict["subtask_id"] == "test_subtask"
        assert subtask_dict["completion_conditions"] == ["test_command", "another_command"]

        # Test JSON conversion
        subtask_json = subtask.json()
        assert isinstance(subtask_json, str)
        assert "test_subtask" in subtask_json

        # Test reconstruction from dict
        new_subtask = SubTask(**subtask_dict)
        assert new_subtask.subtask_id == subtask.subtask_id
        assert new_subtask.completion_conditions == subtask.completion_conditions
