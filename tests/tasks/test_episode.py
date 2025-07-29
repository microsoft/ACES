"""
Unit tests for Episode and related classes.

Tests episode data models, properties, serialization, and state management.
"""

import pytest
import json
from dataclasses import asdict
from datetime import datetime
from unittest.mock import Mock

from saber.server.tasks.episodes import Episode, Step, Action
from saber.server.tasks.episodes.episode_manager import EpisodeState


class TestAction:
    """Test cases for Action class."""

    def test_action_creation_minimal(self):
        """Test creating an action with minimal required fields."""
        action = Action(
            tool_name="test_tool",
            parameters={"param1": "value1"}
        )

        assert action.tool_name == "test_tool"
        assert action.parameters == {"param1": "value1"}
        assert isinstance(action.timestamp, datetime)
        assert action.command is None

    def test_action_creation_with_command(self):
        """Test creating an action with command specified."""
        action = Action(
            tool_name="docker_cli_executor",
            parameters={"command": "file sample.exe"},
            command="file sample.exe"
        )

        assert action.tool_name == "docker_cli_executor"
        assert action.command == "file sample.exe"

    def test_action_creation_with_custom_timestamp(self):
        """Test creating an action with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        action = Action(
            tool_name="test_tool",
            parameters={},
            timestamp=custom_time
        )

        assert action.timestamp == custom_time

    def test_action_serialization(self):
        """Test action serialization to dict and JSON."""
        action = Action(
            tool_name="test_tool",
            parameters={"param1": "value1"},
            command="test_command"
        )

        # Test dict conversion
        action_dict = action.dict()
        assert action_dict["tool_name"] == "test_tool"
        assert action_dict["parameters"] == {"param1": "value1"}
        assert action_dict["command"] == "test_command"
        assert "timestamp" in action_dict

        # Test JSON conversion
        action_json = action.json()
        assert isinstance(action_json, str)
        parsed = json.loads(action_json)
        assert parsed["tool_name"] == "test_tool"

    def test_action_pydantic_validation(self):
        """Test Pydantic validation for Action."""
        # Test that tool_name is required
        with pytest.raises(ValueError):
            Action()  # Missing required tool_name

        # Test valid action
        action = Action(tool_name="valid_tool")
        assert action.tool_name == "valid_tool"
        assert action.parameters == {}  # Default empty dict


class TestStep:
    """Test cases for Step class."""

    def test_step_creation_minimal(self, sample_action):
        """Test creating a step with minimal required fields."""
        step = Step(
            step_number=1,
            action=sample_action,
            response={"success": True, "data": "test_result"}
        )

        assert step.step_number == 1
        assert step.action == sample_action
        assert step.response == {"success": True, "data": "test_result"}
        assert isinstance(step.timestamp, datetime)
        assert step.current_subtask is None
        assert step.completed_subtasks == set()
        assert step.in_progress_subtasks == set()
        assert step.not_visited_subtasks == set()
        assert step.context_snapshot == {}
        assert step.done is False

    def test_step_creation_with_all_fields(self, sample_action):
        """Test creating a step with all fields specified."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)

        step = Step(
            step_number=5,
            timestamp=custom_time,
            action=sample_action,
            response={"success": True},
            current_subtask="active_subtask",
            completed_subtasks={"completed1", "completed2"},
            in_progress_subtasks={"in_progress1"},
            not_visited_subtasks={"not_visited1", "not_visited2"},
            context_snapshot={"context_key": "context_value"},
            done=True
        )

        assert step.step_number == 5
        assert step.timestamp == custom_time
        assert step.current_subtask == "active_subtask"
        assert step.completed_subtasks == {"completed1", "completed2"}
        assert step.in_progress_subtasks == {"in_progress1"}
        assert step.not_visited_subtasks == {"not_visited1", "not_visited2"}
        assert step.context_snapshot == {"context_key": "context_value"}
        assert step.done is True

    def test_step_serialization(self, sample_action):
        """Test step serialization to dict and JSON."""
        step = Step(
            step_number=1,
            action=sample_action,
            response={"success": True},
            completed_subtasks={"completed1"},
            in_progress_subtasks={"in_progress1"}
        )

        # Test dict conversion
        step_dict = step.dict()
        assert step_dict["step_number"] == 1
        assert step_dict["completed_subtasks"] == {"completed1"}
        assert step_dict["in_progress_subtasks"] == {"in_progress1"}

        # Test JSON conversion
        step_json = step.json()
        assert isinstance(step_json, str)
        parsed = json.loads(step_json)
        assert parsed["step_number"] == 1

    def test_step_pydantic_validation(self, sample_action):
        """Test Pydantic validation for Step."""
        # Test that required fields are enforced
        with pytest.raises(ValueError):
            Step()  # Missing required fields

        with pytest.raises(ValueError):
            Step(step_number=1)  # Missing action and response

        # Test valid step
        step = Step(
            step_number=1,
            action=sample_action,
            response={"success": True}
        )
        assert step.step_number == 1


class TestEpisode:
    """Test cases for Episode class."""

    def test_episode_creation_minimal(self):
        """Test creating an episode with minimal required fields."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        assert episode.task_id == "test_task"
        assert episode.session_id == "test_session"
        assert isinstance(episode.episode_id, str)
        assert len(episode.episode_id) > 0  # UUID should be generated
        assert isinstance(episode.start_time, datetime)
        assert episode.end_time is None
        assert episode.state == EpisodeState.CREATED
        assert episode.steps == []
        assert episode.context == {}
        assert episode.metadata == {}
        assert episode.completion_reason is None

    def test_episode_creation_with_all_fields(self):
        """Test creating an episode with all fields specified."""
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        episode = Episode(
            episode_id="custom_episode_id",
            task_id="test_task",
            session_id="test_session",
            start_time=start_time,
            end_time=end_time,
            state=EpisodeState.COMPLETED,
            context={"key": "value"},
            metadata={"meta": "data"},
            completion_reason="success"
        )

        assert episode.episode_id == "custom_episode_id"
        assert episode.start_time == start_time
        assert episode.end_time == end_time
        assert episode.state == EpisodeState.COMPLETED
        assert episode.context == {"key": "value"}
        assert episode.metadata == {"meta": "data"}
        assert episode.completion_reason == "success"

    def test_episode_current_subtask_property_no_steps(self):
        """Test current_subtask property when no steps exist."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        assert episode.current_subtask is None

    def test_episode_current_subtask_property_with_steps(self, sample_action):
        """Test current_subtask property when steps exist."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        step = Step(
            step_number=1,
            action=sample_action,
            response={"success": True},
            current_subtask="active_subtask"
        )

        episode.steps = [step]
        assert episode.current_subtask == "active_subtask"

    def test_episode_completed_subtasks_property(self, sample_action):
        """Test completed_subtasks property."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        # No steps
        assert episode.completed_subtasks == set()

        # With steps
        step = Step(
            step_number=1,
            action=sample_action,
            response={"success": True},
            completed_subtasks={"completed1", "completed2"}
        )

        episode.steps = [step]
        assert episode.completed_subtasks == {"completed1", "completed2"}

    def test_episode_in_progress_subtasks_property(self, sample_action):
        """Test in_progress_subtasks property."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        step = Step(
            step_number=1,
            action=sample_action,
            response={"success": True},
            in_progress_subtasks={"in_progress1", "in_progress2"}
        )

        episode.steps = [step]
        assert episode.in_progress_subtasks == {"in_progress1", "in_progress2"}

    def test_episode_not_visited_subtasks_property(self, sample_action):
        """Test not_visited_subtasks property."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        step = Step(
            step_number=1,
            action=sample_action,
            response={"success": True},
            not_visited_subtasks={"not_visited1", "not_visited2"}
        )

        episode.steps = [step]
        assert episode.not_visited_subtasks == {"not_visited1", "not_visited2"}

    def test_episode_is_complete_property(self):
        """Test is_complete property for different states."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        # Not complete states
        episode.state = EpisodeState.CREATED
        assert episode.is_complete is False

        episode.state = EpisodeState.ACTIVE
        assert episode.is_complete is False

        episode.state = EpisodeState.RESET
        assert episode.is_complete is False

        # Complete states
        episode.state = EpisodeState.COMPLETED
        assert episode.is_complete is True

        episode.state = EpisodeState.FAILED
        assert episode.is_complete is True

        episode.state = EpisodeState.TIMEOUT
        assert episode.is_complete is True

    def test_episode_duration_property_no_end_time(self):
        """Test duration property when episode hasn't ended."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        assert episode.duration is None

    def test_episode_duration_property_with_end_time(self):
        """Test duration property when episode has ended."""
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 10, 5, 30)  # 5 minutes 30 seconds later

        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            start_time=start_time,
            end_time=end_time
        )

        assert episode.duration == 330.0  # 5.5 minutes in seconds

    def test_episode_add_step(self, sample_action):
        """Test adding steps to an episode."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        step1 = Step(
            step_number=0,  # First step should be 0
            action=sample_action,
            response={"success": True}
        )

        episode.add_step(step1)

        assert len(episode.steps) == 1
        assert episode.steps[0] == step1
        assert step1.step_number == 0  # Should remain 0

        # Add another step
        step2 = Step(
            step_number=1,  # Second step should be 1
            action=sample_action,
            response={"success": True}
        )

        episode.add_step(step2)

        assert len(episode.steps) == 2
        assert episode.steps[1] == step2
        assert step2.step_number == 1  # Should remain 1

    def test_episode_get_executed_commands_no_steps(self):
        """Test getting executed commands when no steps exist."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        commands = episode.get_executed_commands()
        assert commands == []

    def test_episode_get_executed_commands_with_steps(self):
        """Test getting executed commands from steps."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )

        # Create actions with commands
        action1 = Action(
            tool_name="docker_cli_executor",
            parameters={"command": "file sample.exe"},
            command="file sample.exe"
        )

        action2 = Action(
            tool_name="docker_cli_executor",
            parameters={"command": "strings sample.exe"},
            command="strings sample.exe"
        )

        action3 = Action(
            tool_name="other_tool",
            parameters={}
            # No command attribute
        )

        step1 = Step(step_number=1, action=action1, response={})
        step2 = Step(step_number=2, action=action2, response={})
        step3 = Step(step_number=3, action=action3, response={})

        episode.steps = [step1, step2, step3]

        commands = episode.get_executed_commands()
        assert commands == ["file sample.exe", "strings sample.exe"]

    def test_episode_serialization(self):
        """Test episode serialization to dict and JSON."""
        episode = Episode(
            task_id="test_task",
            session_id="test_session",
            context={"key": "value"},
            metadata={"meta": "data"}
        )

        # Test dict conversion
        episode_dict = episode.dict()
        assert episode_dict["task_id"] == "test_task"
        assert episode_dict["session_id"] == "test_session"
        assert episode_dict["context"] == {"key": "value"}
        assert episode_dict["metadata"] == {"meta": "data"}
        assert "episode_id" in episode_dict
        assert "start_time" in episode_dict

        # Test JSON conversion
        episode_json = episode.json()
        assert isinstance(episode_json, str)
        parsed = json.loads(episode_json)
        assert parsed["task_id"] == "test_task"

    def test_episode_pydantic_validation(self):
        """Test Pydantic validation for Episode."""
        # Test that required fields are enforced
        with pytest.raises(ValueError):
            Episode()  # Missing required fields

        with pytest.raises(ValueError):
            Episode(task_id="test_task")  # Missing session_id

        # Test valid episode
        episode = Episode(
            task_id="test_task",
            session_id="test_session"
        )
        assert episode.task_id == "test_task"
        assert episode.session_id == "test_session"

    def test_episode_with_complex_steps_and_properties(self, sample_action, sample_command_result):
        """Test episode with complex step structure and property access."""
        episode = Episode(
            task_id="complex_task",
            session_id="complex_session"
        )

        # Add multiple steps with different subtask states
        step1 = Step(
            step_number=1,
            action=sample_action,
            response=asdict(sample_command_result),
            current_subtask="subtask1",
            completed_subtasks=set(),
            in_progress_subtasks={"subtask1"},
            not_visited_subtasks={"subtask2", "subtask3"}
        )

        step2 = Step(
            step_number=2,
            action=sample_action,
            response=asdict(sample_command_result),
            current_subtask="subtask2",
            completed_subtasks={"subtask1"},
            in_progress_subtasks={"subtask2"},
            not_visited_subtasks={"subtask3"}
        )

        episode.add_step(step1)
        episode.add_step(step2)

        # Test that properties return values from latest step
        assert episode.current_subtask == "subtask2"
        assert episode.completed_subtasks == {"subtask1"}
        assert episode.in_progress_subtasks == {"subtask2"}
        assert episode.not_visited_subtasks == {"subtask3"}

        # Test step numbering was set correctly
        assert episode.steps[0].step_number == 1
        assert episode.steps[1].step_number == 2
