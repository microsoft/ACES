"""
Unit tests for Episode class.

Tests meaningful episode functionality - pruned basic data structure tests.
"""

from datetime import datetime

import pytest

from saber.server.base import Action, Episode, EpisodeState, Step


class TestEpisode:
    """Test cases for Episode functionality."""

    def test_episode_add_step(self):
        """Test adding steps to an episode."""
        episode = Episode(task_id="test_task", session_id="test_session")

        # Add first step
        action1 = Action(tool_name="shell", parameters={"arguments": "ls -la"})
        step1 = Step(step_number=1, action=action1, response={"output": "file listing output"}, done=False)

        episode.add_step(step1)
        assert len(episode.steps) == 1
        assert episode.steps[0].step_number == 1
        assert episode.steps[0].action.parameters["arguments"] == "ls -la"

        # Add second step
        action2 = Action(tool_name="shell", parameters={"arguments": "cat file.txt"})
        step2 = Step(step_number=2, action=action2, response={"output": "file contents"}, done=True)

        episode.add_step(step2)
        assert len(episode.steps) == 2
        assert episode.steps[1].step_number == 2
        assert episode.steps[1].done is True

    def test_episode_properties(self):
        """Test episode properties."""
        episode = Episode(task_id="test_task", session_id="test_session")

        # Initially created state
        assert episode.state == EpisodeState.CREATED
        assert episode.end_time is None
        assert not episode.is_complete

        # Check duration when not complete
        assert episode.duration is None

        # Manually change state to active for testing
        episode.state = EpisodeState.ACTIVE

        # Add a step
        action = Action(tool_name="shell", parameters={"arguments": "pwd"})
        step = Step(step_number=1, action=action, response={"output": "/home/user"}, done=False)

        episode.add_step(step)
        assert len(episode.steps) == 1
        assert episode.steps[0].step_number == 1
        assert episode.steps[0].action.parameters["arguments"] == "pwd"

    def test_episode_state_transitions(self):
        """Test episode state transitions."""
        episode = Episode(task_id="test_task", session_id="test_session")

        # Initially created
        assert episode.state == EpisodeState.CREATED
        assert episode.end_time is None

        # Manually transition to completed for testing
        episode.state = EpisodeState.COMPLETED
        episode.end_time = datetime.utcnow()
        assert episode.state == EpisodeState.COMPLETED
        assert episode.end_time is not None
        assert episode.is_complete

        # Test duration calculation
        duration = episode.duration
        assert duration is not None
        assert duration >= 0

    def test_episode_commands(self):
        """Test getting executed commands."""
        episode = Episode(task_id="test_task", session_id="test_session")

        # Add steps with commands
        action1 = Action(tool_name="shell", parameters={"arguments": "ls -la"})
        step1 = Step(step_number=1, action=action1, response={"output": "listing"}, done=False)
        episode.add_step(step1)

        action2 = Action(tool_name="shell", parameters={"arguments": "pwd"})
        step2 = Step(step_number=2, action=action2, response={"output": "/home"}, done=False)
        episode.add_step(step2)

        # Test command extraction
        commands = episode.get_executed_commands()
        assert "ls -la" in commands
        assert "pwd" in commands

    def test_episode_str_representation(self):
        """Test string representation of episode."""
        episode = Episode(task_id="test_task", session_id="test_session")

        str_repr = str(episode)
        assert "test_task" in str_repr
        assert "test_session" in str_repr
