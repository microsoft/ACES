"""
Unit tests for EpisodeManager.

Tests episode lifecycle management, RL interfaces, and state transitions.
Simplified after removing subtask progression tracking.
"""

from datetime import datetime

import pytest

from saber.server.base import Action, CommandResult, Episode, EpisodeState, Step
from saber.server.episodes.episode_manager import EpisodeManager
from saber.server.episodes.exceptions import EpisodeNotFoundException


class TestEpisodeManager:
    """Test cases for EpisodeManager functionality."""

    def test_start_episode_minimal(self):
        """Test starting an episode with minimal parameters."""
        manager = EpisodeManager()

        episode = manager.start_episode(session_id="test_session", task_id="test_task")

        assert isinstance(episode, Episode)
        assert episode.task_id == "test_task"
        assert episode.session_id == "test_session"
        assert episode.state == EpisodeState.ACTIVE
        assert episode.context == {}
        # Check episode is stored properly
        assert episode.episode_id in manager.episodes
        assert manager.episodes[episode.episode_id] == episode
        # Check session mapping
        assert "test_session" in manager.session_episodes
        assert episode.episode_id in manager.session_episodes["test_session"]

    def test_start_episode_with_context(self):
        """Test starting an episode with initial context."""
        manager = EpisodeManager()
        initial_context = {"sample_path": "/data/test.exe", "timeout": 300}

        episode = manager.start_episode(session_id="test_session", task_id="test_task", initial_context=initial_context)

        assert episode.context == initial_context

    def test_start_episode_replaces_existing(self):
        """Test that starting a new episode doesn't replace existing episodes (multi-episode support)."""
        manager = EpisodeManager()

        # Start first episode
        episode1 = manager.start_episode("test_session", "task1")
        episode1_id = episode1.episode_id

        # Start second episode (should be added alongside first, not replace)
        episode2 = manager.start_episode("test_session", "task2")
        episode2_id = episode2.episode_id

        assert episode1_id != episode2_id
        # Both episodes should exist for the session
        assert episode1_id in manager.session_episodes["test_session"]
        assert episode2_id in manager.session_episodes["test_session"]
        assert len(manager.session_episodes["test_session"]) == 2
        assert episode2.task_id == "task2"

    def test_get_current_episode_exists(self):
        """Test getting an existing episode by ID."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        retrieved = manager.get_episode_by_id(episode.episode_id)
        assert retrieved == episode

    def test_step_basic(self):
        """Test basic step functionality."""
        manager = EpisodeManager()

        episode = manager.start_episode("test_session", "test_task")

        # Create mock task with episode config
        from unittest.mock import MagicMock
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.episode_config = {"max_steps": 10}

        # Configure episode manager for task using episode_id
        manager.configure_for_task(episode.episode_id, mock_task)

        action = Action(tool_name="test_tool", parameters={"key": "value"})
        command_result = CommandResult(
            exit_code=0,
            stdout="Test output",
            stderr="",
            execution_time=0.1,
            metadata={"output": "Test output", "file_type": "PE32"},
        )

        step_result = manager.step(episode.episode_id, action, command_result)

        assert hasattr(step_result, 'step')
        assert hasattr(step_result, 'should_terminate')
        assert hasattr(step_result, 'termination_reason')

        step = step_result.step
        assert isinstance(step, Step)
        assert step.action == action
        assert step.response["exit_code"] == 0  # Success is exit_code == 0
        assert step.response["metadata"]["output"] == "Test output"
        assert step.step_number == 0
        assert step.done is False

        # Verify step was added to episode
        assert len(episode.steps) == 1
        assert episode.steps[0] == step

    def test_step_session_not_found(self):
        """Test step with non-existent episode."""
        manager = EpisodeManager()
        action = Action(tool_name="test_tool", parameters={})
        command_result = CommandResult(exit_code=0, stdout="", stderr="", execution_time=0.1)

        with pytest.raises(EpisodeNotFoundException):
            manager.step("nonexistent_episode_id", action, command_result)

    def test_step_increments_step_number(self):
        """Test that step numbers increment correctly."""
        manager = EpisodeManager()

        episode = manager.start_episode("test_session", "test_task")

        # Create mock task with episode config
        from unittest.mock import MagicMock
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.episode_config = {"max_steps": 10}

        # Configure episode manager for task using episode_id
        manager.configure_for_task(episode.episode_id, mock_task)

        action = Action(tool_name="test_tool", parameters={})
        command_result = CommandResult(exit_code=0, stdout="", stderr="", execution_time=0.1)

        step_result1 = manager.step(episode.episode_id, action, command_result)
        step_result2 = manager.step(episode.episode_id, action, command_result)
        step_result3 = manager.step(episode.episode_id, action, command_result)

        assert step_result1.step.step_number == 0
        assert step_result2.step.step_number == 1
        assert step_result3.step.step_number == 2

    @pytest.mark.asyncio
    async def test_end_episode_success(self):
        """Test successfully ending an episode."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task", {"initial": "context"})

        ended_episode = await manager.end_episode(episode.episode_id, "task completed successfully")

        assert ended_episode == episode
        assert episode.state == EpisodeState.COMPLETED
        assert episode.completion_reason == "task completed successfully"
        assert episode.end_time is not None
        # Episode should be moved from active to completed
        assert episode.episode_id not in manager.episodes
        assert episode.episode_id in manager.completed_episodes

    @pytest.mark.asyncio
    async def test_end_episode_not_found(self):
        """Test ending a non-existent episode."""
        manager = EpisodeManager()

        with pytest.raises(EpisodeNotFoundException):
            await manager.end_episode("nonexistent_session", "reason")

    # NOTE: reset_episode method was removed in episode-first refactor
    # Multiple episodes per session makes "reset" concept obsolete
    # Clients should end current episode and start new one if needed

    def test_get_episode_state_exists(self):
        """Test getting episode state when episode exists."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        state = manager.get_episode_state(episode.episode_id)
        assert state == EpisodeState.ACTIVE

    def test_get_episode_state_not_exists(self):
        """Test getting episode state when episode doesn't exist."""
        manager = EpisodeManager()

        state = manager.get_episode_state("nonexistent_episode_id")
        assert state is None

    def test_cleanup_session_with_episode(self):
        """Test cleaning up a session that has an active episode."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        manager.cleanup_session("test_session")

        # Episode should be moved to completed
        assert episode.episode_id not in manager.episodes
        assert "test_session" not in manager.session_episodes

    def test_cleanup_session_no_episode(self):
        """Test cleaning up a session that has no active episode."""
        manager = EpisodeManager()

        # Should not raise an error
        manager.cleanup_session("nonexistent_session")

        assert manager.episodes == {}

    def test_create_step_basic(self):
        """Test basic step creation."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        action = Action(tool_name="test_tool", parameters={"key": "value"})
        response = CommandResult(exit_code=0, stdout="test output", stderr="", execution_time=0.1)

        step = manager.create_step(episode, action, response)

        assert isinstance(step, Step)
        assert step.action == action
        assert step.response["exit_code"] == 0  # Success is exit_code == 0
        assert step.response["stdout"] == "test output"
        assert step.step_number == 0
        assert step.done is False

    def test_update_episode_state_basic(self):
        """Test basic episode state update."""
        manager = EpisodeManager()
        episode = manager.start_episode("test_session", "test_task")

        action = Action(tool_name="test_tool", parameters={})
        step = Step(step_number=1, action=action, response={"success": True})

        manager.update_episode_state(episode, step)

        # Should not crash and episode should remain active
        assert episode.state == EpisodeState.ACTIVE

    @pytest.mark.asyncio
    async def test_multi_episode_orchestration_single_session(self):
        """Test orchestrating multiple concurrent episodes within a single session."""
        manager = EpisodeManager()
        session_id = "multi_episode_session"

        # Define multiple task scenarios
        task_configs = [
            ("reconnaissance_task", {"target": "example.com", "timeout": 300}),
            ("vulnerability_scan_task", {"scan_type": "web", "depth": "full"}),
            ("exploitation_task", {"payload": "reverse_shell", "port": 4444}),
            ("post_exploit_task", {"extract": "credentials", "lateral": True}),
        ]

        # Start multiple episodes concurrently
        episodes = []
        for task_id, context in task_configs:
            episode = manager.start_episode(session_id, task_id, context)
            episodes.append(episode)

            # Verify episode isolation
            assert episode.session_id == session_id
            assert episode.task_id == task_id
            assert episode.context == context
            assert episode.state == EpisodeState.ACTIVE

        # Verify all episodes are tracked under the same session
        assert len(episodes) == 4
        assert len(set(ep.episode_id for ep in episodes)) == 4  # All unique
        assert session_id in manager.session_episodes
        assert len(manager.session_episodes[session_id]) == 4

        # Test episode-specific operations and isolation
        from unittest.mock import MagicMock

        # Configure each episode with different task settings
        for i, episode in enumerate(episodes):
            mock_task = MagicMock()
            mock_task.task_id = episode.task_id
            mock_task.episode_config = {"max_steps": 10 + i * 5}  # Different limits
            await manager.configure_for_task(episode.episode_id, mock_task)

        # Execute steps on multiple episodes simultaneously
        action = Action(tool_name="test_tool", parameters={"episode_specific": True})
        command_result = CommandResult(exit_code=0, stdout="Episode output", stderr="", execution_time=0.1)

        step_results = []
        for episode in episodes:
            step_result = manager.step(episode.episode_id, action, command_result)
            step_results.append(step_result)

            # Verify step isolation
            assert len(episode.steps) == 1
            assert episode.steps[0].action == action

        # Test selective episode termination
        episode_to_end = episodes[1]  # End vulnerability scan episode
        completed_episode = await manager.end_episode(episode_to_end.episode_id, "completed", "scan_complete")

        # Verify termination isolation
        assert completed_episode.state == EpisodeState.COMPLETED
        assert completed_episode.completion_reason == "completed"
        # Check that the result was stored in the final step submission
        final_step = completed_episode.steps[-1]
        assert final_step.action.tool_name == "submission"
        assert final_step.action.parameters["result"] == "scan_complete"

        # Episode should be moved from active to completed but still tracked in session
        assert episode_to_end.episode_id not in manager.episodes  # Removed from active
        assert episode_to_end.episode_id in manager.completed_episodes  # Moved to completed
        assert episode_to_end.episode_id in manager.session_episodes[session_id]  # Still tracked for session

        # Verify other episodes remain active and unaffected
        for episode in episodes:
            if episode.episode_id != episode_to_end.episode_id:
                assert episode.state == EpisodeState.ACTIVE
                assert episode.episode_id in manager.episodes  # Still active
                assert episode.episode_id in manager.session_episodes[session_id]  # Still tracked

        # Test episode lookup isolation
        for episode in episodes:
            if episode.state == EpisodeState.ACTIVE:
                retrieved = manager.get_episode_by_id(episode.episode_id)
                assert retrieved == episode
                assert retrieved.session_id == session_id
            else:
                # Completed episodes should be retrievable from completed_episodes
                retrieved = manager.completed_episodes.get(episode.episode_id)
                assert retrieved == episode

    @pytest.mark.asyncio
    async def test_concurrent_multi_session_multi_episode_orchestration(self):
        """Test orchestrating multiple episodes across different sessions simultaneously."""
        manager = EpisodeManager()

        # Define multiple session scenarios with different episode types
        session_scenarios = {
            "red_team_session": [
                ("network_discovery", {"scope": "internal", "stealth": True}),
                ("lateral_movement", {"technique": "wmi", "creds": "cached"}),
                ("persistence", {"method": "scheduled_task", "persistence_level": "system"}),
            ],
            "blue_team_session": [
                ("threat_hunting", {"indicators": "network_anomalies", "timeframe": "24h"}),
                ("incident_response", {"alert_type": "malware", "severity": "high"}),
            ],
            "forensics_session": [
                ("memory_analysis", {"dump_path": "/evidence/memory.raw", "profile": "win10"}),
                ("disk_forensics", {"image_path": "/evidence/disk.dd", "filesystem": "ntfs"}),
                ("timeline_analysis", {"start": "2024-01-01", "end": "2024-01-31"}),
            ],
        }

        # Start episodes across all sessions
        all_episodes = {}
        total_episodes = 0

        for session_id, tasks in session_scenarios.items():
            session_episodes = []
            for task_id, context in tasks:
                episode = manager.start_episode(session_id, task_id, context)
                session_episodes.append(episode)
                total_episodes += 1

                # Verify session isolation
                assert episode.session_id == session_id
                assert episode.task_id == task_id
                assert episode.context == context

            all_episodes[session_id] = session_episodes

        # Verify cross-session isolation and proper tracking
        assert len(manager.session_episodes) == 3  # Three distinct sessions
        assert total_episodes == 8  # 3 + 2 + 3 = 8 total episodes

        for session_id, episodes in all_episodes.items():
            assert session_id in manager.session_episodes
            assert len(manager.session_episodes[session_id]) == len(episodes)

            # Verify no episode IDs leak between sessions
            session_episode_ids = set(manager.session_episodes[session_id])
            for other_session_id, other_episodes in all_episodes.items():
                if other_session_id != session_id:
                    other_episode_ids = set(ep.episode_id for ep in other_episodes)
                    assert session_episode_ids.isdisjoint(other_episode_ids)

        # Test concurrent episode operations across sessions
        from unittest.mock import MagicMock
        action = Action(tool_name="cross_session_tool", parameters={"global_op": True})
        command_result = CommandResult(exit_code=0, stdout="Cross-session output", stderr="", execution_time=0.2)

        # Configure and execute steps across all episodes
        for session_id, episodes in all_episodes.items():
            for i, episode in enumerate(episodes):
                # Configure with session-specific settings
                mock_task = MagicMock()
                mock_task.task_id = episode.task_id
                mock_task.episode_config = {"max_steps": 15, f"{session_id}_setting": True}
                await manager.configure_for_task(episode.episode_id, mock_task)

                # Execute step
                step_result = manager.step(episode.episode_id, action, command_result)
                assert len(episode.steps) == 1

        # Test selective session cleanup - end all episodes for blue team
        blue_team_episodes = all_episodes["blue_team_session"].copy()
        for episode in blue_team_episodes:
            completed = await manager.end_episode(episode.episode_id, "completed", f"blue_team_{episode.task_id}_done")
            assert completed.state == EpisodeState.COMPLETED

        # Verify blue team episodes are completed but still tracked in session
        for episode in blue_team_episodes:
            assert episode.episode_id not in manager.episodes  # Removed from active
            assert episode.episode_id in manager.completed_episodes  # Moved to completed
            assert episode.episode_id in manager.session_episodes["blue_team_session"]  # Still tracked

        # Verify other sessions remain unchanged (episodes still active)
        assert len(manager.session_episodes["red_team_session"]) == 3  # Unchanged
        assert len(manager.session_episodes["forensics_session"]) == 3  # Unchanged

        # Test adding new episodes to existing sessions during operations
        urgent_episode = manager.start_episode(
            "red_team_session",
            "emergency_cleanup",
            {"urgency": "critical", "cleanup_traces": True}
        )

        # Verify dynamic episode addition
        assert len(manager.session_episodes["red_team_session"]) == 4  # 3 + 1 new
        assert urgent_episode.episode_id in manager.session_episodes["red_team_session"]
        assert urgent_episode.context["urgency"] == "critical"

        # Verify cross-session episode retrieval and isolation
        for session_id, episodes in all_episodes.items():
            for episode in episodes:
                if episode.state == EpisodeState.ACTIVE:
                    retrieved = manager.get_episode_by_id(episode.episode_id)
                    assert retrieved.session_id == session_id
                    assert retrieved.episode_id in manager.session_episodes[session_id]
                else:
                    # Completed episodes should be retrievable from completed_episodes
                    retrieved = manager.completed_episodes.get(episode.episode_id)
                    assert retrieved == episode
                    assert retrieved.episode_id in manager.session_episodes[session_id]
