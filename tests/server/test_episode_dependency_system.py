"""
Test episode dependency system functionality.

Tests the automatic episode attachment based on depends_on_task_id configuration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from saber.server.episodes.episode_manager import EpisodeManager
from saber.server.session_manager import SessionManager
from saber.server.base import Episode, EpisodeState
from saber.server.benchmarks.task import Task


class TestEpisodeDependencySystem:
    """Test automatic episode dependency resolution."""

    @pytest.fixture
    def mock_benchmark_manager(self):
        """Create a mock benchmark manager."""
        manager = MagicMock()

        # Create mock tasks
        independent_task = Task(
            task_id="independent_task",
            domain="test_domain",
            title="Independent Task",
            description="A task with no dependencies",
            prompts={
                "instruction": "test_instruction.md",
                "assistant": "test_assistant.md",
                "submit": "test_submit.md"
            },
            initial_context={"test": "context"},
            depends_on_task_id=None,
            episode_config={"max_steps": 10}
        )

        dependent_task = Task(
            task_id="dependent_task",
            domain="test_domain",
            title="Dependent Task",
            description="A task that depends on independent_task",
            prompts={
                "instruction": "test_instruction.md",
                "assistant": "test_assistant.md",
                "submit": "test_submit.md"
            },
            initial_context={"test": "context"},
            depends_on_task_id="independent_task",
            episode_config={"max_steps": 10}
        )

        another_dependent_task = Task(
            task_id="another_dependent_task",
            domain="test_domain",
            title="Another Dependent Task",
            description="Another task that depends on independent_task",
            prompts={
                "instruction": "test_instruction.md",
                "assistant": "test_assistant.md",
                "submit": "test_submit.md"
            },
            initial_context={"test": "context"},
            depends_on_task_id="independent_task",
            episode_config={"max_steps": 10}
        )

        circular_task = Task(
            task_id="circular_task",
            domain="test_domain",
            title="Circular Task",
            description="A task that creates circular dependency",
            prompts={
                "instruction": "test_instruction.md",
                "assistant": "test_assistant.md",
                "submit": "test_submit.md"
            },
            initial_context={"test": "context"},
            depends_on_task_id="circular_task",  # Self-reference
            episode_config={"max_steps": 10}
        )

        manager.get_task.side_effect = lambda task_id: {
            "independent_task": independent_task,
            "dependent_task": dependent_task,
            "another_dependent_task": another_dependent_task,
            "circular_task": circular_task
        }.get(task_id)

        # Mock get_task_prompt method
        manager.get_task_prompt.return_value = "Test prompt for task"

        return manager

    @pytest.fixture
    def episode_manager(self):
        """Create episode manager instance."""
        return EpisodeManager()

    @pytest.fixture
    def mock_session(self):
        """Create a mock session."""
        session = MagicMock()
        session.session_id = "test_session"
        session.active_episode_ids = []
        session.add_active_episode = MagicMock()
        session.remove_active_episode = MagicMock()
        session.update_activity = MagicMock()
        return session

    @pytest.fixture
    def fast_dependency_config(self):
        """Provide fast dependency configuration for test performance."""
        return {
            "wait_seconds": 1.0,  # Fast timeout for tests
            "retry_interval": 0.1,  # Quick retries
            "max_retry_interval": 0.2  # Low max backoff
        }

    @pytest.fixture
    def session_manager(self, episode_manager, mock_benchmark_manager, fast_dependency_config):
        """Create session manager with mocked dependencies."""
        with patch('saber.server.session_manager.ExecutionManager'), \
             patch('saber.server.session_manager.PolicyManager'), \
             patch('saber.server.session_manager.EvaluationManager'), \
             patch('saber.server.session_manager.BenchmarkManager') as mock_bm_class:

            # Make the BenchmarkManager constructor return our mock
            mock_bm_class.return_value = mock_benchmark_manager

            # Configure fast dependency config for tests
            mock_benchmark_manager.get_dependency_config.return_value = fast_dependency_config

            manager = SessionManager(
                domain_name="test_domain",
                config_dir="/tmp/test"
            )
            manager.episode_manager = episode_manager
            manager.execution_manager = MagicMock()
            manager.execution_manager.configure_for_task = MagicMock()
            manager.policy_manager = MagicMock()
            manager.policy_manager.set_episode_policy = MagicMock()
            manager.evaluation_manager = MagicMock()
            manager.evaluation_manager.configure_for_task = MagicMock()
            manager.evaluation_manager.log_episode_start = AsyncMock()

            # Mock _get_session method
            mock_session = MagicMock()
            mock_session.session_id = "test_session"
            mock_session.active_episode_ids = []
            mock_session.add_active_episode = MagicMock()
            mock_session.update_activity = MagicMock()
            manager._get_session = MagicMock(return_value=mock_session)

            return manager

    @pytest.mark.asyncio
    async def test_independent_task_creates_episode_without_attachment(self, session_manager):
        """Test that tasks without dependencies create episodes normally."""
        # Create episode for independent task
        episode = await session_manager.start_episode("test_session", "independent_task")

        # Verify episode created successfully
        assert episode.task_id == "independent_task"
        assert episode.depends_on_task_id is None
        assert episode.attached_to_episode_id is None
        assert len(episode.attached_episode_ids) == 0
        assert episode.state == EpisodeState.ACTIVE

    @pytest.mark.asyncio
    async def test_dependent_task_attaches_to_running_episode(self, session_manager):
        """Test that dependent tasks automatically attach to running episodes."""
        # First, create an independent episode
        independent_episode = await session_manager.start_episode("test_session", "independent_task")

        # Now create a dependent episode - should auto-attach
        dependent_episode = await session_manager.start_episode("test_session", "dependent_task")

        # Verify dependency attachment
        assert dependent_episode.task_id == "dependent_task"
        assert dependent_episode.depends_on_task_id == "independent_task"
        assert dependent_episode.attached_to_episode_id == independent_episode.episode_id

        # Verify the independent episode tracks the attachment
        updated_independent = session_manager.episode_manager.get_episode_by_id(independent_episode.episode_id)
        assert dependent_episode.episode_id in updated_independent.attached_episode_ids

    @pytest.mark.asyncio
    async def test_dependent_task_fails_when_no_running_episodes_available(self, session_manager):
        """Test that dependent tasks fail when no target episodes are running."""
        # Try to create dependent episode without any independent episodes running
        with pytest.raises(ValueError) as exc_info:
            await session_manager.start_episode("test_session", "dependent_task")

        # Verify error message is clear
        assert "no available episodes with required dependency task_id independent_task after waiting 1.0s" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_multiple_dependent_episodes_can_attach_to_same_target(self, session_manager):
        """Test that multiple episodes with different task_ids can attach to the same target episode."""
        # Create an independent episode
        independent_episode = await session_manager.start_episode("test_session", "independent_task")

        # Create first dependent episode
        dependent_episode_1 = await session_manager.start_episode("test_session", "dependent_task")

        # Create second dependent episode with different task_id - should still be able to attach
        dependent_episode_2 = await session_manager.start_episode("test_session", "another_dependent_task")

        # Both should attach to the same independent episode
        assert dependent_episode_1.attached_to_episode_id == independent_episode.episode_id
        assert dependent_episode_2.attached_to_episode_id == independent_episode.episode_id

        # Independent episode should track both attachments
        updated_independent = session_manager.episode_manager.get_episode_by_id(independent_episode.episode_id)
        assert dependent_episode_1.episode_id in updated_independent.attached_episode_ids
        assert dependent_episode_2.episode_id in updated_independent.attached_episode_ids
        assert len(updated_independent.attached_episode_ids) == 2

    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, session_manager):
        """Test that circular dependencies are detected and prevented."""
        # Try to create episode with circular dependency
        with pytest.raises(ValueError) as exc_info:
            await session_manager.start_episode("test_session", "circular_task")

        # Verify circular dependency is detected
        assert "Circular dependency detected" in str(exc_info.value)
        assert "cannot depend on itself" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_dependency_only_attaches_to_active_episodes(self, session_manager, episode_manager):
        """Test that dependencies only attach to ACTIVE episodes, not completed ones."""
        # Create an independent episode and manually mark it as completed
        independent_episode = await session_manager.start_episode("test_session", "independent_task")
        independent_episode.state = EpisodeState.COMPLETED

        # Try to create dependent episode - should fail since target is not ACTIVE
        with pytest.raises(ValueError) as exc_info:
            await session_manager.start_episode("test_session", "dependent_task")

        # Verify it fails due to no available episodes
        assert "no available episodes with required dependency task_id independent_task after waiting 1.0s" in str(exc_info.value)

    def test_episode_dependency_lookup_logic(self, episode_manager):
        """Test the episode dependency lookup logic directly."""
        # Create some test episodes
        session_id = "test_session"

        # Create independent episode
        independent_episode = Episode(
            task_id="independent_task",
            session_id=session_id,
            state=EpisodeState.ACTIVE
        )
        episode_manager.add_episode_to_session(session_id, independent_episode)

        # Create another independent episode
        independent_episode_2 = Episode(
            task_id="independent_task",
            session_id=session_id,
            state=EpisodeState.ACTIVE
        )
        episode_manager.add_episode_to_session(session_id, independent_episode_2)

        # Test finding available episode
        found_episode_id = episode_manager.find_available_episode_for_dependency(
            session_id=session_id,
            target_task_id="independent_task",
            dependent_task_id="dependent_task"
        )

        # Should find one of the independent episodes
        assert found_episode_id in [independent_episode.episode_id, independent_episode_2.episode_id]

        # Test that it prevents circular dependencies
        with pytest.raises(ValueError) as exc_info:
            episode_manager.find_available_episode_for_dependency(
                session_id=session_id,
                target_task_id="same_task",
                dependent_task_id="same_task"
            )
        assert "Circular dependency detected" in str(exc_info.value)

    def test_episode_attachment_tracking(self, episode_manager):
        """Test that episode attachment tracking works correctly."""
        session_id = "test_session"

        # Create episodes
        target_episode = Episode(
            task_id="target_task",
            session_id=session_id,
            state=EpisodeState.ACTIVE
        )
        dependent_episode = Episode(
            task_id="dependent_task",
            session_id=session_id,
            state=EpisodeState.ACTIVE,
            depends_on_task_id="target_task"
        )

        episode_manager.add_episode_to_session(session_id, target_episode)
        episode_manager.add_episode_to_session(session_id, dependent_episode)

        # Attach episodes
        episode_manager.attach_episode_to_episode(dependent_episode.episode_id, target_episode.episode_id)

        # Verify attachment tracking
        assert dependent_episode.attached_to_episode_id == target_episode.episode_id
        assert dependent_episode.episode_id in target_episode.attached_episode_ids

        # Test prevention of duplicate attachments from same task type
        another_dependent = Episode(
            task_id="dependent_task",
            session_id=session_id,
            state=EpisodeState.ACTIVE,
            depends_on_task_id="target_task"
        )
        episode_manager.add_episode_to_session(session_id, another_dependent)

        # Should not find target episode since it already has attachment from dependent_task
        found_episode_id = episode_manager.find_available_episode_for_dependency(
            session_id=session_id,
            target_task_id="target_task",
            dependent_task_id="dependent_task"
        )

        # Should return None since target already has attachment from dependent_task
        assert found_episode_id is None

    @pytest.mark.asyncio
    async def test_same_task_id_cannot_attach_twice_to_same_target(self, session_manager):
        """Test that only one episode per task_id can attach to a target episode."""
        # Create an independent episode
        independent_episode = await session_manager.start_episode("test_session", "independent_task")

        # Create first dependent episode
        dependent_episode_1 = await session_manager.start_episode("test_session", "dependent_task")

        # Verify first episode attached successfully
        assert dependent_episode_1.attached_to_episode_id == independent_episode.episode_id

        # Try to create second episode with same task_id - should fail
        with pytest.raises(ValueError, match="no available episodes with required dependency task_id independent_task after waiting 1.0s"):
            await session_manager.start_episode("test_session", "dependent_task")

    @pytest.mark.asyncio
    async def test_dependency_retry_logic_with_delayed_episode(self, session_manager):
        """Test that retry logic successfully finds dependencies that become available during wait period."""
        import asyncio

        async def create_dependency_after_delay():
            """Create the required dependency after a short delay."""
            await asyncio.sleep(0.3)  # Wait 300ms then create dependency
            return await session_manager.start_episode("test_session", "independent_task")

        async def create_dependent_episode():
            """Try to create dependent episode (will retry until dependency is found)."""
            return await session_manager.start_episode("test_session", "dependent_task")

        # Start both tasks concurrently
        dependency_task = asyncio.create_task(create_dependency_after_delay())
        dependent_task = asyncio.create_task(create_dependent_episode())

        # Wait for both to complete
        independent_episode, dependent_episode = await asyncio.gather(dependency_task, dependent_task)

        # Verify the dependency attachment worked
        assert dependent_episode.attached_to_episode_id == independent_episode.episode_id
        assert dependent_episode.episode_id in independent_episode.attached_episode_ids
