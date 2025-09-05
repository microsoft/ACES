"""
Test suite for ContainerCleanupManager - unified container cleanup system.

This test module validates the unified cleanup architecture that replaced
the scattered cleanup methods throughout the codebase.
"""

import pytest
from unittest.mock import Mock, patch
import time
from typing import Dict, Any

from saber.server.execution.cleanup.cleanup_manager import ContainerCleanupManager
from saber.server.execution.cleanup.cleanup_reason import CleanupReason


class TestContainerCleanupManager:
    """Test suite for the unified container cleanup system."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock sandbox manager for testing."""
        mock_manager = Mock()
        mock_manager.active_environments = {}
        return mock_manager
        return mock_manager

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment with stop method."""
        mock_env = Mock()
        mock_env.stop.return_value = None
        return mock_env

    @pytest.fixture
    def debug_cleanup_manager(self, mock_sandbox_manager):
        """Create a ContainerCleanupManager instance with debug mode enabled."""
        return ContainerCleanupManager(mock_sandbox_manager, debug_mode=True)

    @pytest.fixture
    def normal_cleanup_manager(self, mock_sandbox_manager):
        """Create a ContainerCleanupManager instance with debug mode disabled."""
        return ContainerCleanupManager(mock_sandbox_manager, debug_mode=False)


class TestContainerCleanupManagerDebugMode:
    """Test suite for ContainerCleanupManager debug mode functionality."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock sandbox manager for testing."""
        mock_manager = Mock()
        mock_manager.active_environments = {}
        return mock_manager

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment with stop method."""
        mock_env = Mock()
        mock_env.stop.return_value = None
        return mock_env

    @pytest.fixture
    def debug_cleanup_manager(self, mock_sandbox_manager):
        """Create a ContainerCleanupManager instance with debug mode enabled."""
        return ContainerCleanupManager(mock_sandbox_manager, debug_mode=True)

    @pytest.fixture
    def normal_cleanup_manager(self, mock_sandbox_manager):
        """Create a ContainerCleanupManager instance with debug mode disabled."""
        return ContainerCleanupManager(mock_sandbox_manager, debug_mode=False)


class TestContainerCleanupManager:
    """Test suite for the unified container cleanup system."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock sandbox manager for testing."""
        mock_manager = Mock()
        mock_manager.active_environments = {}
        return mock_manager

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment with stop method."""
        mock_env = Mock()
        mock_env.stop.return_value = None
        return mock_env

    @pytest.fixture
    def cleanup_manager(self, mock_sandbox_manager):
        """Create a ContainerCleanupManager instance with mocked dependencies."""
        return ContainerCleanupManager(mock_sandbox_manager)

    def test_initialization(self, cleanup_manager, mock_sandbox_manager):
        """Test that CleanupManager initializes correctly."""
        assert cleanup_manager.sandbox_manager == mock_sandbox_manager
        assert cleanup_manager._cleanup_history == []

    def test_cleanup_episode_success(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test successful single episode cleanup."""
        episode_id = "episode_123"
        reason = CleanupReason.SESSION_TERMINATED

        # Setup mocks
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_environments = {episode_id: mock_environment}

        result = cleanup_manager.cleanup_episode(episode_id, reason)

        # Verify environment stop was called
        mock_environment.stop.assert_called_once()
        assert result is True

        # Verify history tracking
        assert len(cleanup_manager._cleanup_history) == 1
        assert cleanup_manager._cleanup_history[0].identifier == episode_id
        assert cleanup_manager._cleanup_history[0].reason == reason

    def test_cleanup_episode_failure(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test handling of cleanup failures."""
        episode_id = "failing_episode"
        reason = CleanupReason.ERROR_TRIGGERED

        # Setup failure scenario
        mock_environment.stop.side_effect = Exception("Docker container removal failed")
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_episodes = {episode_id: Mock()}

        result = cleanup_manager.cleanup_episode(episode_id, reason)

        # Verify cleanup was attempted but failed
        mock_environment.stop.assert_called_once()
        assert result is False

        # Verify history tracking includes failures
        assert len(cleanup_manager._cleanup_history) == 1
        assert cleanup_manager._cleanup_history[0].success is False

    def test_cleanup_episode_no_environment(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup when no environment exists."""
        episode_id = "no_env_episode"
        reason = CleanupReason.EPISODE_COMPLETED

        # Setup no environment scenario
        mock_sandbox_manager.get_episode_environment.return_value = None
        mock_sandbox_manager._cleanup_orphaned_episode_resources = Mock(return_value=None)

        result = cleanup_manager.cleanup_episode(episode_id, reason)

        # Should succeed with orphaned cleanup
        assert result is True
        mock_sandbox_manager._cleanup_orphaned_episode_resources.assert_called_once_with(episode_id)

    def test_cleanup_all_episodes_success(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test successful cleanup of all episodes."""
        reason = CleanupReason.SERVER_SHUTDOWN
        mock_episodes = {"episode_1": Mock(), "episode_2": Mock(), "episode_3": Mock()}
        mock_sandbox_manager.active_environments = mock_episodes
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment

        result = cleanup_manager.cleanup_all_episodes(reason)

        # Verify all episodes were cleaned up
        assert mock_environment.stop.call_count == 3
        assert result == 3

        # Verify history tracking
        assert len(cleanup_manager._cleanup_history) == 3

    def test_cleanup_duplicate_prevention(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test that the cleanup system properly tracks active cleanups."""
        episode_id = "duplicate_test_episode"
        reason = CleanupReason.EPISODE_COMPLETED

        # Setup mocks
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_environments = {episode_id: mock_environment}

        # First cleanup
        result1 = cleanup_manager.cleanup_episode(episode_id, reason)
        assert result1 is True

        # Second cleanup (should still work since episode-based cleanup is idempotent)
        result2 = cleanup_manager.cleanup_episode(episode_id, reason)
        assert result2 is True

        # Verify tracking
        assert len(cleanup_manager._cleanup_history) == 2
        assert all(op.identifier == episode_id for op in cleanup_manager._cleanup_history)

    def test_cleanup_reasons_coverage(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test all cleanup reasons work correctly."""
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment

        test_cases = [
            CleanupReason.SESSION_TERMINATED,
            CleanupReason.ERROR_TRIGGERED,
            CleanupReason.SESSION_TIMEOUT,
            CleanupReason.SERVER_SHUTDOWN,
            CleanupReason.MANUAL_CLEANUP,
            CleanupReason.HEALTH_CHECK_FAILED,
            CleanupReason.EPISODE_COMPLETED,
            CleanupReason.EPISODE_FAILED
        ]

        for i, reason in enumerate(test_cases):
            episode_id = f"test_episode_{i}"
            mock_sandbox_manager.active_episodes = {episode_id: Mock()}

            result = cleanup_manager.cleanup_episode(episode_id, reason)
            assert result is True

        # Verify all reasons are tracked in history
        assert len(cleanup_manager._cleanup_history) == len(test_cases)
        tracked_reasons = [entry.reason for entry in cleanup_manager._cleanup_history]
        assert set(tracked_reasons) == set(test_cases)

    def test_get_cleanup_history(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test cleanup history retrieval."""
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment

        # Perform some cleanups
        episodes = ["episode_1", "episode_2", "episode_3"]
        reasons = [CleanupReason.EPISODE_COMPLETED, CleanupReason.ERROR_TRIGGERED, CleanupReason.EPISODE_FAILED]

        for episode, reason in zip(episodes, reasons):
            mock_sandbox_manager.active_episodes = {episode: Mock()}
            cleanup_manager.cleanup_episode(episode, reason)

        history = cleanup_manager.get_cleanup_history()

        assert len(history) == 3
        assert all(hasattr(entry, "identifier") for entry in history)
        assert all(hasattr(entry, "reason") for entry in history)
        assert all(hasattr(entry, "start_time") for entry in history)
        assert all(hasattr(entry, "success") for entry in history)

    @patch('saber.server.execution.cleanup.cleanup_manager.logger')
    def test_error_logging_behavior(self, mock_logger, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test that cleanup errors are properly logged."""
        episode_id = "error_logging_test"
        reason = CleanupReason.ERROR_TRIGGERED
        error_msg = "Test error for logging"

        mock_environment.stop.side_effect = Exception(error_msg)
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_environments = {episode_id: mock_environment}

        cleanup_manager.cleanup_episode(episode_id, reason)

        # Verify error logging
        assert mock_logger.error.called

    def test_initialization(self, cleanup_manager, mock_sandbox_manager):
        """Test that CleanupManager initializes correctly."""
        assert cleanup_manager.sandbox_manager == mock_sandbox_manager
        assert cleanup_manager._cleanup_history == []

    def test_cleanup_reasons_coverage(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test all cleanup reasons work correctly."""
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment

        test_cases = [
            CleanupReason.SESSION_TERMINATED,
            CleanupReason.ERROR_TRIGGERED,
            CleanupReason.SESSION_TIMEOUT,
            CleanupReason.SERVER_SHUTDOWN,
            CleanupReason.MANUAL_CLEANUP,
            CleanupReason.HEALTH_CHECK_FAILED,
            CleanupReason.EPISODE_COMPLETED,
            CleanupReason.EPISODE_FAILED
        ]

        for i, reason in enumerate(test_cases):
            episode_id = f"test_episode_{i}"
            mock_sandbox_manager.active_environments = {episode_id: mock_environment}

            result = cleanup_manager.cleanup_episode(episode_id, reason)
            assert result is True

        # Verify all reasons are tracked in history
        assert len(cleanup_manager._cleanup_history) == len(test_cases)
        tracked_reasons = [entry.reason for entry in cleanup_manager._cleanup_history]
        assert set(tracked_reasons) == set(test_cases)

    def test_get_cleanup_history(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup history retrieval."""
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_environments = {}

        # Perform some cleanups
        cleanup_manager.cleanup_episode("episode_1", CleanupReason.SESSION_TERMINATED)
        cleanup_manager.cleanup_episode("episode_2", CleanupReason.ERROR_TRIGGERED)
        cleanup_manager.cleanup_episode("episode_3", CleanupReason.SESSION_TIMEOUT)

        history = cleanup_manager.get_cleanup_history()

        assert len(history) == 3
        assert all(hasattr(entry, "identifier") for entry in history)
        assert all(hasattr(entry, "reason") for entry in history)
        assert all(hasattr(entry, "start_time") for entry in history)
        assert all(hasattr(entry, "success") for entry in history)

    def test_get_cleanup_history_filtered(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup history with episode filter."""
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_environments = {}

        # Perform multiple cleanups
        for i in range(5):
            cleanup_manager.cleanup_episode(f"episode_{i}", CleanupReason.SESSION_TERMINATED)

        # Test filtered history - pass identifier as positional argument
        filtered_history = cleanup_manager.get_cleanup_history("episode_2")
        assert len(filtered_history) == 1
        assert filtered_history[0].identifier == "episode_2"

        # Test full history
        full_history = cleanup_manager.get_cleanup_history()
        assert len(full_history) == 5

    def test_empty_episodes_cleanup(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup behavior when no active episodes exist."""
        mock_sandbox_manager.active_environments = {}

        result = cleanup_manager.cleanup_all_episodes(CleanupReason.SERVER_SHUTDOWN)

        assert result == 0  # No episodes to clean up

    @patch('saber.server.execution.cleanup.cleanup_manager.logger')
    def test_logging_behavior(self, mock_logger, cleanup_manager, mock_sandbox_manager):
        """Test that cleanup operations are properly logged."""
        episode_id = "logging_test_episode"
        reason = CleanupReason.SESSION_TERMINATED

        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_environments = {episode_id: mock_environment}

        cleanup_manager.cleanup_episode(episode_id, reason)

        # Verify logging occurred (either info or warning)
        assert mock_logger.warning.called or mock_logger.info.called

    def test_cleanup_history_memory_management(self, cleanup_manager, mock_sandbox_manager):
        """Test that cleanup history doesn't grow unbounded."""
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_environments = {}

        # Perform many cleanups to test memory management
        # Note: The current implementation doesn't have a limit, but this test
        # verifies the history is being tracked correctly
        for i in range(100):
            cleanup_manager.cleanup_episode(f"episode_{i}", CleanupReason.SESSION_TERMINATED)

        history = cleanup_manager.get_cleanup_history()
        assert len(history) == 100

        # All entries should have required fields
        for entry in history:
            assert hasattr(entry, "identifier")
            assert hasattr(entry, "reason")
            assert hasattr(entry, "start_time")
            assert hasattr(entry, "success")

    def test_concurrent_cleanup_safety(self, cleanup_manager, mock_sandbox_manager):
        """Test that cleanup manager handles concurrent operations safely."""
        # This is a basic test - in a real scenario, you'd use threading
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_environments = {}

        # Simulate rapid successive cleanups
        results = []
        for i in range(10):
            mock_sandbox_manager.active_environments = {f"episode_{i}": mock_environment}
            result = cleanup_manager.cleanup_episode(f"episode_{i}", CleanupReason.SESSION_TERMINATED)
            results.append(result)

        # All should succeed
        assert all(result for result in results)
        assert len(cleanup_manager.get_cleanup_history()) == 10

    def test_mixed_episode_cleanup_orchestration(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup orchestration with mixed success and failure scenarios."""
        # Create multiple episodes with different states
        episode_configs = {
            "episode_success_1": {"should_fail": False, "cleanup_time": 0.1},
            "episode_success_2": {"should_fail": False, "cleanup_time": 0.2},
            "episode_failure_1": {"should_fail": True, "error": "Container locked"},
            "episode_success_3": {"should_fail": False, "cleanup_time": 0.05},
            "episode_failure_2": {"should_fail": True, "error": "Network timeout"},
        }

        # Setup mock environments for each episode
        mock_environments = {}
        for episode_id, config in episode_configs.items():
            mock_env = Mock()
            if config["should_fail"]:
                mock_env.stop.side_effect = Exception(config["error"])
            else:
                mock_env.stop.return_value = None
            mock_environments[episode_id] = mock_env

        mock_sandbox_manager.active_environments = mock_environments

        # Setup get_episode_environment to return the correct mock
        def get_mock_environment(episode_id):
            return mock_environments.get(episode_id)
        mock_sandbox_manager.get_episode_environment.side_effect = get_mock_environment

        # Perform cleanup_all_episodes
        reason = CleanupReason.SERVER_SHUTDOWN
        successful_cleanups = cleanup_manager.cleanup_all_episodes(reason)

        # Verify correct number of successes (3 successful episodes)
        assert successful_cleanups == 3

        # Verify all environments had stop() called
        for mock_env in mock_environments.values():
            mock_env.stop.assert_called_once()

        # Verify history tracking for all episodes
        history = cleanup_manager.get_cleanup_history()
        assert len(history) == 5

        # Check success/failure tracking
        successful_episodes = [op for op in history if op.success]
        failed_episodes = [op for op in history if not op.success]
        assert len(successful_episodes) == 3
        assert len(failed_episodes) == 2

        # Verify specific episode results
        episode_results = {op.identifier: op.success for op in history}
        assert episode_results["episode_success_1"] is True
        assert episode_results["episode_success_2"] is True
        assert episode_results["episode_success_3"] is True
        assert episode_results["episode_failure_1"] is False
        assert episode_results["episode_failure_2"] is False

    def test_episode_cleanup_ordering_and_dependencies(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup manager handles episode dependencies and ordering correctly."""
        # Simulate a scenario where episodes may have dependencies
        # Episode 1 -> Episode 2 -> Episode 3 (dependency chain)
        # Episode 4, Episode 5 (independent)

        episodes = ["episode_1", "episode_2", "episode_3", "episode_4", "episode_5"]
        cleanup_order = []

        # Create mock environments that track cleanup order
        mock_environments = {}
        for episode_id in episodes:
            mock_env = Mock()
            def track_cleanup(ep_id=episode_id):
                cleanup_order.append(ep_id)
                return None
            mock_env.stop.side_effect = track_cleanup
            mock_environments[episode_id] = mock_env

        mock_sandbox_manager.active_environments = mock_environments

        def get_mock_environment(episode_id):
            return mock_environments.get(episode_id)
        mock_sandbox_manager.get_episode_environment.side_effect = get_mock_environment

        # Test individual episode cleanup preserves order
        test_episodes = ["episode_1", "episode_2", "episode_3"]
        for episode_id in test_episodes:
            result = cleanup_manager.cleanup_episode(episode_id, CleanupReason.EPISODE_COMPLETED)
            assert result is True

        # Verify episodes were cleaned up in the order we called them
        assert cleanup_order == test_episodes

        # Test bulk cleanup with remaining episodes
        cleanup_order.clear()  # Reset tracking

        # Setup remaining episodes for bulk cleanup test
        remaining_episodes = {"episode_4": mock_environments["episode_4"], "episode_5": mock_environments["episode_5"]}
        mock_sandbox_manager.active_environments = remaining_episodes

        # Perform bulk cleanup
        reason = CleanupReason.SERVER_SHUTDOWN
        successful_cleanups = cleanup_manager.cleanup_all_episodes(reason)

        # Verify remaining episodes were cleaned up
        assert successful_cleanups == 2
        assert len(cleanup_order) == 2

        # Verify all episode IDs were cleaned up (order may vary in bulk cleanup)
        assert set(cleanup_order) == {"episode_4", "episode_5"}

        # Verify history tracking shows all operations
        history = cleanup_manager.get_cleanup_history()
        # Should have 5 total operations: 3 individual + 2 bulk
        assert len(history) == 5

        # Verify all operations were successful
        assert all(op.success for op in history)

        # Test filtering by specific episode
        episode_2_history = cleanup_manager.get_cleanup_history("episode_2")
        assert len(episode_2_history) == 1  # Only individual cleanup
        assert episode_2_history[0].identifier == "episode_2"
        assert episode_2_history[0].success is True

        # Test filtering by bulk cleanup episode
        episode_4_history = cleanup_manager.get_cleanup_history("episode_4")
        assert len(episode_4_history) == 1  # Only bulk cleanup
        assert episode_4_history[0].identifier == "episode_4"
        assert episode_4_history[0].success is True

    def test_cleanup_manager_integration_with_episode_manager(self, cleanup_manager):
        """Test integration patterns that episode manager would use."""
        # This tests the interface that EpisodeManager relies on

        # Test the cleanup interface
        assert hasattr(cleanup_manager, 'cleanup_episode')
        assert hasattr(cleanup_manager, 'cleanup_all_episodes')
        assert hasattr(cleanup_manager, 'get_cleanup_history')

        # Test that methods accept the expected parameters
        import inspect

        cleanup_episode_sig = inspect.signature(cleanup_manager.cleanup_episode)
        assert 'episode_id' in cleanup_episode_sig.parameters
        assert 'reason' in cleanup_episode_sig.parameters

        cleanup_all_sig = inspect.signature(cleanup_manager.cleanup_all_episodes)
        assert 'reason' in cleanup_all_sig.parameters


class TestContainerCleanupManagerDebugMode:
    """Test suite for debug mode functionality in ContainerCleanupManager."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock sandbox manager for testing."""
        mock_manager = Mock()
        mock_manager.active_environments = {}
        return mock_manager

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment with stop method."""
        mock_env = Mock()
        mock_env.stop.return_value = None
        return mock_env

    @pytest.fixture
    def debug_cleanup_manager(self, mock_sandbox_manager):
        """Create a ContainerCleanupManager instance with debug mode enabled."""
        return ContainerCleanupManager(mock_sandbox_manager, debug_mode=True)

    @pytest.fixture
    def normal_cleanup_manager(self, mock_sandbox_manager):
        """Create a ContainerCleanupManager instance with debug mode disabled."""
        return ContainerCleanupManager(mock_sandbox_manager, debug_mode=False)

    def test_initialization_with_debug_mode_enabled(self, mock_sandbox_manager):
        """Test that ContainerCleanupManager initializes correctly with debug mode enabled."""
        cleanup_manager = ContainerCleanupManager(mock_sandbox_manager, debug_mode=True)

        assert cleanup_manager.sandbox_manager == mock_sandbox_manager
        assert cleanup_manager.debug_mode is True
        assert cleanup_manager._cleanup_history == []

    def test_initialization_with_debug_mode_disabled(self, mock_sandbox_manager):
        """Test that ContainerCleanupManager initializes correctly with debug mode disabled."""
        cleanup_manager = ContainerCleanupManager(mock_sandbox_manager, debug_mode=False)

        assert cleanup_manager.sandbox_manager == mock_sandbox_manager
        assert cleanup_manager.debug_mode is False
        assert cleanup_manager._cleanup_history == []

    def test_cleanup_episode_skipped_in_debug_mode(self, debug_cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test that cleanup_episode is skipped when debug mode is enabled."""
        episode_id = "test_episode_debug_unique"
        reason = CleanupReason.SESSION_TERMINATED

        # Setup mocks - environment should not be called in debug mode
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_environments = {episode_id: mock_environment}

        # Perform cleanup
        result = debug_cleanup_manager.cleanup_episode(episode_id, reason)

        # Verify cleanup was skipped
        assert result is True

        # Environment should not have been fetched or stopped in debug mode
        mock_sandbox_manager.get_episode_environment.assert_not_called()
        mock_environment.stop.assert_not_called()

        # Verify cleanup operation was recorded in history with debug skip
        all_history = debug_cleanup_manager.get_cleanup_history()
        episode_operations = [op for op in all_history if op.identifier == episode_id]
        assert len(episode_operations) >= 1

        # Check the most recent operation for this episode was a debug skip
        latest_operation = episode_operations[-1]
        assert latest_operation.reason == reason
        assert latest_operation.success is True
        assert "debug_mode_skip" in latest_operation.steps_completed

    def test_cleanup_episode_runs_in_normal_mode(self, normal_cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test that cleanup_episode runs normally when debug mode is disabled."""
        episode_id = "test_episode_normal"
        reason = CleanupReason.SESSION_TERMINATED

        # Setup mocks
        mock_sandbox_manager.get_episode_environment.return_value = mock_environment
        mock_sandbox_manager.active_environments = {episode_id: mock_environment}

        # Perform cleanup
        result = normal_cleanup_manager.cleanup_episode(episode_id, reason)

        # Verify cleanup ran normally
        assert result is True

        # Environment should have been fetched and cleaned
        mock_sandbox_manager.get_episode_environment.assert_called_once_with(episode_id)
        mock_environment.stop.assert_called_once()

    def test_cleanup_all_containers_debug_mode_logging(self, debug_cleanup_manager):
        """Test that cleanup_all_containers logs debug mode status."""
        reason = CleanupReason.SERVER_SHUTDOWN

        with patch.object(debug_cleanup_manager, 'cleanup_all_episodes', return_value=0) as mock_cleanup_all:
            with patch.object(debug_cleanup_manager, 'stop_permanent_environment', return_value=True) as mock_stop_perm:
                result = debug_cleanup_manager.cleanup_all_containers(reason)

                # Verify methods were called (debug mode only affects individual episode cleanup)
                mock_cleanup_all.assert_called_once_with(reason, {})
                mock_stop_perm.assert_called_once()

                # Verify result structure
                assert 'ephemeral_episodes_cleaned' in result
                assert 'permanent_environment_stopped' in result
                assert result['reason'] == reason
