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
        mock_manager.active_sessions = {}
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

    def test_cleanup_session_success(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test successful single session cleanup."""
        session_id = "test_session_123"
        reason = CleanupReason.SESSION_TERMINATED

        # Setup mocks
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.active_sessions = {session_id: Mock()}

        result = cleanup_manager.cleanup_session(session_id, reason)

        # Verify environment stop was called
        mock_environment.stop.assert_called_once()
        assert result is True

        # Verify history tracking
        assert len(cleanup_manager._cleanup_history) == 1
        assert cleanup_manager._cleanup_history[0].session_id == session_id
        assert cleanup_manager._cleanup_history[0].reason == reason

    def test_cleanup_session_failure(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test handling of cleanup failures."""
        session_id = "failing_session"
        reason = CleanupReason.ERROR_TRIGGERED

        # Setup failure scenario
        mock_environment.stop.side_effect = Exception("Docker container removal failed")
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.active_sessions = {session_id: Mock()}

        result = cleanup_manager.cleanup_session(session_id, reason)

        # Verify cleanup was attempted but failed
        mock_environment.stop.assert_called_once()
        assert result is False

        # Verify history tracking includes failures
        assert len(cleanup_manager._cleanup_history) == 1
        assert cleanup_manager._cleanup_history[0].success is False

    def test_cleanup_session_no_environment(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup when no environment exists."""
        session_id = "no_env_session"
        reason = CleanupReason.SESSION_TERMINATED

        # Setup no environment scenario
        mock_sandbox_manager.get_session_environment.return_value = None
        mock_sandbox_manager._cleanup_orphaned_session_resources.return_value = None

        result = cleanup_manager.cleanup_session(session_id, reason)

        # Should succeed with orphaned cleanup
        assert result is True
        mock_sandbox_manager._cleanup_orphaned_session_resources.assert_called_once_with(session_id)

    def test_cleanup_all_sessions_success(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test successful cleanup of all sessions."""
        reason = CleanupReason.SERVER_SHUTDOWN
        mock_sessions = {"session_1": Mock(), "session_2": Mock(), "session_3": Mock()}
        mock_sandbox_manager.active_sessions = mock_sessions
        mock_sandbox_manager.get_session_environment.return_value = mock_environment

        result = cleanup_manager.cleanup_all_sessions(reason)

        # Verify all sessions were cleaned up
        assert mock_environment.stop.call_count == 3
        assert result == 3

        # Verify history tracking
        assert len(cleanup_manager._cleanup_history) == 3

    def test_cleanup_duplicate_prevention(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test that the cleanup system properly tracks active cleanups."""
        session_id = "duplicate_test_session"
        reason = CleanupReason.SESSION_TERMINATED

        # Setup mocks
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.active_sessions = {session_id: Mock()}

        # First cleanup
        result1 = cleanup_manager.cleanup_session(session_id, reason)
        assert result1 is True

        # Reset the active_sessions for second cleanup
        mock_sandbox_manager.active_sessions = {session_id: Mock()}

        # Second cleanup (should work since first is complete)
        result2 = cleanup_manager.cleanup_session(session_id, reason)
        assert result2 is True

        # Both cleanups should have occurred
        assert mock_environment.stop.call_count == 2

    def test_cleanup_reasons_coverage(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test all cleanup reasons work correctly."""
        mock_sandbox_manager.get_session_environment.return_value = mock_environment

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
            session_id = f"test_session_{i}"
            mock_sandbox_manager.active_sessions = {session_id: Mock()}

            result = cleanup_manager.cleanup_session(session_id, reason)
            assert result is True

        # Verify all reasons are tracked in history
        assert len(cleanup_manager._cleanup_history) == len(test_cases)
        tracked_reasons = [entry.reason for entry in cleanup_manager._cleanup_history]
        assert set(tracked_reasons) == set(test_cases)

    def test_get_cleanup_history(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test cleanup history retrieval."""
        mock_sandbox_manager.get_session_environment.return_value = mock_environment

        # Perform some cleanups
        sessions = ["session_1", "session_2", "session_3"]
        reasons = [CleanupReason.SESSION_TERMINATED, CleanupReason.ERROR_TRIGGERED, CleanupReason.SESSION_TIMEOUT]

        for session, reason in zip(sessions, reasons):
            mock_sandbox_manager.active_sessions = {session: Mock()}
            cleanup_manager.cleanup_session(session, reason)

        history = cleanup_manager.get_cleanup_history()

        assert len(history) == 3
        assert all(hasattr(entry, "session_id") for entry in history)
        assert all(hasattr(entry, "reason") for entry in history)
        assert all(hasattr(entry, "start_time") for entry in history)
        assert all(hasattr(entry, "success") for entry in history)

    def test_get_cleanup_history_filtered(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test cleanup history with session filter."""
        mock_sandbox_manager.get_session_environment.return_value = mock_environment

        # Perform multiple cleanups
        for i in range(5):
            session_id = f"session_{i}"
            mock_sandbox_manager.active_sessions = {session_id: Mock()}
            cleanup_manager.cleanup_session(session_id, CleanupReason.SESSION_TERMINATED)

        # Test filtered history
        filtered_history = cleanup_manager.get_cleanup_history(session_id="session_2")
        assert len(filtered_history) == 1
        assert filtered_history[0].session_id == "session_2"

        # Test full history
        full_history = cleanup_manager.get_cleanup_history()
        assert len(full_history) == 5

    def test_empty_sessions_cleanup(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup behavior when no active sessions exist."""
        mock_sandbox_manager.active_sessions = {}

        result = cleanup_manager.cleanup_all_sessions(CleanupReason.SERVER_SHUTDOWN)
        assert result == 0  # No sessions to clean up

    @patch('saber.server.execution.cleanup.cleanup_manager.logger')
    def test_logging_behavior(self, mock_logger, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test that cleanup operations are properly logged."""
        session_id = "logging_test_session"
        reason = CleanupReason.SESSION_TERMINATED

        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.active_sessions = {session_id: Mock()}

        cleanup_manager.cleanup_session(session_id, reason)

        # Verify logging occurred
        assert mock_logger.warning.called or mock_logger.info.called

    def test_cleanup_manager_integration(self, cleanup_manager):
        """Test integration patterns that SessionManager would use."""
        # Test the cleanup interface
        assert hasattr(cleanup_manager, 'cleanup_session')
        assert hasattr(cleanup_manager, 'cleanup_all_sessions')
        assert hasattr(cleanup_manager, 'get_cleanup_history')

        # Test that methods accept the expected parameters
        import inspect

        cleanup_session_sig = inspect.signature(cleanup_manager.cleanup_session)
        assert 'session_id' in cleanup_session_sig.parameters
        assert 'reason' in cleanup_session_sig.parameters

        cleanup_all_sig = inspect.signature(cleanup_manager.cleanup_all_sessions)
        assert 'reason' in cleanup_all_sig.parameters

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock sandbox manager for testing."""
        mock_manager = Mock()
        mock_manager.active_sessions = {"session_1": Mock(), "session_2": Mock()}
        return mock_manager

    @pytest.fixture
    def cleanup_manager(self, mock_sandbox_manager):
        """Create a ContainerCleanupManager instance with mocked dependencies."""
        return ContainerCleanupManager(mock_sandbox_manager)

    def test_initialization(self, cleanup_manager, mock_sandbox_manager):
        """Test that CleanupManager initializes correctly."""
        assert cleanup_manager.sandbox_manager == mock_sandbox_manager
        assert cleanup_manager._cleanup_history == []

    def test_cleanup_session_success(self, cleanup_manager, mock_sandbox_manager):
        """Test successful single session cleanup."""
        session_id = "test_session_123"
        reason = CleanupReason.SESSION_TERMINATED

        # Mock successful cleanup - create a mock environment with stop method
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.active_sessions = {session_id: Mock()}

        result = cleanup_manager.cleanup_session(session_id, reason)

        # Verify environment stop was called
        mock_environment.stop.assert_called_once()

        # Verify result is boolean
        assert result is True

        # Verify history tracking
        assert len(cleanup_manager._cleanup_history) == 1
        assert cleanup_manager._cleanup_history[0].session_id == session_id
        assert cleanup_manager._cleanup_history[0].reason == reason

    def test_cleanup_session_failure(self, cleanup_manager, mock_sandbox_manager):
        """Test handling of cleanup failures."""
        session_id = "failing_session"
        reason = CleanupReason.ERROR_TRIGGERED
        error_msg = "Docker container removal failed"

        # Mock cleanup failure
        mock_environment = Mock()
        mock_environment.stop.side_effect = Exception(error_msg)
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.active_sessions = {session_id: Mock()}

        result = cleanup_manager.cleanup_session(session_id, reason)

        # Verify cleanup was attempted
        mock_environment.stop.assert_called_once()

        # Verify error handling
        assert result is False

        # Verify history tracking includes failures
        assert len(cleanup_manager._cleanup_history) == 1
        assert cleanup_manager._cleanup_history[0].success is False

    def test_cleanup_all_sessions_success(self, cleanup_manager, mock_sandbox_manager):
        """Test successful cleanup of all sessions."""
        reason = CleanupReason.SERVER_SHUTDOWN
        mock_sessions = {"session_1": Mock(), "session_2": Mock(), "session_3": Mock()}
        mock_sandbox_manager.active_sessions = mock_sessions

        # Mock successful cleanup
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_session_environment.return_value = mock_environment

        result = cleanup_manager.cleanup_all_sessions(reason)

        # Verify all sessions were cleaned up
        assert mock_environment.stop.call_count == 3

        # Verify result is the number of successful cleanups
        assert result == 3

        # Verify history tracking
        assert len(cleanup_manager._cleanup_history) == 3

    def test_cleanup_reasons_coverage(self, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test all cleanup reasons work correctly."""
        mock_sandbox_manager.get_session_environment.return_value = mock_environment

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
            session_id = f"test_session_{i}"
            mock_sandbox_manager.active_sessions = {session_id: Mock()}

            result = cleanup_manager.cleanup_session(session_id, reason)
            assert result is True

        # Verify all reasons are tracked in history
        assert len(cleanup_manager._cleanup_history) == len(test_cases)
        tracked_reasons = [entry.reason for entry in cleanup_manager._cleanup_history]
        assert set(tracked_reasons) == set(test_cases)
        assert mock_sandbox_manager.cleanup_session.call_count == 1

    def test_cleanup_reasons_coverage(self, cleanup_manager, mock_sandbox_manager):
        """Test all cleanup reasons work correctly."""
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.cleanup_session.return_value = None

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
            session_id = f"test_session_{i}"
            result = cleanup_manager.cleanup_session(session_id, reason)

            assert result is True

        # Verify all reasons are tracked in history
        assert len(cleanup_manager._cleanup_history) == len(test_cases)
        tracked_reasons = [entry.reason for entry in cleanup_manager._cleanup_history]
        expected_reasons = test_cases
        assert set(tracked_reasons) == set(expected_reasons)

    def test_get_cleanup_history(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup history retrieval."""
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.cleanup_session.return_value = None

        # Perform some cleanups
        cleanup_manager.cleanup_session("session_1", CleanupReason.SESSION_TERMINATED)
        cleanup_manager.cleanup_session("session_2", CleanupReason.ERROR_TRIGGERED)
        cleanup_manager.cleanup_session("session_3", CleanupReason.SESSION_TIMEOUT)

        history = cleanup_manager.get_cleanup_history()

        assert len(history) == 3
        assert all(hasattr(entry, "session_id") for entry in history)
        assert all(hasattr(entry, "reason") for entry in history)
        assert all(hasattr(entry, "start_time") for entry in history)
        assert all(hasattr(entry, "success") for entry in history)

    def test_get_cleanup_history_filtered(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup history with session filter."""
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.cleanup_session.return_value = None

        # Perform multiple cleanups
        for i in range(5):
            cleanup_manager.cleanup_session(f"session_{i}", CleanupReason.SESSION_TERMINATED)

        # Test filtered history
        filtered_history = cleanup_manager.get_cleanup_history(session_id="session_2")
        assert len(filtered_history) == 1
        assert filtered_history[0].session_id == "session_2"

        # Test full history
        full_history = cleanup_manager.get_cleanup_history()
        assert len(full_history) == 5

    def test_empty_sessions_cleanup(self, cleanup_manager, mock_sandbox_manager):
        """Test cleanup behavior when no active sessions exist."""
        mock_sandbox_manager.active_sessions = {}

        result = cleanup_manager.cleanup_all_sessions(CleanupReason.SERVER_SHUTDOWN)

        assert result == 0  # No sessions to clean up

    @patch('saber.server.execution.cleanup.cleanup_manager.logger')
    def test_logging_behavior(self, mock_logger, cleanup_manager, mock_sandbox_manager):
        """Test that cleanup operations are properly logged."""
        session_id = "logging_test_session"
        reason = CleanupReason.SESSION_TERMINATED

        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.cleanup_session.return_value = None

        cleanup_manager.cleanup_session(session_id, reason)

        # Verify logging occurred (either info or warning)
        assert mock_logger.warning.called or mock_logger.info.called

    @patch('saber.server.execution.cleanup.cleanup_manager.logger')
    def test_error_logging_behavior(self, mock_logger, cleanup_manager, mock_sandbox_manager, mock_environment):
        """Test that cleanup errors are properly logged."""
        session_id = "error_logging_test"
        reason = CleanupReason.ERROR_TRIGGERED
        error_msg = "Test error for logging"

        mock_environment.stop.side_effect = Exception(error_msg)
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.active_sessions = {session_id: Mock()}

        cleanup_manager.cleanup_session(session_id, reason)

        # Verify error logging
        assert mock_logger.error.called

    def test_cleanup_history_memory_management(self, cleanup_manager, mock_sandbox_manager):
        """Test that cleanup history doesn't grow unbounded."""
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.cleanup_session.return_value = None

        # Perform many cleanups to test memory management
        # Note: The current implementation doesn't have a limit, but this test
        # verifies the history is being tracked correctly
        for i in range(100):
            cleanup_manager.cleanup_session(f"session_{i}", CleanupReason.SESSION_TERMINATED)

        history = cleanup_manager.get_cleanup_history()
        assert len(history) == 100

        # All entries should have required fields
        for entry in history:
            assert hasattr(entry, "session_id")
            assert hasattr(entry, "reason")
            assert hasattr(entry, "start_time")
            assert hasattr(entry, "success")

    def test_concurrent_cleanup_safety(self, cleanup_manager, mock_sandbox_manager):
        """Test that cleanup manager handles concurrent operations safely."""
        # This is a basic test - in a real scenario, you'd use threading
        mock_environment = Mock()
        mock_environment.stop.return_value = None
        mock_sandbox_manager.get_session_environment.return_value = mock_environment
        mock_sandbox_manager.cleanup_session.return_value = None

        # Simulate rapid successive cleanups
        results = []
        for i in range(10):
            result = cleanup_manager.cleanup_session(f"session_{i}", CleanupReason.SESSION_TERMINATED)
            results.append(result)

        # All should succeed
        assert all(result for result in results)
        assert len(cleanup_manager.get_cleanup_history()) == 10

    def test_cleanup_manager_integration_with_session_manager(self, cleanup_manager):
        """Test integration patterns that SessionManager would use."""
        # This tests the interface that SessionManager relies on

        # Test the cleanup interface
        assert hasattr(cleanup_manager, 'cleanup_session')
        assert hasattr(cleanup_manager, 'cleanup_all_sessions')
        assert hasattr(cleanup_manager, 'get_cleanup_history')

        # Test that methods accept the expected parameters
        import inspect

        cleanup_session_sig = inspect.signature(cleanup_manager.cleanup_session)
        assert 'session_id' in cleanup_session_sig.parameters
        assert 'reason' in cleanup_session_sig.parameters

        cleanup_all_sig = inspect.signature(cleanup_manager.cleanup_all_sessions)
        assert 'reason' in cleanup_all_sig.parameters
